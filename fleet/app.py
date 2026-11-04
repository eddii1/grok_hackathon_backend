import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from dotenv import dotenv_values
from pydantic import TypeAdapter, ValidationError

from .agent import AgentTools, StubAgent
from .contracts import ConfirmResponse, IncomingEvent, StopAll
from .perception import annotate, detect
from .registry import DeviceRegistry
from .grok import GrokAgent, XaiProvider
from .capture import CaptureRequest, CaptureService
from .range_tests import GroundReconnectRequest, RangeTestRequest, RangeTests
from .scans import ScanRequest, ScanService
from .pilot import PilotService, register_pilot

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
incoming = TypeAdapter(IncomingEvent)


class EventHub:
    def __init__(self):
        self.queues: set[asyncio.Queue] = set()

    async def emit(self, event: dict):
        for queue in tuple(self.queues):
            if queue.full():
                # Disconnect slow clients instead of blocking STOP and the simulator.
                self.queues.discard(queue)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(None)
            else:
                queue.put_nowait(event)


def create_app(config_path: Path | None = None, media_dir: Path | None = None,
               agent_mode: str | None = None, provider=None) -> FastAPI:
    settings = {**dotenv_values(ROOT / ".env"), **os.environ}
    key = settings.get("XAI_API_KEY", "")
    mode = agent_mode or settings.get("FLEET_AGENT", "auto")
    if mode not in {"auto", "grok", "stub"}:
        raise ValueError("FLEET_AGENT must be auto, grok or stub")
    mode = ("grok" if key or provider else "stub") if mode == "auto" else mode
    model = settings.get("XAI_MODEL") or "grok-4.7"
    config_path = Path(config_path or os.getenv("FLEET_CONFIG", ROOT / "devices.json")).resolve()
    media_dir = Path(media_dir or os.getenv("FLEET_MEDIA_DIR", ROOT / "runtime")).resolve()
    hub = EventHub()
    missions: set[asyncio.Task] = set()
    agents: set[GrokAgent] = set()

    async def cancel_missions():
        tasks = tuple(missions)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    @asynccontextmanager
    async def lifespan(app):
        app.state.provider = provider
        if mode == "grok" and provider is None and key:
            app.state.provider = XaiProvider(key, model)
        registry = DeviceRegistry.from_config(config_path)
        try:
            await registry.start(strict=False)
        except BaseException:
            if app.state.provider is not None:
                await app.state.provider.close()
            raise
        app.state.registry = registry
        app.state.captures = CaptureService(registry, app.state.provider, media_dir, model)
        app.state.range_tests = RangeTests(registry, media_dir)
        app.state.scans = ScanService(registry, app.state.provider, media_dir, model)
        app.state.pilot = PilotService(registry,media_dir,app.state.provider,model)
        pilot_persistence = asyncio.create_task(app.state.pilot.background())
        app.state.cancel_missions = cancel_missions
        async def tick():
            previous = time.monotonic()
            count = 0
            while True:
                now = time.monotonic()
                for device in registry.devices.values():
                    device.tick(min(now - previous, .2))
                previous = now
                if count % 10 == 0:
                    for state in registry.list_states():
                        await hub.emit({"type": "device_state", "device": state})
                count += 1
                await asyncio.sleep(.05)
        ticker = asyncio.create_task(tick())
        try:
            yield
        finally:
            app.state.pilot.stop_all()
            pilot_persistence.cancel()
            await asyncio.gather(pilot_persistence,return_exceptions=True)
            ticker.cancel()
            with suppress(asyncio.CancelledError):
                await ticker
            await cancel_missions()
            await app.state.pilot.close()
            await app.state.scans.stop_all()
            await app.state.captures.cancel_all()
            await app.state.range_tests.cancel_all()
            await registry.stop_all()
            for device in registry.devices.values():
                device.close()
            if app.state.provider is not None:
                await app.state.provider.close()

    app = FastAPI(title="Grok Fleet Backend", version="0.1.0", lifespan=lifespan)
    for name in ("photos", "snapshots"):
        (media_dir / name).mkdir(parents=True, exist_ok=True)
        app.mount(f"/{name}", StaticFiles(directory=media_dir / name), name=name)
    app.mount("/test-assets", StaticFiles(directory=ROOT / "fleet" / "web"), name="test-assets")

    @app.get("/", include_in_schema=False)
    async def test_console():
        return FileResponse(ROOT / "fleet" / "web" / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=204)

    @app.get("/health")
    async def health():
        states = app.state.registry.list_states()
        real_count = sum(s.get('simulated') is False for s in states)
        device_mode = ('real' if real_count == len(states) else 'mixed') if real_count else 'simulator'
        return {"status": "ok", "agent": mode, "mode": device_mode, "model": model if mode == "grok" else None,
                "agent_ready": mode == "stub" or app.state.provider is not None,
                "message": "" if app.state.provider else "Add XAI_API_KEY to backend/.env and restart for real Grok missions."}

    @app.get("/devices")
    async def devices():
        return app.state.registry.list_states()

    @app.post('/captures', status_code=202)
    async def create_capture(request: CaptureRequest):
        if any(not task.done() for task in missions) or app.state.scans.active or app.state.pilot.active or app.state.registry._reconnect_lock.locked():
            raise HTTPException(409, 'Another mission is active')
        return app.state.captures.create(request)

    @app.get('/captures/{capture_id}')
    async def get_capture(capture_id: str):
        return app.state.captures.get(capture_id)

    @app.post('/captures/{capture_id}/cancel')
    async def cancel_capture(capture_id: str):
        return await app.state.captures.cancel(capture_id)

    @app.post('/devices/{device_id}/reconnect')
    async def reconnect_device(device_id: str, request: GroundReconnectRequest):
        if not request.grounded:
            raise HTTPException(422, 'Confirm the drone is grounded before reconnecting')
        if app.state.captures.active or app.state.scans.active or app.state.pilot.active or any(not task.done() for task in missions):
            raise HTTPException(409, 'Cancel active capture/ToF tests before reconnecting')
        try:
            return await app.state.registry.reconnect_ground(device_id)
        except KeyError:
            raise HTTPException(404, 'Unknown device')
        except ValueError as exc:
            raise HTTPException(409, str(exc))
        except Exception:
            raise HTTPException(503, 'Ground reconnect failed. Check TT Wi-Fi, power and other SDK clients; retry shortly.')

    @app.post('/range-tests', status_code=202)
    async def create_range_test(request: RangeTestRequest):
        raise HTTPException(410, 'Forward ToF is retired: hardware is unusable. It is not a navigation input.')

    @app.post('/scans', status_code=202)
    async def create_scan(request: ScanRequest):
        if app.state.captures.active or app.state.pilot.active or any(not task.done() for task in missions) or app.state.registry._reconnect_lock.locked():
            raise HTTPException(409, 'Another capture/mission/connection operation is active')
        return app.state.scans.create(request)

    @app.get('/scans/{scan_id}')
    async def get_scan(scan_id: str):
        return app.state.scans.get(scan_id)

    @app.get('/scans/{scan_id}/map.jpg')
    async def scan_map(scan_id: str):
        app.state.scans._job(scan_id)
        return Response(app.state.scans.maps[scan_id].bird(), media_type='image/jpeg', headers={'Cache-Control':'no-store'})

    @app.post('/scans/{scan_id}/heartbeat')
    async def scan_heartbeat(scan_id: str):
        return app.state.scans.heartbeat(scan_id)

    @app.post('/scans/{scan_id}/stop')
    async def stop_scan(scan_id: str):
        return await app.state.scans.stop(scan_id)

    @app.get('/range-tests/{test_id}')
    async def get_range_test(test_id: str):
        return app.state.range_tests.get(test_id)

    @app.get('/range-tests')
    async def list_range_tests():
        return [{key: job[key] for key in ('id','status','object_label','reference_distance_mm','created_at','summary')}
                for job in reversed(list(app.state.range_tests.jobs.values()))]

    @app.post('/range-tests/{test_id}/cancel')
    async def cancel_range_test(test_id: str):
        return await app.state.range_tests.cancel(test_id)

    @app.get("/video/{device_id}")
    async def video(device_id: str):
        try:
            device = app.state.registry.get(device_id)
        except KeyError:
            raise HTTPException(404, "Unknown device")
        async def frames():
            while True:
                frame = app.state.registry.get(device_id).latest_frame()
                if frame is not None:
                    jpeg = annotate(frame[0], detect(frame[0]))
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                await asyncio.sleep(1 / 15)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def websocket(ws: WebSocket):
        await ws.accept()
        queue = asyncio.Queue(maxsize=256)
        hub.queues.add(queue)
        tools = AgentTools(app.state.registry, hub.emit, media_dir,
                           vision=app.state.provider.vision if app.state.provider else None)
        stub = StubAgent(tools)
        agent = GrokAgent(tools, app.state.provider) if app.state.provider else None
        if agent:
            agents.add(agent)
        current = None
        async def sender():
            while True:
                event = await queue.get()
                if event is None:
                    await ws.close(code=1013)
                    return
                await ws.send_json(event)
        async def run_message(text):
            status = "completed"
            try:
                await hub.emit({"type": "mission_status", "status": "started"})
                if text.startswith("/demo "):
                    await stub.run(text)
                elif mode == "grok" and agent is None:
                    await hub.emit({"type": "chat", "role": "assistant", "text": "Grok is not configured. Add XAI_API_KEY to backend/.env and restart."})
                elif agent:
                    await agent.run(text)
                else:
                    await stub.run(text)
            except asyncio.CancelledError:
                status = "cancelled"
                raise
            except Exception as exc:
                status = "failed"
                logger.warning("Mission failed (%s)", type(exc).__name__)
                await hub.emit({"type": "chat", "role": "assistant", "text": "Mission failed or timed out; stop/landing was requested. Check device state: touchdown may be unconfirmed."})
            finally:
                await hub.emit({"type": "mission_status", "status": status})
        sender_task = asyncio.create_task(sender())
        try:
            for state in app.state.registry.list_states():
                queue.put_nowait({"type": "device_state", "device": state})
            while True:
                raw = await ws.receive_text()
                try:
                    event = incoming.validate_python(json.loads(raw))
                except (ValidationError, ValueError):
                    queue.put_nowait({"type": "chat", "role": "assistant", "text": "Invalid event. Expected user_message, confirm_response, or stop_all."})
                    continue
                if isinstance(event, StopAll):
                    app.state.pilot.stop_all()
                    app.state.registry.request_stop()
                    await app.state.scans.stop_all()
                    await cancel_missions()
                    await app.state.captures.cancel_all()
                    await app.state.range_tests.cancel_all()
                    for active_agent in agents:
                        active_agent.reset()
                    results = await app.state.registry.stop_all()
                    label = 'STOP requested; landing unconfirmed' if any(r['airborne_possible'] for r in results) else 'STOP completed'
                    await hub.emit({"type": "chat", "role": "assistant", "text": f"{label}: {results}"})
                elif isinstance(event, ConfirmResponse):
                    if not tools.respond(event.request_id, event.approved):
                        queue.put_nowait({"type": "chat", "role": "assistant", "text": "Unknown or expired confirmation request."})
                elif any(not task.done() for task in missions) or app.state.captures.active or app.state.scans.active or app.state.pilot.active or app.state.registry._reconnect_lock.locked():
                    queue.put_nowait({"type": "chat", "role": "assistant", "text": "A mission is already active. Confirm it or use STOP."})
                else:
                    await hub.emit({"type": "chat", "role": "user", "text": event.text})
                    current = asyncio.create_task(run_message(event.text))
                    missions.add(current)
                    current.add_done_callback(missions.discard)
        except WebSocketDisconnect:
            pass
        finally:
            if agent:
                agents.discard(agent)
            hub.queues.discard(queue)
            sender_task.cancel()
            with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                await sender_task
            if current is not None and not current.done():
                current.cancel()
                await asyncio.gather(current, return_exceptions=True)

    register_pilot(app,lambda:app.state.captures.active or app.state.scans.active
                   or any(not task.done() for task in missions) or app.state.registry._reconnect_lock.locked())
    return app


app = create_app()
