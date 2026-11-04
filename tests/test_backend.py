import asyncio
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fleet.agent import AgentTools
from fleet.app import create_app
from fleet.contracts import DeviceAdapter
from fleet.perception import annotate, detect, follow_command, photo_ready
from fleet.registry import DeviceRegistry
from fleet.simulator import SimAdapter


@pytest.fixture
def app(tmp_path):
    config = tmp_path / "devices.json"
    config.write_text(json.dumps({"devices": [{"id": "drone-1", "adapter": "sim"}, {"id": "drone-2", "adapter": "sim"}]}))
    return create_app(config, tmp_path / "media", agent_mode="stub")


def receive_until(ws, predicate):
    for _ in range(40):
        event = ws.receive_json()
        if predicate(event):
            return event
    pytest.fail("Expected event not received")


def test_health_devices_and_shutdown(app):
    with TestClient(app) as client:
        assert client.get("/health").json()["agent"] == "stub"
        assert client.get("/").status_code == 200
        assert "Mission control" in client.get("/").text
        assert client.get("/test-assets/console.js").status_code == 200
        assert client.get("/favicon.ico").status_code == 204
        assert len(client.get("/devices").json()) == 2
        assert client.get("/video/missing").status_code == 404
        device = app.state.registry.get("drone-1")
        asyncio.run(device.takeoff())
    assert device.state()["flying"] is False
    assert device.state()["connected"] is False


def test_ws_confirmed_demo_saves_real_jpeg_and_lands(app):
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "user_message", "text": "/demo drone-1"})
        request = receive_until(ws, lambda e: e["type"] == "confirm_request")
        assert not app.state.registry.get("drone-1").state()["flying"]
        ws.send_json({"type": "confirm_response", "request_id": request["request_id"], "approved": True})
        photo = receive_until(ws, lambda e: e["type"] == "photo")
        response = client.get(photo["url"])
        assert response.status_code == 200
        assert response.content.startswith(b"\xff\xd8")
        receive_until(ws, lambda e: e["type"] == "device_state" and e["device"]["id"] == "drone-1" and not e["device"]["flying"])
        commands = app.state.registry.get("drone-1").commands
        assert [c["command"] for c in commands] == ["takeoff", "land"]


@pytest.mark.parametrize("approved", [False, True])
def test_stop_cancels_pending_confirmation_and_other_flying_device(app, approved):
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        asyncio.run(app.state.registry.get("drone-2").takeoff())
        ws.send_json({"type": "user_message", "text": "/demo drone-1"})
        request = receive_until(ws, lambda e: e["type"] == "confirm_request")
        ws.send_json({"type": "stop_all"})
        receive_until(ws, lambda e: e["type"] == "chat" and e["text"].startswith("STOP completed"))
        ws.send_json({"type": "confirm_response", "request_id": request["request_id"], "approved": approved})
        receive_until(ws, lambda e: e["type"] == "chat" and "expired confirmation" in e["text"])
        assert all(not state["flying"] for state in client.get("/devices").json())
        assert "takeoff" not in [c["command"] for c in app.state.registry.get("drone-1").commands]
        assert [c["command"] for c in app.state.registry.get("drone-2").commands][-2:] == ["hover", "land"]


def test_reject_invalid_events_and_takeoff(app):
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text("not-json")
        receive_until(ws, lambda e: e["type"] == "chat" and "Invalid event" in e["text"])
        ws.send_json({"type": "user_message", "text": "/demo drone-1"})
        request = receive_until(ws, lambda e: e["type"] == "confirm_request")
        ws.send_json({"type": "confirm_response", "request_id": request["request_id"], "approved": "false"})
        receive_until(ws, lambda e: e["type"] == "chat" and "Invalid event" in e["text"])
        ws.send_json({"type": "confirm_response", "request_id": request["request_id"], "approved": False})
        receive_until(ws, lambda e: e["type"] == "chat" and "rejected" in e["text"])
        assert not app.state.registry.get("drone-1").state()["flying"]


def test_simulator_commands_limits_and_camera():
    async def scenario():
        device = SimAdapter("test")
        assert isinstance(device, DeviceAdapter)
        device.battery_pct = 24
        with pytest.raises(ValueError, match="25%"):
            await device.takeoff()
        device.battery_pct = 25
        await device.takeoff()
        await device.move(20, 30, 100)
        await device.rotate(400)
        assert device.state()["height_cm"] == 150
        assert device.state()["yaw_deg"] == 40
        assert device.position == [20, 30]
        device.rc(100, -100, 100, 100)
        assert device.velocity == (40, -40, 40, 40)
        device.tick(10)
        assert device.state()["height_cm"] == 200
        with pytest.raises(ValueError):
            await device.move(0, 0, 1)
        assert device.velocity == (0, 0, 0, 0)
        frame, timestamp = device.latest_frame()
        assert frame.shape == (360, 640, 3) and timestamp > 0
        assert detect(frame) == []
        target = {"track_id": 1, "box": (20, 20, 60, 100), "confidence": .9}
        assert annotate(frame, [target]).startswith(b"\xff\xd8")
        assert follow_command(target, frame.shape) == (0, 0, 0, 0)
        assert photo_ready(target, frame.shape) is False
        await device.emergency()
        assert not device.state()["flying"]
        with pytest.raises(ValueError):
            device.rc(0, 0, 0, 0)
        device.close()
        assert device.latest_frame() is None
    asyncio.run(scenario())


def test_recorded_video_loops(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48))
    assert writer.isOpened()
    for value in (30, 200):
        writer.write(np.full((48, 64, 3), value, np.uint8))
    writer.release()
    device = SimAdapter("recording", path)
    means = []
    for _ in range(3):
        device._frame_at = 0
        means.append(device.latest_frame()[0].mean())
    device.close()
    assert means[0] == pytest.approx(means[2], abs=5)
    assert means[1] > means[0] + 100


def test_confirmation_timeout_is_rejection(tmp_path):
    async def scenario():
        device = SimAdapter("drone-1")
        async def emit(event):
            pass
        tools = AgentTools(DeviceRegistry([device]), emit, tmp_path, confirmation_timeout=.01)
        assert (await tools.takeoff("drone-1"))["approved"] is False
        assert tools.pending == {}
        assert not device.flying
    asyncio.run(scenario())


def test_registry_attempts_land_even_when_hover_fails():
    class BrokenHover(SimAdapter):
        async def hover(self):
            raise RuntimeError("hover failed")
    async def scenario():
        device = BrokenHover("test")
        await device.takeoff()
        results = await DeviceRegistry([device]).stop_all()
        assert results[0]["errors"] == ["hover failed"]
        assert not device.flying
    asyncio.run(scenario())


def test_invalid_registry_config(tmp_path):
    config = tmp_path / "devices.json"
    config.write_text(json.dumps({"devices": [{"id": "same", "adapter": "sim"}] * 2}))
    with pytest.raises(ValueError, match="unique"):
        DeviceRegistry.from_config(config)
