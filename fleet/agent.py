"""JSON-returning tool stubs and a deterministic simulator demo, without Grok."""
import asyncio
from pathlib import Path
from uuid import uuid4

from .perception import annotate
from .registry import DeviceRegistry


class AgentTools:
    def __init__(self, registry: DeviceRegistry, emit, media_dir: Path, confirmation_timeout: float = 30, vision=None):
        self.registry, self.emit, self.media_dir = registry, emit, media_dir
        self.confirmation_timeout = confirmation_timeout
        self.pending: dict[str, asyncio.Future] = {}
        self.vision = vision
        self.photos: dict[str, Path] = {}

    async def _confirm(self, device_id: str, reason: str, image_url: str = "") -> bool:
        request_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self.emit({"type": "confirm_request", "request_id": request_id,
                             "device_id": device_id, "image_url": image_url, "reason": reason})
            return await asyncio.wait_for(future, timeout=self.confirmation_timeout)
        except TimeoutError:
            return False
        finally:
            self.pending.pop(request_id, None)

    def respond(self, request_id: str, approved: bool) -> bool:
        future = self.pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(approved)
        return True

    async def list_devices(self) -> dict:
        capabilities = ["takeoff", "land", "video", "take_photo"]
        if self.vision:
            capabilities += ["describe_scene", "verify_photo"]
        return {"devices": [{**state, "capabilities": capabilities,
                             "simulated": True} for state in self.registry.list_states()]}

    async def describe_scene(self, device_id: str) -> dict:
        from .grok import SceneDescription
        frame = self.registry.get(device_id).latest_frame()
        if frame is None:
            raise ValueError("No camera frame available")
        if self.vision is None:
            return {"stub": True, "description": "Vision is unavailable in stub mode"}
        return {"simulated": True, **await self.vision(annotate(frame[0], []),
                "Describe what is actually visible in this simulator camera image. Mention if it is a synthetic frame.", SceneDescription)}

    async def takeoff(self, device_id: str) -> dict:
        device = self.registry.get(device_id)
        approved = await self._confirm(device_id, "SIMULATOR: approve takeoff to 50 cm?")
        if not approved:
            return {"approved": False, "status": "rejected"}
        await device.takeoff()
        return {"approved": True, "device": device.state()}

    async def land(self, device_id: str) -> dict:
        device = self.registry.get(device_id)
        await device.land()
        return {"device": device.state()}

    async def search(self, device_id: str, description: str) -> dict:
        self.registry.get(device_id)
        return {"stub": True, "status": "not_found", "candidates": [], "description": description}

    async def confirm_target(self, device_id: str, track_id: int, reason: str) -> dict:
        self.registry.get(device_id)
        # The detector stub has no valid tracks. Never approve a fabricated target.
        return {"stub": True, "approved": False, "reason": "No tracked targets in stub perception"}

    async def follow(self, device_id: str, track_id: int, max_seconds: float) -> dict:
        await self.registry.get(device_id).hover()
        return {"stub": True, "status": "not_implemented"}

    async def take_photo(self, device_id: str) -> dict:
        frame = self.registry.get(device_id).latest_frame()
        if frame is None:
            raise ValueError("No camera frame available")
        photo_id = uuid4().hex
        target = self.media_dir / "photos" / f"{photo_id}.jpg"
        target.write_bytes(annotate(frame[0], []))
        self.photos[photo_id] = target
        await self.emit({"type": "photo", "photo_id": photo_id, "url": f"/photos/{photo_id}.jpg",
                         "caption": "Simulator camera capture"})
        return {"photo_id": photo_id}

    async def verify_photo(self, photo_id: str, description: str) -> dict:
        from .grok import PhotoVerification
        if photo_id not in self.photos:
            raise ValueError("Unknown photo from this session")
        if self.vision is None:
            return {"stub": True, "match": False, "confidence": 0.0, "reason": "Vision is unavailable in stub mode"}
        result = await self.vision(self.photos[photo_id].read_bytes(),
                                  f"Does the image match this description? {description}", PhotoVerification)
        return {"simulated": True, **result}

    async def report(self, text: str, photo_ids: list[str]) -> dict:
        await self.emit({"type": "chat", "role": "assistant", "text": text})
        return {"text": text, "photo_ids": photo_ids}


class StubAgent:
    def __init__(self, tools: AgentTools):
        self.tools = tools

    async def run(self, text: str):
        if not text.startswith("/demo "):
            await self.tools.report("Stub agent: no Grok calls or person detection yet. Send '/demo drone-1' to test confirmed simulated takeoff, a camera capture, and landing.", [])
            return
        device_id = text.removeprefix("/demo ").strip()
        device = self.tools.registry.get(device_id)
        if device.state()["flying"]:
            raise ValueError("Device is already flying")
        try:
            result = await self.tools.takeoff(device_id)
            await self.tools.emit({"type": "tool_call", "name": "takeoff", "args": {"device_id": device_id}, "result": result})
            if not result["approved"]:
                await self.tools.report("Simulated takeoff rejected or confirmation timed out.", [])
                return
            photo = await self.tools.take_photo(device_id)
            await self.tools.report("Simulator demo complete. Camera captured; landing. No target selection or AI verification was performed.", [photo["photo_id"]])
        finally:
            await device.land()
