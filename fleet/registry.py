import asyncio
import json
from pathlib import Path

from .contracts import DeviceAdapter, DeviceState
from .simulator import SimAdapter


class DeviceRegistry:
    def __init__(self, devices: list[DeviceAdapter]):
        if len({d.id for d in devices}) != len(devices):
            raise ValueError("Device IDs must be unique")
        self.devices = {d.id: d for d in devices}

    @classmethod
    def from_config(cls, path: Path):
        devices = []
        try:
            for item in json.loads(path.read_text(encoding="utf-8"))["devices"]:
                if item["adapter"] != "sim":
                    raise ValueError("Only the sim adapter is implemented")
                if not item["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in item["id"]):
                    raise ValueError("Device IDs must contain only letters, numbers, '-' or '_'")
                video = item.get("video_path")
                devices.append(SimAdapter(item["id"], (path.parent / video).resolve() if video else None, item.get("battery_pct", 100)))
            return cls(devices)
        except Exception:
            for device in devices:
                device.close()
            raise

    def get(self, device_id: str) -> DeviceAdapter:
        return self.devices[device_id]

    def list_states(self) -> list[DeviceState]:
        return [d.state() for d in self.devices.values()]

    async def stop_all(self) -> list[dict]:
        async def stop(device):
            errors = []
            if device.state()["flying"]:
                for action in (device.hover, device.land):
                    try:
                        await asyncio.wait_for(action(), timeout=3)
                    except Exception as exc:
                        errors.append(str(exc))
            return {"device_id": device.id, "errors": errors}
        return await asyncio.gather(*(stop(d) for d in self.devices.values()))
