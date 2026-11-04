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
        self._reconnect_lock = asyncio.Lock()

    @classmethod
    def from_config(cls, path: Path):
        devices = []
        try:
            for item in json.loads(path.read_text(encoding="utf-8"))["devices"]:
                if not item["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in item["id"]):
                    raise ValueError("Device IDs must contain only letters, numbers, '-' or '_'")
                if item['adapter'] == 'sim':
                    video = item.get("video_path")
                    devices.append(SimAdapter(item["id"], (path.parent / video).resolve() if video else None, item.get("battery_pct", 100)))
                elif item['adapter'] == 'scan_sim':
                    from .scan_simulator import ScanSimAdapter
                    devices.append(ScanSimAdapter(item['id']))
                elif item['adapter'] == 'pilot_sim':
                    from .pilot_sim import PilotSimAdapter
                    devices.append(PilotSimAdapter(item['id']))
                elif item['adapter'] in {'tello','pilot'}:
                    from .tello import TelloAdapter
                    if any(getattr(d, 'simulated', True) is False for d in devices):
                        raise ValueError('Initial hardware integration supports one TT only')
                    if item.get('flight_enabled'):
                        raise ValueError('Browser flight enablement is unavailable; use the supervised hover terminal')
                    if item['adapter']=='pilot':
                        from .pilot_adapter import PilotAdapter
                        devices.append(PilotAdapter(item['id'],item.get('host','192.168.10.1'),item.get('bind','0.0.0.0')))
                    else:
                        devices.append(TelloAdapter(item['id'], item.get('host', '192.168.10.1'), item.get('bind', '0.0.0.0')))
                else:
                    raise ValueError('Unknown device adapter')
            return cls(devices)
        except Exception:
            for device in devices:
                device.close()
            raise

    def get(self, device_id: str) -> DeviceAdapter:
        return self.devices[device_id]

    async def start(self, strict=True):
        try:
            for device in self.devices.values():
                if getattr(device, 'simulated', True) is False or getattr(device,'pilot_capable',False):
                    try:
                        await device.start()
                    except Exception as exc:
                        device.close()
                        device.fault = f'Ground link unavailable ({type(exc).__name__}); reconnect grounded TT'
                        if strict:
                            raise
        except BaseException:
            for device in self.devices.values():
                device.close()
            raise

    async def reconnect_ground(self, device_id):
        from .tello import TelloAdapter
        if self._reconnect_lock.locked():
            raise ValueError('A ground reconnect is already in progress')
        async with self._reconnect_lock:
            old = self.get(device_id)
            if getattr(old, 'simulated', True) or old.flight_enabled or old._armed or old.state()['airborne_possible']:
                raise ValueError('Reconnect is restricted to an unarmed, grounded real monitor')
            old.connecting = True
            old.close()
            try:
                if hasattr(old.transport, 'wait_closed'):
                    # Cleanup only: this worker never sends a motion command.
                    await asyncio.to_thread(old.transport.wait_closed)
                if getattr(old,'pilot_capable',False):
                    from .pilot_adapter import PilotAdapter
                    new = PilotAdapter(old.id,old.host,old.bind)
                else:
                    new = TelloAdapter(old.id, old.host, old.bind)
                self.devices[device_id] = new
                await new.start()
                await new.wait_ready(timeout=10)
                return new.state()
            except Exception as exc:
                current = self.get(device_id)
                current.close()
                current.fault = f'Ground reconnect failed ({type(exc).__name__}); check Wi-Fi and other Tello clients'
                raise
            finally:
                old.connecting = False

    def request_stop(self):
        # Latch before awaiting mission cancellation or any SDK acknowledgement.
        for device in self.devices.values():
            if getattr(device, 'simulated', True) is False or getattr(device,'pilot_capable',False):
                device.trip('STOP requested')

    def list_states(self) -> list[DeviceState]:
        return [d.state() for d in self.devices.values()]

    async def stop_all(self) -> list[dict]:
        self.request_stop()
        async def stop(device):
            errors = []
            if getattr(device, 'simulated', True) is False or getattr(device,'pilot_capable',False):
                try:
                    await device.stop_all()
                except Exception as exc:
                    errors.append(str(exc))
            elif device.state()["flying"]:
                for action in (device.hover, device.land):
                    try:
                        await asyncio.wait_for(action(), timeout=3)
                    except Exception as exc:
                        errors.append(str(exc))
            return {"device_id": device.id, "errors": errors,
                    'airborne_possible': device.state()['flying']}
        return await asyncio.gather(*(stop(d) for d in self.devices.values()))
