"""In-memory drone, synthetic camera or looping local video. No hardware I/O."""
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from .contracts import DeviceState

logger = logging.getLogger(__name__)


class SimAdapter:
    kind = "drone"

    def __init__(self, device_id: str, video_path: Path | None = None, battery_pct: int = 100):
        self.id = device_id
        self.battery_pct = battery_pct
        self.flying = False
        self.height = 0.0
        self.yaw = 0.0
        self.position = [0.0, 0.0]
        self.velocity = (0, 0, 0, 0)
        self.connected = True
        self.commands: list[dict] = []
        self._capture = None
        self._frame: np.ndarray | None = None
        self._frame_at = 0.0
        self._last_rc = time.monotonic()
        if video_path is not None:
            self._capture = cv2.VideoCapture(str(video_path))
            if not self._capture.isOpened():
                self.close()
                raise ValueError(f"Cannot open simulator video: {video_path}")

    def _log(self, name: str, **args):
        entry = {"time": time.time(), "device_id": self.id, "command": name, "args": args}
        self.commands.append(entry)
        # Bounded diagnostic history; this is not the persistent mission log.
        del self.commands[:-1000]
        logger.info("sim command %s", entry)

    def state(self) -> DeviceState:
        return dict(id=self.id, kind="drone", connected=self.connected, flying=self.flying,
                    battery_pct=self.battery_pct, height_cm=round(self.height), yaw_deg=round(self.yaw) % 360)

    def latest_frame(self) -> tuple[np.ndarray, float] | None:
        if not self.connected:
            return None
        now = time.time()
        fps = (self._capture.get(cv2.CAP_PROP_FPS) if self._capture else 15) or 15
        if self._frame is None or now - self._frame_at >= 1 / fps:
            if self._capture:
                ok, frame = self._capture.read()
                if not ok:
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self._capture.read()
                if not ok:
                    return None
            else:
                frame = np.full((360, 640, 3), (40, 30, 20), dtype=np.uint8)
                cv2.putText(frame, f"SIMULATOR - {self.id}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, .8, (80, 230, 180), 2)
                cv2.putText(frame, f"height {round(self.height)} cm | yaw {round(self.yaw)}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 230, 230), 1)
            self._frame, self._frame_at = frame, now
        return self._frame.copy(), self._frame_at

    def _require_flight(self):
        if not self.connected or not self.flying:
            raise ValueError("Device must be connected and flying")

    async def takeoff(self) -> None:
        if not self.connected or self.battery_pct < 25:
            raise ValueError("Takeoff requires a connected device and at least 25% battery")
        if not self.flying:
            self.flying, self.height = True, 50.0
        self._log("takeoff")

    async def land(self) -> None:
        self.flying, self.height, self.velocity = False, 0.0, (0, 0, 0, 0)
        self._log("land")

    async def hover(self) -> None:
        self.velocity = (0, 0, 0, 0)
        self._log("hover")

    async def move(self, forward_cm: int, left_cm: int, up_cm: int) -> None:
        self._require_flight()
        if not 0 <= self.height + up_cm <= 200:
            await self.hover()
            raise ValueError("Requested height is outside 0..200 cm")
        self.height += up_cm
        self.position[0] += forward_cm
        self.position[1] += left_cm
        self._log("move", forward_cm=forward_cm, left_cm=left_cm, up_cm=up_cm)

    async def rotate(self, cw_deg: int) -> None:
        self._require_flight()
        self.yaw = (self.yaw + cw_deg) % 360
        self._log("rotate", cw_deg=cw_deg)

    def rc(self, left_right: int, fwd_back: int, up_down: int, yaw: int) -> None:
        self._require_flight()
        self.velocity = tuple(max(-40, min(40, int(v))) for v in (left_right, fwd_back, up_down, yaw))
        self._last_rc = time.monotonic()
        self._log("rc", values=self.velocity)

    def tick(self, seconds: float) -> None:
        if not self.flying:
            return
        if time.monotonic() - self._last_rc > 1:
            self.velocity = (0, 0, 0, 0)
        lr, fb, ud, yaw = self.velocity
        self.height = max(0, min(200, self.height + ud * seconds))
        self.yaw = (self.yaw + yaw * seconds) % 360
        self.position[0] += fb * seconds
        self.position[1] += lr * seconds

    async def emergency(self) -> None:
        await self.land()
        self._log("emergency")

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
        self.connected = False
