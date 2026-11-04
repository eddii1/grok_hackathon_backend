"""Drones the agent can fly. Both have the same three methods: send(cmd) -> reply, frame() -> jpeg, close()."""

import socket
import threading
import time

import cv2
import numpy as np

from fleet.grid import jpg, parse_cell


class SimDrone:
    """Flies on the top-down overview: the camera is a crop around (and ahead of) the current cell."""

    def __init__(self, overview: np.ndarray, start: str, cols: int, rows: int):
        self.img, self.cols, self.rows = overview, cols, rows
        self.c, self.r = parse_cell(start)

    def send(self, cmd: str) -> str:
        word = cmd.split()[0]
        c = self.c + {"right": 1, "left": -1}.get(word, 0)
        r = self.r + {"forward": -1, "back": 1}.get(word, 0)
        if not (0 <= c < self.cols and 0 <= r < self.rows):
            return "error out of area"
        self.c, self.r = c, r
        return "ok"

    def frame(self) -> bytes:
        h, w = self.img.shape[:2]
        cw, ch = w / self.cols, h / self.rows
        x0, y0 = int(max(0, (self.c - 0.75) * cw)), int(max(0, (self.r - 1.5) * ch))
        x1, y1 = int(min(w, (self.c + 1.75) * cw)), int(min(h, (self.r + 1) * ch))
        return jpg(cv2.resize(self.img[y0:y1, x0:x1], (960, 720)))

    def close(self) -> None:
        pass


class TelloDrone:
    """RoboMaster TT / Tello over the raw SDK. Every command waits for its reply, so moves never overlap."""

    def __init__(self, ip: str = "192.168.10.1"):
        self.addr = (ip, 8889)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind(("", 9000))
        self.s.settimeout(20)
        self.latest = None
        for cmd in ("command", "streamon"):
            if self.send(cmd) != "ok":
                raise RuntimeError(f"drone did not accept '{cmd}'")
        threading.Thread(target=self._video, daemon=True).start()

    def send(self, cmd: str) -> str:
        self.s.sendto(cmd.encode(), self.addr)
        try:
            return self.s.recvfrom(1024)[0].decode().strip()
        except TimeoutError:
            return "timeout"

    def _video(self) -> None:
        cap = cv2.VideoCapture("udp://@0.0.0.0:11111")
        while True:
            ok, f = cap.read()
            if ok:
                self.latest = f

    def frame(self) -> bytes:
        time.sleep(0.5)  # let the picture settle after the move
        return jpg(self.latest) if self.latest is not None else b""

    def close(self) -> None:
        self.send("land")
