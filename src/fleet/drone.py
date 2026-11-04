"""One drone object for the whole backend: every command goes through here, is safety-checked, logged and traced.

    d = Drone(SimBackend("room_360.jpg"))        # or Drone(TelloBackend())
    d.takeoff(); d.spin(); d.forward(50); d.turn_right(45); d.land()
    d.frame()  # newest camera image      d.pose()  # x, y (m), z height (m), yaw (deg clockwise from north)
    d.log      # every command: time, args, reply, pose before and after (also runs/<id>/commands.jsonl)

Attach a LiveMap with d.attach(mapper) and every new frame + pose is mapped automatically.
"""

import json
import socket
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from fleet.tracing import span

MAX_HEIGHT_M = 2.0
MIN_BATTERY = 25


# ---------- backends: the only code that knows the hardware ----------


class SimBackend:
    """A camera inside a 360 photo. Turning changes the view; moving only changes the pose (one photo = one spot)."""

    W, H, FOV_DIAG = 1280, 720, 82.6
    YAW_RATE = 90.0  # deg/s at rc yaw 100
    SPEED = 0.5  # m/s at rc 100

    def __init__(self, pano: str):
        self.pano = cv2.imread(pano)
        self.state = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "bat": 100, "flying": False}
        self.rc_cmd = (0, 0, 0, 0)
        f = np.hypot(self.W / 2, self.H / 2) / np.tan(np.radians(self.FOV_DIAG / 2))
        self._u, self._v, self._f = (
            *np.meshgrid(np.arange(self.W) - self.W / 2 + 0.5, np.arange(self.H) - self.H / 2 + 0.5),
            f,
        )
        threading.Thread(target=self._physics, daemon=True).start()

    def _physics(self) -> None:
        last = time.time()
        while True:
            time.sleep(0.03)
            dt, last = time.time() - last, time.time()
            lr, fb, ud, yaw = self.rc_cmd
            if not self.state["flying"]:
                continue
            a = np.radians(self.state["yaw"])
            self.state["yaw"] = (self.state["yaw"] + yaw / 100 * self.YAW_RATE * dt) % 360
            v_fwd, v_right = fb / 100 * self.SPEED * dt, lr / 100 * self.SPEED * dt
            self.state["x"] += v_fwd * np.sin(a) + v_right * np.cos(a)
            self.state["y"] += v_fwd * np.cos(a) - v_right * np.sin(a)
            self.state["z"] = max(0.2, self.state["z"] + ud / 100 * self.SPEED * dt)

    def send(self, cmd: str) -> str:
        word, *args = cmd.split()
        s = self.state
        if word in ("command", "streamon", "stop", "emergency"):
            if word == "emergency":
                s.update(flying=False, z=0.0)
            return "ok"
        if word == "takeoff":
            s.update(flying=True, z=1.0)
        elif word == "land":
            s.update(flying=False, z=0.0)
        elif word in ("forward", "back", "left", "right", "up", "down"):
            m = int(args[0]) / 100
            a = np.radians(s["yaw"])
            fwd = {"forward": m, "back": -m}.get(word, 0)
            right = {"right": m, "left": -m}.get(word, 0)
            s["x"] += fwd * np.sin(a) + right * np.cos(a)
            s["y"] += fwd * np.cos(a) - right * np.sin(a)
            s["z"] += {"up": m, "down": -m}.get(word, 0)
            time.sleep(m / self.SPEED)
        elif word in ("cw", "ccw"):
            deg = int(args[0])
            s["yaw"] = (s["yaw"] + (deg if word == "cw" else -deg)) % 360
            time.sleep(deg / self.YAW_RATE)
        elif word == "battery?":
            return str(s["bat"])
        else:
            return "error unknown command"
        return "ok"

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self.rc_cmd = (lr, fb, ud, yaw)

    def frame(self) -> np.ndarray:
        a = np.radians(self.state["yaw"])
        u, v, f = self._u, self._v, self._f
        xr, zr = u * np.cos(a) + f * np.sin(a), -u * np.sin(a) + f * np.cos(a)
        lon, lat = np.arctan2(xr, zr), np.arctan2(-v, np.hypot(xr, zr))
        ph, pw = self.pano.shape[:2]
        mx = ((lon / (2 * np.pi) + 0.5) * pw % pw).astype(np.float32)
        my = ((0.5 - lat / np.pi) * ph).astype(np.float32)
        return cv2.remap(self.pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def telemetry(self) -> dict:
        s = self.state
        return {"x": s["x"], "y": s["y"], "z": s["z"], "yaw": s["yaw"], "bat": s["bat"]}


class TelloBackend:
    """RoboMaster TT / Tello over the raw SDK. Commands wait for their reply; state and video stream in."""

    def __init__(self, ip: str = "192.168.10.1"):
        self.addr = (ip, 8889)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", 9000))
        self.sock.settimeout(20)
        self.lock = threading.Lock()
        self._frame, self._state = None, {}
        for cmd in ("command", "streamon"):
            if self.send(cmd) != "ok":
                raise RuntimeError(f"drone did not accept '{cmd}'")
        threading.Thread(target=self._video, daemon=True).start()
        threading.Thread(target=self._state_loop, daemon=True).start()

    def send(self, cmd: str) -> str:
        with self.lock:
            self.sock.sendto(cmd.encode(), self.addr)
            try:
                return self.sock.recvfrom(1024)[0].decode().strip()
            except TimeoutError:
                return "timeout"

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        self.sock.sendto(f"rc {lr} {fb} {ud} {yaw}".encode(), self.addr)  # rc has no reply

    def _video(self) -> None:
        cap = cv2.VideoCapture("udp://@0.0.0.0:11111")
        while True:
            ok, f = cap.read()
            if ok:
                self._frame = f

    def _state_loop(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("", 8890))
        while True:
            msg = s.recv(1024).decode()
            self._state = dict(kv.split(":", 1) for kv in msg.strip().strip(";").split(";") if ":" in kv)

    def frame(self) -> np.ndarray | None:
        return self._frame

    def telemetry(self) -> dict:
        s = self._state
        num = lambda k, d=0.0: float(s.get(k, d))  # noqa: E731
        return {
            "yaw": num("yaw") % 360,
            "z": num("h") / 100,
            "bat": int(num("bat", 0)),
            "tof_cm": num("tof"),
            "mid": int(num("mid", -2)),
            "pad_x": num("x"),
            "pad_y": num("y"),
        }


# ---------- the drone ----------


class WebcamBackend:
    """A laptop camera carried by a person, who plays the drone.

    Heading comes from the image: consecutive frames are compared and the sideways shift is turned into degrees
    (no sensor needed). Position comes from moves: every move is shown to the person as an instruction
    ("carry the camera forward 50 cm") and counts once they press Done. Turns finish by themselves when the
    camera has turned far enough. While a move is pending the map pauses, so walking does not smear it.
    """

    FOV_DIAG = 78.0
    TURN_TOLERANCE = 12.0  # degrees

    def __init__(self, index: int = 0):
        self.index = index
        self.yaw, self.turned = 0.0, 0.0  # heading, and total degrees turned (for spins)
        self._frame, self._prev = None, None
        self.instruction: str | None = None  # what the person must do now (shown in the viewer)
        self.progress = None  # optional callable -> "turned 40 of 90 degrees"
        self.moving = False  # True while the person walks: the mapper skips frames
        self._done, self._cancelled = threading.Event(), False
        self._one = threading.Lock()  # one instruction at a time
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:  # reopen until the camera works (e.g. after macOS camera permission is granted)
            cap = cv2.VideoCapture(self.index)
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                self._track(f)
                self._frame = f
            cap.release()
            print("camera not available - check System Settings > Privacy & Security > Camera; retrying")
            time.sleep(3)

    def _track(self, f: np.ndarray) -> None:
        """Heading from the image. ORB features of each frame are matched against a reference frame and RANSAC keeps
        the shift most of the scene agrees on, so a person walking through the view does not turn the camera
        (too few features, e.g. a blank wall: phase correlation of the whole image instead).
        Measuring against a reference (not the previous frame) means noise does not add up while still, and a
        frame that cannot be matched (blank wall, blur) loses nothing. The reference moves on every ~8 degrees."""
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 180))
        if not hasattr(self, "_orb"):
            self._orb = cv2.ORB_create(600, fastThreshold=10)
            self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            self._misses = 0
        kp, des = self._orb.detectAndCompute(g, None)
        ref = self._prev  # (keypoints, descriptors, yaw at ref, turned at ref, image)
        if ref is None:
            self._prev = (kp, des, self.yaw, self.turned, g)
            return
        dx = None
        if des is not None and ref[1] is not None:
            matches = self._bf.match(ref[1], des)
            if len(matches) >= 15:
                a = np.float32([ref[0][m.queryIdx].pt for m in matches])
                b = np.float32([kp[m.trainIdx].pt for m in matches])
                M, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M is not None and inliers.sum() >= 12:
                    dx = M[0, 2] + M[0, 1] * 90 + (M[0, 0] - 1) * 160  # shift of the image centre
        if dx is None:  # too few features (blank wall, curtain): whole-image phase correlation instead
            if not hasattr(self, "_win"):
                self._win = cv2.createHanningWindow((320, 180), cv2.CV_32F)
            (sx, _), response = cv2.phaseCorrelate(ref[4].astype(np.float32), g.astype(np.float32), self._win)
            if response > 0.05 and abs(sx) < 120:
                dx = sx
        if dx is None:
            self._misses += 1
            if self._misses > 30:  # lost for ~1 s (camera covered, very fast turn): start again from here
                self._prev, self._misses = (kp, des, self.yaw, self.turned, g), 0
            return
        self._misses = 0
        focal = np.hypot(160, 90) / np.tan(np.radians(self.FOV_DIAG / 2))
        d = -np.degrees(np.arctan(dx / focal))  # turning right moves the scene left
        self.yaw, self.turned = (ref[2] + d) % 360, ref[3] + d
        if abs(d) > 8:
            self._prev = (kp, des, self.yaw, self.turned, g)

    def _ask(self, text: str, until=None, timeout: float = 120) -> str:
        """Show an instruction and wait for Done (or for `until()` to become true)."""
        if not self._one.acquire(blocking=False):
            return "busy: another instruction is still open"
        self._done.clear()
        self._cancelled = False
        self.instruction = text
        t = time.time()
        try:
            while time.time() - t < timeout:
                if self._done.wait(0.1):
                    return "stopped" if self._cancelled else "ok"
                if until and until():
                    return "ok"
            return "timeout"
        finally:
            self.instruction = self.progress = None
            self._one.release()

    def done(self) -> None:
        """The person pressed Done."""
        self._done.set()

    def cancel(self) -> None:
        """The mission was stopped: release the person from the current instruction."""
        self._cancelled = True
        self._done.set()

    def send(self, cmd: str) -> str:
        word, *args = cmd.split()
        if word in ("command", "streamon", "stop", "emergency", "takeoff", "land", "battery?"):
            return "100" if word == "battery?" else "ok"
        if word in ("cw", "ccw"):
            target = (self.yaw + (1 if word == "cw" else -1) * int(args[0])) % 360
            side = "right" if word == "cw" else "left"
            start = self.turned
            self.progress = lambda: f"turned {abs(self.turned - start):.0f} of {args[0]} degrees"
            return self._ask(
                f"Turn the camera {side} {args[0]} degrees",
                until=lambda: abs((self.yaw - target + 540) % 360 - 180) < self.TURN_TOLERANCE,
            )
        if word == "spin":
            start = self.turned
            self.progress = lambda: f"turned {max(0, self.turned - start):.0f} of 360 degrees"
            return self._ask(
                "Turn slowly right, a full circle, standing in one spot", until=lambda: self.turned - start >= 350
            )
        if word in ("forward", "back", "left", "right", "up", "down"):
            what = {
                "forward": "forward",
                "back": "backward",
                "left": "to your left",
                "right": "to your right",
                "up": "up",
                "down": "down",
            }[word]
            self.moving = True
            try:
                return self._ask(f"Carry the camera {what} {args[0]} cm without turning it, then press Done")
            finally:
                self.moving = False
        return "error unknown command"

    def rc(self, lr: int, fb: int, ud: int, yaw: int) -> None:
        pass

    def frame(self) -> np.ndarray | None:
        return self._frame

    def telemetry(self) -> dict:
        return {"yaw": self.yaw, "z": 1.0, "bat": 100}


class Drone:
    def __init__(self, backend, log_dir: Path | None = None):
        self.b = backend
        self.x = self.y = 0.0  # position: counted from executed moves (Tello has no GPS)
        self.flying = False
        self.log: list[dict] = []
        self.log_file = open(log_dir / "commands.jsonl", "a") if log_dir else None
        self._mapper = None
        threading.Thread(target=self._keepalive, daemon=True).start()

    # pose and sensors
    def pose(self) -> dict:
        t = self.b.telemetry()
        if isinstance(self.b, SimBackend):
            self.x, self.y = t["x"], t["y"]
        return {"x": round(self.x, 3), "y": round(self.y, 3), "z": round(t["z"], 2), "yaw": round(t["yaw"], 1) % 360}

    def battery(self) -> int:
        return int(self.b.telemetry()["bat"])

    def frame(self) -> np.ndarray | None:
        return self.b.frame()

    # the one place commands are sent
    def _do(self, cmd: str, **meta) -> str:
        before = self.pose()
        with span(f"drone {cmd}", kind="TOOL", input={"cmd": cmd, "pose": before}) as done:
            t = time.time()
            reply = self.b.send(cmd)
            if reply != "ok" and cmd.split()[0] not in ("battery?",):
                self.b.send("stop")  # never retry a move blindly: hover instead
            after = self.pose()
            entry = {
                "t": round(t, 2),
                "cmd": cmd,
                "reply": reply,
                "seconds": round(time.time() - t, 2),
                "before": before,
                "after": after,
                **meta,
            }
            self.log.append(entry)
            if self.log_file:
                self.log_file.write(json.dumps(entry) + "\n")
                self.log_file.flush()
            done({"reply": reply, "pose": after})
        return reply

    def _refuse(self, cmd: str, reason: str) -> str:
        """Blocked by a safety rule: nothing is sent to the drone, but it is still logged."""
        entry = {
            "t": round(time.time(), 2),
            "cmd": cmd,
            "reply": f"refused: {reason}",
            "seconds": 0,
            "before": self.pose(),
            "after": self.pose(),
        }
        self.log.append(entry)
        if self.log_file:
            self.log_file.write(json.dumps(entry) + "\n")
            self.log_file.flush()
        return entry["reply"]

    def _move(self, word: str, cm: int) -> str:
        cm = int(round(cm))
        if not 20 <= cm <= 500:
            return self._refuse(f"{word} {cm}", "outside 20-500 cm")
        if not self.flying:
            return self._refuse(f"{word} {cm}", "not flying")
        if word == "up" and self.pose()["z"] + cm / 100 > MAX_HEIGHT_M:
            return self._refuse(f"{word} {cm}", f"above {MAX_HEIGHT_M} m")
        yaw = np.radians(self.pose()["yaw"])
        reply = self._do(f"{word} {cm}")
        if reply == "ok" and not isinstance(self.b, SimBackend):
            m = cm / 100
            fwd = {"forward": m, "back": -m}.get(word, 0)
            right = {"right": m, "left": -m}.get(word, 0)
            self.x += fwd * np.sin(yaw) + right * np.cos(yaw)
            self.y += fwd * np.cos(yaw) - right * np.sin(yaw)
        return reply

    # commands
    def takeoff(self) -> str:
        if self.battery() < MIN_BATTERY:
            return self._refuse("takeoff", f"battery {self.battery()}% < {MIN_BATTERY}%")
        reply = self._do("takeoff")
        self.flying = reply == "ok"
        return reply

    def land(self) -> str:
        self.b.rc(0, 0, 0, 0)
        reply = self._do("land")
        self.flying = False
        return reply

    def hover(self) -> str:
        self.b.rc(0, 0, 0, 0)
        return self._do("stop")

    def emergency(self) -> str:
        self.flying = False
        return self._do("emergency")

    def forward(self, cm: int) -> str:
        return self._move("forward", cm)

    def back(self, cm: int) -> str:
        return self._move("back", cm)

    def left(self, cm: int) -> str:
        return self._move("left", cm)

    def right(self, cm: int) -> str:
        return self._move("right", cm)

    def up(self, cm: int) -> str:
        return self._move("up", cm)

    def down(self, cm: int) -> str:
        return self._move("down", cm)

    def turn_right(self, deg: int) -> str:
        return self._do(f"cw {int(np.clip(deg, 1, 360))}") if self.flying else self._refuse(f"cw {deg}", "not flying")

    def turn_left(self, deg: int) -> str:
        return self._do(f"ccw {int(np.clip(deg, 1, 360))}") if self.flying else self._refuse(f"ccw {deg}", "not flying")

    def rc(self, lr: int = 0, fb: int = 0, ud: int = 0, yaw: int = 0, seconds: float = 0.0) -> str:
        """Smooth motion (each -100..100) for `seconds`, then stop. Speeds are capped at 40 for safety."""
        if not self.flying:
            return self._refuse(f"rc {lr} {fb} {ud} {yaw}", "not flying")
        lr, fb, ud, yaw = (int(np.clip(v, -40, 40)) for v in (lr, fb, ud, yaw))
        with span("drone rc", kind="TOOL", input={"rc": [lr, fb, ud, yaw], "seconds": seconds}) as done:
            before = self.pose()
            self.b.rc(lr, fb, ud, yaw)
            if seconds:
                time.sleep(seconds)
                self.b.rc(0, 0, 0, 0)
            entry = {
                "t": round(time.time(), 2),
                "cmd": f"rc {lr} {fb} {ud} {yaw}",
                "seconds": seconds,
                "reply": "sent",
                "before": before,
                "after": self.pose(),
            }
            self.log.append(entry)
            if self.log_file:
                self.log_file.write(json.dumps(entry) + "\n")
            done(entry)
        return "ok"

    def spin(self, yaw_speed: int = 40) -> str:
        """One full turn in place while the mapper watches (~10 s at speed 40; faster speeds blur more)."""
        if isinstance(self.b, WebcamBackend):
            return self._do("spin") if self.flying else self._refuse("spin", "not flying")
        seconds = 360 / (SimBackend.YAW_RATE * yaw_speed / 100)
        return self.rc(yaw=yaw_speed, seconds=seconds)

    # background
    def attach(self, mapper, hz: float = 5.0) -> None:
        """Feed every new frame and its pose into a LiveMap."""
        self._mapper = mapper

        def feed():
            while self._mapper is mapper:
                f = self.frame()
                if f is not None and not getattr(self.b, "moving", False):  # a person walking: pose is stale
                    mapper.add(f, self.pose())
                time.sleep(1 / hz)

        threading.Thread(target=feed, daemon=True).start()

    def _keepalive(self) -> None:
        """The Tello lands by itself after 15 s without a command; poke it while flying."""
        while True:
            time.sleep(5)
            if self.flying and isinstance(self.b, TelloBackend):
                self.b.send("battery?")
