"""Opt-in team service. Talks to the existing drone viewer and rover HTTP bridge; owns no SDK sockets."""

import argparse
import copy
import json
import math
import threading
import time
from http.client import HTTPException as HTTPStreamError
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from fleet.grid import cell_name, jpg
from fleet.harness import decision_config, draw_box, shortest_path
from fleet.llm import ask_json
from fleet.mapper import depth

app = FastAPI()
drone_url = "http://127.0.0.1:8001"
rover_url = "http://127.0.0.1:5000"
lock = threading.RLock()
DEFAULT_CAL = {
    "x": 0.0,
    "y": 0.0,
    "yaw": 0.0,
    "camera_height": 0.3,
    "body_width": 0.6,
    "body_height": 0.5,
    "fov_diag": 82.6,
    "y_sign": 1,
    "yaw_sign": -1,
}
state = {
    "rover_running": False,
    "calibration": None,
    "drone_goal": "",
    "rover_goal": "",
    "status": "Enter a goal and start",
    "rover": "idle",
    "candidate": None,
    "events": [],
}
generation = 0
latest = {"frame": None, "depth_time": 0.0, "clearance": 0.0, "cloud": [], "pose": None}
rover_voxels = {}
candidate_image = None
rover_map = None
rover_steps = []
rover_version = 0


class RoverMap:
    """Ground support and body-height obstacles from ONLY the rover camera. Never reads the drone grid."""

    n, half, cell = 51, 12.75, 0.5

    def __init__(self, pose):
        self.floor = np.zeros((self.n, self.n), dtype=bool)
        self.obstacles = np.zeros_like(self.floor)
        self.pose, self.trail, self.frames, self.route = pose, [], 0, []
        c, r = self.index(pose["x"], pose["y"])
        self.floor[r, c] = True  # operator-verified starting footprint, not assumed free surrounding space

    def index(self, x, y):
        return int(np.clip((x + self.half) / self.cell, 0, self.n - 1)), int(
            np.clip((self.half - y) / self.cell, 0, self.n - 1)
        )

    def grid(self):
        return np.where(self.obstacles, 2, np.where(self.floor, 1, 0))

    def integrate(self, east, north, up, valid, pose, body_height):
        cols = np.floor((east + self.half) / self.cell).astype(int)
        rows = np.floor((self.half - north) / self.cell).astype(int)
        inside = valid & (cols >= 0) & (cols < self.n) & (rows >= 0) & (rows < self.n)
        support = inside & (abs(up) < 0.08)
        solid = inside & (up > 0.08) & (up < body_height + 0.1)
        for mask, target in ((support, self.floor), (solid, self.obstacles)):
            counts = np.zeros_like(self.floor, dtype=int)
            np.add.at(counts, (rows[mask], cols[mask]), 1)
            target |= counts >= 3
        self.pose, self.frames = dict(pose), self.frames + 1
        if not self.trail or math.hypot(pose["x"] - self.trail[-1][0], pose["y"] - self.trail[-1][1]) > 0.05:
            self.trail.append((pose["x"], pose["y"]))
            self.trail = self.trail[-2000:]

    def path_to(self, target, pose):
        if not (-self.half <= target["x"] < self.half and -self.half < target["y"] <= self.half):
            return None
        grid = self.grid()
        start, goal = self.index(pose["x"], pose["y"]), self.index(target["x"], target["y"])
        if grid[goal[1], goal[0]] != 1:
            return None
        return shortest_path(grid, start, goal, {})

    def ascii(self):
        grid = self.grid()
        return "Columns A-AY, rows 1-51, 50 cm cells. ? unknown, . ground-free, # rover-body obstacle.\n" + "\n".join(
            f"{r + 1}:" + "".join("?.#"[int(v)] for v in row) for r, row in enumerate(grid)
        )


def graph_update():
    global rover_version
    rover_version += 1


def http(base, path, body=None, timeout=1.5):
    payload = json.dumps(body).encode() if body is not None else None
    req = Request(base + path, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise HTTPException(502, f"Upstream {base}{path} returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise HTTPException(
            503, f"Backend unavailable at {base}. Keep its server running in a separate terminal."
        ) from exc


def note(lane, text):
    with lock:
        state[lane] = text
        state["events"] = (state["events"] + [{"t": time.time(), "lane": lane, "text": text}])[-80:]


def rover_stop():
    try:
        http(rover_url, "/api/control/stop", {})
    except Exception:
        pass  # bridge-side 0.8 s watchdog is the final fallback, never renew stale velocity


def stop_rover(reason="Rover stopped by operator"):
    """Stops ONLY the rover agent. The drone keeps its own mission."""
    global generation
    with lock:
        generation += 1
        state["rover_running"] = False
    rover_stop()
    note("rover", reason)


def stop_drone():
    try:
        http(drone_url, "/cmd/stop", {})
    except Exception:
        pass


def active(gen):
    return state["rover_running"] and gen == generation


def fresh(data):
    now = data["server_time"]
    return data["status"] == "CONNECTED" and all(
        0 <= now - data.get(k, 0) < 1 for k in ("pose_time", "attitude_time", "frame_time")
    )


def world(raw):
    c = state["calibration"]
    a = math.radians(c["yaw"])
    x, y = raw["x"] - c["raw"]["x"], c["y_sign"] * (raw["y"] - c["raw"]["y"])
    return {
        "x": c["x"] + x * math.sin(a) - y * math.cos(a),
        "y": c["y"] + x * math.cos(a) + y * math.sin(a),
        "z": c["camera_height"],
        "yaw": (c["yaw"] + c["yaw_sign"] * (raw["yaw"] - c["raw"]["yaw"])) % 360,
    }


def perceive(gen):
    """Rover camera -> own ground map, colour cloud and a depth-based forward clearance. Never reads drone data."""
    while active(gen):
        try:
            sampled = time.monotonic()
            data = http(rover_url, "/api/fleet")
            if not fresh(data):
                raise ValueError("stale rover camera or odometry")
            with urlopen(rover_url + "/api/fleet/frame", timeout=1) as response:
                captured = float(response.headers["X-Captured-At"])
                image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
            if data["server_time"] - captured > 1 or image is None:
                raise ValueError("stale or invalid rover frame")
            pose = world(data["robot"])
            d = depth(image)
            h, w = d.shape
            v, u = np.mgrid[:h:4, :w:4]
            z = d[::4, ::4]
            focal = np.hypot(w / 2, h / 2) / np.tan(np.radians(state["calibration"]["fov_diag"] / 2))
            x, up = (u - w / 2) * z / focal, pose["z"] - (v - h / 2) * z / focal
            valid = np.isfinite(z) & (z > 0.05) & (z < 4.5)
            c = state["calibration"]
            corridor = valid & (abs(x) < c["body_width"] / 2 + 0.1)
            solid = corridor & (up > 0.05) & (up < c["body_height"] + 0.1)
            clearance = max(0.0, float(np.min(z[solid])) - 0.35) if solid.any() else 2.0
            a = math.radians(pose["yaw"])
            east = pose["x"] + x * math.cos(a) + z * math.sin(a)
            north = pose["y"] - x * math.sin(a) + z * math.cos(a)
            colors = image[::4, ::4][..., ::-1] / 255
            cloud = np.column_stack([east[valid], north[valid], up[valid], colors[valid]])[::3]
            with lock:
                if active(gen):
                    for point in np.round(cloud, 3).tolist():
                        rover_voxels[tuple(round(v * 10) for v in point[:3])] = point
                    while len(rover_voxels) > 30000:
                        rover_voxels.pop(next(iter(rover_voxels)))
                    latest.update(
                        frame=jpg(image),
                        depth_time=sampled - max(0, data["server_time"] - captured),
                        clearance=clearance,
                        cloud=list(rover_voxels.values()),
                        pose=pose,
                        depth_pose=pose,
                    )
                    rover_map.integrate(east, north, up, valid, pose, c["body_height"])
        except Exception as exc:
            with lock:
                latest["clearance"] = 0.0
            state["rover_perception"] = f"perception holding: {str(exc)[:120]}"
        # ~1.4 Hz, not 7: the drone's own mapper shares this Mac's GPU/CPU, and starving it made the drone
        # explore worse in multi. The rover's motion guard is its 10 Hz ToF; depth here is only a second check.
        time.sleep(ROVER_DEPTH_PERIOD)


ROVER_SCHEMA = {
    "type": "object",
    "properties": {
        "seen": {"type": "string"},
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["forward", "turn_left", "turn_right", "found"]},
        "amount": {"type": "number"},
        "evidence": {"type": "string"},
        "box": {
            "type": "array",
            "items": {"type": "number"},
            "description": "found: tight box around the goal [x_min, y_min, x_max, y_max], fractions 0-1; else []",
        },
    },
    "required": ["seen", "thought", "action", "amount", "evidence", "box"],
    "additionalProperties": False,
}
ROVER_DEPTH_PERIOD = 0.7  # s between rover depth frames
ROVER_SPEED, ROVER_TURN = 0.45, 90.0  # m/s, deg/s: explore fast, the ToF/depth guard still brakes


def drive_for(gen, vx, vz, seconds, forward):
    """Renew a velocity inside the bridge's 0.8 s watchdog; abort on stop, stale data or an obstacle ahead."""
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            if not active(gen):
                return "stopped by operator"
            data = http(rover_url, "/api/fleet", timeout=0.5)
            if data["status"] != "CONNECTED":
                return "rover disconnected"
            if forward and (data["collision_warning"] or latest["clearance"] < 0.15):
                return f"obstacle ahead (ToF {data['current_distance_mm']} mm)"
            if not http(rover_url, "/api/control/drive", {"vx": vx, "vy": 0, "vz": vz}, timeout=0.5).get("success"):
                return "rover refused motion"
            time.sleep(0.2)
        return ""
    finally:
        rover_stop()


def rover_agent(gen):
    """The rover's OWN look -> think -> act loop. It never waits for, follows or reads the drone."""
    global candidate_image
    cfg, history = decision_config(), []
    while active(gen):
        frame = latest["frame"]
        if not frame or time.monotonic() - latest["depth_time"] > 1.5:
            note("rover", state.get("rover_perception") or "waiting for a fresh rover camera frame")
            time.sleep(0.5)
            continue
        try:
            data = http(rover_url, "/api/fleet")
        except Exception as exc:
            note("rover", f"holding: {str(exc)[:120]}")
            time.sleep(0.5)
            continue
        tof = data["current_distance_mm"]
        with lock:
            number = rover_steps[-1]["step"] + 1 if rover_steps else 1
            pose = latest["pose"]
            think = {
                "id": f"r{number}.think",
                "kind": "think",
                "label": "Grok is reading the rover view",
                "status": "running",
                "model": cfg["model"],
            }
            step = {
                "step": number,
                "title": "Rover observing",
                "trace": "",
                "nodes": [
                    {
                        "id": f"r{number}.observe",
                        "kind": "observe",
                        "label": "Rover camera + own ground map",
                        "status": "done",
                        "detail": f"Pose {pose}; ToF {tof} mm; depth clearance {latest['clearance']:.2f} m",
                    },
                    think,
                ],
            }
            rover_steps.append(step)
            del rover_steps[:-100]
            graph_update()
        note("rover", "thinking")
        prompt = (
            f"You are the ROVER agent: a ground robot searching a room ON YOUR OWN. A drone searches separately; "
            f"ignore it. Goal: find {state['rover_goal']}.\n"
            f"Your camera is about {state['calibration']['camera_height']} m "
            f"above the floor. Forward ToF distance: {tof} mm (0 or >4000 = nothing in range). Depth-estimated "
            f"clear distance ahead: {latest['clearance']:.2f} m. "
            f"Your recent actions, oldest first: {history[-8:] or 'none'}.\n"
            "Choose ONE action: forward (amount = metres, 0.3-2.5), turn_left or turn_right (amount = degrees, "
            "30-180), or found (ONLY if the goal clearly fits something in THIS image; give the evidence and a tight "
            "box around it). Your job is to EXPLORE and keep MOVING: prefer long forward drives into open floor and "
            "unseen parts of the room; turn only when the way ahead is blocked (under ~0.5 m) or you have already "
            "seen this view. Do not turn back and forth. The motors brake by themselves before obstacles."
        )
        try:
            reply, secs = ask_json(cfg, prompt, frame, ROVER_SCHEMA)
        except Exception as exc:
            with lock:
                think.update(status="failed", detail=f"Vision unavailable: {str(exc)[:120]}")
                graph_update()
            note("rover", "vision unavailable; retrying")
            time.sleep(1)
            continue
        action, amount = reply["action"], float(reply["amount"] or 0)
        with lock:
            if not active(gen):
                think.update(status="done", detail="stopped while thinking")
                graph_update()
                return
            think.update(
                status="done",
                label=f"{action} {amount:g}",
                detail=reply["thought"],
                seconds=round(secs, 1),
                reply=reply,
            )
            step["title"] = f"{action} {amount:g}"
            graph_update()
        if action == "found":
            with lock:
                step["nodes"].append(
                    {
                        "id": f"r{number}.confirm",
                        "kind": "confirm",
                        "status": "waiting",
                        "label": "Waiting for operator confirmation",
                        "detail": reply["evidence"],
                    }
                )
                graph_update()
                boxed = draw_box(cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR), reply.get("box"))
                candidate_image = jpg(boxed)
                state["candidate"] = {"device": "rover", "evidence": reply["evidence"]}
            stop_rover("possible find; waiting for operator review")
            return
        if action == "forward":
            metres = max(0.3, min(2.5, amount))
            vx, vz, seconds, label = ROVER_SPEED, 0.0, metres / ROVER_SPEED, f"Drive forward {metres:.1f} m"
        else:
            degrees = max(30.0, min(180.0, amount))
            vz = ROVER_TURN if action == "turn_right" else -ROVER_TURN
            vx, seconds, label = 0.0, degrees / ROVER_TURN, f"Turn {'right' if vz > 0 else 'left'} {degrees:.0f} deg"
        act = {"id": f"r{number}.act", "kind": "act", "label": label, "status": "running", "detail": ""}
        with lock:
            step["nodes"].append(act)
            graph_update()
        note("rover", label.lower())
        try:
            outcome = drive_for(gen, vx, vz, seconds, forward=vx > 0)
        except Exception as exc:
            outcome = f"drive error: {str(exc)[:100]}"
            rover_stop()
        with lock:
            act.update(status="refused" if outcome else "done", detail=outcome or "completed")
            graph_update()
        history.append(f"{label} -> {outcome or 'done'}")
        note("rover", f"{label.lower()}: {outcome or 'done'}")
        time.sleep(0.05)
    rover_stop()


def start_rover(goal):
    """Starts the rover agent now, independent of the drone. Origin = where the rover sits at start."""
    global generation, rover_map
    if state["rover_running"]:
        return
    data = http(rover_url, "/api/fleet")
    if not fresh(data):
        raise HTTPException(409, "rover needs live camera and odometry (check /api/debug/stream on the bridge)")
    with lock:
        generation += 1
        gen = generation
        state["calibration"] = {**(state["calibration"] or DEFAULT_CAL), "raw": data["robot"]}
        state.update(rover_running=True, rover_goal=goal, candidate=None, rover="starting")
        rover_voxels.clear()
        latest.update(frame=None, depth_time=0.0, cloud=[], pose=world(data["robot"]), depth_pose=None, clearance=0.0)
        rover_map = RoverMap(latest["pose"])
        rover_steps.clear()
        graph_update()
    for job in (perceive, rover_agent):
        threading.Thread(target=job, args=(gen,), daemon=True).start()


def start_drone(goal):
    state["drone_goal"] = goal
    drone = http(drone_url, "/state")
    if not drone["running"]:
        http(drone_url, "/agent/start", {"goal": goal})


@app.get("/state")
def snapshot():
    with lock:
        return {**state, "running": state["rover_running"], "rover_pose": latest["pose"]}


@app.post("/calibrate")
def calibrate(body: Annotated[dict, Body()]):
    """Optional: overrides the default rover dimensions/start offset. Applied at the next rover start."""
    if state["rover_running"]:
        raise HTTPException(409, "stop the rover before recalibrating")
    try:
        values = {
            k: float(body[k]) for k in ("x", "y", "yaw", "camera_height", "body_width", "body_height", "fov_diag")
        }
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(400, "provide numeric pose, dimensions and camera FOV") from exc
    if not all(math.isfinite(v) for v in values.values()):
        raise HTTPException(400, "values must be finite numbers")
    if not all(0.05 <= values[k] <= 1.5 for k in ("camera_height", "body_width", "body_height")):
        raise HTTPException(400, "dimensions must be 0.05-1.5 m")
    if not 20 <= values["fov_diag"] <= 140:
        raise HTTPException(400, "measured diagonal camera FOV must be 20-140 degrees")
    signs = {k: int(body.get(k, default)) for k, default in (("y_sign", 1), ("yaw_sign", -1))}
    if any(abs(v) != 1 for v in signs.values()):
        raise HTTPException(400, "axis signs must be +1 or -1")
    state["calibration"] = {**values, **signs}
    return {"ok": True}


def goal_of(body, key="goal", fallback=""):
    goal = str(body.get(key) or body.get("goal") or "").strip() or fallback
    if not goal:
        raise HTTPException(400, "enter a goal first")
    return goal


@app.post("/start")
def start(body: Annotated[dict, Body()]):
    """Starts BOTH agents in parallel, each on its own. One failing does not block the other."""
    problems = []
    state["events"] = []
    for name, job in (("rover", start_rover), ("drone", start_drone)):
        try:
            job(goal_of(body, f"{name}_goal", state[f"{name}_goal"]))
        except HTTPException as exc:
            problems.append(f"{name}: {exc.detail}")
    note("status", "; ".join(problems) or "Both agents running independently")
    return {"ok": not problems, "problems": problems}


@app.post("/start/{device}")
def start_one(device: str, body: Annotated[dict, Body()]):
    if device not in ("rover", "drone"):
        raise HTTPException(404)
    goal = goal_of(body, f"{device}_goal", state[f"{device}_goal"])
    (start_rover if device == "rover" else start_drone)(goal)
    note("status", f"{device} started")
    return {"ok": True}


@app.post("/stop")
def stop():
    stop_rover("stopped (stop all)")
    stop_drone()
    note("status", "Both stopped; drone hovering")
    return {"ok": True}


@app.post("/stop/{device}")
def stop_one(device: str):
    if device == "rover":
        stop_rover()
    elif device == "drone":
        stop_drone()
    else:
        raise HTTPException(404)
    note("status", f"{device} stopped; the other agent keeps running")
    return {"ok": True}


@app.post("/candidate/{answer}")
def confirm_candidate(answer: str):
    if not state["candidate"] or answer not in ("yes", "no"):
        raise HTTPException(409, "no candidate waiting for confirmation")
    state["candidate"]["confirmed"] = answer == "yes"
    with lock:
        for step in rover_steps:
            for node in step["nodes"]:
                if node["kind"] == "confirm" and node["status"] == "waiting":
                    node.update(
                        status="done" if answer == "yes" else "rejected",
                        label="Operator confirmed" if answer == "yes" else "Operator rejected",
                    )
        graph_update()
    if answer == "no":
        state["candidate"] = None
        start_rover(state["rover_goal"])  # wrong find: the rover resumes its own search
        note("status", "Rover candidate rejected; rover resumed searching")
    else:
        note("status", "Rover find confirmed by operator")
    return {"ok": True}


@app.get("/candidate.jpg")
def candidate_photo():
    if not candidate_image or not state["candidate"] or state["candidate"]["device"] != "rover":
        raise HTTPException(404)
    return Response(candidate_image, media_type="image/jpeg")


@app.post("/drone/ready")
def ready():
    return http(drone_url, "/agent/ready", {})


@app.get("/drone/state")
def drone_state():
    return http(drone_url, "/state")


@app.get("/drone/graph")
def drone_graph():
    return http(drone_url, "/graph")


@app.get("/rover/graph")
def rover_graph():
    with lock:
        return {
            "goal": state["rover_goal"],
            "steps": copy.deepcopy(rover_steps),
            "phoenix": None,
            "version": rover_version,
        }


@app.get("/rover/state")
def rover_state():
    battery = rover_battery()  # outside the lock: a slow bridge must not stall the agents
    with lock:
        confirmed = (
            state["candidate"] and state["candidate"]["device"] == "rover" and state["candidate"].get("confirmed")
        )
        return {
            "kind": "rover",
            "mode": "agent",
            "running": state["rover_running"],
            "goal": state["rover_goal"],
            "status": "ended: found" if confirmed else state["rover"] if state["rover_running"] else "ended: stopped",
            "confirm": None,
            "plan": None,
            "waiting": False,
            "approve_each": False,
            "instruction": None,
            "progress": None,
            "pose": latest["pose"] or dict(x=0, y=0, z=0, yaw=0),
            "battery": battery,
            "log": [],
        }


@app.post("/drone/auto/{on}")
def drone_auto(on: str):
    """on: the drone agent acts without asking; off: each drone action waits for operator approval."""
    return http(drone_url, f"/auto/{'on' if on == 'on' else 'off'}", {})


def rover_battery():
    try:
        return http(rover_url, "/api/status", timeout=0.5).get("battery")
    except Exception:
        return None


@app.post("/drone/height/{direction}")
def adjust_height(direction: str):
    if direction not in ("up", "down") or not http(drone_url, "/state").get("height_waiting"):
        raise HTTPException(409, "height adjustments are only allowed during the height check")
    return http(drone_url, f"/cmd/{direction}?arg=20", {})


@app.post("/manual/{device}/{name}")
def manual(device: str, name: str, arg: int = 20):
    """Operator takeover affects ONLY the named device."""
    if device == "drone":
        return http(drone_url, f"/manual/{name}?arg={arg}", {})
    if device == "rover" and name == "stop":
        stop_rover()
        return {"ok": True}
    raise HTTPException(400, "use the rover dashboard for manual rover driving")


MAP_POINTS = 12000  # the 3D view only needs a sketch of the room; full-size clouds made the page lag


@app.get("/points")
def points():
    """Drone scene, thinned and rounded (about 10x smaller JSON), plus the rover's pose and track as a polyline."""
    scene = http(drone_url, "/points", timeout=3)
    pts = scene["points"]
    step = max(1, math.ceil(len(pts) / MAP_POINTS))
    scene["points"] = [[round(v, 2) for v in p] for p in pts[::step]]
    with lock:
        trail = rover_map.trail[::4] + rover_map.trail[-1:] if rover_map and rover_map.trail else []
        scene["rover_trail"] = [[round(x, 2), round(y, 2)] for x, y in trail]
        scene["rover"] = latest["pose"]
    return Response(json.dumps(scene, separators=(",", ":")), media_type="application/json")


@app.get("/rover/points")
def rover_points():
    with lock:
        if rover_map is None:
            raise HTTPException(409, "calibrate the rover first")
        m, grid = rover_map, rover_map.grid()
        path = m.route
        return {
            "points": latest["cloud"][:25000],
            "pose": m.pose,
            "rover": latest["pose"],
            "state": {
                "drone_cell": cell_name(*m.index(m.pose["x"], m.pose["y"])),
                "frames_used": m.frames,
                "cells": {
                    "unknown": int((grid == 0).sum()),
                    "free": int((grid == 1).sum()),
                    "obstacle": int((grid == 2).sum()),
                },
            },
            "grid": grid.tolist(),
            "n": m.n,
            "half": m.half,
            "cell": m.cell,
            "visited": list({cell_name(*m.index(x, y)) for x, y in m.trail}),
            "plan": {"action": "goto", "label": "rover ground route", "path": path} if path else None,
        }


@app.get("/video/{device}")
def video(device: str):
    if device not in ("drone", "rover"):
        raise HTTPException(404)
    base, path = (drone_url, "/video") if device == "drone" else (rover_url, "/video_feed")

    def stream():
        sent = 0.0
        while True:
            try:
                with urlopen(base + path, timeout=3) as upstream:
                    buffer = b""
                    while chunk := upstream.read1(65536):
                        buffer += chunk
                        end = buffer.rfind(b"\xff\xd9")
                        if end < 0:
                            if len(buffer) > 4_000_000:
                                raise ValueError("oversized camera frame")
                            continue
                        start = buffer.rfind(b"\xff\xd8", 0, end)
                        image_bytes, buffer = buffer[start : end + 2], buffer[end + 2 :]
                        if start < 0 or time.monotonic() - sent < 0.1:
                            continue  # discard older frames instead of forwarding a latency backlog
                        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                        if image is None:
                            continue
                        h, w = image.shape[:2]
                        if w > 640:
                            image = cv2.resize(image, (640, max(1, round(h * 640 / w))))
                        image_bytes = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 65])[1].tobytes()
                        sent = time.monotonic()
                        yield b"--fleet\r\nContent-Type: image/jpeg\r\n\r\n" + image_bytes + b"\r\n"
            except (OSError, HTTPStreamError, ValueError):
                pass
            # Do not leave a disconnected camera's old picture looking live.
            offline = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(
                offline,
                f"{device.upper()} CAMERA RECONNECTING",
                (25, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (90, 160, 240),
                2,
            )
            yield b"--fleet\r\nContent-Type: image/jpeg\r\n\r\n" + jpg(offline) + b"\r\n"
            time.sleep(1)

    return StreamingResponse(
        stream(),
        media_type="multipart/x-mixed-replace; boundary=fleet",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


def main():
    global drone_url, rover_url
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-url", default=drone_url)
    parser.add_argument("--rover-url", default=rover_url)
    args = parser.parse_args()
    drone_url, rover_url = args.drone_url.rstrip("/"), args.rover_url.rstrip("/")
    import torch

    torch.set_num_threads(2)  # leave the CPU to the drone server running next to us
    uvicorn.run(app, host="127.0.0.1", port=8002)
