"""SEEK app: camera, live map (2D as Grok sees it, 3D point cloud), manual control and the Grok agent.
http://localhost:8001

    uv run fleet-viewer --webcam 0          # your laptop camera: you carry it and do what the app says
    uv run fleet-viewer --sim room_360.jpg  # simulated drone inside a 360 photo
    uv run fleet-viewer --video flight.mp4  # a recorded drone flight (spin replays it)
    uv run fleet-viewer --tello             # the real drone, found on the laptop's network (hotspot)
    uv run fleet-viewer --tello 192.168.10.1  # the real drone at this address (laptop on its Wi-Fi)
    ... --goal "a red mug"                  # start straight in agent mode with this goal

The map is built live from the camera the whole time. Two modes, switchable at any moment:
  - Manual: you send the commands (buttons).
  - Agent: type what to find and press Start; Grok decides every move until you confirm a find or press Stop.
With the webcam every command becomes an instruction on screen ("turn right 90 degrees"); turns finish by
themselves when the camera has turned, moves when you press Done.
"""

import argparse
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

from fleet import tracing
from fleet.drone import MAX_HEIGHT_M, MIN_HEIGHT_M, Drone, SimBackend, TelloBackend, VideoBackend, WebcamBackend
from fleet.grid import jpg
from fleet.harness import Seek, decision_config
from fleet.mapper import LiveMap, depth

app = FastAPI(telemetry={"tracing": False, "metrics": False, "logs": False})  # keep viewer polls out of Phoenix
ctx: dict = {"mode": "manual", "seek": None, "approve_each": True, "hop_cells": 2}
manual_lock = threading.Lock()


# ---------- agent control ----------


def start_agent(goal: str) -> None:
    busy = ctx.get("busy")
    if (busy and busy.is_alive()) or ctx["drone"].landing:
        raise HTTPException(409, "finish the current command or landing first")
    old = ctx["seek"]
    if old and not old.status.startswith("ended"):
        raise HTTPException(409, "stop the current search first")
    stop_agent()
    ctx["drone"].holding = False
    out = ctx["out"] / f"mission-{time.strftime('%H%M%S')}"
    seek = Seek(ctx["drone"], ctx["map"], goal, out, decision_config())
    seek.approve_each = ctx["approve_each"]
    seek.hop_cells = ctx["hop_cells"]
    ctx.update(mode="agent", seek=seek, manual_status="")
    print(f"agent: searching for {goal!r}   files: {out}")
    threading.Thread(target=seek.run, daemon=True).start()


def stop_agent() -> None:
    if ctx["seek"] is not None:
        ctx["seek"].stop()


@app.post("/mode/{mode}")
def set_mode(mode: str) -> dict:
    if mode not in ("manual", "agent"):
        raise HTTPException(400, "mode is manual or agent")
    if mode == "manual":
        stop_agent()
    ctx["mode"] = mode
    return {"mode": mode}


@app.post("/agent/start")
def agent_start(goal: str = Body(..., embed=True)) -> dict:
    if not goal.strip():
        raise HTTPException(400, "type what to find")
    start_agent(goal.strip())
    return {"ok": True}


@app.post("/agent/stop")
def agent_stop() -> dict:
    stop_agent()
    return {"ok": True}


@app.post("/agent/ready")
def agent_ready() -> dict:
    """The operator finished setting altitude; only now may recon and Grok start."""
    seek, d = ctx["seek"], ctx["drone"]
    if not seek or not seek.height_waiting or seek.stopped:
        raise HTTPException(409, "not waiting for the flight height")
    busy = ctx.get("busy")
    if busy and busy.is_alive():
        raise HTTPException(409, "wait for the height adjustment to finish")
    if not d.flying or d.landing or d.holding:
        raise HTTPException(409, "drone is not ready to begin")
    seek._height_ready.set()
    return {"ok": True}


@app.post("/confirm/{answer}")
def confirm(answer: str) -> dict:
    if ctx["seek"] is not None:
        ctx["seek"].answer(answer == "yes")
    return {"ok": True}


@app.post("/approve")
def approve(ok: bool = Body(...), note: str = Body("")) -> dict:
    """The operator approved or rejected the agent's next action."""
    if ctx["seek"] is not None:
        ctx["seek"].decide(ok, note)
    return {"ok": True}


@app.post("/auto/{on}")
def auto(on: str) -> dict:
    """auto on: the agent acts without asking. auto off: every action waits for approval."""
    ctx["approve_each"] = on != "on"
    if ctx["seek"] is not None:
        ctx["seek"].approve_each = ctx["approve_each"]
    return {"approve_each": ctx["approve_each"]}


@app.post("/map/reset")
def map_reset() -> dict:
    """A new place: forget the map (not while the agent is searching)."""
    seek = ctx["seek"]
    if seek is not None and not seek.status.startswith("ended"):
        raise HTTPException(409, "stop the agent first")
    ctx["map"].reset()
    ctx["seek"] = None
    return {"ok": True}


@app.post("/done")
def done() -> dict:
    """The person carrying the camera finished the instruction."""
    if hasattr(ctx["drone"].b, "done"):
        ctx["drone"].b.done()
    return {"ok": True}


@app.get("/graph")
def graph() -> dict:
    """The current mission as a graph: steps, each a chain of nodes (observe, think, check, approve, act...)."""
    seek = ctx["seek"]
    return {
        "goal": seek.goal if seek else "",
        "steps": seek.graph_snapshot() if seek else [],
        "phoenix": ctx.get("phoenix"),
        "mission": seek.session if seek else "",
        "version": seek.gver if seek else 0,
    }


@app.get("/graph/stream")
def graph_stream() -> StreamingResponse:
    """Server-sent events: the whole graph every time it changes (new node, status, Grok tokens)."""

    def gen():
        ver, sid = -1, None
        while True:
            seek = ctx["seek"]
            if seek is None:
                time.sleep(0.5)
                if sid is not None or ver == -1:
                    ver, sid = 0, None
                    yield f"data: {json.dumps(graph())}\n\n"
                continue
            if id(seek) != sid:
                sid, ver = id(seek), -1
            new = seek.wait_graph(ver) if ver != -1 else seek.gver
            if new == ver and ctx["seek"] is seek:
                yield ": ping\n\n"
                continue
            ver = new
            yield f"data: {json.dumps(graph())}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/files/{name}")
@app.get("/found/{name}")
def found_photo(name: str) -> FileResponse:
    seek = ctx["seek"]
    path = seek.out / Path(name).name if seek else None
    if not path or not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


# ---------- state for the page ----------


def map_seek() -> Seek | None:
    """Manual destinations and agent actions publish their plans through the same map endpoints."""
    manual = ctx.get("manual")
    return manual if manual and manual.plan else ctx["seek"]


def kind_is_drone(d) -> bool:
    return isinstance(d.b, TelloBackend)


@app.get("/tof")
def tof() -> dict:
    """The forward distance sensor (TT top module), in cm: a check that the wall guard works."""
    d = ctx["drone"]
    if not kind_is_drone(d):
        return {"tof_cm": None}
    return {"tof_cm": d.b.tof_cm(), "raw": d.b.last_tof}


@app.get("/state")
def state() -> dict:
    seek, d = ctx["seek"], ctx["drone"]
    active = map_seek()
    plan = active.plan if active else None
    running = seek is not None and not seek.status.startswith("ended")
    return {
        "kind": ctx["kind"],
        "mode": ctx["mode"],
        "running": running,
        "height_waiting": bool(seek and seek.height_waiting and not seek.stopped),
        "busy": bool(ctx.get("busy") and ctx["busy"].is_alive()),
        "goal": seek.goal if seek else "",
        "status": seek.status if seek else "",
        "manual_status": ctx.get("manual_status", ""),
        "confirm": seek.confirm if seek else None,
        "plan": {k: v for k, v in plan.items() if k != "path"} if plan else None,
        "waiting": bool(seek and seek.plan and seek.status.startswith("waiting for your approval")),
        "approve_each": ctx["approve_each"],
        "turns": seek.turns[-30:] if seek else [],
        "instruction": getattr(d.b, "instruction", None),
        "progress": d.b.progress() if getattr(d.b, "progress", None) else None,
        "pose": d.pose(),
        "battery": d.battery() if kind_is_drone(d) else None,
        "log": d.log[-15:],
    }


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


@app.get("/video")
def video() -> StreamingResponse:
    def gen():
        while True:
            f = ctx["drone"].frame()
            if f is not None:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg(cv2.resize(f, (640, 360))) + b"\r\n"
            time.sleep(1 / 15)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")


@app.get("/map.png")
def map_png() -> Response:
    seek = map_seek()
    visits, plan = (seek.mem.visits, seek.plan) if seek else (None, None)
    return Response(ctx["map"].bird(visited=visits, plan=plan), media_type="image/jpeg")


@app.get("/points")
def points() -> JSONResponse:
    """Everything the 3D view draws: the point cloud, the drone, the 50 cm cells (the same grid the agent's
    path checks use) and the agent's planned action."""
    m, seek = ctx["map"], map_seek()
    current_plan = seek.plan if seek else None  # one snapshot: action threads can clear it during this request
    plan = None
    if current_plan:
        plan = {k: current_plan.get(k) for k in ("action", "label", "heading")}
        plan["path"] = [[int(c), int(r)] for c, r in current_plan.get("path") or []]
        plan["route"] = [[int(c), int(r)] for c, r in current_plan.get("route", current_plan.get("path")) or []]
    return JSONResponse(
        {
            "points": m.points(max_points=25000),
            "pose": m.pose,
            "state": m.state(),
            "grid": m.grid().tolist(),
            "n": m.n,
            "half": m.half,
            "cell": m.cell,
            "visited": list(seek.mem.visits) if seek else [],
            "plan": plan,
        }
    )


@app.get("/settings")
def get_settings() -> dict:
    return {"speed": ctx["drone"].speed, "hop_cells": ctx["hop_cells"],
            "takeoff_height_m": ctx["drone"].takeoff_height_m}


@app.post("/settings")
def set_settings(speed: int | None = None, hop_cells: int | None = None,
                 takeoff_height_m: float | None = None) -> dict:
    """Operator flight settings. The map grid (50 cm cells) never changes: a hop is just how many cells the drone
    flies before it looks ahead again, and speed is how fast it flies them."""
    reply = "ok"
    if takeoff_height_m is not None:
        if not math.isfinite(takeoff_height_m) or not MIN_HEIGHT_M <= takeoff_height_m <= MAX_HEIGHT_M:
            raise HTTPException(400, f"takeoff height must be {MIN_HEIGHT_M}-{MAX_HEIGHT_M} m")
        ctx["drone"].takeoff_height_m = takeoff_height_m  # next takeoff only; never moves the drone implicitly
    if hop_cells is not None:
        ctx["hop_cells"] = max(1, min(10, hop_cells))  # SDK forward limit: 500 cm
        if ctx["seek"] is not None:
            ctx["seek"].hop_cells = ctx["hop_cells"]
    if speed is not None:
        reply = ctx["drone"].set_speed(speed)
    return {"reply": reply, **get_settings()}


@app.post("/goto/{cell}")
def manual_goto(cell: str) -> dict:
    """The operator picked a cell on the 3D map: fly there over known-free cells, with the same depth look-ahead."""
    d = ctx["drone"]
    if not d.flying:
        return {"reply": "not flying: take off first"}
    busy = ctx.get("busy")
    if busy is not None and busy.is_alive():
        return {"reply": f"busy: still doing '{ctx['busy_name']}' - finish it first"}
    took_over = ctx["seek"] is not None and not ctx["seek"].status.startswith("ended")
    if took_over:
        stop_agent()
        ctx["mode"] = "manual"
    m = ctx.get("manual")
    if m is None or m.stopped:  # its own Seek: a stopped mission's Seek refuses to fly
        m = ctx["manual"] = Seek(d, ctx["map"], "manual goto", ctx["out"] / "manual", decision_config())
    m.hop_cells = ctx["hop_cells"]
    path, why = m.plan_goto(cell.upper(), manual=True)
    if path is None:
        return {"reply": why}
    d.holding = False
    plan = m.goto_plan(path)
    plan["text"] = f"Fly to {cell.upper()}"
    m.plan = plan  # publish before starting: even the first map poll sees the destination
    ctx["busy"] = threading.Thread(target=lambda: print(f"manual goto {cell}: {m.act(plan)}"), daemon=True)
    ctx["busy_name"] = f"goto {cell}"
    ctx["busy"].start()
    return {"reply": f"{'took over from the agent: ' if took_over else ''}flying to {cell.upper()}"}


@app.post("/cmd/{name}")
def command(name: str, arg: int = 0) -> dict:
    d = ctx["drone"]
    if name in ("stop", "land", "emergency"):
        with manual_lock:
            ctx["manual_gen"] = ctx.get("manual_gen", 0) + 1  # cancel keyboard commands still waiting to run
    if name == "stop":
        stop_agent()
        if ctx.get("manual"):
            ctx["manual"].stop()
        ctx["mode"] = "manual"
        reply = d.hover()  # never queue behind busy: STOP must interrupt the current command
        return {"reply": "holding position: search stopped" if reply in ("ok", "sent") else reply}
    if name in ("land", "emergency") and ctx.get("manual"):
        ctx["manual"].stop()  # never let a manual route resume after LAND/KILL
    took_over = False
    height_adjust = bool(ctx["seek"] and ctx["seek"].height_waiting and name in ("up", "down"))
    if ctx["seek"] is not None and not ctx["seek"].status.startswith("ended") and not height_adjust:
        stop_agent()  # any manual command takes over from the agent: you are flying now
        ctx["mode"], took_over = "manual", True
    if name == "land" and not d.stop_for_land():
        return {"reply": "already landing"}
    calls = {
        "takeoff": d.takeoff,
        "land": d.land,
        "spin": d.spin,
        "forward": lambda: d.forward(arg),
        "back": lambda: d.back(arg),
        "left": lambda: d.left(arg),
        "right": lambda: d.right(arg),
        "up": lambda: d.up(arg),
        "down": lambda: d.down(arg),
        "turn_left": lambda: d.turn_left(arg),
        "turn_right": lambda: d.turn_right(arg),
        "emergency": d.emergency,
    }
    if name not in calls:
        return {"reply": f"unknown command {name}"}
    if name == "emergency":
        if hasattr(d.b, "cancel"):
            d.b.cancel()  # a person holding the camera: drop the current instruction
        return {"reply": d.emergency()}
    # Commands can take long (a webcam move waits for Done), so they run in the background and this request
    # returns at once: a request held open per button would use up the browser's 6 connections per site and
    # freeze the camera stream and the page. One command at a time.
    busy = ctx.get("busy")
    if busy is not None and busy.is_alive() and name != "land":  # land always goes: it queues right behind
        return {"reply": f"busy: still doing '{ctx['busy_name']}' - finish it first"}
    if name != "land":
        d.holding = False  # a new, explicit operator action releases STOP/HOVER
    ctx["busy"] = threading.Thread(target=calls[name], daemon=True)
    ctx["busy_name"] = name
    ctx["busy"].start()
    return {"reply": "took over from the agent: manual control" if took_over else "started"}


@app.post("/manual/{name}")
def keyboard_command(name: str, arg: int = 20) -> dict:
    """Newest operator input wins. Brake now, discard old motion, then execute only the latest command."""
    if name in ("stop", "land"):
        return command(name)
    if name not in ("forward", "back", "left", "right", "up", "down", "turn_left", "turn_right"):
        raise HTTPException(400, "not a keyboard flight command")
    if not (1 <= arg <= 90 if name.startswith("turn_") else 20 <= arg <= 100):
        raise HTTPException(400, "keyboard moves are 20-100 cm; turns are 1-90 degrees")
    d, seek = ctx["drone"], ctx["seek"]
    if d.landing or not d.flying:
        return {"reply": "landing: manual motion refused" if d.landing else "not flying: take off first"}
    # Height tweaks during the mandatory preflight pause preserve the pending goal, not any autonomy.
    height_adjust = bool(seek and seek.height_waiting and not seek.stopped and name in ("up", "down"))
    with manual_lock:
        ctx["manual_gen"] = gen = ctx.get("manual_gen", 0) + 1
        old_busy = ctx.get("busy")
        if not height_adjust:
            stop_agent()
            ctx["mode"] = "manual"
        if ctx.get("manual"):
            ctx["manual"].stop()
        d.hover()  # immediately interrupts SDK flight, even when another command owns the send lock
        ctx["manual_status"] = f"manual: {name} {arg}"
        tracing.event("manual override", command=name, arg=arg, generation=gen, pose=d.pose())

        def run():
            if old_busy and old_busy.is_alive():
                old_busy.join(2)
            settled = d.motion_done.wait(2)  # an interrupted agent move must finish updating its pose/map latch
            with manual_lock:
                if gen != ctx.get("manual_gen") or d.landing or not d.flying:
                    return
                if not settled or (old_busy and old_busy.is_alive()):
                    ctx["manual_status"] = "holding: previous command has not stopped; press the desired key again"
                    return
                d.holding = False
            reply = getattr(d, name)(arg)
            if gen == ctx.get("manual_gen"):
                ctx["manual_status"] = f"manual: {name} {arg} ({reply})"

        ctx["busy"] = threading.Thread(target=run, daemon=True)
        ctx["busy_name"] = f"manual {name} {arg}"
        ctx["busy"].start()
    return {"reply": "started"}


PAGE = """<!doctype html><meta charset=utf-8><title>SEEK</title>
<style>
:root{--bg:#0b0f14;--panel:#111821;--line:#1e2a36;--text:#dbe4ee;--dim:#7f8fa0;--blue:#1f6feb;--red:#d1242f;
--green:#2da44e;--amber:#ffd479}
*{box-sizing:border-box}
body{margin:0;font:13px system-ui;background:var(--bg);color:var(--text);display:grid;
grid-template-columns:minmax(0,1fr) minmax(0,1.35fr);grid-template-rows:auto minmax(0,.8fr) minmax(0,1.25fr);height:100vh;
gap:6px;padding:6px;overflow:hidden}
#view3d{grid-column:2;grid-row:2 / span 2}#legend{position:absolute;right:10px;top:6px;font-size:11px;color:var(--dim);
display:flex;gap:10px}#legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.p{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden;position:relative;min-height:0}
.p h3{margin:0;padding:6px 10px;font-size:12px;color:var(--dim);font-weight:600}
button{background:var(--blue);color:#fff;border:0;border-radius:5px;padding:6px 10px;cursor:pointer;font:inherit}
button:disabled{opacity:.35;cursor:default}.stop{background:var(--red)}.go{background:var(--green)}
#bar{grid-column:span 2;display:flex;align-items:center;gap:8px;padding:8px 10px}
#bar b{font-size:15px;letter-spacing:1px;margin-right:6px}
.seg{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{background:transparent;border-radius:0;color:var(--dim)}.seg button.on{background:var(--blue);color:#fff}
#goalin{flex:1;min-width:120px;background:#0b121a;color:var(--text);border:1px solid var(--line);border-radius:5px;
padding:7px 9px;font:inherit}#status{color:var(--amber);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:32%}
#side{display:flex;flex-direction:column}#camwrap{position:relative;flex:1 1 0;min-height:0;display:flex;
justify-content:center;background:#000}#cam{max-width:100%;max-height:100%;object-fit:contain;display:block}
#ins{position:absolute;inset:auto 8px 8px 8px;background:rgba(255,212,121,.95);color:#111;border-radius:6px;
padding:10px 12px;font-size:16px;font-weight:600;display:none;align-items:center;gap:10px}
#ins span{flex:1}#ins button{background:#111;font-size:14px}
#st{padding:4px 10px;font-size:11px;color:#9fb0c0}
#btns{display:flex;flex-wrap:wrap;gap:4px;padding:4px 6px;align-items:center}#flash{color:var(--amber);font-size:12px;margin-left:6px}
#log{margin:0;padding:4px 10px;font-size:11px;line-height:14px;color:#9fb0c0;flex:0 0 46px;overflow:auto;
white-space:pre}
#three{position:absolute;inset:28px 0 0 0}#msg{position:absolute;top:40px;left:12px;color:#ff9b9b;font-size:12px}
#agent{display:flex;flex-direction:column}#turns{padding:0 10px}
#graph{padding:2px 10px 10px;overflow:auto;flex:1}
.st{position:relative;padding:0 0 10px 16px;border-left:2px solid #263241;margin-left:6px}
.st:last-child{border-left-color:transparent}
.st:before{content:'';position:absolute;left:-7px;top:3px;width:12px;height:12px;border-radius:50%;background:#263241}
.st.now:before{background:#ffd479;box-shadow:0 0 0 4px rgba(255,212,121,.2)}
.sh{display:flex;gap:8px;align-items:baseline;margin-bottom:5px}.sh b{font-size:13px}
.sh .n{color:var(--dim);font-size:11px;letter-spacing:.5px}.sh a{color:#7cc4ff;font-size:11px;margin-left:auto;text-decoration:none}
.chain{display:flex;flex-wrap:wrap;align-items:stretch;gap:4px}
.arr{color:#3a4a5c;align-self:center;font-size:13px;margin-right:2px}.pair{display:flex;align-items:stretch}
.nd{background:#0d141c;border:1px solid #2a3644;border-left:3px solid #4a5a6c;border-radius:6px;padding:5px 8px;
 min-width:92px;max-width:260px;cursor:pointer}
.nd:hover{border-color:#4a5a6c}.nd.sel{outline:2px solid #7cc4ff}
.nd .k{font-size:10px;letter-spacing:.6px;color:var(--dim);text-transform:uppercase}
.nd .l{font-weight:600;font-size:12px}.nd .d{font-size:11px;color:#9fb0c0;display:-webkit-box;-webkit-line-clamp:2;
 -webkit-box-orient:vertical;overflow:hidden}
.nd.done{border-left-color:#2da44e}.nd.refused,.nd.rejected,.nd.failed{border-left-color:#d1242f}
.nd.refused .l,.nd.rejected .l,.nd.failed .l{color:#ff9b9b}
.nd.running{border-left-color:#1f6feb;animation:pulse 1.2s infinite}.nd.waiting{border-left-color:#ffd479;animation:pulse 1.2s infinite}
@keyframes pulse{50%{background:#152131}}
.det{margin-top:6px;background:#0b1118;border:1px solid var(--line);border-radius:6px;padding:8px;font-size:12px}
.det .row{margin-bottom:4px}.det .s{color:var(--dim)}.det img{height:90px;border-radius:4px;margin:4px 4px 0 0;cursor:zoom-in}
.det pre{white-space:pre-wrap;font-size:11px;color:#9fb0c0;margin:4px 0 0;max-height:160px;overflow:auto}
.t{border-top:1px solid var(--line);padding:6px 0}.t b{color:#7cc4ff}.t .r{color:#9fb0c0}.t .s{color:var(--dim)}
#plan{display:none;margin:0 10px 8px;padding:10px;border:2px solid #3cdc3c;border-radius:6px;background:#0f2014}
#plan .pl{font-size:14px;margin-bottom:4px}#plan b{color:#7ef07e}#plan .s{color:var(--dim);margin-bottom:8px}
.prow{display:flex;gap:6px}.prow input{flex:1;background:#0b121a;color:var(--text);border:1px solid var(--line);
border-radius:5px;padding:6px 8px;font:inherit}
#autol{color:var(--dim);white-space:nowrap;display:flex;align-items:center;gap:4px;cursor:pointer}
.ghost{background:transparent;border:1px solid var(--line);color:var(--dim)}
#confirm{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);align-items:center;justify-content:center;z-index:9}
#confirm .box{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;width:min(640px,92vw)}
#confirm img{width:100%;border-radius:4px;margin:8px 0}#confirm .row{display:flex;gap:8px;justify-content:flex-end}
</style>
<div class=p id=bar><b>SEEK</b>
 <div class=seg><button id=m-manual onclick="setMode('manual')">Manual</button><button id=m-agent onclick="setMode('agent')">Agent</button></div>
 <input id=goalin placeholder="What should it find? e.g. a red mug" onkeydown="if(event.key==='Enter')start()">
 <label id=autol title="Off: you approve every action before it runs"><input type=checkbox id=auto onchange="post('/auto/'+(this.checked?'on':'off'))"> automatic</label>
 <button class=go id=startb onclick="start()">Start</button><button class=stop id=stopb onclick="stopAgent()">Stop agent</button>
 <button class=ghost id=resetb onclick="post('/map/reset')" title="Forget the map: start over in a new place">New map</button>
 <span id=status></span></div>
<div class=p id=side><h3>CAMERA</h3>
 <div id=camwrap><img id=cam src="/video"><div id=ins><span id=instext></span><button onclick="post('/done')">Done</button></div></div>
 <div id=st></div>
 <div id=btns>
  <button class=man data-k=drone onclick="cmd('takeoff')">Take off</button><button class=man onclick="cmd('spin')">Spin 360</button>
  <button class=man onclick="cmd('forward',50)">Fwd 50</button><button class=man onclick="cmd('back',50)">Back 50</button>
  <button class=man onclick="cmd('left',50)">Left 50</button><button class=man onclick="cmd('right',50)">Right 50</button>
  <button class=man onclick="cmd('turn_left',45)">Turn L 45</button><button class=man onclick="cmd('turn_right',45)">Turn R 45</button>
  <button class=man data-k=drone onclick="cmd('land')">Land</button><button class=stop onclick="cmd('emergency')">STOP</button><span id=flash></span>
 </div><pre id=log></pre></div>
<div class=p id=view3d><h3>3D MAP, BUILT LIVE (drag to rotate, right-drag to pan, scroll to zoom)</h3>
 <div id=legend><span><i style="background:#d9dee5"></i>free</span><span><i style="background:#c0392b"></i>obstacle</span>
 <span><i style="background:#2a3440"></i>unknown</span><span><i style="background:#4a90d9"></i>visited</span>
 <span><i style="background:#3cdc3c"></i>Grok's plan</span></div>
 <div id=three></div><div id=msg>loading 3D view...</div></div>
<div class=p id=agent><h3>GROK DECISIONS (click a step to open it)</h3>
 <div id=plan><div class=pl>Grok wants to: <b id=ptext></b></div><div class=s id=pwhy></div>
  <div class=prow><input id=pnote placeholder="Why not? (optional, Grok reads it)" onkeydown="if(event.key==='Enter')decide(false)">
  <button class=stop onclick="decide(false)">Reject</button><button class=go onclick="decide(true)">Approve</button></div></div>
 <div id=turns></div><div id=graph></div></div>
<div id=confirm><div class=box><b>Grok thinks it found it. Is this right?</b><div id=cev></div><img id=cimg>
 <div class=row><button class=stop onclick="answer('no')">No, keep searching</button><button class=go onclick="answer('yes')">Yes, that's it</button></div></div></div>
<script>addEventListener('error',e=>{const m=document.getElementById('msg');if(m)m.textContent='3D view error: '+e.message},true)</script>
<script type=importmap>{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
// three.js axes: x = east, y = up, z = south (map rows grow southwards). Map cell (c, r) centre:
// x = -half + (c + .5) * cell, z = -half + (r + .5) * cell.
const el=document.getElementById('three');const r=new THREE.WebGLRenderer({antialias:true});el.appendChild(r.domElement);
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0b0f14);
const cam=new THREE.PerspectiveCamera(55,1,0.05,100);cam.position.set(0,7.5,6.5);
const ctl=new OrbitControls(cam,r.domElement);ctl.target.set(0,0,-0.3);ctl.enableDamping=true;
const COLS='ABCDEFGHIJKLMNOPQRSTUVWXYZ';const cellName=(c,r)=>COLS[c]+(r+1);
// the drone: a group turned by yaw, with a cone pointing where it looks
const drone=new THREE.Group();const cone=new THREE.Mesh(new THREE.ConeGeometry(0.12,0.35,16),
 new THREE.MeshBasicMaterial({color:0x3399ff}));cone.rotation.x=-Math.PI/2;cone.material.depthTest=false;cone.renderOrder=6;drone.add(cone);scene.add(drone);
const stem=new THREE.Mesh(new THREE.CylinderGeometry(0.015,0.015,1,8),new THREE.MeshBasicMaterial({color:0x7cc4ff,depthTest:false,transparent:true,opacity:.8}));
stem.renderOrder=6;scene.add(stem);
let cloud=null;const mat=new THREE.PointsMaterial({size:0.04,vertexColors:true,transparent:true,opacity:.55,depthWrite:false});
// the floor: one canvas texture with every cell, its state and its name
const PX=96;let floor=null,floorKey='';const fc=document.createElement('canvas');const fctx=fc.getContext('2d');
const tex=new THREE.CanvasTexture(fc);tex.anisotropy=8;tex.colorSpace=THREE.SRGBColorSpace;
function drawFloor(j){const n=j.n;fc.width=fc.height=n*PX;const vis=new Set(j.visited);
 const path=new Set((j.plan&&j.plan.path||[]).map(([c,r])=>c+','+r));const tgt=j.plan&&j.plan.path&&j.plan.path.length?j.plan.path[j.plan.path.length-1]:null;
 for(let r=0;r<n;r++)for(let c=0;c<n;c++){const g=j.grid[r][c],x=c*PX,y=r*PX,name=cellName(c,r);
  fctx.fillStyle=g===2?'#7a2a24':g===1?(vis.has(name)?'#3d6f9e':'#a9b1bb'):'#1a222c';fctx.fillRect(x,y,PX,PX);
  if(path.has(c+','+r)){fctx.fillStyle='rgba(60,220,60,.55)';fctx.fillRect(x,y,PX,PX)}
  fctx.strokeStyle='rgba(0,0,0,.55)';fctx.lineWidth=2;fctx.strokeRect(x+1,y+1,PX-2,PX-2);
  fctx.font='bold 26px system-ui';fctx.textAlign='left';fctx.textBaseline='top';
  fctx.fillStyle=g===1&&!vis.has(name)?'rgba(10,15,20,.75)':g===0?'rgba(255,255,255,.22)':'rgba(255,255,255,.6)';fctx.fillText(name,x+9,y+8)}  // corner, so a path over the centre never hides it
 const dc=j.state.drone_cell,dcc=COLS.indexOf(dc[0]),dcr=+dc.slice(1)-1;fctx.strokeStyle='#7cc4ff';fctx.lineWidth=8;
 fctx.strokeRect(dcc*PX+4,dcr*PX+4,PX-8,PX-8);
 if(tgt){fctx.strokeStyle='#3cdc3c';fctx.lineWidth=10;fctx.strokeRect(tgt[0]*PX+5,tgt[1]*PX+5,PX-10,PX-10)}
 tex.needsUpdate=true;
 if(!floor||floor.userData.n!==n){if(floor)scene.remove(floor);const size=n*j.cell;
  floor=new THREE.Mesh(new THREE.PlaneGeometry(size,size),new THREE.MeshBasicMaterial({map:tex,transparent:true,opacity:.9,depthWrite:false}));
  floor.rotation.x=-Math.PI/2;floor.position.y=-0.002;floor.userData.n=n;scene.add(floor)}}
// a text label that always faces the camera
function label(text,color,w=1.1){const c=document.createElement('canvas');c.width=512;c.height=128;const x=c.getContext('2d');
 x.fillStyle='rgba(8,12,16,.85)';x.beginPath();x.roundRect(4,14,504,100,22);x.fill();x.strokeStyle=color;x.lineWidth=6;x.stroke();
 x.fillStyle=color;x.font='bold 58px system-ui';x.textAlign='center';x.textBaseline='middle';x.fillText(text,256,66);
 const t=new THREE.CanvasTexture(c);t.colorSpace=THREE.SRGBColorSpace;
 const s=new THREE.Sprite(new THREE.SpriteMaterial({map:t,depthTest:false}));s.scale.set(w,w/4,1);s.renderOrder=10;return s}

// Grok's planned action in 3D: path at flight height + a column on the target, an arrow for look, a ring for spin
const planGroup=new THREE.Group();scene.add(planGroup);let planKey='';const GREEN=0x3cdc3c;
function centre(j,c,rr){return new THREE.Vector3(-j.half+(c+.5)*j.cell,0,-j.half+(rr+.5)*j.cell)}
function drawPlan(j){planGroup.clear();const p=j.plan;if(!p)return;planGroup.renderOrder=5;const h=Math.max(0.3,j.pose.z),FL=0.04;
 const here=new THREE.Vector3(j.pose.x,h,-j.pose.y);const gm=new THREE.MeshBasicMaterial({color:GREEN,depthTest:false,transparent:true});
 if(p.path&&p.path.length>1){const pts=p.path.map(([c,rr])=>centre(j,c,rr).setY(FL));  // on the floor, over its cells
  const curve=new THREE.CurvePath();for(let i=0;i<pts.length-1;i++)curve.add(new THREE.LineCurve3(pts[i],pts[i+1]));
  planGroup.add(new THREE.Mesh(new THREE.TubeGeometry(curve,64*(pts.length-1),0.05,8,false),gm));
  for(const q of pts.slice(1,-1))planGroup.add(new THREE.Mesh(new THREE.SphereGeometry(0.06,12,8),gm).translateX(q.x).translateY(q.y).translateZ(q.z));
  const t=pts[pts.length-1];const col=new THREE.Mesh(new THREE.BoxGeometry(j.cell*.92,1.2,j.cell*.92),
   new THREE.MeshBasicMaterial({color:GREEN,transparent:true,opacity:.22,depthWrite:false,depthTest:false}));col.position.set(t.x,.6,t.z);planGroup.add(col);
  const edges=new THREE.LineSegments(new THREE.EdgesGeometry(col.geometry),new THREE.LineBasicMaterial({color:GREEN,depthTest:false}));
  edges.position.copy(col.position);planGroup.add(edges);
  const s=label(p.label.toUpperCase(),'#7ef07e');s.position.set(t.x,1.45,t.z);planGroup.add(s)}
 else if(p.heading!==null&&p.heading!==undefined){const a=p.heading*Math.PI/180;
  const dir=new THREE.Vector3(Math.sin(a),0,-Math.cos(a));const from=here.clone().setY(FL),to=from.clone().addScaledVector(dir,1.25);
  planGroup.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.LineCurve3(from,to),8,0.05,8,false),gm));
  const head=new THREE.Mesh(new THREE.ConeGeometry(0.16,0.4,16),gm);head.position.copy(to).addScaledVector(dir,0.2);
  head.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir);planGroup.add(head);
  const s=label(p.label.toUpperCase(),'#7ef07e');s.position.copy(here).addScaledVector(dir,1.6).setY(0.5);planGroup.add(s)}
 else if(p.action==='spin'){const ring=new THREE.Mesh(new THREE.TorusGeometry(0.55,0.03,8,48),gm);ring.rotation.x=Math.PI/2;
  ring.position.copy(here).setY(FL);ring.renderOrder=5;planGroup.add(ring);const s=label('SPIN 360','#7ef07e');s.position.copy(here).setY(0.9);planGroup.add(s)}}
let focusKey='',glide=null;
function focusPlan(j){const p=j.plan;if(!p)return;const k=p.label+JSON.stringify(p.path);if(k===focusKey)return;focusKey=k;
 const here=new THREE.Vector3(j.pose.x,0,-j.pose.y);let far=here.clone().add(new THREE.Vector3(0,0,-1));
 if(p.path&&p.path.length>1){const [c,rr]=p.path[p.path.length-1];far=centre(j,c,rr)}
 else if(p.heading!==null&&p.heading!==undefined){const a=p.heading*Math.PI/180;far=here.clone().add(new THREE.Vector3(Math.sin(a)*1.6,0,-Math.cos(a)*1.6))}
 const mid=here.clone().add(far).multiplyScalar(.5);const span=Math.max(1.5,here.distanceTo(far));
 const dir=cam.position.clone().sub(ctl.target);dir.y=Math.max(dir.y,1.45*Math.hypot(dir.x,dir.z));dir.normalize();
 const dist=Math.min(10,Math.max(5,span*2.2+3.5));
 glide={t0:performance.now(),fromT:ctl.target.clone(),toT:mid,fromP:cam.position.clone(),toP:mid.clone().addScaledVector(dir,dist)}}
function stepGlide(){if(!glide)return;const t=Math.min(1,(performance.now()-glide.t0)/700),e=t*(2-t);
 ctl.target.lerpVectors(glide.fromT,glide.toT,e);cam.position.lerpVectors(glide.fromP,glide.toP,e);if(t===1)glide=null}
ctl.addEventListener('start',()=>{glide=null});  // the operator takes over: stop gliding
function size(){const w=el.clientWidth,h=el.clientHeight;r.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix()}
addEventListener('resize',size);new ResizeObserver(size).observe(el);size();
async function load(){const j=await (await fetch('/points',{signal:AbortSignal.timeout(5000)})).json();const n=j.points.length;
 const pos=new Float32Array(n*3),col=new Float32Array(n*3);
 j.points.forEach((p,i)=>{pos.set([p[0],p[2],-p[1]],i*3);col.set([p[3],p[4],p[5]],i*3)});
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(pos,3));
 g.setAttribute('color',new THREE.BufferAttribute(col,3));
 if(cloud){scene.remove(cloud);cloud.geometry.dispose()}
 cloud=new THREE.Points(g,mat);scene.add(cloud);
 drone.position.set(j.pose.x,Math.max(0.3,j.pose.z),-j.pose.y);drone.rotation.y=-j.pose.yaw*Math.PI/180;
 const hz=Math.max(0.3,j.pose.z);stem.scale.y=hz;stem.position.set(j.pose.x,hz/2,-j.pose.y);
 const fk=JSON.stringify([j.grid,j.visited,j.plan&&j.plan.path,j.state.drone_cell]);if(fk!==floorKey){floorKey=fk;drawFloor(j)}
 const pk=JSON.stringify([j.plan,j.pose]);if(pk!==planKey){planKey=pk;drawPlan(j)}focusPlan(j);
 document.getElementById('st').textContent=`cell ${j.state.drone_cell}  heading ${Math.round(j.pose.yaw)%360}deg  `+
  `frames mapped ${j.state.frames_used}  free ${j.state.cells.free}  obstacle ${j.state.cells.obstacle}`}
async function loop3d(){try{await load();document.getElementById('msg').textContent=''}catch(e){}setTimeout(loop3d,1000)}
loop3d();
(function loop(){requestAnimationFrame(loop);stepGlide();ctl.update();r.render(scene,cam)})();
</script>
<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function post(url,body){const r=await fetch(url,{method:'POST',signal:AbortSignal.timeout(5000),headers:{'Content-Type':'application/json'},
 body:body?JSON.stringify(body):undefined});refresh();return r.json()}
let flashT=0;
async function cmd(name,arg){let j;try{j=await post('/cmd/'+name+(arg?'?arg='+arg:''))}catch(e){j={reply:'no answer from the app'}}
 if(j.reply!=='started'){$('flash').textContent=j.reply;clearTimeout(flashT);flashT=setTimeout(()=>$('flash').textContent='',4000)}}
function setMode(m){post('/mode/'+m)}
function start(){const g=$('goalin').value.trim();if(!g){$('goalin').focus();return}post('/agent/start',{goal:g})}
function stopAgent(){post('/agent/stop')}
function answer(a){post('/confirm/'+a)}
function decide(ok){const n=$('pnote').value;$('pnote').value='';post('/approve',{ok:ok,note:n})}
let shownConfirm=null;
async function refresh(){let j;try{j=await (await fetch('/state',{signal:AbortSignal.timeout(5000)})).json()}catch(e){return}
 $('m-manual').classList.toggle('on',j.mode==='manual');$('m-agent').classList.toggle('on',j.mode==='agent');
 const agent=j.mode==='agent';$('goalin').style.display=$('startb').style.display=$('autol').style.display=agent?'':'none';
 $('auto').checked=!j.approve_each;$('resetb').disabled=j.running;
 $('plan').style.display=j.waiting?'block':'none';
 if(j.plan){$('ptext').textContent=j.plan.text;$('pwhy').textContent='Why: '+j.plan.thought}
 $('stopb').style.display=agent&&j.running?'':'none';$('startb').disabled=j.running;
 if(j.running&&document.activeElement!==$('goalin'))$('goalin').value=j.goal;
 $('status').textContent=j.goal?`${j.running?'searching':'last mission'}: ${j.goal} - ${j.status}`:'';
 document.querySelectorAll('.man').forEach(b=>{b.disabled=agent&&j.running;
  b.style.display=b.dataset.k==='drone'&&j.kind==='webcam'?'none':''});
 $('ins').style.display=j.instruction?'flex':'none';
 if(j.instruction)$('instext').textContent=j.instruction+(j.progress?`  -  ${j.progress}`:'');
 $('log').textContent=j.log.slice().reverse().map(e=>`${e.cmd.padEnd(14)} ${String(e.reply).padEnd(8)} -> x ${e.after.x} y ${e.after.y} yaw ${e.after.yaw}`).join('\\n');
 $('turns').innerHTML=j.goal?'':`<div class=s style="padding:8px 0">${j.running&&j.status.startsWith('first look')?(j.kind==='webcam'?'Looking around first: follow the instruction on the camera.':'Looking around first: a full turn to build the map.'):j.running?'':agent?'Type what to find and press Start.':'Manual mode: use the buttons. Switch to Agent to let Grok search.'}</div>`;
 const c=j.confirm;$('confirm').style.display=c?'flex':'none';
 if(c&&shownConfirm!==c.photo){shownConfirm=c.photo;$('cimg').src='/found/'+c.photo;
  $('cev').textContent=`In ${c.cell}: ${c.evidence}`}}
(async function poll(){await refresh();setTimeout(poll,700)})();
// ---- the mission graph: each step a chain of nodes; click a node for everything behind it ----
const KIND={observe:'Observe',think:'Grok',check:'Safety check',approve:'You',act:'Drone',verify:'Verify',confirm:'You confirm'};
let graphKey='',openNode=null,lastGraph=null;
function nodeHtml(n){return `<div class="nd ${n.status}${openNode===n.id?' sel':''}" onclick="toggleNode('${n.id}')">`+
 `<div class=k>${KIND[n.kind]||n.kind}${n.seconds?' &middot; '+n.seconds+'s':''}</div><div class=l>${esc(n.label)}</div>`+
 (n.detail?`<div class=d>${esc(n.detail)}</div>`:'')+`</div>`}
function detailHtml(n){if(!n)return'';let h=`<div class=det><div class=row><b>${KIND[n.kind]||n.kind}: ${esc(n.label)}</b> <span class=s>(${n.status})</span></div>`;
 if(n.detail)h+=`<div class=row>${esc(n.detail)}</div>`;
 if(n.seen)h+=`<div class=row><span class=s>Grok saw:</span> ${esc(n.seen)}</div>`;
 if(n.model)h+=`<div class=row><span class=s>model:</span> ${esc(n.model)}</div>`;
 const imgs=(n.images||[]).concat(n.photo?[n.photo]:[]);
 if(imgs.length)h+=`<div class=row>${imgs.map(f=>`<a href="/files/${f}" target=_blank><img src="/files/${f}" title="${f}"></a>`).join('')}</div>`;
 if(n.prompt)h+=`<div class=row><a href="/files/${n.prompt}" target=_blank style="color:#7cc4ff">full prompt sent to Grok</a></div>`;
 if(n.reply)h+=`<pre>${esc(JSON.stringify(n.reply,null,1))}</pre>`;
 return h+'</div>'}
function toggleNode(id){openNode=openNode===id?null:id;drawGraph(lastGraph,true)}
function drawGraph(g,force){if(!g)return;const k=JSON.stringify(g)+openNode;if(!force&&k===graphKey)return;graphKey=k;
 const box=$('graph'),atEnd=box.scrollHeight-box.scrollTop-box.clientHeight<40;
 box.innerHTML=g.steps.map((st,si)=>{const now=si===g.steps.length-1&&st.nodes.some(n=>n.status==='running'||n.status==='waiting');
  const sel=st.nodes.find(n=>n.id===openNode);
  return `<div class="st${now?' now':''}"><div class=sh><span class=n>${st.step?'STEP '+st.step:'START'}</span><b>${esc(st.title)}</b>`+
   (g.phoenix&&st.trace?`<a href="${g.phoenix}${st.trace}" target=_blank>trace in Phoenix</a>`:'')+`</div>`+
   `<div class=chain>${st.nodes.map((n,k)=>`<div class=pair>${k?'<span class=arr>&rarr;</span>':''}${nodeHtml(n)}</div>`).join('')}</div>${detailHtml(sel)}</div>`}).join('');
 if(atEnd)box.scrollTop=box.scrollHeight}
(async function pollGraph(){try{lastGraph=await (await fetch('/graph',{signal:AbortSignal.timeout(5000)})).json();drawGraph(lastGraph)}catch(e){}
 setTimeout(pollGraph,800)})();
// camera watchdog: if the picture has not changed for 4 s, reconnect the stream
(function(){const img=$('cam'),c=document.createElement('canvas');c.width=32;c.height=18;const x=c.getContext('2d');
 let last=0,same=0;setInterval(()=>{let h=0;try{x.drawImage(img,0,0,32,18);const d=x.getImageData(0,0,32,18).data;
  for(let i=0;i<d.length;i+=8)h=(h*31+d[i])|0}catch(e){}
  same=h===last?same+1:0;last=h;if(same>=4){same=0;img.src='/video?'+Date.now()}},1000)})();
</script>"""


def phoenix_project_url(name: str = "seek") -> str | None:
    """Link prefix for this project's traces in the local Phoenix, if it is running."""
    import json
    import urllib.request

    base = "http://localhost:6006"
    try:
        with urllib.request.urlopen(f"{base}/v1/projects", timeout=2) as r:
            for p in json.load(r)["data"]:
                if p["name"] == name:
                    return f"{base}/projects/{p['id']}/traces/"
    except Exception:
        return None
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--webcam", type=int, help="camera index (0 = built-in)")
    src.add_argument("--sim", help="360 photo for the simulated drone")
    src.add_argument("--video", help="recorded drone flight: spin replays it, turns show the frame facing that way")
    src.add_argument(
        "--tello", nargs="?", const="auto", metavar="IP", help="real drone; no IP = find it on this network"
    )
    p.add_argument("--goal", help="start in agent mode, searching for this")
    p.add_argument("--fov", type=float, help="diagonal field of view in degrees (default: 78 webcam, 82.6 drone)")
    p.add_argument("--port", type=int, default=8001)
    a = p.parse_args()

    out = Path("runs") / f"seek-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    tracing.setup(out)
    print("loading depth model...")
    depth(np.zeros((360, 640, 3), np.uint8))  # load once at startup, not during the mission

    if a.webcam is not None:
        backend, fov, kind = WebcamBackend(a.webcam), a.fov or WebcamBackend.FOV_DIAG, "webcam"
    else:
        backend = SimBackend(a.sim) if a.sim else VideoBackend(a.video) if a.video else TelloBackend(a.tello)
        fov, kind = a.fov or 82.6, "drone"
    drone = Drone(backend, log_dir=out)
    m = LiveMap(fov_diag=fov)
    drone.attach(m)
    if kind == "webcam":
        drone.takeoff()  # a hand-held camera is always "flying"
    ctx.update(kind=kind, drone=drone, map=m, out=out, phoenix=phoenix_project_url())
    if a.goal:
        start_agent(a.goal)
    print(f"SEEK: http://localhost:{a.port}   files: {out}   traces: http://localhost:6006 (project seek)")
    try:
        uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning", timeout_graceful_shutdown=2)
    finally:
        stop_agent()
        m.save_ply(str(out / "points.ply"))
        tracing.flush()


if __name__ == "__main__":
    main()
