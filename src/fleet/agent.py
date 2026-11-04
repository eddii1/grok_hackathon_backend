"""SEEK agent: plan a route from one overview photo, fly it with a check after every step, replan if needed.

    uv run fleet-agent area.jpg --goal "person in a red jacket" --start C5                 # plan only
    uv run fleet-agent area.jpg --goal "..." --start C5 --fly sim                          # fly on the photo
    uv run fleet-agent area.jpg --goal "..." --start C5 --fly tello                        # fly for real
    uv run fleet-agent area.jpg ... --corners "x,y;x,y;x,y;x,y"                             # angled photo

The loop (each box is a span in Phoenix, http://localhost:6006, project "seek"):

    mission (AGENT)
      plan (CHAIN)          overview + grid -> Grok -> route JSON -> deterministic checks -> moves
      execute (CHAIN)
        step N (TOOL)       move one cell -> hover -> frame -> Grok check: continue / abort / found
      plan (CHAIN)          replan from where the drone is, with what it has already seen
      execute ...
"""

import argparse
import os
import time
from pathlib import Path

import cv2

from fleet import tracing
from fleet.drones import SimDrone, TelloDrone
from fleet.grid import COLS, clean_route, draw_grid, jpg, to_moves, top_down
from fleet.llm import ask_json, config
from fleet.tracing import event, span

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "obstacles": {"type": "array", "items": {"type": "string"}},
        "likely_cells": {"type": "array", "items": {"type": "string"}},
        "route": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"cell": {"type": "string"}, "scan": {"type": "boolean"}},
                "required": ["cell", "scan"],
                "additionalProperties": False,
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["obstacles", "likely_cells", "route", "reason"],
    "additionalProperties": False,
}

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["continue", "abort", "found"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
    "additionalProperties": False,
}

PLAN_PROMPT = """You plan the route of an indoor search drone.
The photo shows the search area from above with a grid: columns {cols} (left to right), rows 1-{rows} \
(row 1 = far side, row {rows} = near side). Each cell is about {cw} x {cd} cm of floor.
The drone is in cell {start}, facing the far side.
Goal: {goal}
{memory}
Return:
- obstacles: cells the drone must not enter (furniture, walls, anything the drone could hit).
- likely_cells: cells where the goal most likely is, best first.
- route: cells in visiting order, starting with {start}. Consecutive cells must share an edge \
(no diagonals). Put scan=true where the drone should stop and look around. \
Reach the likely cells early; at most {max_steps} cells.
- reason: one sentence."""

CHECK_PROMPT = """You are the camera check of a search drone, after step {i} of {n} (now in cell {cell}).
Goal: {goal}
Answer:
- found: the goal is clearly visible in this frame.
- abort: only for danger: the way ahead is blocked within about 1 m, or the frame is black or unusable.
- continue: everything else, including when the goal is not visible yet.
Be brief in reason."""


def plan(
    cfg: dict,
    overview,
    goal: str,
    start: str,
    grid: tuple,
    area: tuple,
    max_steps: int,
    visited: list[str],
    notes: list[str],
    out: Path,
) -> list[dict]:
    """One planning round: Grok proposes, code checks. Returns flyable moves."""
    cols, rows = grid
    cw, cd = area[0] // cols, area[1] // rows
    memory = ""
    if visited:
        memory = (
            f"Already visited (the goal was not seen there): {', '.join(visited)}.\n"
            f"Observations so far: {' | '.join(notes[-8:]) or 'none'}\n"
        )
    with span("plan", input={"goal": goal, "start": start, "visited": visited}) as done:
        gridded = draw_grid(overview, cols, rows)
        (out / f"plan-{len(list(out.glob('plan-*.jpg')))}.jpg").write_bytes(jpg(gridded))
        prompt = PLAN_PROMPT.format(
            cols=f"A-{COLS[cols - 1]}",
            rows=rows,
            cw=cw,
            cd=cd,
            start=start,
            goal=goal,
            memory=memory,
            max_steps=max_steps,
        )
        proposal, secs = ask_json(cfg, prompt, jpg(gridded), PLAN_SCHEMA)
        with span("validate", input=proposal) as checked:
            route, issues = clean_route(proposal, start, cols, rows, max_steps)
            moves = to_moves(route, cw, cd)
            checked({"issues": issues, "moves": moves})
        result = {
            "likely": proposal["likely_cells"],
            "obstacles": proposal["obstacles"],
            "reason": proposal["reason"],
            "moves": moves,
            "issues": issues,
            "llm_seconds": round(secs, 1),
        }
        done(result)
    print(f"\nPLAN from {start} ({secs:.1f}s): {proposal['reason']}")
    print(f"  likely {proposal['likely_cells']}  obstacles {proposal['obstacles']}")
    print("  " + " | ".join(f"{m['cmd']}->{m['to']}{'*' if m['scan'] else ''}" for m in moves))
    if issues:
        print("  fixed:", "; ".join(issues))
    return moves


def execute(cfg: dict, drone, moves: list[dict], goal: str, out: Path) -> dict:
    """Fly the moves one by one. After each: hover, frame, Grok check. Stops on found, abort or error."""
    check_cfg = cfg
    if os.getenv("CHECK_MODEL"):  # the per-step check can use a faster model than the planner
        check_cfg = {**cfg, "model": os.getenv("CHECK_MODEL"), "reasoning": os.getenv("CHECK_REASONING")}
    visited, notes = [], []
    with span("execute", input={"moves": len(moves)}) as done:
        result = {"end": "route done", "cell": None}
        for i, m in enumerate(moves, 1):
            with span(f"step {i}", kind="TOOL", input=m) as step_done:
                reply = drone.send(m["cmd"])
                event("move", cmd=m["cmd"], reply=reply)
                if reply != "ok":
                    drone.send("stop")
                    result = {"end": "error", "cell": visited[-1] if visited else None, "detail": reply}
                    step_done({"reply": reply})
                    break
                visited.append(m["to"])
                for t in range(4 if m["scan"] and isinstance(drone, TelloDrone) else 1):
                    if t:
                        drone.send("cw 90")
                    frame = drone.frame()
                    (out / f"step{len(list(out.glob('step*.jpg'))):03d}-{m['to']}.jpg").write_bytes(frame)
                    prompt = CHECK_PROMPT.format(i=i, n=len(moves), cell=m["to"], goal=goal)
                    check, secs = ask_json(check_cfg, prompt, frame, CHECK_SCHEMA)
                    print(
                        f"  step {i:2d} {m['cmd']:<11} -> {m['to']:<4} {check['decision']:<8} "
                        f"{secs:4.1f}s  {check['reason']}"
                    )
                    if check["decision"] != "continue":
                        break
                step_done({"reply": reply, **check, "check_seconds": round(secs, 1)})
                notes.append(f"{m['to']}: {check['reason']}")
                if check["decision"] in ("found", "abort"):
                    result = {"end": check["decision"], "cell": m["to"], "detail": check["reason"]}
                    break
        result.update(visited=visited, notes=notes)
        done(result)
    return result


def mission(cfg, drone, overview, goal, start, grid, area, max_steps, replans, out: Path) -> dict:
    """Plan -> execute -> replan from the current cell until found, aborted or out of replans."""
    with span("mission", kind="AGENT", input={"goal": goal, "start": start, "grid": grid, "area": area}) as done:
        visited, notes, cell = [], [], start
        result = {"end": "no plan"}
        try:
            if isinstance(drone, TelloDrone) and drone.send("takeoff") != "ok":
                result = {"end": "takeoff failed"}
                done(result)
                return result
            for round_ in range(replans + 1):
                moves = plan(cfg, overview, goal, cell, grid, area, max_steps, visited, notes, out)
                if drone is None or not moves:
                    result = {"end": "planned only" if drone is None else "no moves left", "moves": moves}
                    break
                result = execute(cfg, drone, moves, goal, out)
                visited += result["visited"]
                notes += result["notes"]
                cell = result["cell"] or (visited[-1] if visited else cell)
                if result["end"] != "route done":
                    break
                event("replan", round=round_ + 1, at=cell)
        finally:
            if drone is not None:
                drone.close()
        result["rounds"] = round_ + 1
        done(result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="one photo of the whole search area")
    p.add_argument("--goal", required=True)
    p.add_argument("--start", default="C5")
    p.add_argument("--fly", choices=["sim", "tello"], help="omit to plan only")
    p.add_argument("--corners", help="floor corners in pixels: far-left;far-right;near-right;near-left")
    p.add_argument("--grid", default="5x5", help="columns x rows")
    p.add_argument("--area", default="400x400", help="width x depth in cm")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--replans", type=int, default=2)
    p.add_argument("--provider")
    a = p.parse_args()

    grid = tuple(map(int, a.grid.split("x")))
    area = tuple(map(int, a.area.split("x")))
    if min(area[0] // grid[0], area[1] // grid[1]) < 20:
        raise SystemExit("Cells must be at least 20 cm (the Tello's minimum move).")
    cfg = config(a.provider)
    out = Path("runs") / f"mission-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    tracing.setup(out)

    overview = top_down(cv2.imread(a.image), a.corners, area)
    start = a.start.upper()
    drone = None
    if a.fly == "sim":
        drone = SimDrone(overview, start, *grid)
    elif a.fly == "tello":
        drone = TelloDrone()
    result = mission(cfg, drone, overview, a.goal, start, grid, area, a.max_steps, a.replans, out)
    tracing.flush()
    print(f"\nRESULT: {result['end']} {result.get('cell') or ''} after {result['rounds']} plan(s)")
    print(f"files: {out}   traces: http://localhost:6006 (project seek)")


if __name__ == "__main__":
    main()
