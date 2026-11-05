"""SEEK agent: Grok searches with the live 3D map. Grok picks the next action, code checks and flies it.

Started and stopped from the viewer (uv run fleet-viewer --webcam 0, then type a goal and press Start).
It keeps searching until the operator confirms a find or presses Stop: there is no give-up.

Each turn Grok gets:
  - a top-down photo of the 3D map (colour of the highest point per 10 cm, cell names, grey = unknown,
    red outline = obstacle, blue tint = visited, orange = flown trail, blue arrow = drone)
  - the same grid as text (. free, # obstacle, ? unknown, v visited, D drone)
  - the current camera frame
  - memory: every cell visited (how often, which headings looked at, what was seen) and the last result
and answers with ONE action: goto <cell>, look <heading>, spin, found.
A found pauses the agent until the operator confirms it; a rejection is fed back and the search goes on.

Code makes it safe and keeps Grok from looping:
  - goto: shortest path over known-free cells only (unknown is never flown into), visited cells cost
    more so new ground wins, and the map is re-checked before every straight leg.
  - goto to a cell already visited twice is refused while unvisited cells remain; repeated cells on a
    path are reported back.
  - every pose the drone has been at is stored (memory.jsonl), so the map and Grok see the same history.
"""

import copy
import heapq
import json
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from openinference.instrumentation import using_session

from fleet.drone import Drone, TelloBackend
from fleet.grid import cell_name, jpg, parse_cell
from fleet.llm import ask_json, config
from fleet.mapper import LiveMap, depth, flight_clearance
from fleet.tracing import event, span

HEADINGS = {0: "north", 90: "east", 180: "south", 270: "west"}
STEP = {(0, -1): 0, (1, 0): 90, (0, 1): 180, (-1, 0): 270}  # (dcol, drow) -> heading

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "seen": {"type": "string", "description": "what the camera frame shows, a few words"},
        "thought": {"type": "string", "description": "one sentence: why this action"},
        "action": {"type": "string", "enum": ["goto", "look", "spin", "found"]},
        "cell": {"type": "string", "description": "goto: target cell, found: cell of the goal, else empty"},
        "heading": {
            "type": "integer",
            "enum": [0, 90, 180, 270],
            "description": "look: direction to face; found in the spin views: direction of that view",
        },
        "evidence": {"type": "string", "description": "found: what in the frame matches the goal"},
        "found_in": {
            "type": "string",
            "enum": ["camera", "spin views", "none"],
            "description": "found: image 2 (camera) or image 3 (spin views); else none",
        },
    },
    "required": ["seen", "thought", "action", "cell", "heading", "evidence", "found_in"],
    "additionalProperties": False,
}

CHUNK_CELLS = 2  # a straight leg is flown in hops of at most this many cells (1 m), looking before each
AHEAD_MARGIN = 0.6  # m: keep at least this much space in front after a hop
MIN_HOP = 3  # cells (50 cm each): a goto flies at least 1.5 m (short hops explore little)
NEW_CELLS = 3  # a decision that reveals fewer new map cells than this made no exploring progress

PROMPT = """You control an indoor search drone. Goal: find {goal}.

Image 1 is the drone's map seen from above, built from its camera: cells are 50 cm, columns A-{last_col} \
west to east, rows 1-{n} north to south. Grey = never seen, red outline = obstacle, blue tint = already \
visited, orange line = path flown, blue arrow = drone and its heading.
Image 2 is what the camera sees right now (the drone faces {heading_name}).
The camera sees AHEAD of the drone, not the floor beneath it: the ~1 m look-ahead reference is cell \
{view_cell}. The map already projects objects using metric depth; do NOT add another metre to their positions.

Map as text (. free, # obstacle, ? unknown, v visited, D drone):
{ascii}

Drone: cell {cell}, heading {heading} ({heading_name}), height {height} m. This is decision {turn}.
Frontier (unknown cells next to free ones, where exploring pays off): {frontier}
Reachable free cells not visited yet: {unvisited}
Next scan corner: {corner}
Temporarily avoid these failed destinations: {blocked}
Memory (cell: visits, headings looked at, what was seen there):
{memory}
Last action: {last}
{explore}

Choose ONE action:
- found: the goal is clearly visible. Say where: found_in "camera" (image 2) or "spin views" (image 3, then \
heading = the direction of that view). Every find is checked: the drone turns to face it if \
needed and the live camera must clearly show it. Give its cell and the evidence (what exactly matches the description).
- look: turn to a heading (0 north, 90 east, 180 south, 270 west) to see that side from here.
- spin: turn 360 degrees in place to map all around (costly; useful in a new spot).
- goto: fly to a free cell (., v) at least {min_hop} cells away (shorter hops are refused: the camera already \
sees that far). Only free cells are flyable, so to explore unknown space go to a free cell next to the \
frontier, then look or spin. Prefer far cells you have not visited; do not go back and forth.
There is no giving up: if every reachable cell was visited, go next to the frontier and spin or look \
to discover more of the map. The map grows as the camera sees more.
How to explore well (you are judged on how fast the goal is found):
- Cover new ground every decision. A goto toward the biggest grey (unknown) area beats a look or spin in a \
spot you already know.
- After arriving in a new cell, look toward the unknown side first (cheap); spin only where most \
directions are unknown.
- Never look the same way twice from the same cell, never spin twice in the same area, never go back to \
where you came from unless everything else is explored.
- Scan corner to corner in long passes, not short moves near the centre. Prefer the next scan \
corner above, but choose a DIFFERENT reachable corner if its approach was blocked. Never retry an \
unchanged failed route. Once there, look toward \
unknown space; subsequent passes cover the other corners and remaining unvisited ground.
Also say in "seen" what image 2 shows."""


VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "boolean"},
        "evidence": {"type": "string", "description": "what exactly in the image matches or does not match"},
    },
    "required": ["match", "evidence"],
    "additionalProperties": False,
}

VERIFY_PROMPT = """A search drone is looking for: {goal}
Its planner thinks it saw it: {claim}
Look at this live camera image only. Is {goal} clearly visible in it? Answer match true only if every part of \
the description fits something you can see (a similar but different person or object is false). In evidence, \
say what matches and where in the image, or what is missing."""


def line_cells(a: tuple, b: tuple) -> list:
    """Cells a straight flight from cell a's centre to cell b's centre passes over (a excluded), in order."""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 5) + 1
    out: list = []
    for i in range(1, n + 1):
        t = i / n
        c = (int(round(a[0] + (b[0] - a[0]) * t)), int(round(a[1] + (b[1] - a[1]) * t)))
        if c != a and (not out or out[-1] != c):
            out.append(c)
    return out


def straight_legs(grid: np.ndarray, path: list) -> list[tuple[int, list, float]]:
    """Turn a grid path into as few straight legs as the known-free cells allow: from each corner, fly straight to
    the farthest later path cell whose straight line crosses only free cells. Each leg: (heading deg, cells, length
    in cells). One turn and one long forward instead of a turn every 50 cm."""
    legs, i = [], 0
    while i < len(path) - 1:
        j = i + 1
        for k in range(len(path) - 1, i + 1, -1):
            if all(grid[r, c] == 1 for c, r in line_cells(path[i], path[k])):
                j = k
                break
        (c0, r0), (c1, r1) = path[i], path[j]
        heading = int(round(np.degrees(np.arctan2(c1 - c0, r0 - r1)))) % 360  # 0 north (row up), 90 east
        legs.append((heading, line_cells(path[i], path[j]), float(np.hypot(c1 - c0, r1 - r0))))
        i = j
    return legs


class Memory:
    """Where the drone has been and what it saw there. Saved to memory.jsonl as it grows."""

    def __init__(self, out: Path):
        self.visits: dict[str, int] = {}
        self.looked: dict[str, set[int]] = {}
        self.seen: dict[str, list[str]] = {}
        self.file = open(out / "memory.jsonl", "a")

    def log(self, kind: str, **data) -> None:
        self.file.write(json.dumps({"t": round(time.time(), 2), "kind": kind, **data}) + "\n")
        self.file.flush()

    def visit(self, cell: str, pose: dict) -> None:
        self.visits[cell] = self.visits.get(cell, 0) + 1
        self.log("pose", cell=cell, pose=pose, visits=self.visits[cell])

    def look(self, cell: str, heading: int, seen: str = "") -> None:
        self.looked.setdefault(cell, set()).add(heading)
        if seen:
            notes, note = self.seen.setdefault(cell, []), f"{HEADINGS[heading]}: {seen}"
            if note not in notes:
                notes.append(note)

    def text(self) -> str:
        cells = sorted(set(self.visits) | set(self.looked), key=lambda c: parse_cell(c)[::-1])
        if not cells:
            return "nothing yet"
        lines = []
        for c in cells:
            looked = ",".join(HEADINGS[h] for h in sorted(self.looked.get(c, ()))) or "-"
            saw = "; saw " + " | ".join(self.seen[c][-3:]) if c in self.seen else ""
            lines.append(f"{c}: visited {self.visits.get(c, 0)}x, looked {looked}{saw}")
        return "\n".join(lines)


def partial_field(text: str, key: str) -> str | None:
    """The value of string field `key` in a JSON object that is still streaming in (None if not started)."""
    m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    if not m:
        return None
    v = m.group(1)
    try:
        return json.loads('"' + v.rstrip("\\") + '"')
    except ValueError:
        return v


def shortest_path(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], visits: dict[str, int]
) -> list[tuple[int, int]] | None:
    """Dijkstra over free cells (4-connected). Visited cells cost more, so new ground is preferred."""
    n = grid.shape[0]
    dist, prev, todo = {start: 0.0}, {}, [(0.0, start)]
    while todo:
        d, cur = heapq.heappop(todo)
        if cur == goal:
            path = [cur]
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            return path[::-1]
        if d > dist[cur]:
            continue
        for dc, dr in STEP:
            nxt = (cur[0] + dc, cur[1] + dr)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or grid[nxt[1], nxt[0]] != 1:
                continue
            nd = d + 1 + 0.5 * visits.get(cell_name(*nxt), 0)
            if nd < dist.get(nxt, 1e9):
                dist[nxt], prev[nxt] = nd, cur
                heapq.heappush(todo, (nd, nxt))
    return None


def decision_config(provider: str | None = None) -> dict:
    """The model that makes the decisions: SEEK_MODEL (fast, ~2 s) if set, else the default model."""
    cfg = config(provider)
    if os.getenv("SEEK_MODEL"):
        cfg = {**cfg, "model": os.getenv("SEEK_MODEL"), "reasoning": os.getenv("SEEK_REASONING")}
    return cfg


class Seek:
    """One search mission. run() blocks until the operator confirms a find or stop() is called."""

    def __init__(self, drone: Drone, m: LiveMap, goal: str, out: Path, cfg: dict):
        self.drone, self.map, self.goal, self.out, self.cfg = drone, m, goal, out, cfg
        out.mkdir(parents=True, exist_ok=True)
        self.mem = Memory(out)
        self.last = "none (mission start)"
        self.stale = 0  # decisions in a row that revealed almost nothing new
        self.known = 0  # cells seen (free or obstacle) at the previous decision
        self.views: tuple[str, bytes] | None = None  # (cell, 2x2 sheet) from the latest spin
        self.turns: list[dict] = []  # every decision + what came of it (the viewer shows these)
        self.status = "starting"
        self.confirm: dict | None = None  # a find waiting for the operator
        self.plan: dict | None = None  # drawn on the map while awaiting approval AND while executing
        self.approve_each = True  # the operator approves every action before it runs
        self.session = f"{out.parent.name}/{out.name}"  # Phoenix groups one mission's step traces by this
        self._stop, self._answered, self._answer = threading.Event(), threading.Event(), False
        self._decided, self._decision = threading.Event(), (False, "")
        self._height_ready = threading.Event()
        self.height_waiting = False
        self.graph: list[dict] = []  # the mission as a graph: one entry per step, each a chain of nodes
        self._glock = threading.Condition()  # guards the graph; notified on every change
        self.trace = ""  # the mission's Phoenix trace id
        self.flown_m = 0.0
        self.blocked_targets: dict[str, float] = {}  # cooldown after a failed route, not a permanent obstacle
        self.hop_cells = CHUNK_CELLS  # cells flown per hop between depth look-aheads (operator setting)
        self.gver = 0  # graph version, bumped on every change (the page streams it)

    # ---------- control from the viewer ----------

    def stop(self) -> None:
        if not self.status.startswith("ended"):
            self.status = "stopping (finishing the current step)..."
        self._stop.set()
        self.plan = None
        self._answered.set()
        self._decided.set()
        self._height_ready.set()
        with self._glock:
            self._changed()
        if hasattr(self.drone.b, "cancel"):
            self.drone.b.cancel()  # a person waiting on an instruction: release it

    def answer(self, ok: bool) -> None:
        self._answer = ok
        self._answered.set()

    def decide(self, ok: bool, note: str = "") -> None:
        """The operator approved or rejected the planned action."""
        self._decision = (ok, note.strip())
        self._decided.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ---------- the mission graph (what the dashboard draws) ----------
    # Each step is a chain: observe -> think (Grok) -> check (safety) -> approve (you) -> act,
    # or for a find: observe -> think -> verify (live camera) -> confirm (you).
    # Node status: running, waiting (for you), done, refused, rejected, failed.

    def begin(self, step: int, title: str, trace: str = "") -> None:
        with self._glock:
            self.graph.append({"step": step, "title": title, "trace": trace, "nodes": []})
            self._changed()

    def node(self, kind: str, label: str, status: str = "done", **detail) -> dict:
        with self._glock:
            nodes = self.graph[-1]["nodes"]
            n = {"id": f"{len(self.graph) - 1}.{len(nodes)}", "kind": kind, "label": label, "status": status, **detail}
            nodes.append(n)
            self._changed()
        return n

    def update(self, n: dict, **changes) -> None:
        with self._glock:
            n.update(changes)
            self._changed()

    def _changed(self) -> None:
        self.gver += 1
        self._glock.notify_all()

    def wait_graph(self, ver: int, timeout: float = 15) -> int:
        """Block until the graph is newer than `ver` (or timeout); returns the current version."""
        with self._glock:
            self._glock.wait_for(lambda: self.gver != ver, timeout)
            return self.gver

    def streamer(self, n: dict):
        """on_text callback for a streamed Grok answer: shows the raw JSON and the fields read so far on `n`."""
        last = [0.0]

        def on_text(text: str) -> None:
            now = time.time()
            if now - last[0] < 0.05:  # ~20 updates/s is plenty for the page
                return
            last[0] = now
            fields = {k: partial_field(text, k) for k in ("seen", "thought", "action", "cell", "evidence")}
            self.update(n, stream=text, partial={k: v for k, v in fields.items() if v is not None})

        return on_text

    def graph_snapshot(self) -> list[dict]:
        with self._glock:
            return copy.deepcopy(self.graph)

    # ---------- where the drone is ----------

    def here(self) -> tuple[str, int]:
        p = self.drone.pose()
        return cell_name(*self.map.cell_index(p["x"], p["y"])), int(round(p["yaw"] / 90) * 90) % 360

    def settle(self, seconds: float = 0.45) -> None:
        """Hold still so the mapper sees the new view before anything is decided."""
        time.sleep(seconds)

    # ---------- actions (each returns a one-line result for Grok) ----------

    def face(self, heading: int) -> str:
        if self.stopped:
            return "mission stopped"
        delta = (heading - self.drone.pose()["yaw"] + 540) % 360 - 180
        if abs(delta) < 10:
            return "ok"
        return self.drone.turn_right(round(delta)) if delta > 0 else self.drone.turn_left(round(-delta))

    def look(self, heading: int) -> str:
        reply = self.face(heading)
        self.settle()
        return f"now facing {HEADINGS[heading]}" if reply == "ok" else f"turn failed: {reply}"

    def spin(self) -> str:
        """Turn 360 while the mapper watches; keep the frames closest to N/E/S/W as a 2x2 sheet for Grok."""
        if self.stopped:
            return "spin failed: mission stopped"
        shots, spinning = [], [True]

        def sample():
            while spinning[0]:
                f = self.drone.frame()
                if f is not None:
                    shots.append((self.drone.pose()["yaw"], f))
                time.sleep(0.1)

        t = threading.Thread(target=sample)
        t.start()
        reply = self.drone.spin()
        spinning[0] = False
        t.join()
        self.settle()
        cell, _ = self.here()
        tiles = []
        for h, name in HEADINGS.items():
            self.mem.look(cell, h)
            yaw, f = min(shots, key=lambda s: abs((s[0] - h + 540) % 360 - 180), default=(0, None))
            tile = cv2.resize(f, (480, 270)) if f is not None else np.zeros((270, 480, 3), np.uint8)
            cv2.putText(tile, f"{name} from {cell}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(tile, f"{name} from {cell}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(tile)
        self.views = (cell, jpg(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])))
        return "spun 360, map updated" if reply == "ok" else f"spin failed: {reply}"

    def unvisited_reachable(self) -> list[str]:
        """Free cells the drone could fly to but has not been in yet."""
        g, cell = self.map.grid(), parse_cell(self.here()[0])
        seen, todo = {cell}, [cell]
        while todo:
            c, r = todo.pop()
            for dc, dr in STEP:
                nb = (c + dc, r + dr)
                if 0 <= nb[0] < self.map.n and 0 <= nb[1] < self.map.n and g[nb[1], nb[0]] == 1 and nb not in seen:
                    seen.add(nb)
                    todo.append(nb)
        return [cell_name(*p) for p in seen if not self.mem.visits.get(cell_name(*p))]

    def plan_goto(self, target: str, manual: bool = False) -> tuple[list | None, str]:
        """Check a goto and find its path. Returns (path, "") or (None, why it is refused).
        `manual`: the operator picked it, so the search rules (min distance, revisits) do not apply."""
        try:
            goal = parse_cell(target)
        except (ValueError, IndexError):
            return None, f"refused: '{target}' is not a cell"
        if not (0 <= goal[0] < self.map.n and 0 <= goal[1] < self.map.n):
            return None, f"refused: {target} is outside the map"
        if not manual and self.blocked_targets.get(target, 0) > time.monotonic():
            return None, f"refused: the approach to {target} just failed; choose another corner"
        cell, _ = self.here()
        if target == cell:
            return None, f"refused: already in {target}"
        here = parse_cell(cell)
        if not manual and max(abs(goal[0] - here[0]), abs(goal[1] - here[1])) < MIN_HOP and any(
            max(abs(c[0] - here[0]), abs(c[1] - here[1])) >= MIN_HOP
            for c in map(parse_cell, self.unvisited_reachable())
        ):
            return None, f"refused: {target} is too close, go at least {MIN_HOP} cells ({MIN_HOP * 50} cm) away"
        if not manual and self.mem.visits.get(target, 0) >= 2 and self.unvisited_reachable():
            return None, f"refused: {target} already visited {self.mem.visits[target]}x, search somewhere new"
        g = self.map.grid()
        if g[goal[1], goal[0]] != 1:
            return (
                None,
                f"refused: {target} is {['unknown', 'free', 'an obstacle'][int(g[goal[1], goal[0]])]}, not free",
            )
        path = shortest_path(g, parse_cell(cell), goal, self.mem.visits)
        if path is None:
            return None, f"refused: no known-free path from {cell} to {target}"
        return path, ""

    def scan_corner(self) -> str | None:
        """Choose a far, unvisited corner of the reachable free area, then work through the remaining ground."""
        here = parse_cell(self.here()[0])
        cells = sorted(
            p
            for p in map(parse_cell, self.unvisited_reachable())
            if p != here and self.blocked_targets.get(cell_name(*p), 0) <= time.monotonic()
        )
        if not cells:
            return None

        def distance(p):
            return (p[0] - here[0]) ** 2 + (p[1] - here[1]) ** 2

        far = [p for p in cells if max(abs(p[0] - here[0]), abs(p[1] - here[1])) >= MIN_HOP]
        cells = far or cells
        left, right = min(p[0] for p in cells), max(p[0] for p in cells)
        top, bottom = min(p[1] for p in cells), max(p[1] for p in cells)
        corners = [
            min(cells, key=lambda p: ((p[0] - c) ** 2 + (p[1] - r) ** 2, -distance(p)))
            for c, r in ((left, top), (right, top), (left, bottom), (right, bottom))
        ]
        # A distant corner with a direct approach beats an even farther corner behind a long detour.
        options = []
        for p in corners:
            path, _ = self.plan_goto(cell_name(*p), manual=True)
            if path:
                options.append((distance(p) ** 0.5 - 0.4 * (len(path) - 1), p))
        return cell_name(*max(options)[1]) if options else None

    def goto_plan(self, path: list) -> dict:
        """Publish the same straight legs the executor will fly, rather than a misleading grid staircase."""
        legs = straight_legs(self.map.grid(), path)
        target = cell_name(*path[-1])
        return {
            "action": "goto",
            "label": f"goto {target}",
            "path": [path[0]] + [p for _, cells, _ in legs for p in cells],
            "route": [path[0]] + [cells[-1] for _, cells, _ in legs],
            "text": f"Scan to {target} ({sum(dist for _, _, dist in legs) * self.map.cell:.1f} m)",
        }

    def goto(self, path: list) -> str:
        """Keep the destination through up to two local detours, without asking Grok to choose again."""
        target = cell_name(*path[-1])
        result = "stopped: no route"
        for attempt in range(3):
            self.plan = {**(self.plan or {}), **self.goto_plan(path)}
            result = self.fly_path(path)
            if not result.startswith("stopped") or self.stopped or not self.drone.flying or self.drone.holding:
                break
            if attempt == 2 or "drone said" in result or "no reliable" in result:
                break
            new_path, why = self.plan_goto(target, manual=True)
            if new_path is None:
                result += f"; cannot detour: {why}"
                break
            if new_path == path:
                break
            event("reroute", target=target, reason=result, attempt=attempt + 1)
            path = new_path
        if result.startswith("stopped") and not self.stopped:
            self.blocked_targets[target] = time.monotonic() + 45
        return result

    def fly_path(self, path: list) -> str:
        """Fly a checked path, one straight leg per move command, re-checking the map before each leg."""
        target = cell_name(*path[-1])
        names = [cell_name(*p) for p in path]
        repeats = [c for c in names[1:] if self.mem.visits.get(c)]
        legs = straight_legs(self.map.grid(), path)  # any-angle straight lines, not a staircase
        result = f"arrived in {target} via {'>'.join(names)}"
        for _, cells, _ in legs:
            here = self.here()[0]
            if self.stopped:
                result = f"stopped in {here}: mission stopped"
                break
            if any(self.map.grid()[r, c] == 2 for c, r in cells):  # the map changed on the way
                result = f"stopped in {here}: the way to {target} is now blocked"
                break
            end_x, end_y = self.map.cell_centre(*cells[-1])
            stop = ""
            for _ in range(100):  # bounded even if the position estimate fails to advance
                if self.stopped or not self.drone.flying or self.drone.holding:
                    stop = "mission stopped"
                    break
                pose = self.drone.pose()
                dx, dy = end_x - pose["x"], end_y - pose["y"]
                remaining = float(np.hypot(dx, dy))
                if remaining < 0.2:
                    break
                h = int(round(np.degrees(np.arctan2(dx, dy)))) % 360
                reply = self.face(h)
                if reply != "ok":
                    stop = f"drone said '{reply}'"
                    break
                self.settle(0.35)
                before = self.ahead()
                if before is None:
                    stop = "no reliable forward depth; hold and look another way"
                    break
                # Hop is a MAXIMUM, not an all-or-nothing move. Keep the existing 60 cm stand-off.
                cm = int(np.floor(min(self.hop_cells * self.map.cell, remaining, before - AHEAD_MARGIN) * 100))
                if cm < 20:
                    self.map.record_blockage(before, self.drone.pose())
                    stop = f"something {before:.1f} m ahead; no safe 20 cm hop with {AHEAD_MARGIN:.1f} m clearance"
                    event("hop blocked", target=target, depth_m=round(before, 2), pose=self.drone.pose())
                    break
                if self.stopped or not self.drone.flying:
                    stop = "mission stopped"
                    break
                pose = self.drone.pose()
                a = np.radians(pose["yaw"])
                end = (pose["x"] + cm / 100 * np.sin(a), pose["y"] + cm / 100 * np.cos(a))
                hop = line_cells(self.map.cell_index(pose["x"], pose["y"]), self.map.cell_index(*end))
                grid = self.map.grid()
                if any(grid[r, c] != 1 for c, r in hop):
                    stop = f"the next hop to {target} is no longer known-free"
                    break
                event("hop planned", target=target, commanded_m=cm / 100, depth_m=round(before, 2), pose=pose)
                reply = self.drone.forward(cm)
                if reply != "ok":
                    stop = f"drone said '{reply}'"
                    break
                self.flown_m += cm / 100
                actual = self.drone.pose()
                crossed = line_cells(
                    self.map.cell_index(pose["x"], pose["y"]), self.map.cell_index(actual["x"], actual["y"])
                )
                for c, r in crossed:
                    self.mem.visit(cell_name(c, r), self.drone.pose())
                    self.mem.look(cell_name(c, r), int(round(h / 90) * 90) % 360)
                self.settle(0.35)
                after = self.ahead()
                if before is not None and after is not None and before < 4:  # how far it really went, by depth
                    event(
                        "hop",
                        commanded_m=cm / 100,
                        depth_before=round(before, 2),
                        depth_after=round(after, 2),
                        moved_by_depth=round(before - after, 2),
                    )
            else:
                stop = "position estimate did not converge to the waypoint"
            if stop:
                result = f"stopped in {self.here()[0]}: {stop}"
                break
        if repeats:
            result += f"; flew over cells visited before: {','.join(repeats)}"
        return result

    def ahead(self) -> float | None:
        """Check the swept body corridor and integrate this SAME settled view before the next hop."""
        frame = self.drone.frame()
        if frame is None:
            return None
        d = depth(frame)
        self.map.integrate(frame, d, self.drone.pose())
        clearance = flight_clearance(d, self.map.fov_diag)
        tof = self.drone.b.tof_cm() if isinstance(self.drone.b, TelloBackend) else None
        event("clearance", depth_m=clearance, tof_cm=tof, pose=self.drone.pose())
        return min(clearance, tof / 100) if clearance is not None and tof is not None else clearance

    def prepare(self, a: dict) -> dict | str:
        """Turn Grok's answer into a concrete plan (drawn on the map), or a refusal."""
        act = a["action"]
        if act == "goto":
            target = a["cell"].upper()
            path, why = self.plan_goto(target)
            if path is None:
                corner = self.scan_corner()
                if corner and corner != target:
                    path, why = self.plan_goto(corner)
            if path is None:
                return why
            return self.goto_plan(path)
        if act == "look":
            return {
                "action": "look",
                "label": f"look {HEADINGS[a['heading']]}",
                "heading": a["heading"],
                "text": f"Turn to face {HEADINGS[a['heading']]} ({a['heading']} deg)",
            }
        return {"action": "spin", "label": "spin", "text": "Turn a full circle here to map all around"}

    def approve(self, plan: dict, a: dict) -> tuple[bool, str]:
        """Show the plan and wait for the operator (unless approve_each is off)."""
        self.plan = {**plan, "thought": a["thought"]}
        ok = False
        try:
            if not self.approve_each:
                ok = not self.stopped
                return ok, "auto-approved" if ok else "mission stopped"
            self._decided.clear()
            self.status = f"waiting for your approval: {plan['label']}"
            with span("operator approval", kind="TOOL", input={"plan": plan["text"], "why": a["thought"]}) as done:
                while not self._decided.wait(0.2):
                    if not self.approve_each:  # switched to automatic while waiting
                        self._decision = (True, "auto-approved")
                        break
                ok, note = self._decision if not self.stopped else (False, "mission stopped")
                done({"approved": ok, "note": note})
            return ok, note
        finally:
            if self.stopped or not ok:
                self.plan = None

    def act(self, plan: dict) -> str:
        if self.stopped:
            return "stopped: mission stopped"
        self.status = f"doing: {plan['label']}"
        self.plan = plan
        try:
            if plan["action"] == "goto":
                return self.goto(plan["path"])
            if plan["action"] == "look":
                return self.look(plan["heading"])
            return self.spin()
        finally:
            self.plan = None

    # ---------- the loop ----------

    def turn(self, i: int, on_text=None) -> dict:
        cell, heading = self.here()
        frame = self.drone.frame()
        if frame is not None:  # map exactly what the camera sees now, before deciding (not only the background feed)
            with span("depth: map the current view", kind="TOOL"):
                self.map.integrate(frame, depth(frame), self.drone.pose())
        counts = self.map.counts()
        known = counts["free"] + counts["obstacle"]
        self.stale = self.stale + 1 if known - self.known < NEW_CELLS else 0
        self.known = known
        bird = self.map.bird(visited=self.mem.visits)
        (self.out / f"turn{i:02d}-map.jpg").write_bytes(bird)
        (self.out / f"turn{i:02d}-cam.jpg").write_bytes(jpg(frame))
        st = self.map.state()
        prompt = PROMPT.format(
            goal=self.goal,
            last_col=cell_name(self.map.n - 1, 0)[:-1],
            n=self.map.n,
            heading=heading,
            heading_name=HEADINGS[heading],
            view_cell=st["camera_view"]["cell"],
            ascii=self.map.ascii(self.mem.visits),
            cell=cell,
            height=st["height_m"],
            turn=i,
            frontier=", ".join(st["frontier"][:40]) or "none",
            memory=self.mem.text(),
            last=self.last,
            min_hop=MIN_HOP,
            explore=(
                f"Exploring: the last {self.stale} decisions revealed almost nothing new and {self.goal} is not found "
                f"yet. Go FAR: pick a goto target {MIN_HOP + 2}+ cells away, toward the biggest unknown (grey) area "
                "or the frontier on the other side of the map, then spin there."
                if self.stale >= 1
                else ""
            ),
            unvisited=", ".join(sorted(self.unvisited_reachable(), key=lambda c: parse_cell(c)[::-1])) or "none",
            corner=self.scan_corner() or "none: look toward unknown space to extend the free area",
            blocked=", ".join(c for c, until in self.blocked_targets.items() if until > time.monotonic()) or "none",
        )
        images = [bird, jpg(cv2.resize(frame, (960, 540)))]
        if self.views:
            prompt += (
                f"\nImage 3: the four views from your latest spin, in {self.views[0]} (north, east / south, west)."
            )
            images.append(self.views[1])
            (self.out / f"turn{i:02d}-views.jpg").write_bytes(self.views[1])
        names = ["map", "cam", "views"][: len(images)]
        (self.out / f"turn{i:02d}-prompt.txt").write_text(
            f"model: {self.cfg['model']}  reasoning: {self.cfg.get('reasoning')}\n"
            f"images: {', '.join(f'turn{i:02d}-{n}.jpg' for n in names)}\n\n{prompt}\n"
        )
        with span("think", input={"cell": cell, "heading": heading, "last": self.last}) as done:
            a, secs = ask_json(self.cfg, prompt, images, ACTION_SCHEMA, on_text)
            done({**a, "seconds": round(secs, 1)})
        self.record = {
            "turn": i,
            "cell": cell,
            "heading": heading,
            "model": self.cfg["model"],
            "seconds": round(secs, 1),
            "reply": a,
        }
        self.mem.look(cell, heading, a["seen"])
        print(
            f"[{i:2d}] {cell} {HEADINGS[heading]:<5} {secs:4.1f}s  {a['action']} {a['cell'] or ''}"
            f"{a['heading'] if a['action'] == 'look' else ''}  - {a['thought']}"
        )
        return a

    def publish(self, result: str) -> None:
        """Write turnNN-reply.json (Grok's answer + what the harness did with it) and show it in the viewer."""
        rec = {**self.record, "result": result}
        (self.out / f"turn{rec['turn']:02d}-reply.json").write_text(json.dumps(rec, indent=1))
        self.turns.append(rec)
        self.mem.log("action", **rec)
        event("action", **rec)
        print(f"     -> {result}")

    def ask_operator(self, a: dict) -> bool:
        """Pause on a find until the operator says yes or no (or stops the mission)."""
        photo = self.out / f"found-{len(self.turns):02d}.jpg"
        photo.write_bytes(jpg(self.drone.frame()))
        self._answered.clear()
        self.confirm = {"cell": a["cell"], "evidence": a["evidence"], "photo": photo.name}
        self.status = "found? waiting for the operator"
        with span("operator", kind="TOOL", input=self.confirm) as done:
            self._answered.wait()
            ok = self._answer and not self.stopped
            done({"confirmed": ok})
        self.confirm = None
        return ok

    def run(self) -> dict:
        """One mission = one Phoenix trace, every step a child of the mission span:
        mission: find <goal> (AGENT)
          recon: look around -> step 1: goto H8 -> step 2: look east -> ...
            each step: think (Grok call) -> verify -> operator approval -> act (drone commands)"""
        result = {"end": "stopped"}
        with (
            using_session(self.session),
            span(
                f"mission: find {self.goal}", kind="AGENT", input={"goal": self.goal}, **{"session.id": self.session}
            ) as mission,
        ):
            self.trace = mission.trace_id
            try:
                self.begin(-1, "takeoff and height check", self.trace)
                self.status = "taking off" if not self.drone.flying else "preparing flight"
                takeoff = self.node("act", self.status, "running")
                if not self.drone.flying and self.drone.takeoff() != "ok":
                    result = {"end": "takeoff failed"}
                    self.update(takeoff, status="failed", detail="takeoff failed")
                    return result
                self.update(takeoff, status="done", detail=f"airborne at {self.drone.pose()['z']:.1f} m")
                if self.stopped:
                    return result
                self.status = "adjust height, then press Begin search"
                self.height_waiting = True
                height = self.node(
                    "approve", "set flight height", "waiting",
                    detail="Use Up / Down to adjust altitude. No scan or Grok reasoning until you press Begin search.",
                )
                self._height_ready.wait()  # deliberately required even when Automatic is on
                self.height_waiting = False
                self.update(
                    height, status="rejected" if self.stopped else "done",
                    detail="mission stopped" if self.stopped else f"height confirmed: {self.drone.pose()['z']:.1f} m",
                )
                if self.stopped or not self.drone.flying or self.drone.holding:
                    return result
                self.map.reset()  # discard the takeoff/height-adjustment map; build at the chosen altitude
                with span("recon: look around", kind="CHAIN", input={"goal": self.goal}) as done:
                    self.settle(1.0)
                    if self.stopped:
                        return result
                    self.mem.visit(self.here()[0], self.drone.pose())
                    self.status = "first look around (spin)"
                    self.begin(0, f"recon: find {self.goal}", self.trace)
                    n = self.node("act", "look around (spin 360)", "running", detail="build the map before deciding")
                    self.last = self.spin()  # bootstrap the map before the first decision
                    self.update(n, status="done", detail=self.last)
                    done(self.last)
                i = 0
                while not self.stopped:
                    i += 1
                    if self.step(i):
                        result = {"end": "found", "cell": self.turns[-1]["reply"]["cell"]}
                        break
            except Exception as e:  # never leave a drone in the air on a crash
                result = {"end": f"error: {e}", "landing": self.drone.land()}
                raise
            finally:
                self.status = f"ended: {result['end']}"
                self.height_waiting = False
                self.plan = None
                with self._glock:
                    self._changed()
                event("mission end", goal=self.goal, **result, visits=self.mem.visits)
                mission({**result, "steps": len(self.turns)})
        return result

    def approve_with_node(self, plan: dict, a: dict) -> tuple[bool, str]:
        n = self.node("approve", "waiting for you" if self.approve_each else "automatic", "waiting")
        ok, note = self.approve(plan, a)
        auto = note == "auto-approved"
        self.update(
            n,
            label="automatic" if auto else "you approved" if ok else "you rejected",
            status="done" if ok else "rejected",
            detail=note if note and not auto else "",
        )
        return ok, note

    def act_with_node(self, plan: dict) -> str:
        n = self.node("act", plan["label"], "running", detail=plan["text"])
        self.flown_m = 0.0  # metres flown forward by this action (the graph's flight path shows it)
        with span(f"act: {plan['label']}", kind="TOOL", input=plan["text"]) as acted:
            result = self.act(plan)
            acted(result)
        failed = result.startswith("stopped") or "failed" in result
        self.update(n, status="failed" if failed else "done", detail=result, flown=round(self.flown_m, 1))
        return result

    def verify(self, a: dict, n: dict | None = None) -> tuple[bool, str]:
        """A focused second look at a possible find: one yes/no question about the live camera frame."""
        frame = jpg(cv2.resize(self.drone.frame(), (960, 540)))
        prompt = VERIFY_PROMPT.format(goal=self.goal, claim=a["evidence"])
        with span("verify on the live camera", kind="CHAIN", input={"claim": a["evidence"]}) as done:
            v, secs = ask_json(self.cfg, prompt, frame, VERIFY_SCHEMA, self.streamer(n) if n else None)
            done({**v, "seconds": round(secs, 1)})
        print(f"     verify {'yes' if v['match'] else 'no'} ({secs:.1f}s): {v['evidence']}")
        return v["match"], v["evidence"]

    def check_find(self, a: dict) -> str | None:
        """A possible find: face it if it was only in the spin views, then verify on the live camera.
        Returns None when it is verified (ask the operator), else what to tell Grok."""
        cell, heading = self.here()
        if a.get("found_in") != "camera" and heading != a["heading"]:
            plan = self.prepare({**a, "action": "look"})
            plan["text"] = f"Turn to face {HEADINGS[a['heading']]} to check a possible find: {a['evidence']}"
            self.node("check", "seen only in the spin views: turn to face it first", detail=plan["text"])
            ok, note = self.approve_with_node(plan, a)
            if self.stopped:
                return "mission stopped"
            if not ok:
                return "the operator rejected turning to check your find" + (f": {note}" if note else "")
            self.act_with_node(plan)
        self.status = "checking a possible find on the live camera"
        v = self.node("verify", "checking on the live camera", "running", detail=a["evidence"])
        ok, evidence = self.verify(a, v)
        self.update(
            v,
            label="verified: it matches" if ok else "not it",
            status="done" if ok else "refused",
            detail=evidence,
            stream=None,
            partial=None,
        )
        if ok:
            a["evidence"] = evidence
            return None
        cell, heading = self.here()
        self.mem.look(cell, heading, f"checked, NOT the goal: {evidence}")
        return (
            f"you thought you saw {self.goal} ({a['evidence']}); facing {HEADINGS[heading]} from {cell} the "
            f"live camera does not clearly show it ({evidence}). Search elsewhere."
        )

    def step(self, i: int) -> bool:
        """One decision: think, (operator approves), act. Returns True when the operator confirmed a find."""
        with span(f"step {i}", kind="AGENT", input={"goal": self.goal, "last": self.last}) as done:
            self.status = f"thinking (decision {i})"
            self.begin(i, "thinking", self.trace)
            cell, heading = self.here()
            images = [f"turn{i:02d}-map.jpg", f"turn{i:02d}-cam.jpg"] + (
                [f"turn{i:02d}-views.jpg"] if self.views else []
            )
            self.node(
                "observe", f"{cell}, facing {HEADINGS[heading]}", detail=f"last result: {self.last}", images=images
            )
            think = self.node("think", "Grok is deciding", "running", model=self.cfg["model"])
            a = self.turn(i, self.streamer(think))
            label = a["action"] + (
                f" {a['cell']}"
                if a["action"] in ("goto", "found")
                else f" {HEADINGS[a['heading']]}"
                if a["action"] == "look"
                else ""
            )
            done.rename(f"step {i}: {label}")
            self.update(
                think,
                status="done",
                label=label,
                detail=a["thought"],
                seen=a["seen"],
                seconds=self.record["seconds"],
                reply=a,
                prompt=f"turn{i:02d}-prompt.txt",
                stream=None,
                partial=None,
            )
            with self._glock:
                self.graph[-1]["title"] = label
            if self.stopped:
                done("stopped")
                return False
            if a["action"] == "found":
                why_not = self.check_find(a)
                if why_not is not None:
                    self.last = why_not
                else:
                    c = self.node("confirm", "is this it? waiting for you", "waiting", detail=a["evidence"])
                    yes = self.ask_operator(a)
                    self.update(
                        c,
                        label="you confirmed it" if yes else "you said: not it",
                        status="done" if yes else "rejected",
                        photo=f"found-{len(self.turns):02d}.jpg",
                    )
                if why_not is None and yes:
                    self.publish("found - verified on the live camera and confirmed by the operator")
                    done(self.turns[-1])
                    return True
                elif why_not is None:
                    self.last = (
                        f"the operator checked your find in {a['cell']} and said it is NOT {self.goal} "
                        "(maybe a reflection, picture or similar object); keep searching"
                    )
                    self.mem.look(self.here()[0], self.here()[1], f"NOT the goal: {a['evidence']}")
            else:
                plan = self.prepare(a)
                if isinstance(plan, str):
                    self.last = plan  # refused by the checks: nothing to approve, Grok hears why
                    self.node("check", "refused", "refused", detail=plan)
                else:
                    safe = (
                        f"path over known-free cells: {plan['text']}"
                        if plan["action"] == "goto"
                        else f"turn in place, no path to check: {plan['text']}"
                    )
                    self.node("check", "safe to fly" if plan["action"] == "goto" else "turn in place", detail=safe)
                    ok, note = self.approve_with_node(plan, a)
                    if self.stopped:
                        self.last = "mission stopped"
                    elif not ok:
                        self.last = (
                            f"the operator rejected your plan ({plan['text']})"
                            + (f": {note}" if note else "")
                            + ". Choose something else."
                        )
                    else:
                        self.last = self.act_with_node(plan)
            self.publish(self.last)
            done(self.turns[-1])
        return False
