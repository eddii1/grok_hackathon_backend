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

from fleet.drone import Drone
from fleet.grid import cell_name, jpg, parse_cell
from fleet.llm import ask_json, config
from fleet.mapper import LiveMap, depth
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

MIN_HOP = 2  # cells (50 cm each): a goto flies at least 1 m
NEW_CELLS = 3  # a decision that reveals fewer new map cells than this made no exploring progress

PROMPT = """You control an indoor search drone. Goal: find {goal}.

Image 1 is the drone's map seen from above, built from its camera: cells are 50 cm, columns A-{last_col} \
west to east, rows 1-{n} north to south. Grey = never seen, red outline = obstacle, blue tint = already \
visited, orange line = path flown, blue arrow = drone and its heading.
Image 2 is what the camera sees right now (the drone faces {heading_name}).

Map as text (. free, # obstacle, ? unknown, v visited, D drone):
{ascii}

Drone: cell {cell}, heading {heading} ({heading_name}), height {height} m. This is decision {turn}.
Frontier (unknown cells next to free ones, where exploring pays off): {frontier}
Reachable free cells not visited yet: {unvisited}
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
        self.plan: dict | None = None  # the next action, drawn on the map, waiting for approval
        self.approve_each = True  # the operator approves every action before it runs
        self.session = f"{out.parent.name}/{out.name}"  # Phoenix groups one mission's step traces by this
        self._stop, self._answered, self._answer = threading.Event(), threading.Event(), False
        self._decided, self._decision = threading.Event(), (False, "")
        self.graph: list[dict] = []  # the mission as a graph: one entry per step, each a chain of nodes
        self._glock = threading.Condition()  # guards the graph; notified on every change
        self.trace = ""  # the mission's Phoenix trace id
        self.gver = 0  # graph version, bumped on every change (the page streams it)

    # ---------- control from the viewer ----------

    def stop(self) -> None:
        if not self.status.startswith("ended"):
            self.status = "stopping (finishing the current step)..."
        self._stop.set()
        self._answered.set()
        self._decided.set()
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

    def settle(self, seconds: float = 0.8) -> None:
        """Hold still so the mapper sees the new view before anything is decided."""
        time.sleep(seconds)

    # ---------- actions (each returns a one-line result for Grok) ----------

    def face(self, heading: int) -> str:
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

    def plan_goto(self, target: str) -> tuple[list | None, str]:
        """Check a goto and find its path. Returns (path, "") or (None, why it is refused)."""
        try:
            goal = parse_cell(target)
        except (ValueError, IndexError):
            return None, f"refused: '{target}' is not a cell"
        if not (0 <= goal[0] < self.map.n and 0 <= goal[1] < self.map.n):
            return None, f"refused: {target} is outside the map"
        cell, _ = self.here()
        if target == cell:
            return None, f"refused: already in {target}"
        here = parse_cell(cell)
        if max(abs(goal[0] - here[0]), abs(goal[1] - here[1])) < MIN_HOP and any(
            max(abs(c[0] - here[0]), abs(c[1] - here[1])) >= MIN_HOP
            for c in map(parse_cell, self.unvisited_reachable())
        ):
            return None, f"refused: {target} is too close, go at least {MIN_HOP} cells ({MIN_HOP * 50} cm) away"
        if self.mem.visits.get(target, 0) >= 2 and self.unvisited_reachable():
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

    def goto(self, path: list) -> str:
        """Fly a checked path, one straight leg per move command, re-checking the map before each leg."""
        target = cell_name(*path[-1])
        names = [cell_name(*p) for p in path]
        repeats = [c for c in names[1:] if self.mem.visits.get(c)]
        legs: list[tuple[int, list]] = []  # straight legs: (heading, cells)
        for a, b in zip(path, path[1:], strict=False):
            h = STEP[(b[0] - a[0], b[1] - a[1])]
            if legs and legs[-1][0] == h:
                legs[-1][1].append(b)
            else:
                legs.append((h, [b]))
        result = f"arrived in {target} via {'>'.join(names)}"
        for h, cells in legs:
            here = self.here()[0]
            if self.stopped:
                result = f"stopped in {here}: mission stopped"
                break
            if any(self.map.grid()[r, c] == 2 for c, r in cells):  # the map changed on the way
                result = f"stopped in {here}: the way to {target} is now blocked"
                break
            reply = self.face(h)
            if self.stopped or not self.drone.flying:  # LAND or a takeover during the turn: no forward after it
                result = f"stopped in {here}: mission stopped"
                break
            if reply == "ok":
                reply = self.drone.forward(round(self.map.cell * 100 * len(cells)))
            if reply != "ok":
                result = f"stopped in {here}: drone said '{reply}'"
                break
            self.settle(0.5)
            for c, r in cells:
                self.mem.visit(cell_name(c, r), self.drone.pose())
                self.mem.look(cell_name(c, r), h)
        if repeats:
            result += f"; flew over cells visited before: {','.join(repeats)}"
        return result

    def prepare(self, a: dict) -> dict | str:
        """Turn Grok's answer into a concrete plan (drawn on the map), or a refusal."""
        act = a["action"]
        if act == "goto":
            path, why = self.plan_goto(a["cell"])
            if path is None:
                return why
            names = [cell_name(*p) for p in path]
            return {
                "action": "goto",
                "label": f"goto {a['cell']}",
                "path": path,
                "text": f"Go to {a['cell']} via {' > '.join(names)} ({(len(path) - 1) * 50} cm)",
            }
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
        try:
            if not self.approve_each:
                return True, "auto-approved"
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
            if self.stopped or not self._decision[0]:
                self.plan = None

    def act(self, plan: dict) -> str:
        self.status = f"doing: {plan['label']}"
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
            last_col=cell_name(self.map.n - 1, 0)[0],
            n=self.map.n,
            heading=heading,
            heading_name=HEADINGS[heading],
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
                if self.stale >= 2
                else ""
            ),
            unvisited=", ".join(sorted(self.unvisited_reachable(), key=lambda c: parse_cell(c)[::-1])) or "none",
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
                with span("recon: look around", kind="CHAIN", input={"goal": self.goal}) as done:
                    if not self.drone.flying and self.drone.takeoff() != "ok":
                        result = {"end": "takeoff failed"}
                        done(result)
                        return result
                    self.settle(1.0)
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
                self.plan = None
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
        with span(f"act: {plan['label']}", kind="TOOL", input=plan["text"]) as acted:
            result = self.act(plan)
            acted(result)
        failed = result.startswith("stopped") or "failed" in result
        self.update(n, status="failed" if failed else "done", detail=result)
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
