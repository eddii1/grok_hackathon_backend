"""Eval: does Grok's planned route reach the target sooner than a fixed lawnmower sweep?

    uv run fleet-eval eval/cases.json

cases.json: [{"image", "goal", "start", "target", "corners"?}]  (target = the cell the person stands in)
Each case is one trace in Phoenix (project "seek-eval").
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from fleet import tracing
from fleet.agent import plan
from fleet.grid import lawnmower, top_down
from fleet.llm import config
from fleet.tracing import span


def steps_to(cells: list[str], target: str) -> int | None:
    return cells.index(target) if target in cells else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases")
    p.add_argument("--grid", default="5x5")
    p.add_argument("--area", default="400x400")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--provider")
    a = p.parse_args()

    grid = tuple(map(int, a.grid.split("x")))
    area = tuple(map(int, a.area.split("x")))
    cfg = config(a.provider)
    out = Path("runs") / f"eval-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    tracing.setup(out, project="seek-eval")

    rows = []
    for case in json.loads(Path(a.cases).read_text()):
        start, target = case["start"].upper(), case["target"].upper()
        with span("eval case", input=case) as done:
            overview = top_down(cv2.imread(case["image"]), case.get("corners"), area)
            moves = plan(cfg, overview, case["goal"], start, grid, area, a.max_steps, [], [], out)
            cells = [start] + [m["to"] for m in moves]
            row = {
                "image": case["image"],
                "target": target,
                "route": cells,
                "grok_steps": steps_to(cells, target),
                "sweep_steps": steps_to(lawnmower(start, *grid), target),
            }
            done(row)
        rows.append(row)
        print(f"  target {target}: grok {row['grok_steps']} steps, sweep {row['sweep_steps']} steps")
    (out / "eval.json").write_text(json.dumps(rows, indent=2))
    tracing.flush()

    hit = [r for r in rows if r["grok_steps"] is not None]
    grok = np.mean([r["grok_steps"] for r in hit]) if hit else float("nan")
    sweep = np.mean([r["sweep_steps"] for r in rows if r["sweep_steps"] is not None])
    print(
        f"\nGrok route reached the target in {len(hit)}/{len(rows)} cases, mean {grok:.1f} steps "
        f"(sweep: {sweep:.1f}). Details: {out}/eval.json"
    )


if __name__ == "__main__":
    main()
