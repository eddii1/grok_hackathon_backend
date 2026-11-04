"""Recon: from N photos taken while the drone spins in place, build a top-down map of the room.

    uv run fleet-recon --pano room_360.jpg --goal "person in a red jacket"     # simulator: cut 8 views from a 360 photo
    uv run fleet-recon --images shots/ --goal "..."                           # real photos named *_<yaw>.jpg

Two parts, each a span in Phoenix (project "seek"):
- geometry: a depth model gives distances per photo; each photo is placed by its yaw (the drone spins
  in one spot, so this needs no matching between photos) and flattened into a grid:
  0 unknown, 1 free, 2 obstacle. Distances are the model's estimate, not calibrated.
- meaning: one Grok call over all photos returns objects, people and hidden spots, each with a bearing
  inside its photo. Code turns bearing + the depth along that bearing into a grid cell.

Map convention: the drone is at the centre facing north (yaw 0), yaw grows clockwise.
Columns A.. go west to east, row 1 is the north edge.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from fleet import tracing
from fleet.grid import cell_name, jpg
from fleet.llm import ask_json, config
from fleet.mapper import DEPTH_MODEL
from fleet.tracing import event, span

FOV_DIAG = 82.6  # Tello / RoboMaster TT camera
W, H = 1280, 720
FOCAL = np.hypot(W / 2, H / 2) / np.tan(np.radians(FOV_DIAG / 2))
HFOV_HALF = np.degrees(np.arctan(W / 2 / FOCAL))  # ~37.4 deg each side

SEMANTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "views": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "yaw": {"type": "integer"},
                    "objects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "bearing": {"type": "number"},
                                "is_person": {"type": "boolean"},
                                "description": {"type": "string"},
                            },
                            "required": ["label", "bearing", "is_person", "description"],
                            "additionalProperties": False,
                        },
                    },
                    "hidden": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"what": {"type": "string"}, "bearing": {"type": "number"}},
                            "required": ["what", "bearing"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["yaw", "objects", "hidden"],
                "additionalProperties": False,
            },
        },
        "room": {"type": "string"},
    },
    "required": ["views", "room"],
    "additionalProperties": False,
}

SEMANTIC_PROMPT = """These {n} photos were taken by a drone spinning in place in one room. Each photo is labelled \
with its yaw (0 = north, clockwise). Search goal: {goal}

For each photo list:
- objects: furniture, doors, openings and every person. bearing = degrees from the photo centre \
(-{h:.0f} = left edge, 0 = centre, +{h:.0f} = right edge). For people, describe clothing and what they hold.
- hidden: places someone could be that the drone cannot see into (behind a sofa, a doorway, a corner), \
with bearing.
room: one sentence describing the room."""


# ---------- inputs ----------


def views_from_pano(path: str, n: int) -> list[tuple[int, np.ndarray]]:
    """Cut n level camera views (Tello field of view) from an equirectangular 360 photo."""
    pano = cv2.imread(path)
    ph, pw = pano.shape[:2]
    u, v = np.meshgrid(np.arange(W) - W / 2 + 0.5, np.arange(H) - H / 2 + 0.5)
    views = []
    for k in range(n):
        yaw = round(360 * k / n)
        a = np.radians(yaw)
        xr, zr = u * np.cos(a) + FOCAL * np.sin(a), -u * np.sin(a) + FOCAL * np.cos(a)
        lon, lat = np.arctan2(xr, zr), np.arctan2(-v, np.hypot(xr, zr))
        px = ((lon / (2 * np.pi) + 0.5) * pw).astype(int) % pw
        py = np.clip(((0.5 - lat / np.pi) * ph).astype(int), 0, ph - 1)
        views.append((yaw, pano[py, px]))
    return views


def views_from_dir(path: str) -> list[tuple[int, np.ndarray]]:
    files = sorted(Path(path).glob("*.jpg"), key=lambda f: int(f.stem.rsplit("_", 1)[-1]))
    return [(int(f.stem.rsplit("_", 1)[-1]), cv2.resize(cv2.imread(str(f)), (W, H))) for f in files]


# ---------- geometry ----------


def depth(img: np.ndarray) -> np.ndarray:
    from fleet.mapper import depth as model_depth

    return model_depth(img)


def to_points(d: np.ndarray, yaw: int) -> np.ndarray:
    """Pixels -> (east, north, metres below camera), using the photo's yaw."""
    v, u = np.mgrid[0:H:4, 0:W:4]
    z = d[::4, ::4]
    x, y = (u - W / 2) * z / FOCAL, (v - H / 2) * z / FOCAL
    a = np.radians(yaw)
    return np.stack([(x * np.cos(a) + z * np.sin(a)).ravel(), (-x * np.sin(a) + z * np.cos(a)).ravel(), y.ravel()], 1)


def occupancy(points: np.ndarray, cam_h: float, cell: float, half: float, max_range: float) -> np.ndarray:
    """0 unknown, 1 free, 2 obstacle. Rows run north to south, columns west to east."""
    n = int(2 * half / cell)
    rng = np.hypot(points[:, 0], points[:, 1])
    points = points[rng < max_range]
    height = cam_h - points[:, 2]
    occ, free = np.zeros((n, n), int), np.zeros((n, n), int)

    def idx(e, nn):
        return ((e + half) / cell).astype(int), ((half - nn) / cell).astype(int)  # column, row

    def add(grid, e, nn):
        c, r = idx(e, nn)
        k = (c >= 0) & (c < n) & (r >= 0) & (r < n)
        np.add.at(grid, (r[k], c[k]), 1)

    solid = (height > 0.3) & (height < 2.0)
    add(occ, points[solid, 0], points[solid, 1])
    dist = np.maximum(np.hypot(points[:, 0], points[:, 1]), 1e-6)
    for t in np.linspace(0, 1, 30)[:-1]:  # rays from the drone to each point are free, stopping one cell short
        tt = np.clip(t, 0, np.maximum(0, 1 - cell / dist))
        add(free, points[:, 0] * tt, points[:, 1] * tt)
    return np.where(occ >= 15, 2, np.where(free > 0, 1, 0))


def place(yaw: int, bearing: float, d: np.ndarray, cell: float, half: float) -> tuple[str, float, float]:
    """Bearing inside a photo -> grid cell, using the median depth of that image column."""
    u = int(np.clip(W / 2 + FOCAL * np.tan(np.radians(bearing)), 0, W - 1))
    dist = float(np.median(d[H // 3 : 2 * H // 3, max(0, u - 10) : u + 10]))
    world = np.radians(yaw + bearing)
    e, nn = dist * np.sin(world), dist * np.cos(world)
    n = int(2 * half / cell)
    c, r = int(np.clip((e + half) / cell, 0, n - 1)), int(np.clip((half - nn) / cell, 0, n - 1))
    return cell_name(c, r), round(e, 2), round(nn, 2)


# ---------- output ----------


def render(grid: np.ndarray, things: list[dict], cell: float) -> np.ndarray:
    n = grid.shape[0]
    px = 48
    img = np.full((n * px, n * px, 3), 255, np.uint8)
    colour = {0: (170, 170, 170), 1: (255, 255, 255), 2: (40, 40, 40)}
    for r in range(n):
        for c in range(n):
            cv2.rectangle(img, (c * px, r * px), ((c + 1) * px, (r + 1) * px), colour[int(grid[r, c])], -1)
            cv2.rectangle(img, (c * px, r * px), ((c + 1) * px, (r + 1) * px), (210, 210, 210), 1)
            cv2.putText(
                img, cell_name(c, r), (c * px + 3, r * px + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 120, 0), 1
            )
    mid = n * px // 2
    cv2.drawMarker(img, (mid, mid), (255, 0, 0), cv2.MARKER_TRIANGLE_UP, 22, 3)
    for t in things:
        x, y = int(mid + t["east"] / cell * px), int(mid - t["north"] / cell * px)
        col = (0, 0, 230) if t.get("is_person") else (0, 150, 0) if t["kind"] == "object" else (0, 140, 255)
        cv2.circle(img, (x, y), 7, col, -1)
        cv2.putText(img, t["label"][:18], (x + 9, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return img


# ---------- recon ----------


def recon(
    cfg: dict,
    views: list[tuple[int, np.ndarray]],
    goal: str,
    out: Path,
    cam_h: float = 1.2,
    cell: float = 0.5,
    half: float = 4.0,
    max_range: float = 4.5,
) -> dict:
    with span("recon", input={"views": [y for y, _ in views], "goal": goal, "cell_m": cell}) as done:
        for yaw, img in views:
            (out / f"view_{yaw:03d}.jpg").write_bytes(jpg(img))

        with span("geometry", input={"model": DEPTH_MODEL}) as geo_done:
            t = time.time()
            depths = {yaw: depth(img) for yaw, img in views}
            points = np.concatenate([to_points(d, yaw) for yaw, d in depths.items()])
            grid = occupancy(points, cam_h, cell, half, max_range)
            counts = {k: int((grid == v).sum()) for k, v in (("unknown", 0), ("free", 1), ("obstacle", 2))}
            geo_done({"seconds": round(time.time() - t, 1), "cells": counts})

        with span("meaning") as sem_done:
            labelled = []
            for yaw, img in views:
                im = img.copy()
                cv2.putText(im, f"yaw {yaw}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
                cv2.putText(im, f"yaw {yaw}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                labelled.append(jpg(cv2.resize(im, (640, 360))))
            prompt = SEMANTIC_PROMPT.format(n=len(views), goal=goal, h=HFOV_HALF)
            sem, secs = ask_json(cfg, prompt, labelled, SEMANTIC_SCHEMA)
            things = []
            for v in sem["views"]:
                if v["yaw"] not in depths:
                    continue
                for o in v["objects"]:
                    c, e, nn = place(v["yaw"], o["bearing"], depths[v["yaw"]], cell, half)
                    things.append(
                        {
                            "kind": "object",
                            "label": o["label"],
                            "is_person": o["is_person"],
                            "description": o["description"],
                            "cell": c,
                            "east": e,
                            "north": nn,
                            "seen_at_yaw": v["yaw"],
                        }
                    )
                for h in v["hidden"]:
                    c, e, nn = place(v["yaw"], h["bearing"], depths[v["yaw"]], cell, half)
                    things.append(
                        {
                            "kind": "hidden",
                            "label": h["what"],
                            "cell": c,
                            "east": e,
                            "north": nn,
                            "seen_at_yaw": v["yaw"],
                        }
                    )
            sem_done({"seconds": round(secs, 1), "room": sem["room"], "things": things})

        n = grid.shape[0]
        result = {
            "room": sem["room"],
            "cell_m": cell,
            "size": n,
            "drone_cell": cell_name(n // 2, n // 2),
            "grid": grid.tolist(),
            "things": things,
        }
        (out / "map.json").write_text(json.dumps(result, indent=2))
        (out / "map.png").write_bytes(jpg(render(grid, things, cell)))
        event("map", path=str(out / "map.png"))
        done({k: v for k, v in result.items() if k != "grid"})
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pano", help="equirectangular 360 photo (simulator)")
    src.add_argument("--images", help="folder of photos named *_<yaw>.jpg")
    p.add_argument("--goal", default="a person")
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--provider")
    a = p.parse_args()

    cfg = config(a.provider)
    out = Path("runs") / f"recon-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    tracing.setup(out)
    views = views_from_pano(a.pano, a.views) if a.pano else views_from_dir(a.images)
    result = recon(cfg, views, a.goal, out)
    tracing.flush()
    print(f"room: {result['room']}")
    for t in result["things"]:
        tag = "PERSON" if t.get("is_person") else t["kind"]
        print(f"  {t['cell']:<4} {tag:<7} {t['label']}" + (f" - {t['description']}" if t.get("description") else ""))
    print(f"map: {out}/map.png   traces: http://localhost:6006")


if __name__ == "__main__":
    main()
