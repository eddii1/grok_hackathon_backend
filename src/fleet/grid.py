"""The search area as a grid of floor cells, and the deterministic route checks. No LLM, no drone.

Cells: columns A.. left to right, rows 1.. with row 1 on the far side (the side the drone faces).
The drone keeps one heading and moves one cell at a time: forward/back/left/right.
"""

import cv2
import numpy as np

COLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_cell(name: str) -> tuple[int, int]:
    return COLS.index(name[0].upper()), int(name[1:]) - 1


def cell_name(c: int, r: int) -> str:
    return f"{COLS[c]}{r + 1}"


def top_down(img: np.ndarray, corners: str | None, area: tuple[int, int]) -> np.ndarray:
    """Warp the photo so the flight area fills the frame as seen from above.

    corners: floor corners in pixels, "x,y;x,y;x,y;x,y" ordered far-left, far-right, near-right, near-left.
    Without corners the photo is used as is, which only works for photos taken from above.
    """
    if not corners:
        return img
    src = np.float32([float(v) for v in corners.replace(";", ",").split(",")]).reshape(4, 2)
    w, h = area[0] * 2, area[1] * 2  # 2 px per cm
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))


def draw_grid(img: np.ndarray, cols: int, rows: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for c in range(1, cols):
        cv2.line(out, (w * c // cols, 0), (w * c // cols, h), (0, 255, 255), 2)
    for r in range(1, rows):
        cv2.line(out, (0, h * r // rows), (w, h * r // rows), (0, 255, 255), 2)
    scale = max(h, w) / 1200
    for c in range(cols):
        for r in range(rows):
            org = (w * c // cols + 6, h * r // rows + int(30 * scale))
            cv2.putText(out, cell_name(c, r), org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
            cv2.putText(out, cell_name(c, r), org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 2)
    return out


def jpg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def clean_route(plan: dict, start: str, cols: int, rows: int, max_steps: int) -> tuple[list[dict], list[str]]:
    """Make the planned route flyable: in bounds, no obstacles, starts at start, one cell per step."""
    issues = []
    blocked = set()
    for o in plan["obstacles"]:
        try:
            blocked.add(parse_cell(o))
        except (ValueError, IndexError):
            pass
    route = [{"cell": parse_cell(start), "scan": False}]
    for step in plan["route"]:
        try:
            c, r = parse_cell(step["cell"])
        except (ValueError, IndexError):
            issues.append(f"bad cell {step['cell']}")
            continue
        if not (0 <= c < cols and 0 <= r < rows):
            issues.append(f"out of bounds {step['cell']}")
            continue
        if (c, r) in blocked:
            issues.append(f"obstacle {step['cell']}")
            continue
        pc, pr = route[-1]["cell"]
        if (c, r) == (pc, pr):
            route[-1]["scan"] |= step["scan"]
            continue
        if abs(c - pc) + abs(r - pr) > 1:  # fill the gap: columns first, then rows
            path = [(x, pr) for x in range(pc, c, 1 if c > pc else -1)][1:]
            path += [(c, y) for y in range(pr, r, 1 if r > pr else -1)]
            if any(p in blocked for p in path):
                issues.append(f"gap through obstacle, skipped {step['cell']}")
                continue
            issues.append(f"filled jump {cell_name(pc, pr)}->{step['cell']}")
            route += [{"cell": p, "scan": False} for p in path]
        route.append({"cell": (c, r), "scan": step["scan"]})
    if len(route) > max_steps + 1:
        issues.append(f"cut to {max_steps} steps")
    return route[: max_steps + 1], issues


def to_moves(route: list[dict], cell_w: int, cell_d: int) -> list[dict]:
    """Adjacent cells -> one Tello command each. A lower row number is forward."""
    moves = []
    for a, b in zip(route, route[1:], strict=False):
        dc, dr = b["cell"][0] - a["cell"][0], b["cell"][1] - a["cell"][1]
        if dc:
            cmd = f"right {cell_w}" if dc > 0 else f"left {cell_w}"
        else:
            cmd = f"forward {cell_d}" if dr < 0 else f"back {cell_d}"
        moves.append({"cmd": cmd, "to": cell_name(*b["cell"]), "scan": b["scan"]})
    return moves


def lawnmower(start: str, cols: int, rows: int) -> list[str]:
    """Baseline route: sweep rows from the start row outwards, alternating direction."""
    c0, r0 = parse_cell(start)
    order = []
    for i, r in enumerate(list(range(r0, -1, -1)) + list(range(r0 + 1, rows))):
        cs = list(range(cols)) if (i % 2 == 0) == (c0 < cols / 2) else list(range(cols - 1, -1, -1))
        order += [cell_name(c, r) for c in cs]
    return order
