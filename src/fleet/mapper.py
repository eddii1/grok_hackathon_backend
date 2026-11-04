"""Live mapper: camera frames + drone pose in, a growing 3D point map and 2D floor grid out.

    mapper = LiveMap()
    mapper.add(frame_bgr, {"x": 0, "y": 0, "z": 1.2, "yaw": 0})   # non-blocking, call as often as you like
    mapper.render()   # 2D grid PNG with the drone
    mapper.points()   # coloured 3D points for the viewer
    mapper.state()    # what Grok gets

Coordinates: x = east, y = north, z = up (metres); yaw = degrees clockwise from north. The camera is
level and looks along yaw. Depth is the model's estimate (not calibrated).

Grid cells (50 cm of floor each): 0 unknown, 1 free, 2 obstacle. A cell is an obstacle when something
30 cm - 2 m above the floor was seen in it more often than the camera saw through it.
"""

import threading
import time

import cv2
import numpy as np

from fleet.grid import cell_name, jpg
from fleet.tracing import event

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"  # Apache-2.0, ~0.2 s per frame on MPS
_model = None
_model_lock = threading.Lock()


def depth(img: np.ndarray) -> np.ndarray:
    """Metres per pixel (model estimate), same size as img."""
    global _model
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    with _model_lock:
        if _model is None:
            _model = (
                AutoImageProcessor.from_pretrained(DEPTH_MODEL),
                AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(dev).eval(),
            )
    proc, model = _model
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    with torch.no_grad():
        out = model(**proc(images=pil, return_tensors="pt").to(dev))
    size = [img.shape[:2]]
    return proc.post_process_depth_estimation(out, target_sizes=size)[0]["predicted_depth"].float().cpu().numpy()


class LiveMap:
    def __init__(
        self,
        fov_diag: float = 82.6,
        half: float = 4.25,  # odd cell count, so the take-off point (0, 0) is a cell centre
        cell: float = 0.5,
        voxel: float = 0.1,
        max_range: float = 4.5,
        min_sharpness: float = 30.0,
    ):
        self.fov_diag, self.half, self.cell, self.voxel = fov_diag, half, cell, voxel
        self.max_range, self.min_sharpness = max_range, min_sharpness
        n, nv, nz = int(2 * half / cell), int(2 * half / voxel), int(2.5 / voxel)
        self.n = n
        self.hits = np.zeros((nv, nv, nz), np.uint32)  # voxel hit counts  [col(east), row(south), z]
        self.colour = np.zeros((nv, nv, nz, 3), np.float32)  # summed BGR
        self.occ = np.zeros((n, n), np.float32)  # frames in which a cell looked solid
        self.free = np.zeros((n, n), np.float32)  # frames in which the camera saw through a cell
        self.pose = {"x": 0.0, "y": 0.0, "z": 1.2, "yaw": 0.0}
        self.trail: list[tuple[float, float]] = []
        self.frames_used = self.frames_skipped = 0
        self.lock = threading.Lock()
        self._latest = None
        self._running = True
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------- input ----------

    def add(self, frame: np.ndarray, pose: dict) -> None:
        """Hand over the newest frame and the pose it was taken at. Older unprocessed frames are dropped."""
        self._latest = (frame, dict(pose))

    def _worker(self) -> None:
        last_log = time.time()
        while self._running:
            item, self._latest = self._latest, None
            if item is None:
                time.sleep(0.02)
                continue
            frame, pose = item
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < self.min_sharpness:  # motion blur: skip
                self.frames_skipped += 1
                continue
            self.integrate(frame, depth(frame), pose)
            if time.time() - last_log > 3:
                last_log = time.time()
                event("map", frames=self.frames_used, skipped=self.frames_skipped, cells=self.counts(), pose=self.pose)

    # ---------- fusion ----------

    def integrate(self, frame: np.ndarray, d: np.ndarray, pose: dict) -> None:
        h, w = d.shape
        focal = np.hypot(w / 2, h / 2) / np.tan(np.radians(self.fov_diag / 2))
        step = max(1, w // 320)
        v, u = np.mgrid[0:h:step, 0:w:step]
        z = d[::step, ::step]
        x, y = (u - w / 2) * z / focal, (v - h / 2) * z / focal  # camera: x right, y down, z forward
        a = np.radians(pose["yaw"])
        east = pose["x"] + x * np.cos(a) + z * np.sin(a)
        north = pose["y"] - x * np.sin(a) + z * np.cos(a)
        up = pose["z"] - y
        ok = z.ravel() < self.max_range
        east, north, up = east.ravel()[ok], north.ravel()[ok], up.ravel()[ok]
        bgr = frame[::step, ::step].reshape(-1, 3)[ok].astype(np.float32)

        with self.lock:
            # 3D voxels (for the viewer)
            vc = ((east + self.half) / self.voxel).astype(int)
            vr = ((self.half - north) / self.voxel).astype(int)
            vz = (up / self.voxel).astype(int)
            nv, nz = self.hits.shape[0], self.hits.shape[2]
            k = (vc >= 0) & (vc < nv) & (vr >= 0) & (vr < nv) & (vz >= 0) & (vz < nz)
            np.add.at(self.hits, (vc[k], vr[k], vz[k]), 1)
            np.add.at(self.colour, (vc[k], vr[k], vz[k]), bgr[k])

            # 2D grid: one vote per cell per frame, so many frames of the same wall do not swamp free space
            solid = (up > 0.3) & (up < 2.0)
            seen_solid = self._cells(east[solid], north[solid], min_points=5)
            dx, dy = east - pose["x"], north - pose["y"]
            dist = np.maximum(np.hypot(dx, dy), 1e-6)
            through = np.zeros((self.n, self.n), bool)
            for t in np.linspace(0, 1, 25)[:-1]:
                tt = np.clip(t, 0, np.maximum(0, 1 - self.cell / dist))
                through |= self._cells(pose["x"] + dx * tt, pose["y"] + dy * tt, min_points=1)
            self.occ += seen_solid
            self.free += through & ~seen_solid

            self.pose = dict(pose)
            if not self.trail or np.hypot(self.trail[-1][0] - pose["x"], self.trail[-1][1] - pose["y"]) > 0.05:
                self.trail.append((pose["x"], pose["y"]))
            self.frames_used += 1

    def _cells(self, east: np.ndarray, north: np.ndarray, min_points: int) -> np.ndarray:
        c = ((east + self.half) / self.cell).astype(int)
        r = ((self.half - north) / self.cell).astype(int)
        k = (c >= 0) & (c < self.n) & (r >= 0) & (r < self.n)
        counts = np.zeros((self.n, self.n), int)
        np.add.at(counts, (r[k], c[k]), 1)
        return counts >= min_points

    # ---------- outputs ----------

    def grid(self) -> np.ndarray:
        with self.lock:
            obstacle = (self.occ >= 2) & (self.occ >= 0.5 * self.free)
            return np.where(obstacle, 2, np.where(self.free > 0, 1, 0))

    def counts(self) -> dict:
        g = self.grid()
        return {"unknown": int((g == 0).sum()), "free": int((g == 1).sum()), "obstacle": int((g == 2).sum())}

    def cell_of(self, x: float, y: float) -> str:
        c = int(np.clip((x + self.half) / self.cell, 0, self.n - 1))
        r = int(np.clip((self.half - y) / self.cell, 0, self.n - 1))
        return cell_name(c, r)

    def points(self, max_points: int = 30000, min_hits: int = 2) -> list[list[float]]:
        """[[x, y, z, r, g, b], ...] for the 3D viewer (colour 0-1)."""
        with self.lock:
            idx = np.argwhere(self.hits >= min_hits)
            if len(idx) > max_points:
                idx = idx[np.argsort(-self.hits[tuple(idx.T)])[:max_points]]
            cols = self.colour[tuple(idx.T)] / self.hits[tuple(idx.T)][:, None] / 255.0
        x = idx[:, 0] * self.voxel - self.half + self.voxel / 2
        y = self.half - idx[:, 1] * self.voxel - self.voxel / 2
        z = idx[:, 2] * self.voxel + self.voxel / 2
        return np.round(np.column_stack([x, y, z, cols[:, 2], cols[:, 1], cols[:, 0]]), 3).tolist()

    def render(self, px: int = 40) -> bytes:
        g = self.grid()
        n = self.n
        img = np.full((n * px, n * px, 3), 255, np.uint8)
        colour = {0: (175, 175, 175), 1: (255, 255, 255), 2: (40, 40, 40)}
        for r in range(n):
            for c in range(n):
                cv2.rectangle(img, (c * px, r * px), ((c + 1) * px, (r + 1) * px), colour[int(g[r, c])], -1)
                cv2.rectangle(img, (c * px, r * px), ((c + 1) * px, (r + 1) * px), (215, 215, 215), 1)
                cv2.putText(
                    img, cell_name(c, r), (c * px + 2, r * px + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 120, 0), 1
                )

        def to_px(x, y):
            return int((x + self.half) / self.cell * px), int((self.half - y) / self.cell * px)

        with self.lock:
            trail, pose = list(self.trail), dict(self.pose)
        for a, b in zip(trail, trail[1:], strict=False):
            cv2.line(img, to_px(*a), to_px(*b), (255, 140, 0), 2)
        cx, cy = to_px(pose["x"], pose["y"])
        a = np.radians(pose["yaw"])
        tip = (int(cx + 22 * np.sin(a)), int(cy - 22 * np.cos(a)))
        cv2.circle(img, (cx, cy), 8, (255, 0, 0), -1)
        cv2.arrowedLine(img, (cx, cy), tip, (255, 0, 0), 3, tipLength=0.4)
        return jpg(img)

    def cell_index(self, x: float, y: float) -> tuple[int, int]:
        """(col, row) of a point in metres, clipped to the map."""
        return (
            int(np.clip((x + self.half) / self.cell, 0, self.n - 1)),
            int(np.clip((self.half - y) / self.cell, 0, self.n - 1)),
        )

    def cell_centre(self, c: int, r: int) -> tuple[float, float]:
        return -self.half + (c + 0.5) * self.cell, self.half - (r + 0.5) * self.cell

    def ascii(self, visited: dict[str, int] | None = None) -> str:
        """The grid as text for Grok: one character per 50 cm cell, row 1 = north."""
        g, visited = self.grid(), visited or {}
        with self.lock:
            dc, dr = self.cell_index(self.pose["x"], self.pose["y"])
        lines = ["    " + " ".join(cell_name(c, 0)[0] for c in range(self.n))]
        for r in range(self.n):
            row = []
            for c in range(self.n):
                ch = "?.#"[int(g[r, c])]
                if ch == "." and visited.get(cell_name(c, r)):
                    ch = "v"
                row.append("D" if (c, r) == (dc, dr) else ch)
            lines.append(f"{r + 1:>3} " + " ".join(row))
        return "\n".join(lines)

    def bird(self, px: int = 48, visited: dict[str, int] | None = None, plan: dict | None = None) -> bytes:
        """Top-down photo of the 3D map: each 10 cm column shows the colour of its highest point.
        Grid lines and cell names on top, unknown cells greyed, obstacles outlined red, trail and drone."""
        with self.lock:
            hits = self.hits >= 2
            top = np.where(hits.any(2), hits.shape[2] - 1 - np.argmax(hits[:, :, ::-1], 2), -1)  # [col, row]
            cc, rr = np.nonzero(top >= 0)
            col = np.zeros((self.hits.shape[0], self.hits.shape[1], 3), np.uint8)
            zz = top[cc, rr]
            col[cc, rr] = np.clip(self.colour[cc, rr, zz] / self.hits[cc, rr, zz][:, None], 0, 255)
            trail, pose = list(self.trail), dict(self.pose)
        g, visited = self.grid(), visited or {}
        seen = cv2.resize((top >= 0).T.astype(np.uint8), (self.n * px, self.n * px), interpolation=cv2.INTER_NEAREST)
        points = cv2.resize(col.transpose(1, 0, 2), (self.n * px, self.n * px), interpolation=cv2.INTER_NEAREST)
        base = np.array([(60, 60, 60), (225, 225, 225), (25, 25, 25)], np.uint8)[np.kron(g, np.ones((px, px), int))]
        img = np.where(seen[..., None] > 0, points, base)  # real colours where there are points, else cell state
        tint = img.copy()
        for r in range(self.n):
            for c in range(self.n):
                p0, p1 = (c * px, r * px), ((c + 1) * px - 1, (r + 1) * px - 1)
                if g[r, c] == 0:
                    cv2.rectangle(tint, p0, p1, (60, 60, 60), -1)
                elif visited.get(cell_name(c, r)):
                    cv2.rectangle(tint, p0, p1, (255, 190, 100), -1)
        img = cv2.addWeighted(img, 0.6, tint, 0.4, 0)
        for r in range(self.n):
            for c in range(self.n):
                p0, p1 = (c * px, r * px), ((c + 1) * px - 1, (r + 1) * px - 1)
                cv2.rectangle(img, p0, p1, (0, 0, 230) if g[r, c] == 2 else (70, 70, 70), 2 if g[r, c] == 2 else 1)
                cv2.putText(
                    img, cell_name(c, r), (c * px + 3, r * px + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3
                )
                cv2.putText(
                    img, cell_name(c, r), (c * px + 3, r * px + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1
                )

        def to_px(x, y):
            return int((x + self.half) / self.cell * px), int((self.half - y) / self.cell * px)

        for a, b in zip(trail, trail[1:], strict=False):
            cv2.line(img, to_px(*a), to_px(*b), (0, 140, 255), 3)  # BGR orange, as the prompt says
        cx, cy = to_px(pose["x"], pose["y"])
        a = np.radians(pose["yaw"])
        cv2.circle(img, (cx, cy), 9, (255, 0, 0), -1)
        cv2.arrowedLine(
            img, (cx, cy), (int(cx + 30 * np.sin(a)), int(cy - 30 * np.cos(a))), (255, 0, 0), 3, tipLength=0.4
        )
        if plan:  # what the agent wants to do next, in green, for the operator to approve
            green = (60, 220, 60)
            if plan.get("path"):
                pts = [(int((c + 0.5) * px), int((r + 0.5) * px)) for c, r in plan["path"]]
                pts[0] = (cx, cy)
                cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 0, 0), 7)
                cv2.polylines(img, [np.array(pts, np.int32)], False, green, 4)
                c, r = plan["path"][-1]
                cv2.rectangle(img, (c * px + 2, r * px + 2), ((c + 1) * px - 3, (r + 1) * px - 3), green, 4)
            elif plan.get("heading") is not None:
                h = np.radians(plan["heading"])
                tip = (int(cx + 2.2 * px * np.sin(h)), int(cy - 2.2 * px * np.cos(h)))
                cv2.arrowedLine(img, (cx, cy), tip, (0, 0, 0), 8, tipLength=0.3)
                cv2.arrowedLine(img, (cx, cy), tip, green, 4, tipLength=0.3)
            elif plan.get("action") == "spin":
                cv2.circle(img, (cx, cy), int(px * 0.9), (0, 0, 0), 7)
                cv2.circle(img, (cx, cy), int(px * 0.9), green, 4)
        return jpg(img)

    def reset(self) -> None:
        """Forget everything seen so far (a new place)."""
        with self.lock:
            for a in (self.hits, self.colour, self.occ, self.free):
                a[:] = 0
            self.trail.clear()
            self.frames_used = self.frames_skipped = 0

    def state(self) -> dict:
        """Compact view for Grok: where the drone is, what is around it."""
        g = self.grid()
        with self.lock:
            pose = dict(self.pose)
        c = int(np.clip((pose["x"] + self.half) / self.cell, 0, self.n - 1))
        r = int(np.clip((self.half - pose["y"]) / self.cell, 0, self.n - 1))
        around = {}
        for name, (dc, dr) in {"north": (0, -1), "east": (1, 0), "south": (0, 1), "west": (-1, 0)}.items():
            cc, rr = c + dc, r + dr
            ok = 0 <= cc < self.n and 0 <= rr < self.n
            around[name] = ["unknown", "free", "obstacle"][int(g[rr, cc])] if ok else "edge"
        frontier = [
            cell_name(cc, rr)
            for rr in range(self.n)
            for cc in range(self.n)
            if g[rr, cc] == 0
            and any(
                0 <= cc + dc < self.n and 0 <= rr + dr < self.n and g[rr + dr, cc + dc] == 1
                for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
        ]
        return {
            "drone_cell": cell_name(c, r),
            "heading_deg": round(pose["yaw"]) % 360,
            "height_m": round(pose["z"], 2),
            "neighbours": around,
            "cells": self.counts(),
            "frontier": frontier,
            "frames_used": self.frames_used,
        }

    def save_ply(self, path: str) -> None:
        pts = self.points(max_points=200000, min_hits=1)
        with open(path, "w") as f:
            f.write(
                f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
            )
            for x, y, z, r, g, b in pts:
                f.write(f"{x} {y} {z} {int(r * 255)} {int(g * 255)} {int(b * 255)}\n")

    def stop(self) -> None:
        self._running = False
