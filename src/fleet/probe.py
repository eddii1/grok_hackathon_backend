"""Ask a vision LLM about drone footage.

    uv run fleet-probe clip.mp4                          # describe 5 frames
    uv run fleet-probe clip.mp4 --find "person in red jacket"
    uv run fleet-probe frame.jpg --prompt "How many people?"
    uv run fleet-probe clip.mp4 --provider fireworks     # override LLM_PROVIDER

Provider, model and keys come from .env (see .env.example).
"""

import argparse
import json
import time
from pathlib import Path

import cv2

from fleet.llm import ask, config

DESCRIBE = (
    "You are the camera of a drone. Describe what you see, briefly: the setting, and every person "
    "(left/centre/right, near/far, clothing colours, what they hold)."
)
FIND = (
    'You are the camera of a drone. Target: "{target}". Answer only with JSON: '
    '{{"found": bool, "position": "left|centre|right|none", "distance": "near|medium|far|none", '
    '"confidence": 0-1, "reason": "short"}}'
)


def frames(source: str, n: int) -> list[tuple[int, bytes]]:
    """n evenly spaced JPEG frames from a video, or the image itself."""
    img = cv2.imread(source)
    if img is not None:
        return [(0, cv2.imencode(".jpg", img)[1].tobytes())]
    cap = cv2.VideoCapture(source)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit(f"Cannot read frames from {source}")
    picks = {round(i * (total - 1) / max(n - 1, 1)) for i in range(n)}
    out = []
    for i in range(total):
        if not cap.grab():
            break
        if i in picks:
            ok, f = cap.retrieve()
            if ok:
                out.append((i, cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="video file or image")
    p.add_argument("--find", help="target description, returns JSON")
    p.add_argument("--prompt", help="custom question")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--provider", help="xai, fireworks, ... (default: LLM_PROVIDER)")
    a = p.parse_args()

    cfg = config(a.provider)
    prompt = a.prompt or (FIND.format(target=a.find) if a.find else DESCRIBE)
    run = Path("runs") / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    print(f"{cfg['provider']} / {cfg['model']} ({cfg['api']}) -> {run}")
    for idx, jpeg in frames(a.source, a.frames):
        (run / f"frame{idx}.jpg").write_bytes(jpeg)
        text, secs = ask(cfg, prompt, jpeg)
        print(f"\n[frame {idx}] {secs:.1f}s\n{text}")
        row = {k: cfg[k] for k in ("provider", "model", "api")}
        row |= {"source": a.source, "frame": idx, "prompt": prompt, "seconds": round(secs, 2), "answer": text}
        with open(run / "results.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
