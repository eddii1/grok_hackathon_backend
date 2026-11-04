"""Ask a vision LLM about drone footage.

    uv run fleet-probe clip.mp4                          # describe 5 frames
    uv run fleet-probe clip.mp4 --find "person in red jacket"
    uv run fleet-probe frame.jpg --prompt "How many people?"
    uv run fleet-probe clip.mp4 --provider fireworks     # override LLM_PROVIDER

Provider, model and keys come from .env (see .env.example).
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from openai import OpenAI

# Default base URL and API style per provider. Any other OpenAI-compatible provider
# works by setting <NAME>_BASE_URL, <NAME>_API_KEY and <NAME>_MODEL.
DEFAULTS = {
    "xai": ("https://api.x.ai/v1", "responses"),  # xAI marks chat completions as deprecated
    "fireworks": ("https://api.fireworks.ai/inference/v1", "chat"),
}

DESCRIBE = (
    "You are the camera of a drone. Describe what you see, briefly: the setting, and every person "
    "(left/centre/right, near/far, clothing colours, what they hold)."
)
FIND = (
    'You are the camera of a drone. Target: "{target}". Answer only with JSON: '
    '{{"found": bool, "position": "left|centre|right|none", "distance": "near|medium|far|none", '
    '"confidence": 0-1, "reason": "short"}}'
)


def config(provider: str | None) -> dict:
    load_dotenv()
    name = (provider or os.getenv("LLM_PROVIDER", "xai")).lower()
    env = lambda k: os.getenv(f"{name.upper()}_{k}")  # noqa: E731
    base_url, api = DEFAULTS.get(name, (None, "chat"))
    cfg = {
        "provider": name,
        "base_url": env("BASE_URL") or base_url,
        "api": env("API") or api,
        "model": env("MODEL"),
        "key": env("API_KEY"),
    }
    names = {"base_url": "BASE_URL", "model": "MODEL", "key": "API_KEY"}
    missing = [f"{name.upper()}_{v}" for k, v in names.items() if not cfg[k]]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} in .env")
    return cfg


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


def ask(cfg: dict, prompt: str, jpeg: bytes) -> tuple[str, float]:
    client = OpenAI(api_key=cfg["key"], base_url=cfg["base_url"])
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    t = time.time()
    if cfg["api"] == "responses":
        r = client.responses.create(
            model=cfg["model"],
            store=False,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": url},
            ]}],
        )
        text = r.output_text
    else:
        r = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": url}},
            ]}],
        )
        text = r.choices[0].message.content
    return text, time.time() - t


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
