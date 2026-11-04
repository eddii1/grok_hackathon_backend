"""LLM access for every part of the backend: one config, one call, any OpenAI-compatible provider.

Provider, model and keys come from .env (see .env.example). Calls are traced automatically
once fleet.tracing.setup() has run.
"""

import base64
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

# Default base URL and API style per provider. Any other OpenAI-compatible provider
# works by setting <NAME>_BASE_URL, <NAME>_API_KEY and <NAME>_MODEL.
DEFAULTS = {
    "xai": ("https://api.x.ai/v1", "responses"),  # xAI marks chat completions as deprecated
    "fireworks": ("https://api.fireworks.ai/inference/v1", "chat"),
}


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
        "reasoning": env("REASONING"),  # optional: low|medium|high
    }
    names = {"base_url": "BASE_URL", "model": "MODEL", "key": "API_KEY"}
    missing = [f"{name.upper()}_{v}" for k, v in names.items() if not cfg[k]]
    if missing:
        raise SystemExit(f"Set {', '.join(missing)} in .env")
    return cfg


def ask(cfg: dict, prompt: str, jpeg: bytes | list[bytes], schema: dict | None = None) -> tuple[str, float]:
    """One vision call with one or more images. With `schema`, the model must answer with JSON matching it."""
    client = OpenAI(api_key=cfg["key"], base_url=cfg["base_url"], timeout=60)
    jpegs = [jpeg] if isinstance(jpeg, bytes) else jpeg
    urls = ["data:image/jpeg;base64," + base64.b64encode(j).decode() for j in jpegs]
    effort = cfg.get("reasoning")
    t = time.time()
    if cfg["api"] == "responses":
        extra = {"reasoning": {"effort": effort}} if effort else {}
        if schema:
            fmt = {"type": "json_schema", "name": "out", "schema": schema, "strict": True}
            extra["text"] = {"format": fmt}
        r = client.responses.create(
            model=cfg["model"],
            store=False,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        *[{"type": "input_image", "image_url": u} for u in urls],
                    ],
                }
            ],
            **extra,
        )
        text = r.output_text
    else:
        extra = {"reasoning_effort": effort} if effort else {}
        if schema:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema, "strict": True},
            }
        r = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *[{"type": "image_url", "image_url": {"url": u}} for u in urls],
                    ],
                }
            ],
            **extra,
        )
        text = r.choices[0].message.content
    return text, time.time() - t


def ask_json(cfg: dict, prompt: str, jpeg: bytes | list[bytes], schema: dict) -> tuple[dict, float]:
    text, secs = ask(cfg, prompt, jpeg, schema)
    text = text[text.find("{") : text.rfind("}") + 1]  # tolerate fences or prose around the JSON
    return json.loads(text), secs
