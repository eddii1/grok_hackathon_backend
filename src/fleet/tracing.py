"""Tracing: every run is visible in Phoenix (http://localhost:6006) and in runs/<id>/events.jsonl.

Phoenix shows the tree (mission -> plan -> LLM call, execute -> step -> move/check -> LLM call),
with prompts, images and JSON outputs. events.jsonl is the same story as plain lines, so a run
can still be read when Phoenix is down.

    uvx --python 3.12 --from arize-phoenix phoenix serve     # start Phoenix once
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from opentelemetry import trace

_tracer = None
_log = None


def setup(run_dir: Path, project: str = "seek") -> None:
    """Start tracing for this process. Phoenix being down only loses the UI, never the run."""
    global _tracer, _log
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from phoenix.otel import register

    provider = register(
        project_name=os.getenv("PHOENIX_PROJECT", project),
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
        batch=True,
        verbose=False,
    )
    from openinference.instrumentation import TraceConfig

    # keep every image Grok gets visible in Phoenix (the default redacts images above 32k base64 chars)
    OpenAIInstrumentor().instrument(tracer_provider=provider, config=TraceConfig(base64_image_max_length=5_000_000))
    _tracer = trace.get_tracer("fleet")
    _log = open(run_dir / "events.jsonl", "a")


def _json(v) -> str:
    return v if isinstance(v, str) else json.dumps(v, default=str)


def event(name: str, **data) -> None:
    """One line in events.jsonl, and an event on the current span."""
    if _log:
        _log.write(json.dumps({"t": round(time.time(), 2), "event": name, **data}, default=str) + "\n")
        _log.flush()
    trace.get_current_span().add_event(name, {k: _json(v) for k, v in data.items()})


@contextmanager
def span(name: str, kind: str = "CHAIN", input=None, **attrs):
    """A Phoenix span. kind: AGENT, CHAIN, TOOL (OpenInference span kinds). Yields a setter for the output."""
    tracer = _tracer or trace.get_tracer("fleet")
    with tracer.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        if input is not None:
            s.set_attribute("input.value", _json(input))
        for k, v in attrs.items():
            s.set_attribute(k, v if isinstance(v, str | int | float | bool) else _json(v))
        event(f"{name}:start", input=input, **attrs)

        def output(value) -> None:
            s.set_attribute("output.value", _json(value))
            event(f"{name}:end", output=value)

        output.rename = s.update_name  # name a span after what it decided, e.g. "step 3: goto H8"
        yield output


def flush() -> None:
    trace.get_tracer_provider().force_flush()
    if _log:
        _log.flush()
