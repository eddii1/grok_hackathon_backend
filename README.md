# grok_hackathon_backend

Backend of the Grok fleet command platform: drone harness, perception, the Grok agent and the FastAPI API that the frontend talks to.

It is a submodule of [`eddii1/grok_hackathon`](https://github.com/eddii1/grok_hackathon). The architecture, the interface contracts and the demo are in its [`docs/plan.md`](https://github.com/eddii1/grok_hackathon/blob/main/docs/plan.md); the drone SDK docs are in its [`docs/hardware/robomaster-tt/`](https://github.com/eddii1/grok_hackathon/tree/main/docs/hardware/robomaster-tt).

**Tasks:** [`TASKS.md`](TASKS.md). Claim one before you start.

## Run without hardware

Python 3.11+ is required. From this `backend/` directory, on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m uvicorn fleet.app:app --host 127.0.0.1 --port 8000
```

On macOS/Linux use `.venv/bin/python` instead. `uv venv` and `uv pip install -e ".[dev]"` also work.

- Health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs
- Devices: http://127.0.0.1:8000/devices
- Camera: http://127.0.0.1:8000/video/drone-1 (MJPEG)
- Events: ws://127.0.0.1:8000/ws
- Photos and snapshots: `/photos/{filename}` and `/snapshots/{filename}`.

The default `devices.json` creates two simulated drones. No drone library, hardware connection, Grok API key, or model download is used. `.env.example` documents the future `XAI_API_KEY` setting; the current stub does not load or use it. Keep the server bound to localhost; authentication is not implemented.

## Try the simulator over WebSocket

In a browser developer console (on a local HTTP page), connect and send the demo command:

```javascript
const ws = new WebSocket('ws://127.0.0.1:8000/ws');
let pending;
ws.onmessage = ({data}) => {
  const event = JSON.parse(data);
  console.log(event);
  if (event.type === 'confirm_request') pending = event.request_id;
};
ws.onopen = () => ws.send(JSON.stringify({type: 'user_message', text: '/demo drone-1'}));
// After reviewing the confirmation request:
ws.send(JSON.stringify({type: 'confirm_response', request_id: pending, approved: true}));
// At any time, including while confirmation is pending:
ws.send(JSON.stringify({type: 'stop_all'}));
```

The deterministic demo confirms takeoff, captures a JPEG and lands. Rejection, a 30-second confirmation timeout, disconnect, STOP and backend shutdown leave the simulated drone landed. One demo mission runs at a time across clients. Only the requesting connection can answer its confirmation. STOP cancels missions and calls the registry directly, without asking the agent. An expired confirmation cannot restart a cancelled mission.

Other messages receive a stub response; natural language planning, person detection, target selection, following and photo verification are not implemented. `confirm_target` rejects because the perception stub has no tracks; `verify_photo` returns `stub: true` and no match. The demo only saves a simulated camera frame.

## Shared modules

- `fleet.contracts`: `DeviceAdapter` protocol, `DeviceState`, `Detection`, backend event types and validated incoming `/ws` events from `docs/plan.md`.
- `fleet.simulator`: `SimAdapter`, command history/logging, generated camera feed and optional looping recording.
- `fleet.registry`: config loading, device lookup, state listing and concurrent hover-then-land STOP.
- `fleet.perception`: importable stubs (`detect`, `annotate`, `follow_command`, `photo_ready`). Only JPEG annotation is implemented.
- `fleet.agent`: JSON-returning tool methods and deterministic `StubAgent`; replace these stubs with the real Grok integration later.
- `fleet.app`: FastAPI app factory, lifecycle cleanup, periodic device events, `/ws`, MJPEG and static media.

To use a recording, add `"video_path": "data/teammates.avi"` to a simulated device in `devices.json`. Paths resolve relative to the config file. The video loops at its recorded FPS. Missing or unreadable recordings fail at startup instead of silently falling back. Without a recording, a clearly labelled synthetic frame shows device ID, height and yaw. Use only consenting teammates' recordings.

Set `FLEET_CONFIG` to a different JSON config path and `FLEET_MEDIA_DIR` to a media output directory before starting the server. Runtime photos are ignored by Git.

Simulated takeoff requires at least 25% battery. RC values are capped at ±40, altitude is bounded to 0–200 cm, and stale RC input stops after one second. These are software simulation checks, not a validated real-drone safety layer: no physics, battery drain, real keepalive, collision avoidance or timeout recovery is modeled. Command history is bounded to the most recent 1,000 commands; persistent mission event logging is a later task.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

Tests cover health, shared stubs, simulator commands and limits, recording loops, confirmed/rejected/timed-out takeoff, JPEG capture, STOP during confirmation, invalid events and shutdown landing. All tests run offline without a drone or Grok credentials.
