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

- Browser test console: http://127.0.0.1:8000/ (chat, video, confirmations, photos and STOP)
- Health: http://127.0.0.1:8000/health
- API docs: http://127.0.0.1:8000/docs
- Devices: http://127.0.0.1:8000/devices
- Camera: http://127.0.0.1:8000/video/drone-1 (MJPEG)
- Events: ws://127.0.0.1:8000/ws
- Photos and snapshots: `/photos/{filename}` and `/snapshots/{filename}`.

The default `devices.json` creates two simulated drones. No drone library, hardware connection or detector model download is used. Keep the server bound to localhost; authentication is not implemented. If you previously started uvicorn, stop it with Ctrl+C and rerun the command to load these changes. The old skeleton had no `/` page, so a 404 there did not mean startup failed.

## Real Grok missions

Copy `.env.example` to `.env` (do not overwrite an existing key), then set `XAI_API_KEY` locally. `.env` is ignored by Git. Restart the server after changing settings. The server reads `backend/.env`; shell environment variables take precedence.

```dotenv
XAI_API_KEY=your-local-key
XAI_MODEL=grok-4.7
FLEET_AGENT=auto
```

`FLEET_AGENT=auto` selects Grok when a key exists and the offline stub otherwise. `grok` explicitly selects Grok (missing credentials produce a configuration error in chat), and `stub` forces offline mode. `/health` reports the selected agent, model and configuration readiness; it does not validate API access or credits. The browser shows the same status. Set `XAI_MODEL` to a vision/tool-capable model available to your xAI account if needed.

Open the console and send: **“Take off with drone-1, describe what you see, take a photo, then land.”** Review the plan, click Approve or Reject when takeoff confirmation appears, and inspect the tool results and photo. STOP and Escape cancel pending work, including a model request, and land the simulators. Sending `/demo drone-1` always runs the offline deterministic check, even when Grok is configured.

The real integration uses the [official xAI SDK](https://github.com/xai-org/xai-sdk-python) with asynchronous tool calling, sequential validated local execution, linked tool-call results, and image input/structured output for `describe_scene` and `verify_photo`. Model guidance follows the [xAI model docs](https://docs.x.ai/developers/models); the model is configurable. A rejected takeoff cannot be retried by the model within that mission. Each mission is bounded to 180 seconds, 12 model rounds and 32 tool calls, with 45-second API timeouts and at most 1,500 output tokens per planning request. Vision requests are capped at 700 output tokens. Conversation context persists per connection for eight user turns, then starts fresh; STOP or failure clears it. API requests use xAI credits. Prompts, conversation context and requested camera images go to xAI; use only appropriate simulator recordings.

`describe_scene(device_id)` is an additional tool for the tracer-bullet mission. It interprets the current frame; it does not provide tracked targets. Person detection, target selection, scanning and following remain stubs: `search` returns not_found, `confirm_target` rejects absent tracks, and `follow` returns not_implemented. Photo verification is now real in Grok mode; it is still a stub in offline mode. The generated camera is explicitly synthetic, so a successful simulator check is not evidence of real-world target selection.

The WebSocket also emits `mission_status` with `started`, `completed`, `cancelled` or `failed`. A completed status means the agent turn ended, not that every requested objective succeeded; inspect tool results and the report. The browser test console is backend development tooling; the separate frontend submodule is unchanged.

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

In offline mode, other messages receive a stub response. The deterministic demo saves a simulated camera frame without Grok planning or verification.

## Shared modules

- `fleet.contracts`: `DeviceAdapter` protocol, `DeviceState`, `Detection`, backend event types and validated incoming `/ws` events from `docs/plan.md`.
- `fleet.simulator`: `SimAdapter`, command history/logging, generated camera feed and optional looping recording.
- `fleet.registry`: config loading, device lookup, state listing and concurrent hover-then-land STOP.
- `fleet.perception`: importable stubs (`detect`, `annotate`, `follow_command`, `photo_ready`). Only JPEG annotation is implemented.
- `fleet.agent`: JSON-returning tool methods, confirmation ownership, session photo IDs and deterministic `StubAgent`.
- `fleet.grok`: official SDK provider, vision schemas, tool allowlist/validation and bounded `GrokAgent` loop.
- `fleet.app`: FastAPI app factory, lifecycle cleanup, periodic device events, `/ws`, MJPEG and static media.

To use a recording, add `"video_path": "data/teammates.avi"` to a simulated device in `devices.json`. Paths resolve relative to the config file. The video loops at its recorded FPS. Missing or unreadable recordings fail at startup instead of silently falling back. Without a recording, a clearly labelled synthetic frame shows device ID, height and yaw. Use only consenting teammates' recordings.

Set `FLEET_CONFIG` to a different JSON config path and `FLEET_MEDIA_DIR` to a media output directory before starting the server. Runtime photos are ignored by Git.

Simulated takeoff requires at least 25% battery. RC values are capped at ±40, altitude is bounded to 0–200 cm, and stale RC input stops after one second. These are software simulation checks, not a validated real-drone safety layer: no physics, battery drain, real keepalive, collision avoidance or timeout recovery is modeled. Command history is bounded to the most recent 1,000 commands; persistent mission event logging is a later task.

## Connect a physical TT: ground check first

The browser currently controls simulators. For the first real TT connection, see the parent repo's [`docs/hardware/robomaster-tt/staged-runs.md`](../docs/hardware/robomaster-tt/staged-runs.md). With the drone grounded, propellers removed for initial checks, laptop Wi-Fi on the TT and tether/Ethernet internet active:

```powershell
uv pip install --python .venv/Scripts/python.exe -e ".[dev,hardware]"
.venv\Scripts\python -m fleet.tt_ground --video
```

This checks SDK replies, sustained telemetry and decoded video, and saves a report/JPEG under ignored `runtime/`. Its hard-coded allowlist contains no motor commands and no Wi-Fi reconfiguration. It does not arm or enable a real drone in the browser.

The restricted real adapter is now implemented in `fleet.tello` / `fleet.tello_transport`, with a separate watchdog thread. Start with its motor-disabled ground check:

```powershell
.venv\Scripts\python -m fleet.supervised_hover
```

See [`hover-trial.md`](../docs/hardware/robomaster-tt/hover-trial.md) for the prepared interactive Windows hover procedure, operator heartbeat, stop keys, limits and remaining physical test gates. `--fly` additionally requires an interactive terminal, the exact arming phrase and live SPACE-key heartbeats. No physical takeoff has been performed during development. The default app remains simulated; `devices.tello-ground.json` is an optional real-camera/telemetry monitor with all real motor commands disabled in the browser. Stop any other SDK process before opening it. Do not use `/demo` for real devices.

The adapter never retries takeoff, latches STOP before awaiting cancellation, and retains an explicit uncertain-flight state until touchdown can be inferred from sustained telemetry after an observed ascent. It rejects translation, rotation and nonzero RC for the first rollout. Faults request bounded hover/landing transmissions and stop healthy-hover keepalives. A sent landing command is not a confirmed landing; inspect the aircraft. This is tested development software, not a certified safety system.

## Existing SEEK dashboard integration

The frontend now connects through a same-origin Vite `/api` proxy. The grounded capture API saves one fresh JPEG, sends those exact bytes plus the description to Grok, and retains a structured assessment or explicit failure/cancellation. Real flight stays disabled. See [`docs/grounded-capture.md`](../docs/grounded-capture.md) in the parent repo for setup, capture endpoints, the manifest format and the optional ground-only expansion ToF query. Saved manifests and images stay in ignored `runtime/` and can be restored by capture ID.

The API stays available when the configured TT is offline. After rejoining its Wi-Fi, use **Reconnect grounded TT** to recreate the SDK/video session. **ToF object tests** collects 3–30 labelled samples with optional ruler-measured references, raw replies, valid-only statistics and JSON export. Zero readings are invalid, `8192` is out-of-range, and cached valid readings expire after two seconds. Tests save under ignored `runtime/range-tests/`; dashboard history is per backend session. The first actual white-card test returned 0/10 usable samples (nine zeros, one timeout); further hardware checks await a charged battery. No approach or recording behavior is enabled.

## Tests

```powershell
.venv\Scripts\python -m pytest -q
```

Tests cover health, shared stubs, simulator commands and limits, recording loops, confirmed/rejected/timed-out takeoff, JPEG capture, STOP during confirmation, invalid events and shutdown landing. All tests run offline without a drone or Grok credentials.

The Grok tests use a scripted provider to verify natural-language routing, vision inputs, tool validation, rejection enforcement, cancellation during a model request, conversation reset, time limits, SDK result IDs and safe error messages. These tests do not validate xAI API access.

For an opt-in live Grok test using one isolated local simulator (requires the dev dependencies and a valid key):

```powershell
.venv\Scripts\python -m fleet.smoke_grok
```

Approve takeoff at the terminal prompt, or explicitly pass `--approve-simulated-takeoff` for an unattended simulator test. It checks list/takeoff/scene/photo/verification/land/report tool completion and the final landed state. It uses xAI credits and temporary media, and does not need an already-running uvicorn server.

Observed live smoke test on 2026-11-04: Grok `grok-4.7` completed all seven required tool steps with operator confirmation, described the synthetic frame, verified the simulator label and landed, in **32.2 seconds**. This is one integration run, not a performance baseline or target-selection accuracy measurement.
