# grok_hackathon_backend

Backend of the Grok fleet command platform: drone harness, perception, the Grok agent and the FastAPI API that the frontend talks to.

It is a submodule of [`eddii1/grok_hackathon`](https://github.com/eddii1/grok_hackathon). The architecture, the interface contracts and the demo are in its [`docs/plan.md`](https://github.com/eddii1/grok_hackathon/blob/main/docs/plan.md); the drone SDK docs are in its [`docs/hardware/robomaster-tt/`](https://github.com/eddii1/grok_hackathon/tree/main/docs/hardware/robomaster-tt).

**Tasks:** [`TASKS.md`](TASKS.md). Claim one before you start.

## Run

```sh
uv sync
cp .env.example .env   # add XAI_API_KEY and/or FIREWORKS_API_KEY
uv run fleet-probe clip.mp4                                  # describe 5 frames with LLM_PROVIDER
uv run fleet-probe clip.mp4 --find "person in red jacket"     # JSON verdict per frame
uv run fleet-probe clip.mp4 --provider fireworks              # same, with GLM on Fireworks
```

Frames and answers are saved under `runs/<timestamp>/`.

Live feed + chat (http://localhost:8000):

```sh
uv run fleet-live --source 0          # webcam
uv run fleet-live --source clip.mp4   # recorded drone video, looped
uv run fleet-live --tello             # live drone (laptop on the drone's Wi-Fi)
```

## SEEK: live map + Grok search agent (http://localhost:8001)

```sh
uv run fleet-viewer --webcam 0          # laptop camera: you carry it and follow the on-screen instructions
uv run fleet-viewer --sim room_360.jpg  # simulated drone inside a 360 photo
uv run fleet-viewer --video flight.mp4  # a recorded drone flight: spin replays it, turns show the frame facing that way
uv run fleet-viewer --tello             # the RoboMaster TT (laptop on its Wi-Fi)
```

The map (depth per frame -> 3D voxels -> 50 cm grid) is built live from the camera. Switch between **Manual**
(buttons) and **Agent** (type a goal, press Start). The agent shows each planned action on the map and waits
for Approve/Reject (tick "automatic" to skip), and keeps searching until you confirm a find. Evidence per
mission: `runs/seek-*/mission-*/` (prompt, images and reply per decision). Traces: Phoenix at
http://localhost:6006, project `seek`, one trace per decision, grouped per mission as a session.

Where to plug in a drone SDK:

| File | What |
|---|---|
| `drone.py` | `Drone`: every command (takeoff, forward/back/left/right cm, turn_left/right deg, spin, land, emergency), safety checks, logging. A backend only needs `send(cmd) -> "ok"`, `rc(...)`, `frame()`, `telemetry()`. Existing: `SimBackend`, `TelloBackend` (raw UDP), `WebcamBackend` (a person plays the drone). |
| `harness.py` | `Seek`: the agent loop. Grok answers `goto <cell>` / `look <heading>` / `spin` / `found`; `prepare()` turns that into a checked plan (shortest path over known-free cells), `act()` flies it through `Drone` in straight legs. |
| `mapper.py` | `LiveMap`: frames + pose -> map; `bird()` is the image Grok gets, `ascii()` the text grid. Cells are 50 cm, take-off is the centre of `I9`, yaw is degrees clockwise from north. |
| `viewer.py` | The app: FastAPI endpoints (`/state`, `/cmd/{name}`, `/agent/start`, `/approve`, ...) and the page. |

Set `SEEK_MODEL` in `.env` (see `.env.example`) for fast decisions.
