# Tasks

The whole team's task list (backend, frontend, pitch) in one place. Architecture, interface contracts and the demo are in the main repo: [`docs/plan.md`](https://github.com/eddii1/grok_hackathon/blob/main/docs/plan.md).

## How to use

- **Claim:** put your GitHub handle after `Owner:` and push before you start. One owner per task; ask in the chat before taking a claimed one.
- **Done:** tick the box when the "Done when" is true and the work is on `main`.
- **Blocked:** add `BLOCKED: <why>` to the line and say it in the chat.
- Team: eddii1, MuhammadAbbasi, Gsnvoff, realMoai, gabrieledigilio.

## Who does what

| Role | People | Tasks |
|---|---|---|
| Hardware (2) | ___, ___ | D1-D6 (drone harness, network, safety, second drone), P1 (footage), P4-P7 (follow and search loops, tuned on the drone) |
| Software (2) | eddii1, ___ | S1-S2, A1-A7 (agent), B1-B3 (API), P2-P3 (detection), E1-E3 (evaluation) |
| Presentation + frontend (1) | ___ | F1-F8 |
| Presentation (1) | ___ | K1-K5, E4 (README) |

**Now:** hardware sends drone footage (mp4 or jpg) to software, and software runs `uv run fleet-probe <file>` on it with Grok and with GLM (A1).

## Milestones

| | By | What works |
|---|---|---|
| M1 | Wed 13:00 | Interface stubs merged, network question answered, each workstream's spike done |
| M2 | Wed 18:00 | Tracer bullet: "take off and tell me what you see" from the UI flies the drone, shows video, Grok describes, drone lands |
| M3 | Wed 23:00 | Grok picks the right person from numbered boxes; follow loop works on the simulator |
| M4 | Thu 09:00 | Full mission on the real drone: search, confirm, follow, photo, verify, report; second device on screen |
| M5 | Thu 12:00 | Feature freeze: evaluation numbers in the README, backup video recorded |
| Demo | Thu 16:00 | Three rehearsals done by 14:00 |

## 0. Setup

- [ ] **S1 Backend skeleton** - Python 3.11+, FastAPI, `.env.example` with `XAI_API_KEY`, run command in the README. Owner: ___ · M1 · Done when: `GET /health` answers locally.
- [ ] **S2 Interface stubs** - `DeviceAdapter`, `DeviceState`, `Detection`, agent tool signatures and `/ws` event types exactly as in `docs/plan.md`, plus stub implementations that return fake data. Owner: ___ · M1 · Done when: every workstream imports them and runs against fakes.

## 1. Drone harness

- [ ] **D1 Network spike (blocks everything)** - TT in station mode (`ap <ssid> <pass>`) on a phone hotspot, laptop on the same hotspot: commands work, video arrives, Grok API reachable. If video fails, set up the fallback (drone in its own Wi-Fi mode, internet for the laptop through a USB-tethered phone) and write down which one we use. Owner: ___ · M1 · Done when: one script flies up 50 cm, shows video and calls Grok in the same run.
- [ ] **D2 TelloAdapter** - `djitellopy` with `retry_count=1`: connect, state from the state stream (battery, height, flying, yaw), `latest_frame` from a background decoder, `takeoff/land/hover/move/rotate/rc/emergency`. Owner: ___ · M2 · Done when: it passes the same smoke script as the SimAdapter.
- [ ] **D3 Safety layer** - inside the adapter: max height 200 cm, `rc` capped at ±40, keepalive under the 15 s auto-land, takeoff refused below 25 % battery, `hover()` on any timeout (never a blind retry), land on backend shutdown. Owner: ___ · M2 · Done when: each limit has a test against the SimAdapter.
- [ ] **D4 SimAdapter** - loops a recorded video as its camera, fakes state, accepts every command and logs it. Owner: ___ · M1 · Done when: the UI and the agent run end to end with no drone.
- [ ] **D5 Device registry** - devices from a config file, `list_states()`, `stop_all()` that hovers then lands every flying device. Owner: ___ · M2 · Done when: two devices (real + sim) show up together.
- [ ] **D6 Second real drone** (if the organisers give us one) - per-drone video port (`port 8890 <video_port>`), both TTs on the hotspot. Owner: ___ · M4 · Done when: two live video feeds at once.

## 2. Perception and control

- [ ] **P1 Record footage** - 2-3 min of drone video of teammates walking, in different clothes, with look-alike distractors (two red tops). Only teammates in frame. Saved under `data/`. Owner: ___ · M1 · Done when: files are on the shared drive or in the repo (if small).
- [ ] **P2 `detect()`** - YOLO nano, people only, ByteTrack IDs (`model.track(..., persist=True)`). Owner: ___ · M2 · Done when: stable track IDs on P1 footage; FPS on a laptop written down.
- [ ] **P3 `annotate()`** - JPEG with large numbered boxes, used for Grok and drawn on the MJPEG stream. Owner: ___ · M2 · Done when: Grok reads the numbers correctly on 5 test frames.
- [ ] **P4 `follow_command()`** - proportional control from box centre and size to `rc` (yaw, up/down, forward/back), with deadbands and caps. Owner: ___ · M3 · Done when: it keeps a walking teammate centred on the real drone for 20 s.
- [ ] **P5 `photo_ready()`** - target centred and big enough for N frames in a row. Owner: ___ · M3 · Done when: it triggers on P1 footage where a human would take the photo.
- [ ] **P6 Follow skill** - 10-20 Hz loop; stops on photo ready, track lost or timeout; hovers on exit. Owner: ___ · M3 · Done when: it runs on the SimAdapter and the real drone.
- [ ] **P7 Search skill** - rotate in steps (for example 45°), collect detections at each step, hand annotated frames to the agent. Owner: ___ · M3 · Done when: a full turn returns every visible person once.

## 3. Agent (Grok)

- [ ] **A1 Vision probe** - `fleet-probe` (done) on real drone footage with Grok and GLM: does it describe the scene and find a described person? Latency per frame. Owner: eddii1 · M1 · Done when: latency and model name are written in the README.
- [ ] **A2 Tools and system prompt** - the tools from `docs/plan.md` as function definitions, and a system prompt that requires operator confirmation before takeoff and before acting on a person. Owner: ___ · M2 · Done when: Grok calls `list_devices` and `takeoff` from a chat message.
- [ ] **A3 Agent loop** - user message, then tool calls, then events to the UI; conversation state; errors shown in the chat. Owner: ___ · M2 · Done when: the tracer bullet (M2) runs.
- [ ] **A4 Target selection** - description + annotated frames in, then `track_id` + reason, or "not found". Owner: ___ · M3 · Done when: it picks the right person on P1 frames, including a distractor.
- [ ] **A5 Confirm flow** - `confirm_target` emits `confirm_request` and waits for `confirm_response` (timeout counts as a reject). Owner: ___ · M3 · Done when: Reject makes the agent pick again or report "not found".
- [ ] **A6 `verify_photo` + `report`** - Grok checks the photo against the description; the final chat message has the photo and the verdict. Owner: ___ · M4 · Done when: the M4 mission ends with a report.
- [ ] **A7 Multi-device** - the agent chooses which device searches; stretch goal: two devices split the room. Owner: ___ · M4 · Done when: the agent assigns a task to a device by itself.

## 4. Platform

Backend (this repo):

- [ ] **B1 API** - `/ws` event hub with the contract from `docs/plan.md`, `/video/{device_id}` MJPEG, static `/photos` and `/snapshots`. Owner: ___ · M2 · Done when: the frontend shows live events and video.
- [ ] **B2 Event log** - append-only JSONL of every event, feeding the UI log, the report and the evaluation. Owner: ___ · M2 · Done when: one mission replays from its log.
- [ ] **B3 STOP path** - `stop_all` goes straight to the device registry, bypassing the agent. Owner: ___ · M2 · Done when: STOP lands the drone while the agent is mid-call.

Frontend ([`grok_hackathon_frontend`](https://github.com/eddii1/grok_hackathon_frontend), `app/`):

- [ ] **F1 App skeleton** - Vite + React, connects to `/ws`, runs on mock events. Owner: ___ · M1 · Done when: mock chat and device events render.
- [ ] **F2 Chat panel** - messages, tool calls shown as small steps. Owner: ___ · M2
- [ ] **F3 Device tiles** - MJPEG video per device plus a state card (battery, height, flying, connected). Owner: ___ · M2
- [ ] **F4 Confirm dialog** - target crop, Grok's reason, Confirm / Reject. Owner: ___ · M3
- [ ] **F5 Mission timeline** - the event log as a readable timeline. Owner: ___ · M4
- [ ] **F6 STOP** - big button plus a keyboard shortcut, always visible. Owner: ___ · M2
- [ ] **F7 Photo results** - photo, Grok's verdict and confidence. Owner: ___ · M4
- [ ] **F8 Product look** - name, logo, consistent theme; it must look like a product, not a hackathon page. Owner: ___ · M5

## 5. Evaluation, pitch and submission

- [ ] **E1 Selection test set** - about 30 staged frames, 2-5 people each, with distractors; a description and the correct box for each. Owner: ___ · M3
- [ ] **E2 Baseline** - picker that chooses the box with the most pixels of the named colour; script that scores Grok vs. baseline on E1. Owner: ___ · M4 · Done when: both accuracies are in the README.
- [ ] **E3 Mission runs** - 10 runs on the real drone: success rate, time to photo, wrong-person photos, operator rejects. Owner: ___ · M5
- [ ] **E4 README** - results, known limitations (including the AGPL detector), roadmap. Owner: ___ · M5
- [ ] **K1 Organisers** - ask now: a second drone, batteries and chargers, where we may fly, hotspot rules. Owner: ___ · M1
- [ ] **K2 Name and one-liner** - product name, tagline, logo. Owner: ___ · M2
- [ ] **K3 Deck** (frontend repo, `slides/`) - problem, live demo, product, market and comparables (Auterion, Anduril Lattice, Skydio, Brinc, Flock Safety; verify every number), business model, go-to-market (search and rescue and security first, police later), regulation (EU AI Act: description not face, human confirms), team, use of the 30k. Owner: ___ · Thu 14:00
- [ ] **K4 Demo script and backup video** - 5-minute script, three rehearsals, backup recording on Thursday morning. Owner: ___ · M5
- [ ] **K5 Portal submission** - project name, description, GitHub URL, demo URL, tech stack, slides URL, video URL. Owner: ___ · Thu 15:30
