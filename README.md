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
