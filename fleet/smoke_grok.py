"""Opt-in live API smoke test against local simulators (uses xAI credits)."""
import argparse
import asyncio
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from dotenv import dotenv_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve-simulated-takeoff", action="store_true",
                        help="Approve the local simulator confirmation automatically")
    args = parser.parse_args()
    import os
    settings = {**dotenv_values(Path(__file__).resolve().parent.parent / ".env"), **os.environ}
    if not settings.get("XAI_API_KEY"):
        parser.exit(2, "Set XAI_API_KEY in backend/.env before running the live test.\n")
    from fastapi.testclient import TestClient
    from .app import create_app

    with TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / "devices.json"
        config.write_text(json.dumps({"devices": [{"id": "drone-1", "adapter": "sim"}]}))
        app = create_app(config, root / "media", agent_mode="grok")
        seen, confirmed = set(), False
        started = time.monotonic()
        with TestClient(app) as client, client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "user_message", "text":
                          "Run a simulator smoke test: list devices, take off drone-1 after operator confirmation, "
                          "describe its actual camera with describe_scene, take a photo, verify that the photo shows "
                          "a simulator label using verify_photo, land, then report. Do not skip any tool. "
                          "Explain the observed results truthfully."})
            deadline = time.monotonic() + 210
            # TestClient's receive is blocking, so a watchdog bounds the live test.
            async def watchdog():
                await asyncio.sleep(205)
                await app.state.cancel_missions()
            watchdog_task = client.portal.start_task_soon(watchdog)
            try:
                while time.monotonic() < deadline:
                    event = ws.receive_json()
                    kind = event["type"]
                    if kind == "confirm_request":
                        print(event["reason"])
                        approved = args.approve_simulated_takeoff or input("Approve simulated takeoff? [yes/no] ").strip().lower() == "yes"
                        ws.send_json({"type": "confirm_response", "request_id": event["request_id"], "approved": approved})
                        confirmed = bool(approved)
                        if not approved:
                            parser.exit(1, "Live test cancelled by operator.\n")
                    elif kind == "tool_call":
                        print(f"Tool: {event['name']}")
                        if "error" in event["result"] or event["result"].get("stub"):
                            raise RuntimeError("Live test received a tool failure or stub result")
                        seen.add(event["name"])
                    elif kind == "chat" and event["role"] == "assistant":
                        print(event["text"])
                        if event["text"].startswith("Mission failed"):
                            raise RuntimeError("Live API mission failed; check credentials, model access and network")
                    elif kind == "mission_status" and event["status"] != "started":
                        if event["status"] != "completed":
                            raise RuntimeError(f"Live mission {event['status']}")
                        required = {"list_devices", "takeoff", "describe_scene", "take_photo", "verify_photo", "land", "report"}
                        if not required <= seen:
                            raise RuntimeError(f"Model skipped required tools: {sorted(required - seen)}")
                        break
                else:
                    raise TimeoutError("Live smoke test did not complete")
                assert confirmed and not client.get("/devices").json()[0]["flying"]
                print(f"PASS: real Grok tools, image analysis, confirmation and landing ({time.monotonic() - started:.1f}s).")
            finally:
                watchdog_task.cancel()


if __name__ == "__main__":
    main()
