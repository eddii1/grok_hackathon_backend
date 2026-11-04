"""Non-flight TT SDK 3.0 probe. No motor or Wi-Fi reconfiguration commands."""
import argparse
import ipaddress
import json
import socket
import time
from pathlib import Path


ALLOWED_COMMANDS = frozenset({"command", "sdk?", "hardware?", "battery?", "streamon", "streamoff"})


class GroundLink:
    def __init__(self, host="192.168.10.1", bind="0.0.0.0", timeout=3.0):
        # One owner per port: do not use SO_REUSEADDR or silently share replies.
        self.host, self.bind, self.timeout = str(ipaddress.IPv4Address(host)), bind, timeout
        self.control = None
        self.telemetry = None

    def __enter__(self):
        try:
            self.control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.control.bind((self.bind, 8889))
            self.control.settimeout(self.timeout)
            self.telemetry = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.telemetry.bind((self.bind, 8890))
            self.telemetry.settimeout(self.timeout)
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()

    def close(self):
        for channel in (self.control, self.telemetry):
            if channel is not None:
                channel.close()

    def command(self, text: str) -> str:
        if text not in ALLOWED_COMMANDS:
            raise ValueError("Command is not on the non-flight allowlist")
        self.control.sendto(text.encode("ascii"), (self.host, 8889))
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            self.control.settimeout(max(.001, deadline - time.monotonic()))
            data, peer = self.control.recvfrom(2048)
            if peer != (self.host, 8889):
                continue
            response = data.decode("utf-8", errors="replace").strip()
            if response.lower().startswith(("error", "unactive")):
                raise RuntimeError(f"Drone rejected non-flight command {text}: {response}")
            return response
        raise TimeoutError(f"No response to {text}; no retry was sent")

    def sample_state(self, seconds: float = 2.0) -> dict:
        # Drop queued packets before measuring a new receive window.
        self.telemetry.setblocking(False)
        try:
            while True:
                self.telemetry.recvfrom(4096)
        except BlockingIOError:
            pass
        count, first, last, state = 0, None, None, {}
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.telemetry.settimeout(max(.001, deadline - time.monotonic()))
            try:
                packet, peer = self.telemetry.recvfrom(4096)
            except socket.timeout:
                break
            if peer[0] != self.host:
                continue
            parsed = parse_state(packet)
            if not {"bat", "h", "yaw"} <= parsed.keys():
                continue
            now = time.monotonic()
            first = now if first is None else first
            last, state = now, parsed
            count += 1
        if count < 2:
            raise RuntimeError("No sustained valid TT telemetry on UDP 8890; do not progress to flight")
        return {"packets": count, "observed_hz": round((count - 1) / max(last - first, .001), 1),
                "state": state}


def parse_state(packet: bytes) -> dict[str, int | float]:
    state = {}
    for entry in packet.decode("ascii", errors="replace").strip().split(";"):
        key, separator, value = entry.partition(":")
        if separator:
            try:
                number = float(value)
                if not (-1e9 < number < 1e9):
                    continue
                state[key] = int(number) if number.is_integer() else number
            except ValueError:
                pass
    return state


def capture_video(link: GroundLink, target: Path) -> dict:
    try:
        import av
    except ImportError:
        raise RuntimeError("Install the hardware extra first: uv pip install -e '.[dev,hardware]'") from None
    import cv2
    try:
        response = link.command("streamon")
        if response.lower() != "ok":
            raise RuntimeError("streamon was not acknowledged with ok")
        started = time.monotonic()
        with av.open("udp://0.0.0.0:11111?fifo_size=500000&overrun_nonfatal=1", timeout=(8, 8)) as container:
            frames = 0
            for frame in container.decode(video=0):
                frames += 1
                # Several decoded frames, not merely incoming UDP packets.
                if frames >= 3:
                    image = frame.to_ndarray(format="bgr24")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(target), image):
                        raise RuntimeError("Could not write TT camera snapshot")
                    return {"decoded_frames": frames, "width": image.shape[1], "height": image.shape[0],
                            "seconds_to_capture": round(time.monotonic() - started, 2), "snapshot": str(target.resolve())}
                if time.monotonic() - started > 12:
                    break
        raise RuntimeError("No usable TT video; do not progress to flight")
    except Exception as exc:
        raise RuntimeError(f"TT camera decode failed ({type(exc).__name__}); check video on UDP 11111") from exc
    finally:
        # Stream cleanup only, never takeoff/land/emergency as a side effect.
        try:
            link.command("streamoff")
        except (OSError, RuntimeError, TimeoutError):
            pass


def run_probe(link: GroundLink, video_path: Path | None = None) -> dict:
    if link.command("command").lower() != "ok":
        raise RuntimeError("Could not enter SDK mode")
    report = {"mode": "ground-only", "host": link.host,
              "sdk": link.command("sdk?"), "hardware": link.command("hardware?"),
              "battery_pct": int(link.command("battery?")), "telemetry": link.sample_state()}
    if video_path is not None:
        report["video"] = capture_video(link, video_path)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.10.1", help="TT IPv4 address; AP mode default is 192.168.10.1")
    parser.add_argument("--bind", default="0.0.0.0", help="Optional laptop TT-network IPv4 address")
    parser.add_argument("--video", action="store_true", help="Decode a few frames and save a local snapshot")
    parser.add_argument("--output", type=Path, default=Path("runtime/tt-ground-report.json"))
    args = parser.parse_args()
    try:
        with GroundLink(args.host, args.bind) as link:
            report = run_probe(link, args.output.with_suffix(".jpg") if args.video else None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print("PASS: ground link only. No motor commands sent; flight readiness is not certified.")
    except (OSError, RuntimeError, ValueError, TimeoutError) as exc:
        parser.exit(1, f"Ground check failed: {exc}\nCheck TT Wi-Fi, local UDP firewall rules and port ownership. No flight commands were sent.\n")


if __name__ == "__main__":
    main()
