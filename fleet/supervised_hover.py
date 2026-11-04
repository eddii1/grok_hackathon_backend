"""Ground preflight by default. --fly additionally requires an interactive Windows terminal."""
import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4

from .tello import SafetyError, TelloAdapter


async def run(args):
    scanning = getattr(args, 'scan', False)
    if scanning:
        from .yaw_scan import SupervisedYawAdapter
        adapter = SupervisedYawAdapter('tt-1', args.host, args.bind, flight_enabled=args.fly)
    else:
        adapter = TelloAdapter('tt-1', args.host, args.bind, flight_enabled=args.fly)
    keyboard_quit = threading.Event()
    go = threading.Event()
    stop = threading.Event()
    pump = None
    result = 'ground-check'
    report_path = Path('runtime') / f'tt-{"scan" if scanning else "hover"}-{uuid4().hex}.json'
    images = []
    sampler = None

    def sample(label):
        latest = adapter.latest_frame()
        if latest is None:
            raise SafetyError('No fresh scan frame; landing required')
        images.append({'label': label, 'frame': latest[0], 'frame_received_at': latest[1],
                       'telemetry_sampled_at': time.time(), 'telemetry': adapter.state(), 'position': None})

    async def collect_images():
        while len(images) < 14:
            sample('during-yaw')
            await asyncio.sleep(.25)

    async def loop_heartbeat():
        while True:
            adapter.tick()
            await asyncio.sleep(.05)

    def keyboard():
        import msvcrt
        while not keyboard_quit.wait(.02):
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == ' ':
                    adapter.operator_heartbeat()
                    go.set()
                elif key.lower() == 'q' or key == '\x1b':
                    adapter.trip('Operator requested landing')
                    stop.set()
                elif key == '\x18':
                    try:
                        adapter.operator_motor_cut()
                        stop.set()
                    except SafetyError:
                        pass

    try:
        await adapter.start()
        pump = asyncio.create_task(loop_heartbeat())
        await adapter.wait_ready()
        print('REAL DRONE:', json.dumps(adapter.state()))
        if not args.fly:
            print('Ground-only adapter check. No motor commands. To prepare flight, read the staged runbook.')
            return
        print('ONE supervised 30-degree clockwise scan, then land.' if scanning else 'ONE supervised stationary hover.')
        print('Automatic SDK takeoff height, not a promised 50 cm. No translation commands.')
        print('Clear controlled area, guards installed, textured well-lit floor, operator and spotter required.')
        print('Keep tapping/holding SPACE for operator heartbeat. Q/Esc = hover + land.')
        print('Ctrl+X = MOTOR CUT: aircraft falls. Use only for imminent danger.')
        print('Maximum 20 s from takeoff request; every stop/fault locks out another takeoff.')
        arming_phrase = 'SCAN30 tt-1' if scanning else 'HOVER tt-1'
        phrase = input(f'After checking the area and aircraft, type {arming_phrase} to arm this one trial: ')
        if phrase != arming_phrase:
            raise SafetyError('Arming phrase did not match; no takeoff')
        adapter.arm_for_supervised_trial()
        # Refresh after the blocking prompt; no flight has started during input().
        adapter.tick()
        threading.Thread(target=keyboard, daemon=True).start()
        print('Press SPACE within 10 seconds to start; keep tapping SPACE until touchdown. Q cancels.')
        deadline = time.monotonic() + 10
        while not go.is_set():
            if stop.is_set() or time.monotonic() > deadline:
                raise SafetyError('Operator did not start trial; no takeoff')
            await asyncio.sleep(.02)
        await adapter.takeoff()
        # ACK alone is not stable hover: require sustained fresh airborne telemetry.
        deadline = time.monotonic() + 5
        stable_at = None
        while True:
            state = adapter.state()
            if state['fault'] or stop.is_set():
                raise SafetyError(state['fault'] or 'Operator stop')
            stable = state['connected'] and 30 <= state['height_cm'] <= 150 and abs(adapter._state.get('vgz', 100)) <= 5
            stable_at = (time.monotonic() if stable_at is None else stable_at) if stable else None
            if stable_at is not None and time.monotonic() - stable_at >= .5:
                break
            if time.monotonic() >= deadline:
                raise SafetyError('No stable hover telemetry observed')
            await asyncio.sleep(.05)
        if scanning:
            print('Stable hover observed. One cw 30 attempt, maximum 4 s; Q/Esc lands immediately.')
            sample('before-yaw')
            sampler = asyncio.create_task(collect_images())
            try:
                await adapter.rotate(30)
            finally:
                sampler.cancel()
                await asyncio.gather(sampler, return_exceptions=True)
                sampler = None
            sample('after-yaw')
        else:
            print(f'Stable hover telemetry observed. Landing in {args.hover_seconds} seconds, or Q/Esc now.')
            until = time.monotonic() + args.hover_seconds
            while time.monotonic() < until and not stop.is_set() and not adapter.fault:
                await asyncio.sleep(.02)
        await adapter.land()
        result = 'touchdown-telemetry-observed'
        print('Low/stable touchdown telemetry observed. OPERATOR: verify aircraft landed and motors stopped.')
    except BaseException as exc:
        result = 'interrupted-or-failed'
        adapter.trip('Trial interrupted or failed')
        raise
    finally:
        if sampler:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
        if adapter.state()['airborne_possible']:
            try:
                await adapter.wait_for_touchdown()
            except SafetyError:
                print('LANDING UNCONFIRMED. Operator must check the aircraft; no further takeoff is allowed.')
        keyboard_quit.set()
        if pump:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
        adapter.close()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if images:
            import hashlib
            import cv2
            photo_dir = report_path.parent/'photos'
            photo_dir.mkdir(parents=True, exist_ok=True)
            for index, entry in enumerate(images):
                frame = entry.pop('frame')
                ok, encoded = cv2.imencode('.jpg', frame)
                if not ok:
                    entry['error'] = 'JPEG encoding failed'
                    continue
                data = encoded.tobytes()
                filename = f'{report_path.stem}-{index:02d}.jpg'
                (photo_dir/filename).write_bytes(data)
                entry.update(photo_url=f'/photos/{filename}', sha256=hashlib.sha256(data).hexdigest(),
                             width=frame.shape[1], height=frame.shape[0])
        report_path.write_text(json.dumps({'result': result, 'state': adapter.state(), 'events': adapter.events,
                                          'kind': 'supervised-yaw-scan' if scanning else 'supervised-hover',
                                          'frames': images, 'assessment': None,
                                          'wire_events': getattr(adapter.transport, 'wire_events', [])}, indent=2), encoding='utf-8')
        print('Local run report:', report_path.resolve())
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='192.168.10.1')
    parser.add_argument('--bind', default='0.0.0.0')
    parser.add_argument('--fly', action='store_true', help='Permit the subsequent interactive one-flight arming procedure')
    parser.add_argument('--hover-seconds', type=int, choices=[1, 2, 3], default=3)
    args = parser.parse_args()
    if args.fly and (sys.platform != 'win32' or not sys.stdin.isatty()):
        parser.error('--fly requires an interactive Windows terminal; piped/unattended flight is disabled')
    try:
        asyncio.run(run(args))
    except (SafetyError, OSError, TimeoutError) as exc:
        parser.exit(1, f'Trial stopped: {type(exc).__name__}: {exc or "No SDK acknowledgement before timeout"}\n')


if __name__ == '__main__':
    main()
