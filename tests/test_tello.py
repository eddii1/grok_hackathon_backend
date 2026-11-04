import asyncio
import json
import time
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fleet.agent import AgentTools, StubAgent
from fleet.app import create_app
from fleet.registry import DeviceRegistry
from fleet.tello import SafetyError, SafetyLimits, TelloAdapter
from fleet.tello_transport import UdpTelloTransport


class Clock:
    def __init__(self):
        self.now = 100.0
    def __call__(self):
        return self.now


class FakeTransport:
    def __init__(self):
        self.sent = []
        self.hold = False
        self.height = 60
        self.closed = False
    async def start(self, on_state, on_frame):
        self.on_state, self.on_frame = on_state, on_frame
        for _ in range(3):
            on_state({'bat': 80, 'h': 0, 'yaw': 0, 'vgz': 0})
            on_frame(np.zeros((48, 64, 3), np.uint8))
    def send(self, command):
        self.sent.append((command, time.monotonic()))
    async def request(self, command, timeout, dispatch=None):
        if dispatch:
            dispatch(lambda: self.send(command))
        else:
            self.send(command)
        if command == 'takeoff' and self.height is not None:
            self.on_state({'bat': 80, 'h': self.height, 'yaw': 0, 'vgz': 0})
        if self.hold:
            await asyncio.Event().wait()
        return 'ok'
    def close(self):
        self.closed = True


async def ready(*, enabled=True, clock=None, limits=None):
    transport = FakeTransport()
    device = TelloAdapter('tt-1', flight_enabled=enabled, transport=transport,
                          clock=clock or time.monotonic, limits=limits)
    await device.start()
    if clock is not None:
        # Deterministic policy tests step the watchdog themselves.
        device._quit.set()
        device._watcher.join(timeout=.2)
    device.operator_heartbeat()
    device.tick()
    if enabled:
        device.arm_for_supervised_trial()
    return device, transport


def commands(transport):
    return [name for name, _ in transport.sent]


@pytest.mark.parametrize('field', ['disabled', 'battery', 'telemetry', 'frame', 'operator', 'loop', 'fault'])
def test_preflight_refusal_sends_no_motor_commands(field):
    async def scenario():
        clock = Clock()
        device, transport = await ready(clock=clock)
        if field == 'disabled': device.flight_enabled = False
        if field == 'battery': device._state['bat'] = 49
        if field == 'telemetry': device._state_at = 0
        if field == 'frame': device._frame_at = 0
        if field == 'operator': device._operator_at = 0
        if field == 'loop': device._loop_at = 0
        if field == 'fault': device.trip('STOP before takeoff')
        with pytest.raises(SafetyError):
            await device.takeoff()
        assert commands(transport) == []
        device.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('reason', ['telemetry', 'video', 'operator', 'loop', 'battery', 'altitude', 'deadline'])
def test_watchdog_trips_and_stops_keepalive(reason):
    async def scenario():
        clock = Clock()
        device, transport = await ready(clock=clock)
        await device.takeoff()
        if reason == 'telemetry': device._state_at = 0
        if reason == 'video': device._frame_at = 0
        if reason == 'operator': device._operator_at = 0
        if reason == 'loop': device._loop_at = 0
        if reason == 'battery': device._state['bat'] = 25
        if reason == 'altitude': device._state['h'] = 201
        if reason == 'deadline':
            clock.now += 20
            device.update_state({'bat': 80, 'h': 60, 'yaw': 0})
            device.update_frame(np.zeros((48, 64, 3), np.uint8))
            device.tick()
            device.operator_heartbeat()
        device.watchdog_step()
        assert device.fault
        assert commands(transport)[:4] == ['takeoff', 'rc 0 0 0 0', 'stop', 'land']
        clock.now += 4
        count = len(transport.sent)
        device.watchdog_step()
        assert len(transport.sent) == count  # bounded safety burst, no endless keepalive
        assert device.state()['airborne_possible']
        with pytest.raises(SafetyError):
            await device.takeoff()
        assert commands(transport).count('takeoff') == 1
        device.close()
    asyncio.run(scenario())


def test_cancelled_or_lost_ack_takeoff_latches_uncertainty_and_cannot_restart():
    async def scenario():
        device, transport = await ready()
        transport.hold, transport.height = True, None
        task = asyncio.create_task(device.takeoff())
        await asyncio.sleep(0)
        assert device.state()['airborne_possible']
        device.trip('STOP while takeoff ACK pending')
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for _ in range(4):
            device.update_state({'bat': 80, 'h': 0, 'yaw': 0})
        assert device.state()['airborne_possible']  # ground data cannot resolve an unobserved/ambiguous takeoff
        with pytest.raises(SafetyError):
            await device.takeoff()
        assert commands(transport).count('takeoff') == 1
        assert commands(transport)[-1] == 'land'
        device.close()
    asyncio.run(scenario())


def test_takeoff_queued_before_stop_is_rechecked_under_the_send_guard():
    async def scenario():
        device, transport = await ready()
        transport.hold = True
        first = asyncio.create_task(device.takeoff())
        await asyncio.sleep(0)
        queued = asyncio.create_task(device.takeoff())
        await asyncio.sleep(0)
        device.trip('STOP before queued transmission')
        first.cancel()
        outcomes = await asyncio.gather(first, queued, return_exceptions=True)
        assert isinstance(outcomes[1], SafetyError)
        assert commands(transport).count('takeoff') == 1
        assert commands(transport)[-1] == 'land'
        device.close()
    asyncio.run(scenario())


def test_healthy_zero_rc_keepalive_stops_after_operator_lease_expires():
    async def scenario():
        clock = Clock()
        device, transport = await ready(clock=clock)
        await device.takeoff()
        device.watchdog_step()
        assert commands(transport)[-1] == 'rc 0 0 0 0'
        clock.now += 2.1
        device.update_state({'bat': 80, 'h': 60, 'yaw': 0, 'vgz': 0})
        device.update_frame(np.zeros((48, 64, 3), np.uint8))
        device.tick()
        device.watchdog_step()
        assert device.fault == 'Stale operator'
        assert commands(transport)[-1] == 'land'
        device.close()
    asyncio.run(scenario())


def test_takeoff_timeout_never_retries():
    async def scenario():
        device, transport = await ready()
        async def lose_ack(command, timeout, dispatch=None):
            dispatch(lambda: transport.send(command))
            raise TimeoutError('Lost ACK')
        transport.request = lose_ack
        with pytest.raises(TimeoutError):
            await device.takeoff()
        assert commands(transport) == ['takeoff', 'rc 0 0 0 0', 'stop', 'land']
        assert device.state()['airborne_possible'] and device.fault
        device.close()
    asyncio.run(scenario())


def test_watchdog_runs_when_asyncio_event_loop_is_blocked():
    async def scenario():
        limits = replace(SafetyLimits(), loop_timeout=.08)
        device, transport = await ready(limits=limits)
        await device.takeoff()
        blocked_at = time.monotonic()
        time.sleep(.25)  # deliberate event-loop stall; watchdog is a separate thread
        assert 'land' in commands(transport)
        landed_request_at = next(t for command, t in transport.sent if command == 'land')
        assert blocked_at < landed_request_at < blocked_at + .25
        assert device.fault == 'Stale event loop'
        device.close()
    asyncio.run(scenario())


def test_touchdown_requires_airborne_observation_and_sustained_ground_frames():
    async def scenario():
        clock = Clock()
        device, transport = await ready(clock=clock)
        await device.takeoff()
        device.trip('Normal end of hover')
        for _ in range(3):
            clock.now += .3
            device.update_state({'bat': 80, 'h': 0, 'yaw': 0, 'vgz': 0})
        await device.wait_for_touchdown()
        assert not device.state()['airborne_possible']
        assert device.fault  # never automatically re-arm
        with pytest.raises(SafetyError):
            await device.takeoff()
        device.close()
    asyncio.run(scenario())


def test_registry_reports_unconfirmed_landing_instead_of_three_second_success():
    async def scenario():
        device, transport = await ready(limits=replace(SafetyLimits(), landing_timeout=.02))
        await device.takeoff()
        results = await DeviceRegistry([device]).stop_all()
        assert results[0]['airborne_possible']
        assert 'Landing unconfirmed' in results[0]['errors'][0]
        assert 'emergency' not in commands(transport)
        device.close()
    asyncio.run(scenario())


def test_translation_rotation_and_nonzero_rc_are_disabled():
    async def scenario():
        device, transport = await ready()
        with pytest.raises(SafetyError): await device.move(20, 0, 0)
        with pytest.raises(SafetyError): await device.rotate(45)
        with pytest.raises(SafetyError): device.rc(1, 0, 0, 0)
        with pytest.raises(SafetyError): await device.emergency()
        assert commands(transport) == []
        device.close()
    asyncio.run(scenario())


def test_monitor_mode_never_sends_motor_commands_even_with_unexpected_height():
    async def scenario():
        device, transport = await ready(enabled=False)
        device.update_state({'bat': 80, 'h': 60, 'yaw': 0})
        device.watchdog_step()
        device.close()
        assert commands(transport) == []
        assert device.state()['airborne_possible']
    asyncio.run(scenario())


def test_flight_enable_switch_alone_cannot_send_motors_before_operator_arming():
    async def scenario():
        transport = FakeTransport()
        device = TelloAdapter('tt-1', transport=transport, flight_enabled=True)
        await device.start()
        device.tick()
        device.operator_heartbeat()
        with pytest.raises(SafetyError, match='not armed'):
            await device.takeoff()
        device.update_state({'bat': 80, 'h': 60, 'yaw': 0, 'vgz': 0})
        device.watchdog_step()
        device.close()
        assert commands(transport) == []
    asyncio.run(scenario())


def test_real_device_is_not_labelled_simulated_and_browser_cannot_arm(tmp_path, monkeypatch):
    made = []
    def factory(device_id, host, bind):
        device = TelloAdapter(device_id, transport=FakeTransport())
        made.append(device)
        return device
    monkeypatch.setattr('fleet.tello.TelloAdapter', factory)
    config = tmp_path / 'devices.json'
    config.write_text(json.dumps({'devices': [{'id': 'tt-1', 'adapter': 'tello'}]}))
    app = create_app(config, tmp_path / 'media', agent_mode='stub')
    with TestClient(app) as client:
        state = client.get('/devices').json()[0]
        assert not state['simulated'] and not state['flight_enabled']
        assert client.get('/health').json()['mode'] != 'simulator'
        async def check_tools():
            async def emit(event): pass
            tools = AgentTools(app.state.registry, emit, tmp_path)
            devices = await tools.list_devices()
            assert 'takeoff' not in devices['devices'][0]['capabilities']
            with pytest.raises(ValueError): await tools.takeoff('tt-1')
            with pytest.raises(ValueError): await StubAgent(tools).run('/demo tt-1')
        asyncio.run(check_tools())
    assert commands(made[0].transport) == []


def test_hover_cli_rejects_piped_flight_before_opening_a_connection(monkeypatch):
    import fleet.supervised_hover as cli
    monkeypatch.setattr('sys.argv', ['supervised_hover', '--fly'])
    monkeypatch.setattr('sys.stdin.isatty', lambda: False)
    monkeypatch.setattr(cli, 'TelloAdapter', lambda *a, **kw: pytest.fail('Must not construct adapter'))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize('text', ['forward 20', 'flip f', 'rc 1 0 0 0', 'takeoff\nland', 'ap SSID password'])
def test_transport_does_not_send_arbitrary_sdk_strings(text):
    transport = UdpTelloTransport()
    with pytest.raises(ValueError):
        transport.send(text)


@pytest.mark.parametrize('field,value', [('max_height_cm',201), ('max_flight_seconds',21),
                                      ('operator_timeout',3), ('min_takeoff_battery',49), ('land_battery',24),
                                      ('frame_timeout',float('nan'))])
def test_safety_policy_cannot_be_relaxed(field, value):
    with pytest.raises(ValueError):
        replace(SafetyLimits(), **{field: value})


def test_motor_cut_is_only_explicit_operator_action_and_failure_leaves_landing_path():
    async def scenario():
        device, transport = await ready()
        await device.takeoff()
        original = transport.send
        def fail_motor_cut(command):
            if command == 'emergency':
                raise OSError('Network send failed')
            original(command)
        transport.send = fail_motor_cut
        with pytest.raises(SafetyError, match='transmission failed'):
            device.operator_motor_cut()
        assert commands(transport)[-1] == 'land'
        assert device.state()['airborne_possible']
        device.close()
    asyncio.run(scenario())


def test_operator_motor_cut_does_not_turn_into_normal_stop_or_resume_keepalives():
    async def scenario():
        device, transport = await ready()
        await device.takeoff()
        device.operator_motor_cut()
        sent = list(transport.sent)
        device.watchdog_step()
        await device.hover()
        assert commands(transport)[-1] == 'emergency'
        assert transport.sent == sent
        assert device.state()['airborne_possible']
        device.close()
    asyncio.run(scenario())


def test_actual_transport_reader_links_replies_and_discards_late_cancelled_results():
    import queue
    import socket
    class MemorySocket:
        def __init__(self):
            self.messages, self.sent, self.closed = queue.Queue(), [], False
        def bind(self, address): pass
        def settimeout(self, timeout): pass
        def sendto(self, data, address):
            self.sent.append(data.decode())
            if data in (b'command', b'streamon'):
                self.messages.put((b'noise', ('192.168.10.99', 8889)))
                self.messages.put((b'ok', ('192.168.10.1', 8889)))
        def recvfrom(self, count):
            if self.closed: raise OSError('closed')
            try: return self.messages.get(timeout=.02)
            except queue.Empty: raise socket.timeout()
        def close(self): self.closed = True
    async def scenario():
        sockets = [MemorySocket(), MemorySocket()]
        transport = UdpTelloTransport(socket_factory=lambda *args: sockets.pop(0))
        transport._video = lambda callback: None
        errors = []
        asyncio.get_running_loop().set_exception_handler(lambda loop, context: errors.append(context))
        await transport.start(lambda state: None, lambda frame: None)
        control = transport.control
        assert control.sent == ['command', 'streamon']
        task = asyncio.create_task(transport.request('takeoff', 1))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        control.messages.put((b'ok', ('192.168.10.1', 8889)))
        await asyncio.sleep(.05)
        assert errors == [] and control.sent.count('takeoff') == 1
        transport.close()
    asyncio.run(scenario())
