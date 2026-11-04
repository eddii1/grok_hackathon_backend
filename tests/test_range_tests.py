import asyncio
import json
import queue
import socket

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from fleet.app import create_app
from fleet.range_tests import RangeTestRequest, RangeTests, summarize
from fleet.registry import DeviceRegistry
from fleet.tello import TelloAdapter
from fleet.tello_transport import UdpTelloTransport
from test_tello import Clock, FakeTransport, commands, ready


def test_udp_reader_waits_for_tof_family_after_unrelated_expansion_packets():
    class MemorySocket:
        def __init__(self):
            self.messages = queue.Queue()
            self.closed = False
        def bind(self, address): pass
        def settimeout(self, timeout): pass
        def sendto(self, data, address):
            replies = [b'ok'] if data in (b'command', b'streamon') else [
                b'unknown command: keepalive', b'version esp32v1.0.0.22', b'tof 300']
            for reply in replies:
                self.messages.put((reply, ('192.168.10.1', 8889)))
        def recvfrom(self, count):
            if self.closed: raise OSError('closed')
            try: return self.messages.get(timeout=.02)
            except queue.Empty: raise socket.timeout()
        def close(self): self.closed = True

    async def scenario():
        transport = UdpTelloTransport(socket_factory=lambda *args: MemorySocket())
        transport._video = lambda callback: None
        try:
            await transport.start(lambda state: None, lambda frame: None)
            assert await transport.request('EXT tof?', 1) == 'tof 300'
            received = [e for e in transport.wire_events if e['direction'] == 'receive']
            assert [e['response'] for e in received if not e['matched']] == [
                'unknown command: keepalive', 'version esp32v1.0.0.22']
        finally:
            transport.close()
            await asyncio.to_thread(transport.wait_closed)
    asyncio.run(scenario())


def test_cached_valid_range_expires_without_relabeling_a_stale_value_as_current():
    async def scenario():
        clock = Clock()
        device, transport = await ready(enabled=False, clock=clock)
        async def answer(*args): return 'tof 300'
        transport.request = answer
        try:
            reading = await device.read_forward_range()
            assert device.state()['forward_range']['distance_mm'] == 300
            clock.now += 2.1
            cached = device.state()['forward_range']
            assert cached['status'] == 'stale' and cached['distance_mm'] is None
            assert cached['raw_response'] == 'tof 300'
            assert reading['status'] == 'valid'  # historical sample remains immutable
        finally: device.close()
    asyncio.run(scenario())


def test_metrics_exclude_invalid_and_out_of_range_samples():
    summary = summarize([{'status': status, 'distance_mm': value} for status, value in
                         [('valid', 280), ('invalid', None), ('out_of_range', None),
                          ('valid', 320), ('unavailable', None)]], 300)
    assert summary['sample_count'] == 5 and summary['valid_count'] == 2
    assert summary['median_mm'] == 300 and summary['stddev_mm'] == 20
    assert summary['mean_absolute_error_mm'] == 20
    assert summary['ready_for_approach'] is False
    assert summarize([])['median_mm'] is None


def test_bench_run_saves_raw_samples_and_never_calls_flight(tmp_path):
    async def scenario():
        device, transport = await ready(enabled=False)
        replies = iter(['tof 300', 'tof 0', 'tof 8192'])
        async def answer(command, timeout):
            transport.send(command)
            device.update_state({'bat': 80, 'h': 0, 'yaw': 0})
            return 'version esp32v1.0.0.22' if command == 'EXT version?' else next(replies)
        transport.request = answer
        tests = RangeTests(DeviceRegistry([device]), tmp_path)
        try:
            created = tests.create(RangeTestRequest(device_id=device.id, object_label='White card',
                                                    reference_distance_mm=300, sample_count=3))
            await tests.tasks[created['id']]
            job = tests.get(created['id'])
            assert job['status'] == 'completed' and job['firmware'].endswith('1.0.0.22')
            assert job['summary']['valid_count'] == 1
            assert [s['raw_response'] for s in job['samples']] == ['tof 300', 'tof 0', 'tof 8192']
            assert json.loads((tests.directory/f"{job['id']}.json").read_text()) == job
            assert set(commands(transport)) == {'EXT version?', 'EXT tof?'}
        finally: device.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('interruption', ['cancel', 'disconnect'])
def test_bench_run_preserves_partial_samples_on_interruption(tmp_path, interruption):
    async def scenario():
        device, transport = await ready(enabled=False)
        sampled = asyncio.Event()
        async def answer(command, timeout):
            if command == 'EXT version?': return 'version esp32v1.0.0.22'
            sampled.set()
            return 'tof 300'
        transport.request = answer
        tests = RangeTests(DeviceRegistry([device]), tmp_path)
        try:
            job = tests.create(RangeTestRequest(device_id=device.id, object_label='Card', sample_count=3))
            await sampled.wait()
            with pytest.raises(HTTPException) as exc:
                tests.create(RangeTestRequest(device_id=device.id, object_label='Second card'))
            assert exc.value.status_code == 409
            if interruption == 'cancel':
                await tests.cancel(job['id'])
            else:
                device._state_at = 0
                await tests.tasks[job['id']]
            result = tests.get(job['id'])
            assert result['status'] == ('cancelled' if interruption == 'cancel' else 'failed')
            assert len(result['samples']) == 1
            assert not tests.active
        finally: device.close()
    asyncio.run(scenario())


def test_backend_survives_offline_startup_and_can_replace_ground_link(tmp_path, monkeypatch):
    made = []
    def factory(device_id, host, bind):
        device = TelloAdapter(device_id, host, bind, transport=FakeTransport())
        if not made:
            async def offline(): raise TimeoutError('offline')
            device.start = offline
        made.append(device)
        return device
    monkeypatch.setattr('fleet.tello.TelloAdapter', factory)
    config = tmp_path/'devices.json'
    config.write_text(json.dumps({'devices': [{'id': 'tt-1', 'adapter': 'tello'}]}))
    app = create_app(config, tmp_path/'media', agent_mode='stub')
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        state = client.get('/devices').json()[0]
        assert state['connection_state'] == 'offline' and not state['connected']
        assert client.post('/devices/tt-1/reconnect', json={'grounded': False}).status_code == 422
        assert client.post('/devices/tt-1/reconnect', json={'grounded': 'true'}).status_code == 422
        response = client.post('/devices/tt-1/reconnect', json={'grounded': True})
        assert response.status_code == 200
        assert response.json()['connected'] and not response.json()['flight_enabled']
        assert app.state.registry.get('tt-1') is made[1] and made[0].transport.closed
        # Late callbacks into the old adapter cannot update its replacement.
        made[0].update_state({'bat': 1, 'h': 0, 'yaw': 0})
        assert client.get('/devices').json()[0]['battery_pct'] == 80
    assert all(commands(device.transport) == [] for device in made)


def test_retired_range_api_never_queries_failed_hardware(tmp_path, monkeypatch):
    made = []
    def factory(device_id, host, bind):
        device = TelloAdapter(device_id, transport=FakeTransport())
        made.append(device)
        return device
    monkeypatch.setattr('fleet.tello.TelloAdapter', factory)
    config = tmp_path/'devices.json'
    config.write_text(json.dumps({'devices': [{'id': 'tt-1', 'adapter': 'tello'}]}))
    app = create_app(config, tmp_path/'media', agent_mode='stub')
    with TestClient(app) as client:
        response = client.post('/range-tests', json={'device_id':'tt-1','object_label':'Card'})
        assert response.status_code == 410 and 'retired' in response.json()['detail']
    assert commands(made[0].transport) == []


@pytest.mark.parametrize('condition', ['armed', 'enabled', 'airborne'])
def test_reconnect_and_sampling_refuse_potential_flight(tmp_path, condition):
    async def scenario():
        device, transport = await ready(enabled=False)
        if condition == 'armed': device._armed = True
        if condition == 'enabled': device.flight_enabled = True
        if condition == 'airborne': device._possible = True
        registry = DeviceRegistry([device])
        try:
            with pytest.raises(ValueError): await registry.reconnect_ground(device.id)
            with pytest.raises(HTTPException) as exc:
                RangeTests(registry, tmp_path).create(RangeTestRequest(device_id=device.id, object_label='Card'))
            assert exc.value.status_code == 409
            assert not transport.closed and commands(transport) == []
        finally:
            device._armed = device._possible = device.flight_enabled = False
            device.close()
    asyncio.run(scenario())
