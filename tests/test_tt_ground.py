import socket

import pytest

from fleet.tt_ground import ALLOWED_COMMANDS, GroundLink, capture_video, parse_state, run_probe


class FakeControl:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.sent = []
    def sendto(self, payload, peer):
        self.sent.append(payload.decode())
    def recvfrom(self, _):
        response = next(self.replies)
        if isinstance(response, Exception):
            raise response
        return response
    def settimeout(self, _):
        pass


@pytest.mark.parametrize('command', ['takeoff', 'land', 'emergency', 'motoron', 'rc 0 0 0 0', 'ap name pass', 'command\ntakeoff'])
def test_motor_and_network_changes_are_unconditionally_rejected(command):
    link = GroundLink()
    link.control = FakeControl([])
    with pytest.raises(ValueError, match='allowlist'):
        link.command(command)
    assert link.control.sent == []


def test_no_command_retry_after_timeout():
    link = GroundLink()
    link.control = FakeControl([socket.timeout('lost packet')])
    with pytest.raises(socket.timeout):
        link.command('command')
    assert link.control.sent == ['command']


def test_replies_from_other_ips_are_ignored():
    link = GroundLink()
    link.control = FakeControl([(b'fake', ('192.168.10.55', 8889)), (b'ok', ('192.168.10.1', 8889))])
    assert link.command('command') == 'ok'


def test_ground_probe_only_queries_expected_values():
    link = GroundLink()
    link.control = FakeControl([(value, ('192.168.10.1', 8889)) for value in [b'ok', b'30', b'RMTT', b'88']])
    link.sample_state = lambda: {'packets': 20, 'state': {'bat': 88, 'h': 0, 'yaw': -4}}
    report = run_probe(link)
    assert report['hardware'] == 'RMTT' and report['battery_pct'] == 88
    assert link.control.sent == ['command', 'sdk?', 'hardware?', 'battery?']
    assert set(link.control.sent) <= ALLOWED_COMMANDS


def test_parse_telemetry_ignores_malformed_values():
    assert parse_state(b'bat:80;h:0;yaw:-10;baro:12.34;other:nan;bad;') == {
        'bat': 80, 'h': 0, 'yaw': -10, 'baro': 12.34}


def test_video_is_stopped_even_when_decoder_fails(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    def fail(*args, **kwargs):
        raise RuntimeError('decoder failed')
    monkeypatch.setitem(sys.modules, 'av', SimpleNamespace(open=fail))
    link = GroundLink()
    link.control = FakeControl([(b'ok', ('192.168.10.1', 8889)), (b'ok', ('192.168.10.1', 8889))])
    with pytest.raises(RuntimeError, match='TT camera decode failed'):
        capture_video(link, tmp_path / 'ground.jpg')
    assert link.control.sent == ['streamon', 'streamoff']


def test_missing_telemetry_cannot_pass():
    class NoState:
        def setblocking(self, value):
            self.blocking = value
        def settimeout(self, value):
            self.blocking = True
        def recvfrom(self, size):
            if not self.blocking:
                raise BlockingIOError()
            raise socket.timeout()
    link = GroundLink()
    link.telemetry = NoState()
    with pytest.raises(RuntimeError, match='sustained valid'):
        link.sample_state(.01)


def test_video_probe_saves_decoded_jpeg_without_motor_commands(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    import numpy as np
    class Container:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def decode(self, **kwargs):
            return [SimpleNamespace(to_ndarray=lambda **kwargs: np.zeros((48, 64, 3), dtype=np.uint8))] * 3
    monkeypatch.setitem(sys.modules, 'av', SimpleNamespace(open=lambda *a, **kw: Container()))
    link = GroundLink()
    link.control = FakeControl([(b'ok', ('192.168.10.1', 8889)), (b'ok', ('192.168.10.1', 8889))])
    target = tmp_path / 'camera.jpg'
    result = capture_video(link, target)
    assert result['decoded_frames'] == 3 and result['width'] == 64
    assert target.read_bytes().startswith(b'\xff\xd8')
    assert link.control.sent == ['streamon', 'streamoff']
