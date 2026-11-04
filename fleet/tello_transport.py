"""Single-owner SDK socket, independent telemetry/video readers. No command retries."""
import asyncio
import ipaddress
import re
import socket
import threading
import time

from .tt_ground import parse_state

ALLOWED = frozenset({'command', 'streamon', 'streamoff', 'takeoff', 'stop', 'land', 'rc 0 0 0 0', 'emergency', 'EXT tof?', 'EXT version?'})


def matches_reply(command, response):
    if command == 'EXT tof?':
        return re.fullmatch(r'tof\s+\d+', response) is not None
    if command == 'EXT version?':
        return re.fullmatch(r'(?:version\s+)?esp32v[\w.\-]+', response) is not None
    return response == 'ok' or response.startswith('error') or response in {'forced stop', 'unactive'}


class UdpTelloTransport:
    def __init__(self, host='192.168.10.1', bind='0.0.0.0', *, socket_factory=socket.socket):
        self.host = str(ipaddress.IPv4Address(host))
        self.bind = bind
        self._socket_factory = socket_factory
        self.control = self.telemetry = None
        self._pending = None
        self._pending_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._request_lock = asyncio.Lock()
        self._quit = threading.Event()
        self._threads = []
        self.wire_events = []
        self._stream_requested = False

    async def start(self, on_state, on_frame):
        try:
            self.control = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            self.control.bind((self.bind, 8889))
            self.control.settimeout(.1)
            self.telemetry = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            self.telemetry.bind((self.bind, 8890))
            self.telemetry.settimeout(.1)
            self._thread(self._responses)
            self._thread(lambda: self._states(on_state))
            if await self.request('command', 3) != 'ok':
                raise RuntimeError('Could not enter SDK mode')
            if await self.request('streamon', 3) != 'ok':
                raise RuntimeError('Could not enable TT video')
            self._thread(lambda: self._video(on_frame))
        except BaseException:
            self.close()
            raise

    def _thread(self, target):
        worker = threading.Thread(target=target, daemon=True)
        self._threads.append(worker)
        worker.start()

    def validate_command(self, command):
        if command not in ALLOWED:
            raise ValueError('SDK command is outside the stationary-hover allowlist')

    def send(self, command):
        self.validate_command(command)
        with self._send_lock:
            self.control.sendto(command.encode('ascii'), (self.host, 8889))
            if command == 'streamon':
                self._stream_requested = True
            elif command == 'streamoff':
                self._stream_requested = False
            self.wire_events.append({'monotonic': time.monotonic(), 'direction': 'send', 'command': command})
            del self.wire_events[:-2000]

    async def request(self, command, timeout, dispatch=None):
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            with self._pending_lock:
                pending = (loop, future, command)
                self._pending = pending
            try:
                if dispatch is None:
                    self.send(command)
                else:
                    dispatch(lambda: self.send(command))
                return await asyncio.wait_for(future, timeout)
            finally:
                with self._pending_lock:
                    if self._pending == pending:
                        self._pending = None

    def _responses(self):
        while not self._quit.is_set():
            try:
                data, peer = self.control.recvfrom(2048)
                if peer != (self.host, 8889):
                    continue
                response = data.decode('ascii', errors='replace').strip().lower()
                with self._pending_lock:
                    pending = self._pending
                matched = bool(pending and matches_reply(pending[2], response))
                self.wire_events.append({'monotonic': time.monotonic(), 'direction': 'receive',
                                         'response': response, 'matched': matched})
                del self.wire_events[:-2000]
                if pending:
                    loop, future, command = pending
                    if not matched:
                        continue
                    def deliver(future=future, data=data):
                        if not future.done():
                            future.set_result(data.decode('ascii', errors='replace').strip().lower())
                    loop.call_soon_threadsafe(deliver)
            except socket.timeout:
                continue
            except (OSError, RuntimeError):
                return

    def _states(self, callback):
        while not self._quit.is_set():
            try:
                packet, peer = self.telemetry.recvfrom(4096)
                if peer[0] == self.host:
                    callback(parse_state(packet))
            except socket.timeout:
                continue
            except OSError:
                return

    def _video(self, callback):
        try:
            import av
            # Decode continuously, retaining only the newest frame in the adapter.
            with av.open('udp://0.0.0.0:11111?fifo_size=100000&overrun_nonfatal=1',
                         timeout=(5, 2), options={'fflags': 'nobuffer', 'flags': 'low_delay'}) as container:
                for frame in container.decode(video=0):
                    if self._quit.is_set():
                        return
                    callback(frame.to_ndarray(format='bgr24'))
        except Exception:
            # Frame freshness becomes unhealthy; the independent watchdog handles flight.
            return

    def close(self):
        self._quit.set()
        if self.control is not None and self._stream_requested:
            try:
                self.send('streamoff')
            except OSError:
                pass
        for channel in (self.control, self.telemetry):
            if channel is not None:
                channel.close()
        # Readers have bounded socket/decoder waits and are daemon threads.

    def wait_closed(self, timeout=7):
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(max(0, deadline-time.monotonic()))
        if any(thread.is_alive() for thread in self._threads):
            raise TimeoutError('Previous video/UDP readers are still closing; retry shortly')
