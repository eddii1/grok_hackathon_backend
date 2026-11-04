"""Single-owner SDK socket, independent telemetry/video readers. No command retries."""
import asyncio
import ipaddress
import socket
import threading
import time

from .tt_ground import parse_state

ALLOWED = frozenset({'command', 'streamon', 'streamoff', 'takeoff', 'stop', 'land', 'rc 0 0 0 0', 'emergency', 'EXT tof?'})


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

    def send(self, command):
        if command not in ALLOWED:
            raise ValueError('SDK command is outside the stationary-hover allowlist')
        with self._send_lock:
            self.control.sendto(command.encode('ascii'), (self.host, 8889))
            self.wire_events.append({'monotonic': time.monotonic(), 'direction': 'send', 'command': command})
            del self.wire_events[:-2000]

    async def request(self, command, timeout, dispatch=None):
        async with self._request_lock:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            with self._pending_lock:
                self._pending = (loop, future)
            try:
                if dispatch is None:
                    self.send(command)
                else:
                    dispatch(lambda: self.send(command))
                return await asyncio.wait_for(future, timeout)
            finally:
                with self._pending_lock:
                    if self._pending == (loop, future):
                        self._pending = None

    def _responses(self):
        while not self._quit.is_set():
            try:
                data, peer = self.control.recvfrom(2048)
                if peer != (self.host, 8889):
                    continue
                self.wire_events.append({'monotonic': time.monotonic(), 'direction': 'receive',
                                         'response': data.decode('ascii', errors='replace').strip()})
                del self.wire_events[:-2000]
                with self._pending_lock:
                    pending = self._pending
                if pending:
                    loop, future = pending
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
        if self.control is not None:
            try:
                self.send('streamoff')
            except OSError:
                pass
        for channel in (self.control, self.telemetry):
            if channel is not None:
                channel.close()
        # Readers have bounded socket/decoder waits and are daemon threads.
