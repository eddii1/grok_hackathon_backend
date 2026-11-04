"""Restricted first-flight adapter. Independent watchdog; movement stays disabled."""
import asyncio
import threading
import time
from dataclasses import dataclass

from .tello_transport import UdpTelloTransport


class SafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class SafetyLimits:
    min_takeoff_battery: int = 50
    land_battery: int = 25
    max_height_cm: int = 200
    max_flight_seconds: float = 20
    telemetry_timeout: float = 2
    frame_timeout: float = 3
    operator_timeout: float = 2
    loop_timeout: float = 1
    takeoff_timeout: float = 15
    landing_timeout: float = 20

    def __post_init__(self):
        if not 50 <= self.min_takeoff_battery <= 100 or not 25 <= self.land_battery < self.min_takeoff_battery:
            raise ValueError('Battery limits may be tightened, not relaxed')
        for name, maximum in {'max_height_cm': 200, 'max_flight_seconds': 20, 'telemetry_timeout': 2,
                              'frame_timeout': 3, 'operator_timeout': 2, 'loop_timeout': 1,
                              'takeoff_timeout': 15, 'landing_timeout': 20}.items():
            if not 0 < getattr(self, name) <= maximum:
                raise ValueError(f'{name} must be positive and at most {maximum}')


class TelloAdapter:
    kind = 'drone'
    simulated = False

    def __init__(self, device_id, host='192.168.10.1', bind='0.0.0.0', *,
                 flight_enabled=False, transport=None, clock=time.monotonic, limits=None):
        self.id, self.flight_enabled = device_id, flight_enabled
        self.host, self.bind = host, bind
        self.transport = transport or UdpTelloTransport(host, bind)
        self.clock, self.limits = clock, limits or SafetyLimits()
        self._lock = threading.RLock()
        self._commands = asyncio.Lock()
        self._quit = threading.Event()
        self._watcher = None
        self._state = {}
        self._state_at = self._frame_at = self._operator_at = self._loop_at = float('-inf')
        self._frame = None
        self._frame_wall = 0
        self._possible = False
        self._seen_airborne = False
        self._takeoff_acked = False
        self._flight_at = 0
        self._landing_at = None
        self._ground_at = None
        self._ground_samples = 0
        self._last_safety_send = float('-inf')
        self._last_keepalive = float('-inf')
        self._started = False
        self.connecting = False
        self.last_range = {'status': 'unavailable', 'distance_mm': None, 'reason': 'not_sampled'}
        self.expansion_version = None
        self._range_at = float('-inf')
        self._armed = False
        self._motor_cut = False
        self._state_samples = self._frame_samples = 0
        self._spent = False
        self.fault = ''
        self.events = []
        self._telemetry_log_at = float('-inf')

    def _record(self, event, **data):
        self.events.append({'monotonic': self.clock(), 'event': event, **data})
        del self.events[:-2000]

    async def start(self):
        self.connecting = True
        try:
            await self.transport.start(self.update_state, self.update_frame)
            self._started = True
            self._loop_at = self.clock()
            self._watcher = threading.Thread(target=self._watch, daemon=True)
            self._watcher.start()
        finally:
            self.connecting = False

    def update_state(self, state):
        if not {'bat', 'h', 'yaw'} <= state.keys():
            return
        if not (0 <= state['bat'] <= 100 and 0 <= state['h'] <= 10000):
            return
        with self._lock:
            self._state, self._state_at = dict(state), self.clock()
            self._state_samples += 1
            if self.clock()-self._telemetry_log_at >= .2:
                self._telemetry_log_at = self.clock()
                self._record('telemetry', battery=state['bat'], height_cm=state['h'],
                             yaw_deg=state['yaw'], vertical_speed=state.get('vgz'))
            if state['h'] >= 20:
                self._seen_airborne = True
                if not self._possible:
                    # Unexpected external flight is not a grounded state.
                    self._possible, self._flight_at = True, self.clock()
                    self.trip('Unexpected airborne telemetry')
            if self._possible and self._landing_at is not None and self._seen_airborne:
                if state['h'] <= 10 and 'vgz' in state and abs(state['vgz']) <= 5:
                    self._ground_at = self.clock() if self._ground_at is None else self._ground_at
                    self._ground_samples += 1
                    if self._ground_samples >= 3 and self.clock() - self._ground_at >= .5:
                        self._possible = False
                        self._record('touchdown_telemetry')
                else:
                    self._ground_at, self._ground_samples = None, 0

    def update_frame(self, frame):
        with self._lock:
            self._frame, self._frame_at, self._frame_wall = frame.copy(), self.clock(), time.time()
            self._frame_samples += 1

    async def wait_ready(self, timeout=10):
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            self.tick()
            if self._state_samples >= 3 and self._frame_samples >= 3 and self.state()['connected'] and self.latest_frame():
                return
            await asyncio.sleep(.05)
        raise SafetyError('No sustained fresh telemetry/video; do not arm')

    def tick(self, seconds=0):
        with self._lock:
            self._loop_at = self.clock()

    def operator_heartbeat(self):
        with self._lock:
            self._operator_at = self.clock()

    def arm_for_supervised_trial(self):
        with self._lock:
            if not self.flight_enabled or not self._started or self.fault or self._spent or self._possible:
                raise SafetyError('Cannot arm this session; inspect ground state and faults')
            self._armed = True
            self._record('operator_armed')

    def _freshness_fault(self):
        now, limits = self.clock(), self.limits
        for label, timestamp, timeout in (
            ('telemetry', self._state_at, limits.telemetry_timeout),
            ('video', self._frame_at, limits.frame_timeout),
            ('operator', self._operator_at, limits.operator_timeout),
            ('event loop', self._loop_at, limits.loop_timeout),
        ):
            if now - timestamp > timeout:
                return f'Stale {label}'
        return ''

    def preflight_errors(self):
        with self._lock:
            errors = []
            if not self.flight_enabled:
                errors.append('Physical flight is disabled')
            if not self._armed:
                errors.append('Operator has not armed this session')
            if not self._started:
                errors.append('Adapter not started')
            if self._state_samples < 3 or self._frame_samples < 3:
                errors.append('Sustained telemetry/video not yet received')
            if self.fault:
                errors.append(self.fault)
            if self._spent:
                errors.append('One flight per adapter session; restart and inspect before another')
            if self._possible or self._state.get('h', 0) >= 20:
                errors.append('Aircraft may already be airborne')
            if self._state.get('bat', 0) < self.limits.min_takeoff_battery:
                errors.append(f'Takeoff requires at least {self.limits.min_takeoff_battery}% battery')
            stale = self._freshness_fault()
            if stale:
                errors.append(stale)
            return errors

    def state(self):
        with self._lock:
            connected = self._started and self.clock() - self._state_at <= self.limits.telemetry_timeout
            reading = dict(self.last_range)
            if self.clock()-self._range_at > 2 and reading.get('status') == 'valid':
                reading.update(status='stale', distance_mm=None, reason='sample_expired')
            return {'id': self.id, 'kind': self.kind,
                    'connected': connected,
                    'flying': self._possible, 'battery_pct': int(self._state.get('bat', 0)),
                    'height_cm': int(self._state.get('h', 0)), 'yaw_deg': int(self._state.get('yaw', 0)),
                    'simulated': False, 'flight_enabled': self.flight_enabled and self._armed and not self._spent and not self.fault,
                    'fault': self.fault, 'airborne_possible': self._possible,
                    'landing_requested': self._landing_at is not None,
                    'connection_state': 'connecting' if self.connecting else 'connected' if connected else 'offline',
                    'link_open': self._started,
                    'telemetry_age_s': round(self.clock()-self._state_at, 2) if self._state_samples else None,
                    'video_age_s': round(self.clock()-self._frame_at, 2) if self._frame_samples else None,
                    'video_ready': self._started and self._frame is not None and self.clock()-self._frame_at <= self.limits.frame_timeout,
                    'forward_range': reading, 'expansion_version': self.expansion_version}

    def latest_frame(self):
        with self._lock:
            if not self._started or self._frame is None or self.clock() - self._frame_at > self.limits.frame_timeout:
                return None
            return self._frame.copy(), self._frame_wall

    async def read_forward_range(self):
        # Optional ground-only query; never share the reply channel with an armed flight.
        async with self._commands:
            with self._lock:
                if self.flight_enabled or self._armed or self._possible or not self.state()['connected']:
                    raise SafetyError('Expansion range query is ground-monitor only')
            started = time.monotonic()
            result = {'status': 'unavailable', 'distance_mm': None, 'sampled_at': time.time(),
                      'raw_response': None, 'raw_value_mm': None, 'reason': 'timeout'}
            try:
                reply = await self.transport.request('EXT tof?', 1)
            except (TimeoutError, OSError):
                reply = ''
            result['sampled_at'] = time.time()
            parts = reply.split()
            if reply:
                result.update(raw_response=reply, reason='unrecognized_response')
            if len(parts) == 2 and parts[0].lower() == 'tof' and parts[1].isdigit():
                distance = int(parts[1])
                result['raw_value_mm'] = distance
                if 0 < distance <= 1200:
                    result.update(status='valid', distance_mm=distance, reason='measured')
                elif distance == 8192 or distance > 1200:
                    result.update(status='out_of_range', reason='sdk_out_of_range' if distance == 8192 else 'outside_supported_range')
                else:
                    result.update(status='invalid', reason='zero_reading_check_module_and_target')
            with self._lock:
                result['ignored_responses'] = [e['response'] for e in getattr(self.transport, 'wire_events', [])
                                               if e.get('direction') == 'receive' and not e.get('matched', True) and e['monotonic'] >= started]
                self.last_range, self._range_at = result, self.clock()
            return result

    async def read_expansion_version(self):
        async with self._commands:
            with self._lock:
                if self.flight_enabled or self._armed or self._possible or not self.state()['connected']:
                    raise SafetyError('Expansion diagnostics are ground-monitor only')
            self.expansion_version = await self.transport.request('EXT version?', 1)
            return self.expansion_version

    def _send(self, command):
        if not self.flight_enabled or not self._armed or self._motor_cut:
            self._record('motor_command_suppressed', command=command)
            return
        try:
            self.transport.send(command)
            self._record('send', command=command)
        except OSError as exc:
            self._record('send_failed', command=command, error=type(exc).__name__)
            if self._landing_at is None:
                self.trip('Command transport send failed')

    def trip(self, reason='STOP requested'):
        """Non-awaiting safety path, callable by the watchdog even with a blocked event loop."""
        with self._lock:
            if not self.fault:
                self.fault = reason
                self._record('fault', reason=reason)
            if not self._possible:
                return
            if self._motor_cut:
                return
            if self._landing_at is None:
                self._landing_at = self.clock()
                self._send('rc 0 0 0 0')
                self._send('stop')
                self._send('land')
                self._last_safety_send = self.clock()

    def watchdog_step(self):
        with self._lock:
            if not self._possible:
                return
            reason = self._freshness_fault()
            if self._state.get('bat', 0) <= self.limits.land_battery:
                reason = 'Low battery'
            if self._state.get('h', 0) > self.limits.max_height_cm:
                reason = 'Altitude trip threshold exceeded'
            if self.clock() - self._flight_at >= self.limits.max_flight_seconds:
                reason = 'Flight deadline exceeded'
            if reason:
                self.trip(reason)
            if self.fault and self._landing_at is not None:
                # A takeoff can take 8+ seconds. Retry only idempotent landing
                # throughout its observation window, not just the first 3 seconds.
                if self.clock() - self._landing_at <= self.limits.landing_timeout and self.clock() - self._last_safety_send >= 1:
                    self._send('land')
                    self._last_safety_send = self.clock()
            elif self._takeoff_acked and self._landing_at is None and self.clock() - self._last_keepalive >= .2:
                command = self.healthy_keepalive()
                if command is not None:
                    self._send(command)
                self._last_keepalive = self.clock()

    def healthy_keepalive(self):
        """Ordinary hover sends zero RC. Fault/landing commands never use this hook."""
        return 'rc 0 0 0 0'

    def _watch(self):
        while not self._quit.wait(.05):
            self.watchdog_step()

    async def takeoff(self):
        async with self._commands:
            def dispatch(send):
                # Fault latching and motion transmission share this lock: STOP cannot
                # race between the last guard and a later queued takeoff packet.
                with self._lock:
                    errors = self.preflight_errors()
                    if errors:
                        raise SafetyError('; '.join(errors))
                    self._possible, self._spent, self._flight_at = True, True, self.clock()
                    self._record('takeoff_requested')
                    send()
            try:
                reply = await self.transport.request('takeoff', self.limits.takeoff_timeout, dispatch)
                with self._lock:
                    if reply != 'ok' or self.fault:
                        raise SafetyError('Takeoff failed or was interrupted; outcome may be uncertain')
                    self._takeoff_acked = True
            except BaseException:
                if self._possible:
                    self.trip('Takeoff interrupted or acknowledgement lost')
                raise

    async def hover(self):
        with self._lock:
            if self._possible and not self.fault and self._landing_at is None:
                self._send('rc 0 0 0 0')

    async def land(self):
        with self._lock:
            if not self._possible:
                return
            self.trip('Landing requested')
        await self.wait_for_touchdown()

    async def wait_for_touchdown(self, timeout=None):
        deadline = self.clock() + (self.limits.landing_timeout if timeout is None else timeout)
        while self._possible and self.clock() < deadline:
            await asyncio.sleep(.05)
        if self._possible:
            raise SafetyError('Landing unconfirmed: aircraft may still be airborne; operator must verify')

    async def move(self, forward_cm: int, left_cm: int, up_cm: int):
        raise SafetyError('Translation is disabled for the initial hover trial')

    async def rotate(self, cw_deg: int):
        raise SafetyError('Rotation is disabled for the initial hover trial')

    def rc(self, left_right: int, fwd_back: int, up_down: int, yaw: int):
        if any((left_right, fwd_back, up_down, yaw)):
            raise SafetyError('Nonzero RC is disabled for the initial hover trial')
        with self._lock:
            if self._possible and not self.fault and self._landing_at is None:
                self._send('rc 0 0 0 0')

    async def emergency(self):
        raise SafetyError('Motor cut is not exposed in this staged adapter; normal STOP is hover then land')

    def operator_motor_cut(self):
        """Only the armed terminal controller binds this, separately from normal STOP."""
        with self._lock:
            if not self.flight_enabled or not self._armed or not self._possible:
                raise SafetyError('No enabled flight session for operator motor cut')
            self.fault, self._motor_cut = 'Operator motor cut; touchdown must be verified', True
            self._landing_at = self.clock()
            self._record('operator_motor_cut')
            try:
                self.transport.send('emergency')
            except OSError as exc:
                self._motor_cut = False
                self._landing_at = None
                self.trip('Motor cut transmission failed')
                raise SafetyError('Motor cut transmission failed; landing requested, verify aircraft') from exc

    async def stop_all(self):
        self.trip('STOP requested')
        await self.wait_for_touchdown()

    def close(self):
        self._record('closing_link', airborne_possible=self._possible,
                     height_cm=self._state.get('h'), telemetry_age_s=self.state()['telemetry_age_s'])
        self.trip('Adapter shutdown')
        self._quit.set()
        self.transport.close()
        self._started = False
