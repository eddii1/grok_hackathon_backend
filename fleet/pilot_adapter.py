"""Dashboard-owned manual RC and one automatic yaw scan, with priority landing.

No keyboard repeat or airborne cloud dependency. The profile keeps a finite
flight budget, an independent watchdog and a short lease on stick commands.
"""
from dataclasses import dataclass
import re

from .tello import SafetyError
from .tello_transport import UdpTelloTransport
from .yaw_scan import SupervisedYawAdapter


@dataclass(frozen=True)
class PilotLimits:
    min_takeoff_battery: int = 50
    land_battery: int = 25
    max_height_cm: int = 200
    max_flight_seconds: float = 60
    telemetry_timeout: float = 2
    frame_timeout: float = 3
    operator_timeout: float = 3
    loop_timeout: float = 1
    takeoff_timeout: float = 15
    landing_timeout: float = 20

    def __post_init__(self):
        caps = dict(min_takeoff_battery=100, land_battery=50, max_height_cm=200,
                    max_flight_seconds=60, telemetry_timeout=2, frame_timeout=3,
                    operator_timeout=3, loop_timeout=1, takeoff_timeout=15, landing_timeout=20)
        if self.min_takeoff_battery < 50 or self.land_battery < 25 or self.land_battery >= self.min_takeoff_battery:
            raise ValueError('Invalid battery limits')
        if any(not 0 < getattr(self,k) <= cap for k,cap in caps.items()):
            raise ValueError('Pilot limits can only be tightened')


class PilotTransport(UdpTelloTransport):
    def validate_command(self, command):
        if command == 'cw 30': return
        if re.fullmatch(r'rc -?\d+ -?\d+ -?\d+ -?\d+', command):
            if all(-20 <= int(v) <= 20 for v in command.split()[1:]): return
            raise ValueError('Manual sticks are limited to SDK units -20..20')
        super().validate_command(command)


class PilotAdapter(SupervisedYawAdapter):
    pilot_capable = True

    def __init__(self, device_id, host='192.168.10.1', bind='0.0.0.0', *, transport=None, clock=None, limits=None):
        kwargs = {'transport':transport or PilotTransport(host,bind), 'limits':limits or PilotLimits(), 'flight_enabled':False}
        if clock is not None: kwargs['clock']=clock
        super().__init__(device_id,host,bind,**kwargs)
        self._landing_authorized = False
        self._sticks = (0,0,0,0)
        self._sticks_at = float('-inf')
        self._stick_sequence = -1
        self._flight_ended_at = None

    def state(self):
        state = super().state()
        state.update(simulated=self.simulated,pilot_capable=True,armed=self._armed,
                     ready_for_control=bool(state['connected'] and state['video_ready'] and self._state_samples>=3 and self._frame_samples>=3),
                     control_enabled=self.flight_enabled and self._armed and not self.fault,
                     sticks=list(self._sticks), flight_limit_s=self.limits.max_flight_seconds,
                     elapsed_flight_s=round((self._flight_ended_at if self._flight_ended_at is not None else self.clock())-self._flight_at,1) if self._spent else 0,
                     landing_unconfirmed=bool(self._possible and self._landing_at is not None
                                              and self.clock()-self._landing_at > self.limits.landing_timeout))
        return state

    def update_state(self,state):
        was_possible=self._possible
        super().update_state(state)
        with self._lock:
            if was_possible and not self._possible:
                self._armed=False
                self.flight_enabled=False
                self._sticks=(0,0,0,0)
                self._flight_ended_at=self.clock()
                self._record('controls_disarmed_after_touchdown')

    def enable(self):
        with self._lock:
            if self._possible or self._state.get('h',100)>10:
                raise SafetyError('Aircraft may be airborne; use Land / recover first')
            if not self.state()['ready_for_control']:
                raise SafetyError('Fresh telemetry and camera are required before arming')
            self.operator_heartbeat()
            self.flight_enabled=True
            self._armed=False
            self._spent=self._yaw_spent=self._motor_cut=False
            self._takeoff_acked=False
            self._landing_at=None
            self._ground_at=None
            self._ground_samples=0
            self._seen_airborne=False
            self._landing_authorized=False
            self._stick_sequence=-1
            self._flight_ended_at=None
            self._sticks=(0,0,0,0)
            self.fault=''
            self.arm_for_supervised_trial()

    def _send(self, command):
        # Explicit Land/recover works even if this process did not arm/take off.
        if self._landing_authorized and not self._motor_cut and command in {'rc 0 0 0 0','stop','land'}:
            try:
                self.transport.send(command);self._record('send',command=command)
            except OSError as exc: self._record('send_failed',command=command,error=type(exc).__name__)
            return
        super()._send(command)

    def trip(self, reason='STOP requested'):
        with self._lock:
            self._sticks=(0,0,0,0)
            if self._armed: self._landing_authorized=True
            super().trip(reason)
            if self._landing_authorized and hasattr(self.transport,'interrupt_pending'):
                self.transport.interrupt_pending()

    def request_land(self, reason='Dashboard Land / recover'):
        with self._lock:
            if self._motor_cut:
                # Preserve the touchdown-observation timestamp after a cut.
                # No later Land/cleanup action may restart or obscure that path.
                self._record('land_suppressed_after_motor_cut')
                return
            self._landing_authorized=True
            self._sticks=(0,0,0,0)
            # Each explicit operator request renews the landing retry window.
            self._landing_at=None
            self.trip(reason)
            if not self._possible:
                self._send('rc 0 0 0 0');self._send('land')
            self._record('operator_land_requested',reason=reason)

    def cut_motors(self):
        with self._lock:
            if not self._started: raise SafetyError('No open TT control link')
            self._sticks=(0,0,0,0)
            self._motor_cut=True
            self.fault='Operator motor cut; verify aircraft and stopped motors'
            self._landing_at=self.clock()
            self._record('operator_motor_cut')
            try:
                self.transport.send('emergency')
                if hasattr(self.transport,'interrupt_pending'): self.transport.interrupt_pending()
            except OSError:
                self._motor_cut=False
                self.request_land('Motor-cut send failed; landing requested')
                raise SafetyError('Motor-cut transmission failed; inspect aircraft') from None

    def set_sticks(self, sequence, values):
        with self._lock:
            if sequence <= self._stick_sequence: return False
            if len(values)!=4 or any(type(v) is not int or not -20<=v<=20 for v in values):
                raise SafetyError('Stick values must be four integers in -20..20')
            if self.fault or self._landing_at is not None or not self._armed or not self._takeoff_acked or self._yaw_active:
                raise SafetyError('Manual sticks need an armed flying session outside automatic scan/landing')
            stale=self._freshness_fault()
            if stale:
                self.trip(stale);raise SafetyError(stale)
            lr,fb,ud,yaw=values
            h=self._state.get('h',0)
            if h>=min(180,self.limits.max_height_cm-10): ud=min(ud,0)
            if h<=25: ud=max(ud,0)
            self._stick_sequence=sequence
            self._sticks=(lr,fb,ud,yaw)
            self._sticks_at=self.clock()
            self._send('rc '+' '.join(map(str,self._sticks)))
            self._last_keepalive=self.clock()
            return True

    def neutral(self):
        with self._lock:
            self._sticks=(0,0,0,0)
            if self._armed and self._possible and not self.fault: self._send('rc 0 0 0 0')

    def healthy_keepalive(self):
        if self._yaw_active: return None
        return 'rc '+' '.join(map(str,self._sticks))

    def watchdog_step(self):
        with self._lock:
            if self._sticks!=(0,0,0,0) and self.clock()-self._sticks_at>.3:
                self._sticks=(0,0,0,0)
                if not self.fault: self._send('rc 0 0 0 0')
            super().watchdog_step()

    async def stop_all(self):
        if self._armed or self._landing_authorized: self.request_land('Global STOP')
        else: self.trip('Global STOP')
        await self.wait_for_touchdown()
