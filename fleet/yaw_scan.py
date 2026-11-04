"""Terminal-only, single 30-degree yaw trial. No translation or generic rotation.

The browser registry continues constructing the base TelloAdapter and its stricter
transport. This class retains all watchdog trips, plus a separate yaw deadline.
"""
import asyncio

from .tello import SafetyError, TelloAdapter
from .tello_transport import UdpTelloTransport

TURN_DEGREES = 30
TURN_SECONDS = 4.0
TURN_ACK_SECONDS = 2.5
LANDING_RESERVE = 8.0


class YawScanTransport(UdpTelloTransport):
    def validate_command(self, command):
        if command != 'cw 30':
            super().validate_command(command)


class SupervisedYawAdapter(TelloAdapter):
    def __init__(self, device_id, host='192.168.10.1', bind='0.0.0.0', **kwargs):
        kwargs['transport'] = kwargs.get('transport') or YawScanTransport(host, bind)
        super().__init__(device_id, host, bind, **kwargs)
        self._yaw_active = self._yaw_spent = False
        self._yaw_at, self._yaw_start = 0., 0.

    def healthy_keepalive(self):
        # Zero-RC could interfere with an SDK yaw manoeuvre. The independent
        # watchdog still sends zero RC / stop / land immediately on every fault.
        return None if self._yaw_active else super().healthy_keepalive()

    def yaw_progress(self):
        return (self._state.get('yaw', 0)-self._yaw_start+180)%360-180

    def watchdog_step(self):
        with self._lock:
            if self._yaw_active:
                progress = self.yaw_progress()
                if progress < -8 or progress > 45:
                    self.trip('Yaw direction/angle trip threshold exceeded')
                if self.clock()-self._yaw_at >= TURN_SECONDS:
                    self.trip('Yaw deadline exceeded')
            super().watchdog_step()

    def _yaw_errors(self):
        errors = []
        if not self.flight_enabled or not self._armed or not self._possible or not self._takeoff_acked:
            errors.append('Yaw needs an armed, acknowledged supervised flight')
        if self.fault or self._landing_at is not None or self._motor_cut:
            errors.append('STOP/fault/landing forbids yaw')
        if self._yaw_spent:
            errors.append('Only one yaw attempt per session; no retry')
        if self._state.get('bat', 0) <= self.limits.land_battery:
            errors.append('Low battery')
        if not 30 <= self._state.get('h', 0) <= min(150, self.limits.max_height_cm) or abs(self._state.get('vgz', 100)) > 5:
            errors.append('Yaw requires low-vertical-speed hover in the 30-150 cm band')
        if self.clock()-self._flight_at+TURN_SECONDS+LANDING_RESERVE > self.limits.max_flight_seconds:
            errors.append('Insufficient flight time remaining for yaw and landing reserve')
        stale = self._freshness_fault()
        if stale: errors.append(stale)
        return errors

    async def rotate(self, cw_deg: int):
        if type(cw_deg) is not int or cw_deg != TURN_DEGREES:
            raise SafetyError('This terminal trial permits exactly one cw 30; all other rotations are disabled')
        async with self._commands:
            dispatched = False
            def dispatch(send):
                nonlocal dispatched
                with self._lock:
                    errors = self._yaw_errors()
                    if errors: raise SafetyError('; '.join(errors))
                    self._yaw_spent = self._yaw_active = True
                    self._yaw_at, self._yaw_start = self.clock(), self._state['yaw']
                    dispatched = True
                    self._record('yaw_requested', degrees=TURN_DEGREES, start_yaw=self._yaw_start)
                    send()
            try:
                reply = await self.transport.request('cw 30', TURN_ACK_SECONDS, dispatch)
                with self._lock:
                    if reply != 'ok' or self.fault:
                        raise SafetyError('Yaw ACK failed/interrupted; rotation outcome is uncertain')
                # An ACK alone cannot prove the turn. Require new telemetry with
                # an approximately 30-degree clockwise change, stable for 0.3 s.
                stable_at, samples, previous, last_sample = None, 0, None, self._state_samples
                while True:
                    with self._lock:
                        if self.fault: raise SafetyError(self.fault)
                        if self.clock()-self._yaw_at >= TURN_SECONDS:
                            raise SafetyError('No stable yaw telemetry before rotation deadline')
                        if self._state_samples != last_sample:
                            last_sample = self._state_samples
                            progress = self.yaw_progress()
                            stable = abs(progress-TURN_DEGREES) <= 8 and (previous is None or abs(progress-previous) <= 3)
                            if stable:
                                stable_at = self.clock() if stable_at is None else stable_at
                                samples += 1
                            else:
                                stable_at, samples = None, 0
                            previous = progress
                            if samples >= 3 and self.clock()-stable_at >= .3:
                                self._record('yaw_telemetry_observed', degrees=progress)
                                break
                    await asyncio.sleep(.02)
            except BaseException:
                if dispatched: self.trip('Yaw interrupted, rejected or acknowledgement lost')
                raise
            finally:
                with self._lock:
                    self._yaw_active = False
                    if dispatched and not self.fault:
                        self._send('rc 0 0 0 0')
