import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import replace

import numpy as np
import pytest

from fleet.tello import SafetyError, SafetyLimits
from fleet.tello_transport import UdpTelloTransport
from fleet.yaw_scan import SupervisedYawAdapter, YawScanTransport


class Transport:
    def __init__(self, mode='ok', yaw=170):
        self.sent, self.mode, self.yaw, self.height = [], mode, yaw, 0
        self.dispatched = asyncio.Event()
    async def start(self, on_state, on_frame):
        self.on_state, self.on_frame = on_state, on_frame
        for _ in range(3): self.feed()
        async def stream():
            while True:
                self.feed()
                await asyncio.sleep(.04)
        self.stream = asyncio.create_task(stream())
    def feed(self):
        self.on_state({'bat':80,'h':self.height,'yaw':self.yaw,'vgz':0})
        self.on_frame(np.zeros((48,64,3),np.uint8))
    def send(self, command):
        self.sent.append(command)
        if command=='cw 30' and self.mode=='send-failure': raise OSError('UDP send failed')
        if command=='land': self.height=0
    async def request(self, command, timeout, dispatch=None):
        if dispatch: dispatch(lambda:self.send(command))
        else: self.send(command)
        if command=='takeoff': self.height=60
        if command=='cw 30':
            self.dispatched.set()
            if self.mode=='timeout': raise TimeoutError('Lost yaw ACK')
            if self.mode=='error-reply': return 'error'
            if self.mode=='hold': await asyncio.Event().wait()
            if self.mode!='no-motion': self.yaw=(self.yaw+30+180)%360-180
        return 'ok'
    def close(self):
        self.closed=True
        self.stream.cancel()


@asynccontextmanager
async def flying(mode='ok', limits=None):
    transport=Transport(mode)
    device=SupervisedYawAdapter('tt-1',flight_enabled=True,transport=transport,limits=limits)
    await device.start()
    async def heartbeats():
        while True:
            device.tick();device.operator_heartbeat()
            await asyncio.sleep(.02)
    pump=asyncio.create_task(heartbeats())
    device.tick();device.operator_heartbeat();device.arm_for_supervised_trial()
    await device.takeoff();await asyncio.sleep(.06)
    try: yield device,transport
    finally:
        device.close();pump.cancel()
        await asyncio.gather(pump,transport.stream,return_exceptions=True)


def test_one_small_rotation_checks_wrapped_yaw_and_cannot_repeat():
    async def scenario():
        async with flying() as (device,transport):
            await device.rotate(30)
            assert device.yaw_progress()==30 and not device._yaw_active
            assert any(e['event']=='yaw_telemetry_observed' for e in device.events)
            with pytest.raises(SafetyError,match='Only one'): await device.rotate(30)
            assert transport.sent.count('cw 30')==1
            await device.land()
            assert not device.state()['airborne_possible']
    asyncio.run(scenario())


@pytest.mark.parametrize('angle',[-30,0,15,45,90,360,True,30.0,'30'])
def test_scan_adapter_rejects_all_unapproved_angles(angle):
    async def scenario():
        device=SupervisedYawAdapter('tt-1')
        with pytest.raises(SafetyError): await device.rotate(angle)
    asyncio.run(scenario())


def test_rotation_transport_is_separate_from_browser_and_hover_allowlist():
    normal,scan=UdpTelloTransport(),YawScanTransport()
    with pytest.raises(ValueError): normal.validate_command('cw 30')
    scan.validate_command('cw 30')
    for text in ('cw 360','cw 30\nland','ccw 30','forward 20','rc 0 0 0 15'):
        with pytest.raises(ValueError): scan.validate_command(text)


@pytest.mark.parametrize('condition',['unarmed','disabled','landing','time-reserve','unstable','height'])
def test_yaw_dispatch_refuses_unready_or_late_flight_without_sending(condition):
    async def scenario():
        async with flying() as (device,transport):
            if condition=='unarmed': device._armed=False
            if condition=='disabled': device.flight_enabled=False
            if condition=='landing': device.trip('Stop before yaw')
            if condition=='time-reserve': device._flight_at-=9
            if condition=='unstable': device._state['vgz']=20
            if condition=='height': device._state['h']=170
            with pytest.raises(SafetyError): await device.rotate(30)
            assert 'cw 30' not in transport.sent
    asyncio.run(scenario())


@pytest.mark.parametrize('reason',['telemetry','video','operator','loop','battery','altitude','deadline',
                                   'yaw-deadline','reverse-yaw','overshoot'])
def test_all_watchdog_faults_interrupt_pending_yaw_and_land(reason):
    async def scenario():
        async with flying('hold') as (device,transport):
            task=asyncio.create_task(device.rotate(30));await transport.dispatched.wait()
            if reason=='telemetry': device._state_at=0
            if reason=='video': device._frame_at=0
            if reason=='operator': device._operator_at=0
            if reason=='loop': device._loop_at=0
            if reason=='battery': device._state['bat']=25
            if reason=='altitude': device._state['h']=201
            if reason=='deadline': device._flight_at-=20
            if reason=='yaw-deadline': device._yaw_at-=4
            if reason=='reverse-yaw': device._state['yaw']=device._yaw_start-15
            if reason=='overshoot': device._state['yaw']=device._yaw_start+46
            device.watchdog_step()
            assert device.fault and transport.sent[-3:]==['rc 0 0 0 0','stop','land']
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            with pytest.raises(SafetyError): await device.rotate(30)
            assert transport.sent.count('cw 30')==1
    asyncio.run(scenario())


def test_stop_rechecks_yaw_queued_behind_another_command():
    async def scenario():
        async with flying() as (device,transport):
            await device._commands.acquire()
            task=asyncio.create_task(device.rotate(30));await asyncio.sleep(0)
            device.trip('STOP before queued rotation');device._commands.release()
            with pytest.raises(SafetyError): await task
            assert 'cw 30' not in transport.sent
    asyncio.run(scenario())


@pytest.mark.parametrize('mode,error',[('timeout',TimeoutError),('error-reply',SafetyError),('send-failure',OSError)])
def test_failed_yaw_request_lands_once_without_retry(mode,error):
    async def scenario():
        async with flying(mode) as (device,transport):
            with pytest.raises(error): await device.rotate(30)
            assert transport.sent.count('cw 30')==1 and 'land' in transport.sent
            with pytest.raises(SafetyError): await device.rotate(30)
    asyncio.run(scenario())


def test_operator_motor_cut_during_yaw_cannot_resume_motion_or_keepalives():
    async def scenario():
        async with flying('hold') as (device,transport):
            task=asyncio.create_task(device.rotate(30));await transport.dispatched.wait()
            device.operator_motor_cut()
            count=len(transport.sent)
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            device.watchdog_step();await device.hover()
            assert transport.sent[-1]=='emergency' and len(transport.sent)==count
            with pytest.raises(SafetyError): await device.rotate(30)
    asyncio.run(scenario())


def test_sdk_ack_without_motion_does_not_count_as_a_scan():
    async def scenario():
        async with flying('no-motion') as (device,transport):
            task=asyncio.create_task(device.rotate(30));await transport.dispatched.wait()
            device._yaw_at-=4;device.watchdog_step()
            with pytest.raises(SafetyError): await task
            assert not any(e['event']=='yaw_telemetry_observed' for e in device.events)
            assert 'land' in transport.sent
    asyncio.run(scenario())


def test_healthy_keepalive_cannot_cancel_yaw_but_stop_still_sends_zero_rc():
    async def scenario():
        async with flying('hold') as (device,transport):
            task=asyncio.create_task(device.rotate(30));await transport.dispatched.wait()
            transport.sent.clear();device._last_keepalive=0;device.watchdog_step()
            assert transport.sent==[]
            device.trip('Operator Q')
            assert transport.sent[:3]==['rc 0 0 0 0','stop','land']
            task.cancel();await asyncio.gather(task,return_exceptions=True)
    asyncio.run(scenario())


def test_yaw_watchdog_survives_a_blocked_asyncio_loop():
    async def scenario():
        async with flying('hold',replace(SafetyLimits(),loop_timeout=.08)) as (device,transport):
            task=asyncio.create_task(device.rotate(30));await transport.dispatched.wait()
            time.sleep(.25)
            assert device.fault=='Stale event loop' and 'land' in transport.sent
            task.cancel();await asyncio.gather(task,return_exceptions=True)
    asyncio.run(scenario())


def test_scan_cli_plan_is_hardware_free_and_piped_flight_is_refused(monkeypatch,capsys):
    import fleet.supervised_scan as cli
    monkeypatch.setattr(cli,'run',lambda *a:pytest.fail('Must not open the drone'))
    monkeypatch.setattr('sys.argv',['supervised_scan','--plan']);cli.main()
    assert json.loads(capsys.readouterr().out)['rotation']=='one cw 30 attempt, no retry'
    monkeypatch.setattr('sys.argv',['supervised_scan','--fly'])
    monkeypatch.setattr('sys.stdin.isatty',lambda:False)
    with pytest.raises(SystemExit) as exc: cli.main()
    assert exc.value.code==2


@pytest.mark.skipif(sys.platform!='win32',reason='Windows terminal keyboard integration')
def test_full_terminal_scan_saves_exact_frames_and_assesses_only_after_connection_close(tmp_path,monkeypatch):
    import hashlib
    import msvcrt
    from types import SimpleNamespace
    import fleet.supervised_hover as hover
    import fleet.supervised_scan as cli
    transport=Transport()
    def factory(*args,**kwargs):
        return SupervisedYawAdapter(*args,**kwargs,transport=transport)
    monkeypatch.setattr('fleet.yaw_scan.SupervisedYawAdapter',factory)
    monkeypatch.setattr('builtins.input',lambda *a:'SCAN30 tt-1')
    monkeypatch.setattr(msvcrt,'kbhit',lambda:True)
    monkeypatch.setattr(msvcrt,'getwch',lambda:' ')
    monkeypatch.chdir(tmp_path)
    class Provider:
        model='fake-grok'
        def __init__(self,*args): assert transport.closed
        async def vision_multi(self,images,prompt,shape):
            assert transport.closed and len(images)>=2
            assert 'No motion tools' in prompt
            return {'summary':'Synthetic test frame sequence','evidence':['Test frame'], 'limitations':['No real flight']}
        async def close(self): pass
    monkeypatch.setattr(cli,'XaiProvider',Provider)
    monkeypatch.setattr(cli,'dotenv_values',lambda *a:{'XAI_API_KEY':'test-only'})
    async def scenario():
        path=await hover.run(SimpleNamespace(scan=True,fly=True,host='192.168.10.1',bind='0.0.0.0'))
        report=json.loads(path.read_text())
        assert report['result']=='touchdown-telemetry-observed'
        assert not report['state']['airborne_possible'] and transport.closed
        assert report['frames'][0]['label']=='before-yaw' and report['frames'][-1]['label']=='after-yaw'
        assert 2 <= len(report['frames']) <= 15
        for item in report['frames']:
            data=(path.parent/'photos'/item['photo_url'].split('/')[-1]).read_bytes()
            assert hashlib.sha256(data).hexdigest()==item['sha256'] and item['position'] is None
        await cli.assess(path)
        assert json.loads(path.read_text())['model']=='fake-grok'
    asyncio.run(scenario())


@pytest.mark.parametrize('reason',['airborne','failed'])
def test_postflight_assessment_rejects_uncertain_landing_before_model_access(tmp_path,monkeypatch,reason):
    import fleet.supervised_scan as cli
    monkeypatch.setattr(cli,'XaiProvider',lambda *a:pytest.fail('Must not access the model'))
    report=tmp_path/'report.json'
    report.write_text(json.dumps({'result':'interrupted-or-failed' if reason=='failed' else 'touchdown-telemetry-observed',
                                  'state':{'airborne_possible':reason=='airborne'}}))
    with pytest.raises(SafetyError): asyncio.run(cli.assess(report))
