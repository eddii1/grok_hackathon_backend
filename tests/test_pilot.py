import asyncio
import json
import time
from contextlib import asynccontextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fleet.app import create_app
from fleet.pilot import PilotMessage,PilotService
from fleet.pilot_adapter import PilotAdapter,PilotLimits,PilotTransport
from fleet.pilot_sim import PilotSimAdapter,PilotSimTransport
from fleet.registry import DeviceRegistry
from fleet.tello import SafetyError
from test_tello import Clock,FakeTransport,commands


@asynccontextmanager
async def plant(tmp_path,delay=1):
    d=PilotSimAdapter('pilot',transport=PilotSimTransport(takeoff_seconds=delay))
    registry=DeviceRegistry([d]);await registry.start();await d.wait_ready()
    s=PilotService(registry,tmp_path);token=s.claim(d.id)
    async def pump():
        while True:
            d.tick();s.command(d.id,token,PilotMessage(action='heartbeat'))
            await asyncio.sleep(.05)
    loop=asyncio.create_task(pump())
    try:yield d,s,token
    finally:
        await s.close();loop.cancel();await asyncio.gather(loop,return_exceptions=True)
        if d._possible:
            try:await d.wait_for_touchdown(timeout=15)
            except SafetyError:pass
        d.close();await asyncio.to_thread(d.transport.wait_closed)


def send(s,d,token,action,**kwargs):s.command(d.id,token,PilotMessage(action=action,**kwargs))


def test_eight_second_takeoff_can_scan_in_dashboard_profile_without_space(tmp_path):
    async def scenario():
        async with plant(tmp_path,8.1) as (d,s,t):
            send(s,d,t,'arm');send(s,d,t,'scan')
            await asyncio.wait_for(s.tasks[d.id],18)
            assert not d._possible and s.records[d.id]['result']=='scan-completed'
            names=[e.get('command') for e in d.transport.wire_events if e['direction']=='send']
            assert names.count('takeoff')==1 and names.count('cw 30')==1
            report=json.loads((tmp_path/f"pilot-{s.records[d.id]['id']}.json").read_text())
            assert report['source']=='simulation' and report['frames'] and not report['state']['airborne_possible']
    asyncio.run(scenario())


def test_land_retries_outlast_busy_takeoff_and_link_remains_open(tmp_path):
    async def scenario():
        async with plant(tmp_path,8.1) as (d,s,t):
            send(s,d,t,'arm');send(s,d,t,'takeoff');await asyncio.sleep(.3)
            s.stop(d.id,'Operator Land during takeoff')
            await asyncio.gather(s.tasks[d.id],return_exceptions=True)
            await d.wait_for_touchdown(timeout=14)
            assert not d._possible and d._started
            wire=d.transport.wire_events
            takeoff=next(e['monotonic'] for e in wire if e.get('command')=='takeoff')
            attempts=[e for e in wire if e.get('command')=='land']
            assert len(attempts)>=8 and any(e['monotonic']-takeoff>8 for e in attempts)
            assert not any(e.get('command')=='streamoff' for e in wire)
            assert not any(e.get('command')=='cw 30' for e in wire)
    asyncio.run(scenario())


def test_browser_disconnect_requests_land_and_cannot_resume(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            send(s,d,t,'arm');send(s,d,t,'takeoff');await s.tasks[d.id]
            s.release(d.id,t)
            with pytest.raises(SafetyError):send(s,d,t,'takeoff')
            await d.wait_for_touchdown(timeout=4)
            assert d._started and not d._possible
    asyncio.run(scenario())


def test_sticks_expire_and_delayed_old_packet_cannot_replace_release():
    async def scenario():
        clock=Clock();transport=FakeTransport();transport.wire_events=[]
        d=PilotAdapter('tt',transport=transport,clock=clock)
        await d.start();d._quit.set();d._watcher.join(.2)
        d.tick();d.operator_heartbeat();d.enable();await d.takeoff()
        assert d.set_sticks(1,[20,10,0,0])
        assert d.set_sticks(3,[0,0,0,0])
        assert not d.set_sticks(2,[20,20,20,20]) and d._sticks==(0,0,0,0)
        d.set_sticks(4,[10,0,0,0]);clock.now+=.31;d.watchdog_step()
        assert d._sticks==(0,0,0,0) and commands(transport)[-1]=='rc 0 0 0 0'
        d.close()
    asyncio.run(scenario())


def test_expired_dashboard_lease_and_unconfirmed_landing_keep_connection_open():
    async def scenario():
        clock=Clock();transport=FakeTransport();transport.wire_events=[]
        d=PilotAdapter('tt',transport=transport,clock=clock)
        await d.start();d._quit.set();d._watcher.join(.2)
        d.tick();d.operator_heartbeat();d.enable();await d.takeoff()
        clock.now+=3.1;d.update_state({'bat':80,'h':60,'yaw':0,'vgz':0});d.update_frame(np.zeros((48,64,3),np.uint8));d.tick()
        d.watchdog_step();assert d.fault=='Stale operator' and 'land' in commands(transport)
        clock.now+=21;d.watchdog_step()
        assert d.state()['landing_unconfirmed'] and d._started and not transport.closed
        count=commands(transport).count('land');d.request_land('Operator retry')
        assert commands(transport).count('land')==count+1
        d.close()
    asyncio.run(scenario())


def test_recovery_land_is_allowed_without_takeoff_arming_but_motion_is_not():
    async def scenario():
        transport=FakeTransport();d=PilotAdapter('tt',transport=transport)
        await d.start();d.update_state({'bat':80,'h':70,'yaw':4,'vgz':0})
        assert commands(transport)==[]
        with pytest.raises(SafetyError):d.set_sticks(1,[0,0,0,20])
        d.request_land('Explicit recovery')
        assert commands(transport)[-3:]==['rc 0 0 0 0','stop','land']
        assert not d._armed and d._possible
        d.cut_motors();assert commands(transport)[-1]=='emergency'
        d.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('payload',[{'action':'sticks','axes':[21,0,0,0]}, {'action':'sticks','axes':[0,0,0,1.5]},
 {'action':'sticks','axes':[True,0,0,0]}, {'action':'takeoff','extra':'raw SDK'}, {'action':'cw 360'}, {'action':'sticks','sequence':-1}])
def test_invalid_control_messages_are_rejected(payload):
    with pytest.raises(ValidationError):PilotMessage.model_validate(payload)


def test_manual_transport_does_not_expose_raw_translation_or_high_rc():
    transport=PilotTransport()
    for valid in ('rc -20 20 0 0','cw 30','land','emergency'):transport.validate_command(valid)
    for invalid in ('rc 21 0 0 0','rc 0 0 0 100','cw 360','forward 20','rc 0 0 0 0\nland'):
        with pytest.raises(ValueError):transport.validate_command(invalid)
    with pytest.raises(ValueError):PilotLimits(max_flight_seconds=61)


def test_one_dashboard_owner_and_emergency_confirmation(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            with pytest.raises(SafetyError):s.claim(d.id)
            with pytest.raises(SafetyError):send(s,d,t,'motor_cut',confirm='')
            send(s,d,t,'arm');send(s,d,t,'takeoff');await s.tasks[d.id]
            send(s,d,t,'motor_cut',confirm='CUT MOTORS')
            assert d._motor_cut
            count=len(d.transport.wire_events)
            send(s,d,t,'sticks',sequence=1,axes=[0,0,0,20])
            assert not any(e.get('command')=='rc 0 0 0 20' for e in d.transport.wire_events[count:])
            s.stop(d.id,'Cleanup after explicit motor cut')
            await d.wait_for_touchdown(timeout=3)
            assert not d._possible
    asyncio.run(scenario())


def test_pilot_api_websocket_manual_flight_logs_and_priority_land(tmp_path):
    config=tmp_path/'devices.json';config.write_text(json.dumps({'devices':[{'id':'pilot','adapter':'pilot_sim'}]}))
    app=create_app(config,tmp_path/'media',agent_mode='stub')
    with TestClient(app) as client:
        assert client.get('/pilot/devices').json()[0]['simulated'] is True
        with client.websocket_connect('/pilot/ws/pilot') as ws:
            for _ in range(15):
                message=ws.receive_json()
                if message.get('device',{}).get('ready_for_control'):break
            assert message['device']['ready_for_control']
            ws.send_json({'action':'arm'});ws.send_json({'action':'takeoff'})
            for _ in range(35):
                ws.send_json({'action':'heartbeat'})
                message=ws.receive_json()
                if message.get('manual_ready'):break
            assert message['manual_ready']
            assert client.post('/captures',json={'device_id':'pilot','description':'room'}).status_code==409
            ws.send_json({'action':'sticks','sequence':1,'axes':[0,10,0,0]})
            for _ in range(10):
                ws.send_json({'action':'heartbeat'});message=ws.receive_json()
                if message.get('device',{}).get('sticks')==[0,10,0,0]:break
            assert message['device']['sticks']==[0,10,0,0]
            response=client.post('/pilot/land/pilot',json={'confirm':'LAND'})
            assert response.status_code==200 and response.json()['requested']
            for _ in range(25):
                ws.send_json({'action':'heartbeat'});message=ws.receive_json()
                if not message.get('device',{}).get('airborne_possible',True):break
            assert message['device']['link_open'] and not message['device']['airborne_possible']
        async def persist():await app.state.pilot.persist('pilot')
        client.portal.call(persist)
        saved=client.get('/pilot/reports').json()[0]
        assert saved['source']=='simulation'
        assert client.get('/pilot/reports/'+saved['name']).status_code==200
        assert client.get('/pilot/reports/.env').status_code==404
        assert client.post('/pilot/land/pilot',json={'confirm':'LAND'},headers={'origin':'https://evil.example'}).status_code==403


def test_runtime_viewer_reads_existing_failures_without_opening_hardware(tmp_path):
    report={'state':{'fault':'Stale operator','airborne_possible':True},'events':[],'wire_events':[]}
    name='tt-scan-'+'a'*32+'.json';(tmp_path/name).write_text(json.dumps(report))
    s=PilotService(DeviceRegistry([]),tmp_path)
    assert s.reports()[0]['fault']=='Stale operator'
    assert s.report(name)==report


def test_completed_scan_report_and_delayed_model_result_do_not_adopt_next_flight_state(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            class Model:
                def __init__(self):self.called=asyncio.Event();self.release=asyncio.Event()
                async def vision_multi(self,images,prompt,shape):
                    assert not d._possible and not d._armed and len(images)>=2
                    self.called.set();await self.release.wait()
                    return {'summary':'Test scan views','evidence':['Synthetic room'],'limitations':['Simulation only']}
            model=Model();s.provider=model;s.model='fake-model'
            send(s,d,t,'arm');send(s,d,t,'scan',assess=True)
            await s.tasks[d.id];await model.called.wait()
            old=s.records[d.id]['id'];path=tmp_path/f'pilot-{old}.json'
            saved=json.loads(path.read_text());assert saved['result']=='scan-completed'
            assert saved['frames'] and not saved['state']['airborne_possible']
            send(s,d,t,'arm');await s.persist(d.id)
            assert json.loads(path.read_text())['state']==saved['state']
            send(s,d,t,'takeoff');await s.tasks[d.id]
            assert d._possible and s.records[d.id]['id']!=old
            model_task=s.assessments[old];model.release.set();await model_task
            completed=json.loads(path.read_text())
            assert completed['state']==saved['state'] and completed['assessment']['summary']=='Test scan views'
            s.stop(d.id);await d.wait_for_touchdown(timeout=4)
    asyncio.run(scenario())
