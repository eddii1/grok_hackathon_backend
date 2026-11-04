import asyncio
import copy
import json
import time

import numpy as np
import pytest

from fleet.exploration import ExperimentalMapping,FAST_MODEL,NavigationDecision,clearance,permits_forward
from fleet.pilot_adapter import PilotLimits
from fleet.tello import SafetyError
from test_pilot import plant,send


class Planner:
    def __init__(self,actions=('scan','advance','advance','scan'),heading=0):
        self.actions=iter(actions);self.models=[];self.heading=heading

    async def vision_multi(self,images,prompt,shape,**kwargs):
        self.models.append(kwargs['model'])
        assert len(images)==2 and all(image[:2]==b'\xff\xd8' for image in images)
        action='finish' if 'Ground preparation only' in prompt else next(self.actions,'finish')
        return {'action':action,'heading':self.heading,'risk':.1,'confidence':.95,'reason':'Synthetic test planner'}


def test_depth_gate_rejects_obstacles_uncertainty_staleness_and_flat_images():
    frame=np.random.default_rng(3).integers(0,255,(108,192,3),dtype=np.uint8)
    distant=clearance(frame,np.full((108,192),3.))
    distant['captured_mono']=10
    assert permits_forward(distant,now=10.1)
    assert not permits_forward(distant,now=11.1)
    close=clearance(frame,np.full((108,192),.8));close['captured_mono']=10
    assert not permits_forward(close,now=10.1)
    invalid=clearance(frame,np.full((108,192),np.nan));invalid['captured_mono']=10
    assert not permits_forward(invalid,now=10.1)
    flat=clearance(np.zeros_like(frame),np.full((108,192),3.));flat['captured_mono']=10
    assert not permits_forward(flat,now=10.1)


def test_twenty_percent_profile_and_strict_navigation_contract():
    from fleet.pilot_adapter import PilotAdapter
    from test_tello import FakeTransport
    assert PilotLimits().min_takeoff_battery==20 and PilotLimits().land_battery==15
    with pytest.raises(ValueError):PilotLimits(min_takeoff_battery=19)
    with pytest.raises(ValueError):PilotLimits(land_battery=14)
    with pytest.raises(ValueError):PilotLimits(max_flight_seconds=121)
    with pytest.raises(ValueError):NavigationDecision(action='raw_rc',heading=0,risk=0.,confidence=1.,reason='bad')
    with pytest.raises(ValueError):NavigationDecision(action='advance',heading=45,risk=0.,confidence=1.,reason='bad')
    d=PilotAdapter('fake',transport=FakeTransport())
    for values in ([0,-8,0,0],[0,20,0,0],[0,8,0,20],[0,0,10,0]):
        with pytest.raises(SafetyError,match='only slow forward'):d.exploration_sticks(values)


def test_complete_two_station_mission_uses_exact_pilot_and_saves_reconstruction(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            planner=Planner(('scan','advance','advance','advance','scan'),heading=90);s.provider=planner
            send(s,d,t,'prepare_map');await asyncio.wait_for(s.tasks[d.id],5)
            mapping=s.explorations[d.id]
            assert mapping.ready() and not d._possible
            assert not any(e.get('command')=='takeoff' for e in d.transport.wire_events)
            send(s,d,t,'arm');send(s,d,t,'map')
            await asyncio.wait_for(s.tasks[d.id],65)
            assert not d._possible and not s.errors[d.id],s.errors
            assert mapping.public['stations']==2
            assert .08<mapping.public['travel_estimate_m']<.4
            assert all(m==FAST_MODEL for m in planner.models)
            assert mapping.mapper.frames_used>30 and len(mapping.mapper.points())>500
            assert len(mapping.public['samples'])>20
            folder=tmp_path/'explorations'/mapping.id
            assert (folder/'map.ply').is_file() and (folder/'report.json').is_file()
            assert len(list(folder.glob('*.jpg')))>20
            report=json.loads((tmp_path/f"pilot-{s.records[d.id]['id']}.json").read_text())
            assert report['mapping']['stations']==2 and report['result']=='mapping-completed'
            assert d.transport.room.position[0]>8
            assert any('reassess' in decision.get('result','') for decision in mapping.public['decisions'])
    asyncio.run(scenario())


def test_mapping_needs_preparation_and_local_invalid_depth_never_translates(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            send(s,d,t,'arm')
            with pytest.raises(SafetyError):send(s,d,t,'map')
            assert not d._possible
            mapping=ExperimentalMapping(d,Planner(),tmp_path)
            mapping.prepared_at=time.monotonic();mapping.public['status']='prepared'
            s.explorations[d.id]=mapping
            send(s,d,t,'takeoff');await s.tasks[d.id]
            mapping.sample={'usable':False,'clearance_m':None,'threshold_m':1.2,'captured_mono':time.monotonic()+.1}
            assert not await mapping.advance()
            assert not any(e.get('command')=='rc 0 8 0 0' for e in d.transport.wire_events)
    asyncio.run(scenario())


def test_manual_takeover_cancels_mapping_then_sticks_and_land_still_work(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            s.provider=Planner();send(s,d,t,'prepare_map');await s.tasks[d.id]
            send(s,d,t,'arm');send(s,d,t,'map')
            for _ in range(70):
                if d._takeoff_acked:break
                await asyncio.sleep(.05)
            send(s,d,t,'manual');await s.tasks[d.id]
            assert s.status(d.id)['manual_ready'] and d._possible and not d.fault
            send(s,d,t,'sticks',sequence=1,axes=[0,0,10,0]);await asyncio.sleep(.15)
            assert d._sticks==(0,0,10,0)
            s.stop(d.id);await d.wait_for_touchdown(timeout=4)
            assert d._started and not d._possible
    asyncio.run(scenario())


def test_stop_preempts_a_hung_grok_decision_and_never_resumes(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            planner=Planner();s.provider=planner
            send(s,d,t,'prepare_map');await s.tasks[d.id]
            entered=asyncio.Event()
            async def hung(*args,**kwargs):
                entered.set();await asyncio.Event().wait()
            planner.vision_multi=hung
            send(s,d,t,'arm');send(s,d,t,'map')
            await asyncio.wait_for(entered.wait(),4)
            before=time.monotonic();s.stop(d.id)
            assert any(e.get('command')=='land' and e['monotonic']>=before for e in d.transport.wire_events)
            await asyncio.wait_for(s.tasks[d.id],3)
            await d.wait_for_touchdown(timeout=4)
            assert d._sticks==(0,0,0,0) and not d._possible
    asyncio.run(scenario())


def test_high_risk_prevents_forward_command_and_lands(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            planner=Planner();s.provider=planner
            send(s,d,t,'prepare_map');await s.tasks[d.id]
            original=planner.vision_multi
            async def risky(*args,**kwargs):
                result=await original(*args,**kwargs);result.update(action='advance',risk=.9);return result
            planner.vision_multi=risky
            send(s,d,t,'arm');send(s,d,t,'map');await asyncio.wait_for(s.tasks[d.id],6)
            assert not d._possible and not any(e.get('command')=='rc 0 8 0 0' for e in d.transport.wire_events)
    asyncio.run(scenario())


def test_dropped_perception_raises_before_motion(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            send(s,d,t,'arm');send(s,d,t,'takeoff');await s.tasks[d.id]
            mapping=ExperimentalMapping(d,Planner(),tmp_path)
            async def failed():raise RuntimeError('Depth worker failed')
            mapping._worker=asyncio.create_task(failed());await asyncio.sleep(0)
            with pytest.raises(RuntimeError):await mapping.advance()
            assert not any(e.get('command')=='rc 0 8 0 0' for e in d.transport.wire_events)
    asyncio.run(scenario())


def test_twenty_percent_really_passes_preflight_but_nineteen_does_not():
    from fleet.pilot_adapter import PilotAdapter
    from test_tello import Clock,FakeTransport
    async def scenario():
        clock=Clock();transport=FakeTransport();transport.wire_events=[]
        d=PilotAdapter('tt',transport=transport,clock=clock)
        await d.start();d._quit.set();d._watcher.join(.2)
        d.tick();d.operator_heartbeat();d.enable()
        d._state['bat']=19
        assert 'Takeoff requires at least 20% battery' in d.preflight_errors()
        d._state['bat']=20
        assert not d.preflight_errors()
        await d.takeoff()
        d._state['bat']=15;d.watchdog_step()
        assert 'battery' in d.fault.lower() and d._sticks==(0,0,0,0)
        d.close()
    asyncio.run(scenario())


def test_grok_timeout_lands_with_no_forward_motion(tmp_path):
    async def scenario():
        async with plant(tmp_path,.3) as (d,s,t):
            planner=Planner();s.provider=planner
            send(s,d,t,'prepare_map');await s.tasks[d.id]
            s.explorations[d.id].MODEL_TIMEOUT=.05
            async def slow(*args,**kwargs):await asyncio.sleep(2)
            planner.vision_multi=slow
            send(s,d,t,'arm');send(s,d,t,'map');await asyncio.wait_for(s.tasks[d.id],6)
            assert not d._possible and s.errors[d.id]
            assert not any(e.get('command')=='rc 0 8 0 0' for e in d.transport.wire_events)
    asyncio.run(scenario())


def test_live_map_export_is_bounded_and_archive_contains_evidence_only(tmp_path):
    import io
    import zipfile
    from fastapi.testclient import TestClient
    from fleet.app import create_app
    config=tmp_path/'devices.json'
    config.write_text(json.dumps({'devices':[{'id':'pilot','adapter':'pilot_sim'}]}))
    with TestClient(create_app(config_path=config,media_dir=tmp_path,agent_mode='stub')) as client:
        service=client.app.state.pilot
        mapping=ExperimentalMapping(service.device('pilot'),Planner(),tmp_path/'explorations')
        service.explorations['pilot']=mapping
        folder=mapping.directory/mapping.id;folder.mkdir(parents=True)
        (folder/'000.jpg').write_bytes(b'fake-jpeg-fixture')
        (folder/'.env').write_text('must never be exported')
        mapping.public['status']='completed'
        assert client.get('/pilot/map/pilot').json()['pose_source']=='simulator'
        assert client.get('/pilot/map/pilot/export').content.splitlines()[0]==b'ply'
        zipped=client.get('/pilot/map/pilot/export?format=zip')
        with zipfile.ZipFile(io.BytesIO(zipped.content)) as bundle:
            assert set(bundle.namelist())=={'000.jpg','report.json','map.ply'}
        mapping.public['status']='running'
        assert client.get('/pilot/map/pilot/export').status_code==409
        assert client.get('/pilot/map/not-a-device/export').status_code==404
