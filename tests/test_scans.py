import asyncio
import json

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from fleet.app import create_app
from fleet.registry import DeviceRegistry
from fleet.scan_simulator import ScanSimAdapter
from fleet.scans import ScanRequest, ScanService, navigable, routes
from fleet.simulator import SimAdapter


def answer(action, heading=0, cell=''):
    return dict(action=action, heading=heading, cell=cell, reason='Inspect an unobserved view', observation='Synthetic room surfaces')


class Decisions:
    def __init__(self, sequence=None, hold=False):
        self.sequence = iter(sequence or [answer('look',90),answer('look',180),answer('look',270),
                                          answer('goto',cell='I8'),answer('look',90),answer('look',180),
                                          answer('look',270),answer('finish')])
        self.received, self.hold = [], hold
        self.called = asyncio.Event()
    async def vision_multi(self, images, prompt, shape):
        self.received.append((images,prompt))
        self.called.set()
        if self.hold: await asyncio.Event().wait()
        return next(self.sequence)
    async def close(self): pass
    async def vision(self, *args): return {}


def test_unknown_and_adjacent_obstacles_block_the_entire_drone_footprint():
    grid = np.ones((7,7), int)
    grid[3,3] = 2
    grid[1,1] = 0
    safe = navigable(grid)
    assert not safe[2:5,2:5].any()
    assert not safe[:3,:3].any()
    assert not safe[0].any()
    path = routes(grid,(1,5))
    assert (5,1) in path
    assert all(safe[r,c] for c,r in path[(5,1)][1:])


def test_map_aware_scan_moves_only_on_observed_grid_then_lands(tmp_path):
    async def scenario():
        device, provider = ScanSimAdapter('scan'), Decisions()
        scans = ScanService(DeviceRegistry([device]), provider, tmp_path, 'scripted-test')
        job = scans.create(ScanRequest(device_id='scan'))
        await scans.tasks[job['id']]
        result = scans.get(job['id'])
        assert result['status'] == 'completed' and result['landed']
        assert result['moves'] == 1 and len(result['headings_seen']) == 4
        assert result['map']['counts']['obstacle'] > 0 and len(result['map']['points']) > 1000
        assert result['pose']['z'] == 0 and result['depth_source'] == 'synthetic-raycast'
        assert [c['command'] for c in device.commands][0] == 'takeoff'
        assert [c['command'] for c in device.commands][-2:] == ['hover','land']
        assert len(provider.received) == 8 and all(len(images)==2 for images,_ in provider.received)
        assert 'allowed_goto' in provider.received[-1][1] and '?' in provider.received[0][1]
        saved = json.loads((tmp_path/'scans'/f"{job['id']}.json").read_text())
        assert saved['map']['points'] and saved['pose']['z'] == 0
        assert (tmp_path/'scans'/f"{job['id']}.ply").exists()
        assert result['limits']['real_flight_enabled'] is False
        device.position = [150, 0]
        assert scans.get(job['id'])['pose'] == result['pose']  # later device use cannot rewrite old evidence
        device.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('interruption',['stop','heartbeat','timeout','provider_failure'])
def test_scan_cancellation_and_faults_land_without_replaying_commands(tmp_path, interruption):
    async def scenario():
        device, provider = ScanSimAdapter('scan'), Decisions(hold=True)
        scans = ScanService(DeviceRegistry([device]),provider,tmp_path,'test',
                            lease_seconds=.01 if interruption=='heartbeat' else 5,
                            mission_seconds=.05 if interruption=='timeout' else 180)
        if interruption == 'provider_failure':
            async def broken(*args): raise RuntimeError('secret-that-must-not-leak')
            provider.vision_multi = broken
        job=scans.create(ScanRequest(device_id='scan'))
        if interruption=='stop':
            await provider.called.wait()
            await scans.stop(job['id'])
        await asyncio.wait_for(scans.tasks[job['id']], 2)
        result=scans.get(job['id'])
        assert result['landed'] and not device.flying
        assert result['status'] in {'cancelled','failed'}
        assert 'secret-that' not in json.dumps(result)
        count=len(device.commands)
        scans.heartbeat(job['id']);await scans.stop(job['id'])
        assert len(device.commands)==count
        assert not scans.active
    asyncio.run(scenario())


def test_prestopped_scan_sends_no_takeoff(tmp_path):
    async def scenario():
        device=ScanSimAdapter('scan')
        scans=ScanService(DeviceRegistry([device]),Decisions(),tmp_path,'test')
        job=scans.create(ScanRequest(device_id='scan'))
        stopped=await scans.stop(job['id'])
        assert stopped['status']=='cancelled' and stopped['landed']
        assert device.commands == []
    asyncio.run(scenario())


def test_unsupported_or_real_devices_are_rejected_before_any_action(tmp_path):
    class RealMonitor:
        id='real'
        simulated=False
        def state(self): return {'connected': True, 'flying': False}
    async def scenario():
        scans=ScanService(DeviceRegistry([RealMonitor(),SimAdapter('flat-video')]),Decisions(),tmp_path,'test')
        for name in ('real','flat-video'):
            with pytest.raises(HTTPException) as exc: scans.create(ScanRequest(device_id=name))
            assert exc.value.status_code == 409
        assert scans.jobs == {}
    asyncio.run(scenario())


def test_unsafe_model_waypoints_and_early_finish_exhaust_budget_without_moving(tmp_path):
    async def scenario():
        device=ScanSimAdapter('scan')
        decisions=[answer('goto',cell='A1'),answer('finish'),answer('goto',cell='Q17')]
        scans=ScanService(DeviceRegistry([device]),Decisions(decisions),tmp_path,'test')
        job=scans.create(ScanRequest(device_id='scan',max_decisions=3))
        await scans.tasks[job['id']]
        result=scans.get(job['id'])
        assert result['status']=='budget_exhausted' and result['landed']
        assert all(r['result'].startswith('Refused') for r in result['decisions'])
        assert not any(c['command']=='move' for c in device.commands)
    asyncio.run(scenario())


def test_simulator_collision_envelope_rejects_a_wall_even_if_a_map_were_wrong():
    async def scenario():
        device=ScanSimAdapter('scan');await device.takeoff()
        device.position=[0,260]
        with pytest.raises(ValueError, match='collision'): await device.move(50,0,0)
        assert device.position == [0,260]
    asyncio.run(scenario())


def test_raw_motor_actions_from_model_fail_closed_and_land(tmp_path):
    async def scenario():
        device = ScanSimAdapter('scan')
        scans = ScanService(DeviceRegistry([device]),Decisions([answer('takeoff')]),tmp_path,'test')
        job = scans.create(ScanRequest(device_id='scan'))
        await scans.tasks[job['id']]
        result = scans.get(job['id'])
        assert result['status'] == 'failed' and result['landed']
        assert [c['command'] for c in device.commands].count('takeoff') == 1
        assert not any(c['command']=='move' for c in device.commands)
    asyncio.run(scenario())


def test_scan_api_serializes_work_and_global_stop_lands(tmp_path):
    config=tmp_path/'devices.json';config.write_text(json.dumps({'devices':[{'id':'scan','adapter':'scan_sim'}]}))
    app=create_app(config,tmp_path/'media',agent_mode='grok',provider=Decisions(hold=True))
    with TestClient(app) as client:
        created=client.post('/scans',json={'device_id':'scan','goal':'Scan the environment'})
        assert created.status_code==202
        scan_id=created.json()['id']
        assert client.post('/scans',json={'device_id':'scan'}).status_code==409
        assert client.post('/captures',json={'device_id':'scan','description':'Room'}).status_code==409
        assert client.post('/scans/'+scan_id+'/heartbeat').status_code==200
        assert client.get('/scans/'+scan_id+'/map.jpg').headers['content-type']=='image/jpeg'
        with client.websocket_connect('/ws') as ws:
            ws.send_json({'type':'stop_all'})
            for _ in range(5):
                if ws.receive_json()['type']=='chat': break
        job=client.get('/scans/'+scan_id).json()
        assert job['status']=='cancelled' and job['landed']
        assert client.get('/scans/missing').status_code==404
