import asyncio
import hashlib
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from fleet.app import create_app
from fleet.capture import CaptureService
from fleet.tello import SafetyError, TelloAdapter
from test_tello import FakeTransport


class VisionProvider:
    def __init__(self, hold=False, fail=False):
        self.received = []
        self.hold, self.fail = hold, fail
    async def vision(self, jpeg, prompt, shape):
        self.received.append((jpeg, prompt, shape))
        if self.hold:
            await asyncio.Event().wait()
        if self.fail:
            raise RuntimeError('secret-credential-must-not-leak')
        return {'match_status':'uncertain','summary':'The described object is not clearly visible.',
                'evidence':['A synthetic test frame is visible.'],'limitations':['Single view only.']}
    async def close(self): pass


def setup(tmp_path, monkeypatch, provider):
    import json
    made=[]
    def factory(device_id, host, bind):
        adapter=TelloAdapter(device_id, transport=FakeTransport())
        made.append(adapter)
        return adapter
    monkeypatch.setattr('fleet.tello.TelloAdapter', factory)
    config=tmp_path/'devices.json'
    config.write_text(json.dumps({'devices':[{'id':'tt-1','adapter':'tello'}]}))
    return create_app(config,tmp_path/'media',agent_mode='grok',provider=provider),made


def terminal(client, capture_id):
    for _ in range(40):
        job=client.get(f'/captures/{capture_id}').json()
        if job['status']!='analyzing': return job
    pytest.fail('Capture did not reach a terminal state')


def test_ground_capture_assesses_exact_served_jpeg_and_preserves_metadata(tmp_path,monkeypatch):
    provider=VisionProvider()
    app,made=setup(tmp_path,monkeypatch,provider)
    with TestClient(app) as client:
        response=client.post('/captures',json={'device_id':'tt-1','description':'A red cup on the table'})
        assert response.status_code==202
        job=terminal(client,response.json()['id'])
        jpeg=client.get(job['photo_url']).content
        assert jpeg==provider.received[0][0]
        assert hashlib.sha256(jpeg).hexdigest()==job['sha256']
        assert job['source']=='real' and job['position'] is None
        assert job['assessment']['match_status']=='uncertain'
        assert 'red cup' in provider.received[0][1]
        assert job['forward_range']['status']=='unavailable'
        assert made[0].transport.sent == []  # retired ToF is never queried by capture
        assert (tmp_path/'media'/'captures'/f"{job['id']}.json").exists()


@pytest.mark.parametrize('condition',['disconnected','airborne','height','stale_frame','armed'])
def test_unsafe_or_unavailable_capture_never_calls_grok(tmp_path,monkeypatch,condition):
    provider=VisionProvider()
    app,made=setup(tmp_path,monkeypatch,provider)
    with TestClient(app) as client:
        adapter=made[0]
        if condition=='disconnected': adapter._state_at=0
        if condition=='airborne': adapter._possible=True
        if condition=='height': adapter._state['h']=11
        if condition=='stale_frame': adapter._frame_at=0
        if condition=='armed': adapter.flight_enabled=True
        response=client.post('/captures',json={'device_id':'tt-1','description':'cup'})
        assert response.status_code in (409,503)
        assert provider.received==[]
        # Restore safe shutdown; this fixture never represents a real flight.
        adapter._possible=False


def test_cancel_retains_image_and_blocks_concurrent_captures(tmp_path,monkeypatch):
    provider=VisionProvider(hold=True)
    app,made=setup(tmp_path,monkeypatch,provider)
    with TestClient(app) as client:
        job=client.post('/captures',json={'device_id':'tt-1','description':'cup'}).json()
        assert client.post('/captures',json={'device_id':'tt-1','description':'another'}).status_code==409
        cancelled=client.post(f"/captures/{job['id']}/cancel").json()
        assert cancelled['status']=='cancelled'
        assert client.get(job['photo_url']).status_code==200
        assert 'takeoff' not in [c for c,_ in made[0].transport.sent]


def test_grok_failure_is_redacted_and_retains_captured_image(tmp_path,monkeypatch):
    app,_=setup(tmp_path,monkeypatch,VisionProvider(fail=True))
    with TestClient(app) as client:
        created=client.post('/captures',json={'device_id':'tt-1','description':'cup'}).json()
        job=terminal(client,created['id'])
        assert job['status']=='failed' and job['assessment'] is None
        assert 'secret-credential' not in str(job)
        assert client.get(job['photo_url']).status_code==200


def test_saved_capture_can_be_restored_without_resending_the_image_to_grok(tmp_path,monkeypatch):
    provider=VisionProvider()
    app,_=setup(tmp_path,monkeypatch,provider)
    with TestClient(app) as client:
        created=client.post('/captures',json={'device_id':'tt-1','description':'cup'}).json()
        job=terminal(client,created['id'])
    restarted=CaptureService(None,None,tmp_path/'media','grok')
    assert restarted.get(job['id'])['assessment']==job['assessment']
    assert len(provider.received)==1
    with pytest.raises(Exception) as exc: restarted.get('../../.env')
    assert exc.value.status_code==404


@pytest.mark.parametrize('reply,status,distance',[('tof 450','valid',450),('tof 8192','out_of_range',None),
                                                   ('tof 1500','out_of_range',None),('ok','unavailable',None),
                                                   ('tof 0','invalid',None)])
def test_expansion_range_is_ground_only_and_does_not_confuse_sentinel_with_distance(reply,status,distance):
    async def scenario():
        adapter=TelloAdapter('tt-1',transport=FakeTransport())
        await adapter.start()
        async def answer(*args,**kwargs): return reply
        adapter.transport.request=answer
        reading=await adapter.read_forward_range()
        assert reading['status']==status and reading['distance_mm']==distance
        adapter.flight_enabled=True
        with pytest.raises(SafetyError): await adapter.read_forward_range()
        adapter.close()
    asyncio.run(scenario())
