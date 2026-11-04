import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from fleet.agent import AgentTools
from fleet.app import create_app
from fleet.grok import GrokAgent, Reply, ToolCall, XaiSession
from fleet.registry import DeviceRegistry
from fleet.simulator import SimAdapter
from test_backend import receive_until


def call(name, /, **arguments):
    return ToolCall(name + '-id', name, json.dumps(arguments))


class ScriptedSession:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.results = []
        self.prompts = []

    def add_user(self, text):
        self.prompts.append(text)

    async def sample(self):
        reply = next(self.replies)
        if reply == 'wait':
            await asyncio.Event().wait()
        if callable(reply):
            return reply(self)
        return reply

    def add_result(self, tool_call, result):
        self.results.append((tool_call, result))


class ScriptedProvider:
    def __init__(self, replies):
        self.replies = replies
        self.sessions = []
        self.closed = False
        self.images = []

    def session(self, specs):
        assert specs
        session = ScriptedSession(self.replies)
        self.sessions.append(session)
        return session

    async def vision(self, jpeg, prompt, shape):
        assert jpeg.startswith(b'\xff\xd8')
        self.images.append(jpeg)
        if shape.__name__ == 'SceneDescription':
            return {"description": "A labelled synthetic simulator image."}
        return {"match": True, "confidence": .99, "reason": "Simulator label visible."}

    async def close(self):
        self.closed = True


def make_app(tmp_path, provider):
    config = tmp_path / 'devices.json'
    config.write_text(json.dumps({'devices': [{'id': 'drone-1', 'adapter': 'sim'}]}))
    return create_app(config, tmp_path / 'media', agent_mode='grok', provider=provider)


def test_natural_language_mission_with_vision_confirmation_and_report(tmp_path):
    provider = ScriptedProvider([
        Reply('I will check the fleet and request takeoff.', [call('list_devices')]),
        Reply('', [call('takeoff', device_id='drone-1')]),
        Reply('', [call('describe_scene', device_id='drone-1'), call('take_photo', device_id='drone-1')]),
        lambda s: Reply('', [call('verify_photo', photo_id=s.results[-1][1]['photo_id'], description='simulator')]),
        Reply('', [call('land', device_id='drone-1'), call('report', text='Captured a synthetic scene and landed.', photo_ids=[])]),
        Reply('Mission complete.', []),
    ])
    app = make_app(tmp_path, provider)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        health = client.get('/health').json()
        assert health['agent'] == 'grok' and health['agent_ready']
        ws.send_json({'type': 'user_message', 'text': 'Take off, describe the camera, take a photo, verify the simulator label, then land.'})
        request = receive_until(ws, lambda e: e['type'] == 'confirm_request')
        assert not app.state.registry.get('drone-1').flying
        ws.send_json({'type': 'confirm_response', 'request_id': request['request_id'], 'approved': True})
        receive_until(ws, lambda e: e['type'] == 'chat' and e['text'] == 'Mission complete.')
        assert not app.state.registry.get('drone-1').flying
        assert len(provider.images) == 2
        assert provider.sessions[0].prompts[0].startswith('Take off')
        assert len(provider.sessions[0].results) == 7
    assert provider.closed


def test_stop_while_model_is_waiting_and_new_mission_has_fresh_context(tmp_path):
    provider = ScriptedProvider([
        Reply('', [call('takeoff', device_id='drone-1')]), 'wait',
    ])
    app = make_app(tmp_path, provider)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({'type': 'user_message', 'text': 'Take off and inspect.'})
        request = receive_until(ws, lambda e: e['type'] == 'confirm_request')
        ws.send_json({'type': 'confirm_response', 'request_id': request['request_id'], 'approved': True})
        receive_until(ws, lambda e: e['type'] == 'tool_call' and e['name'] == 'takeoff')
        ws.send_json({'type': 'stop_all'})
        receive_until(ws, lambda e: e['type'] == 'chat' and e['text'].startswith('STOP completed'))
        assert not app.state.registry.get('drone-1').flying
        provider.replies = [Reply('Fresh session.', [])]
        ws.send_json({'type': 'user_message', 'text': 'What devices are available?'})
        receive_until(ws, lambda e: e['type'] == 'chat' and e['text'] == 'Fresh session.')
        assert len(provider.sessions) == 2


def test_rejection_cannot_be_overridden_by_repeated_model_tool_calls(tmp_path):
    provider = ScriptedProvider([
        Reply('', [call('takeoff', device_id='drone-1')]),
        Reply('', [call('takeoff', device_id='drone-1')]),
        Reply('Rejected.', []),
    ])
    app = make_app(tmp_path, provider)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({'type': 'user_message', 'text': 'Take off.'})
        request = receive_until(ws, lambda e: e['type'] == 'confirm_request')
        ws.send_json({'type': 'confirm_response', 'request_id': request['request_id'], 'approved': False})
        receive_until(ws, lambda e: e['type'] == 'chat' and e['text'] == 'Rejected.')
        assert all(not result['approved'] for _, result in provider.sessions[0].results)
        assert not app.state.registry.get('drone-1').commands


def test_tool_validation_and_round_limit(tmp_path):
    async def scenario():
        events = []
        async def emit(event):
            events.append(event)
        device = SimAdapter('drone-1')
        tools = AgentTools(DeviceRegistry([device]), emit, tmp_path)
        provider = ScriptedProvider([Reply('', [call('__getattribute__', name='registry'),
                                                     call('takeoff', device_id='drone-1', approved=True),
                                                     call('takeoff', device_id=1),
                                                     call('verify_photo', photo_id='../secret', description='x')])])
        agent = GrokAgent(tools, provider, max_rounds=1)
        with pytest.raises(RuntimeError, match='round limit'):
            await agent.run('Invalid tools')
        assert all('error' in result for _, result in provider.sessions[0].results)
        assert agent.session is None
        assert not device.commands
    asyncio.run(scenario())


def test_timeout_lands_and_resets_conversation(tmp_path):
    async def scenario():
        device = SimAdapter('drone-1')
        await device.takeoff()
        async def emit(event):
            pass
        agent = GrokAgent(AgentTools(DeviceRegistry([device]), emit, tmp_path),
                          ScriptedProvider(['wait']), mission_timeout=.01)
        with pytest.raises(TimeoutError):
            await agent.run('Inspect')
        assert not device.flying and agent.session is None
    asyncio.run(scenario())


def test_xai_transport_preserves_tool_call_ids():
    # Build the real SDK request messages without making network calls.
    class Chat:
        def __init__(self):
            self.messages = []
        def append(self, message):
            self.messages.append(message)
    transport = object.__new__(XaiSession)
    transport.chat = Chat()
    transport.add_result(ToolCall('unique-call', 'list_devices', '{}'), {'devices': []})
    assert transport.chat.messages[0].tool_call_id == 'unique-call'
    assert json.loads(transport.chat.messages[0].content[0].text) == {'devices': []}


def test_errors_do_not_expose_credentials_to_websocket(tmp_path):
    class FailingSession(ScriptedSession):
        async def sample(self):
            raise RuntimeError('secret-api-key')
    provider = ScriptedProvider([])
    provider.session = lambda specs: FailingSession([])
    app = make_app(tmp_path, provider)
    with TestClient(app) as client, client.websocket_connect('/ws') as ws:
        ws.send_json({'type': 'user_message', 'text': 'Hello'})
        event = receive_until(ws, lambda e: e['type'] == 'chat' and e['text'].startswith('Mission failed'))
        assert 'secret-api-key' not in event['text']
