"""Run the experimental physical-controller pipeline with real Grok, no sockets."""
import asyncio
import json
import os
from pathlib import Path
from dotenv import dotenv_values

from .grok import XaiProvider
from .exploration import FAST_MODEL
from .pilot import PilotService,PilotMessage
from .pilot_sim import PilotSimAdapter
from .registry import DeviceRegistry


async def main():
    env={**dotenv_values('.env'),**os.environ}
    provider=XaiProvider(env['XAI_API_KEY'],FAST_MODEL)
    d=PilotSimAdapter('pilot-smoke');registry=DeviceRegistry([d])
    await registry.start();await d.wait_ready()
    service=PilotService(registry,Path('runtime'),provider,FAST_MODEL);token=service.claim(d.id)
    async def pump():
        while True:
            d.tick();service.command(d.id,token,PilotMessage(action='heartbeat'));await asyncio.sleep(.05)
    timer=asyncio.create_task(pump())
    try:
        service.command(d.id,token,PilotMessage(action='prepare_map'));await service.tasks[d.id]
        mapping=service.explorations[d.id]
        if not mapping.ready():raise RuntimeError(service.errors[d.id])
        print('Prepared synthetic room with '+FAST_MODEL,flush=True)
        service.command(d.id,token,PilotMessage(action='arm'))
        service.command(d.id,token,PilotMessage(action='map'))
        await service.tasks[d.id]
        if d._possible:await d.wait_for_touchdown(timeout=22)
        print(json.dumps({'landed':not d._possible,'error':service.errors.get(d.id),
                          'mapping':{k:v for k,v in mapping.status().items() if k!='samples'},'frames_used':mapping.mapper.frames_used,
                          'point_count':len(mapping.mapper.points()),'directory':str(mapping.directory/mapping.id)},indent=2),flush=True)
    finally:
        await service.close()
        if d._possible:await d.wait_for_touchdown(timeout=22)
        timer.cancel();await asyncio.gather(timer,return_exceptions=True)
        d.close();await asyncio.to_thread(d.transport.wait_closed);await provider.client.close()


if __name__=='__main__':asyncio.run(main())
