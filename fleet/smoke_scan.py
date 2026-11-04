"""Opt-in live Grok scan of an isolated room simulator. Never opens hardware."""
import asyncio
import json
import os
from pathlib import Path

from dotenv import dotenv_values

from .grok import XaiProvider
from .registry import DeviceRegistry
from .scan_simulator import ScanSimAdapter
from .scans import ScanRequest, ScanService


async def run():
    root = Path(__file__).resolve().parent.parent
    settings = {**dotenv_values(root/'.env'), **os.environ}
    if not settings.get('XAI_API_KEY'): raise SystemExit('Configure XAI_API_KEY locally first')
    provider = XaiProvider(settings['XAI_API_KEY'], settings.get('XAI_MODEL') or 'grok-4.7')
    device = ScanSimAdapter('isolated-scan-sim')
    scans = ScanService(DeviceRegistry([device]), provider, root/'runtime', provider.model)
    try:
        job = scans.create(ScanRequest(device_id=device.id))
        seen = 0
        while scans.active:
            scans.heartbeat(job['id'])
            current = scans.get(job['id'])
            if len(current['decisions']) > seen:
                for record in current['decisions'][seen:]:
                    print(json.dumps({'turn':record['turn'], **record['decision'], 'result':record['result']}), flush=True)
                seen = len(current['decisions'])
            await asyncio.sleep(.5)
        result = scans.get(job['id'])
        print(json.dumps({key:result[key] for key in ('id','status','landed','moves','headings_seen','stop_reason','error')}
                         | {'map_counts':result['map']['counts'],'voxels':len(result['map']['points']),
                            'seconds':round(result['finished_at']-result['created_at'],2)}), flush=True)
        if result['status'] != 'completed' or not result['landed']:
            raise SystemExit('Live scan did not complete; inspect the saved runtime report')
    finally:
        await scans.stop_all()
        await provider.close()
        device.close()


if __name__ == '__main__':
    asyncio.run(run())
