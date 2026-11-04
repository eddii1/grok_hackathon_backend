"""Bounded, labelled ToF bench runs. No flight tools or model calls."""
import asyncio
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field


class RangeTestRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    device_id: str = Field(min_length=1, max_length=80)
    object_label: str = Field(min_length=1, max_length=160)
    reference_distance_mm: int | None = Field(default=None, gt=0, le=3000)
    sample_count: int = Field(default=10, ge=3, le=30)


class GroundReconnectRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    grounded: bool = Field(strict=True)


def summarize(samples, reference=None):
    values = [s['distance_mm'] for s in samples if s['status'] == 'valid' and s['distance_mm'] is not None]
    counts = Counter(s['status'] for s in samples)
    return {'sample_count': len(samples), 'valid_count': len(values), 'status_counts': dict(counts),
            'median_mm': statistics.median(values) if values else None,
            'min_mm': min(values) if values else None, 'max_mm': max(values) if values else None,
            'stddev_mm': round(statistics.pstdev(values), 2) if values else None,
            'mean_absolute_error_mm': round(statistics.mean(abs(v-reference) for v in values), 2) if values and reference else None,
            'ready_for_approach': False}


class RangeTests:
    def __init__(self, registry, media_dir: Path):
        self.registry = registry
        self.directory = media_dir/'range-tests'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jobs, self.tasks = {}, {}

    @property
    def active(self):
        return any(not task.done() for task in self.tasks.values())

    def _save(self, job):
        (self.directory/f"{job['id']}.json").write_text(json.dumps(job, indent=2), encoding='utf-8')

    def create(self, request):
        if self.active:
            raise HTTPException(409, 'A ToF test is already active')
        try:
            device = self.registry.get(request.device_id)
        except KeyError:
            raise HTTPException(404, 'Unknown device')
        state = device.state()
        if getattr(device, 'simulated', True) or device.flight_enabled or device._armed or state['flying'] or state['height_cm'] > 10:
            raise HTTPException(409, 'ToF tests require a real, unarmed, grounded monitor')
        if not state['connected']:
            raise HTTPException(503, 'TT telemetry is stale. Reconnect grounded TT first.')
        if not request.object_label.strip():
            raise HTTPException(422, 'Label the object/surface being tested')
        while len(self.jobs) >= 50:
            oldest = next(iter(self.jobs)); self.jobs.pop(oldest); self.tasks.pop(oldest, None)
        test_id = uuid4().hex
        job = {'id': test_id, 'status': 'sampling', **request.model_dump(), 'created_at': time.time(),
               'samples': [], 'summary': summarize([]), 'firmware': None, 'error': None}
        self.jobs[test_id] = job
        self._save(job)
        self.tasks[test_id] = asyncio.create_task(self._run(device, job))
        return dict(job)

    async def _run(self, device, job):
        try:
            try:
                job['firmware'] = await device.read_expansion_version()
            except (TimeoutError, OSError):
                pass
            for _ in range(job['sample_count']):
                state = device.state()
                if not state['connected'] or state['flying'] or state['height_cm'] > 10:
                    raise RuntimeError('Ground telemetry became unavailable/unsafe; reconnect and repeat')
                sample = await device.read_forward_range()
                job['samples'].append(sample)
                job['summary'] = summarize(job['samples'], job['reference_distance_mm'])
                self._save(job)
                await asyncio.sleep(.2)
            job['status'] = 'completed'
        except asyncio.CancelledError:
            job['status'] = 'cancelled'
        except Exception:
            job['status'], job['error'] = 'failed', 'Ground ToF test interrupted. Check telemetry, module and connection.'
        finally:
            job['finished_at'] = time.time()
            self._save(job)

    def get(self, test_id):
        if test_id not in self.jobs:
            raise HTTPException(404, 'ToF test not found in this session')
        return dict(self.jobs[test_id])

    async def cancel(self, test_id):
        self.get(test_id)
        task = self.tasks[test_id]
        if not task.done():
            task.cancel(); await asyncio.gather(task, return_exceptions=True)
            if self.jobs[test_id]['status'] == 'sampling':
                self.jobs[test_id]['status'] = 'cancelled'; self._save(self.jobs[test_id])
        return self.get(test_id)

    async def cancel_all(self):
        for test_id in tuple(self.jobs):
            await self.cancel(test_id)
