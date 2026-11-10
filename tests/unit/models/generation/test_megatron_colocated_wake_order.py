# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wake ordering of the Megatron inference engine across ranks.

Rank 0's coordinator commands (resume, unpause) reach every rank's engine, and
each engine runs on its own event-loop thread. A rank's local wake work (the
inference weight reshard in ``prepare_for_generation``, the weight transfer
before ``resume_after_refit``) runs on that rank's main thread. No engine may
resume before every rank has finished that work.

Two ranks run the real ``MegatronGenerationMixin`` methods on their own main and
engine-loop threads against a fake coordinator that broadcasts to every engine,
with a threading barrier standing in for ``torch.distributed.barrier`` and one
rank's local work delayed.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import nemo_rl.models.generation.megatron.megatron_worker as mw
from nemo_rl.models.generation.megatron.megatron_worker import (
    MegatronGenerationMixin,
    MegatronGenerationRefitMixin,
)

pytestmark = pytest.mark.mcore

WORLD = 2


class _Cluster:
    def __init__(self):
        self.weights_ready = {r: False for r in range(WORLD)}
        self.thread_rank: dict[int, int] = {}
        self.engines: dict[int, "_Engine"] = {}
        self.barrier = threading.Barrier(WORLD)
        self.violations: list[str] = []
        self.lock = threading.Lock()

    def rank(self) -> int:
        return self.thread_rank[threading.get_ident()]


class _Engine:
    """Engine state and events as MCore's coordinator signals set them."""

    _CLEARS = {
        "RESUMED": ("SUSPENDED",),
        "RUNNING": ("PAUSED",),
        "PAUSED": ("RUNNING",),
        "SUSPENDED": ("RESUMED",),
    }

    def __init__(self, cluster: _Cluster, rank: int, loop: asyncio.AbstractEventLoop):
        self.cluster, self.rank, self.loop = cluster, rank, loop
        self.state = mw.EngineState.SUSPENDED
        self.events: dict = {}

    def _event(self, state):
        return self.events.setdefault(state, asyncio.Event())

    def _set(self, state):
        def apply():
            self.state = state
            for name in self._CLEARS[state.name]:
                self._event(getattr(mw.EngineState, name)).clear()
            self._event(state).set()

        self.loop.call_soon_threadsafe(apply)

    def _signal(self, what, state):
        if not self.cluster.weights_ready[self.rank]:
            with self.cluster.lock:
                self.cluster.violations.append(
                    f"rank {self.rank}: engine {what} before its inference weights were ready"
                )
        self._set(state)

    async def wait_until(self, state):
        await self._event(state).wait()


class _Model:
    config = SimpleNamespace(flash_decode=True)

    def eval(self):
        return self


class _Worker(MegatronGenerationMixin, MegatronGenerationRefitMixin):
    def __init__(self, cluster: _Cluster, rank: int, slow_rank: int):
        self.cluster, self.rank, self.slow_rank = cluster, rank, slow_rank
        self.cfg = {
            "generation": {"mcore_generation_config": {"cuda_graph_impl": "none"}}
        }
        self.is_generation_colocated = True
        self._colocated_reshard_plan = None
        self._inference_engine_initialized = True
        self._inference_engine_asleep = True
        self.inference_model = _Model()
        self._inference_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=self._inference_loop.run_forever, daemon=True)
        thread.start()
        cluster.thread_rank[thread.ident] = rank
        self.dynamic_inference_engine = _Engine(cluster, rank, self._inference_loop)
        cluster.engines[rank] = self.dynamic_inference_engine
        engines = cluster.engines.values
        self.inference_client = SimpleNamespace(
            resume_engines=lambda: [
                e._signal("resumed", mw.EngineState.RESUMED) for e in engines()
            ],
            unpause_engines=lambda: [
                e._signal("unpaused", mw.EngineState.RUNNING) for e in engines()
            ],
        )

    def _inference_model_and_media_parts(self):
        return self.inference_model, None

    def local_weight_update(self):
        if self.rank == self.slow_rank:
            time.sleep(0.3)
        self.cluster.weights_ready[self.rank] = True

    _reshard_into_inference_model = local_weight_update


@pytest.fixture
def cluster(monkeypatch):
    c = _Cluster()
    monkeypatch.setattr(mw, "log_gpu_memory", lambda *a, **k: None)
    monkeypatch.setattr(
        mw.torch.distributed, "barrier", lambda *a, **k: c.barrier.wait(timeout=20)
    )
    monkeypatch.setattr(mw.torch.distributed, "get_rank", lambda *a, **k: c.rank())
    return c


def _wake_after_refit(worker):
    worker.local_weight_update()
    worker.resume_after_refit()


@pytest.mark.parametrize("slow_rank", [0, 1])
@pytest.mark.parametrize(
    "wake",
    [lambda w: w.prepare_for_generation(), _wake_after_refit],
    ids=["prepare_for_generation", "resume_after_refit"],
)
def test_no_engine_resumes_before_every_rank_updated_its_weights(
    cluster, wake, slow_rank
):
    workers = [_Worker(cluster, r, slow_rank) for r in range(WORLD)]
    errors = []

    def main(worker):
        cluster.thread_rank[threading.get_ident()] = worker.rank
        try:
            wake(worker)
        except BaseException as e:  # noqa: BLE001 - surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=main, args=(w,)) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a rank hung"
    assert not errors, errors
    assert cluster.violations == []
    assert all(e.state == mw.EngineState.RUNNING for e in cluster.engines.values())
    assert not any(w._inference_engine_asleep for w in workers)
