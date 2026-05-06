# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Smoke test: cross-process NCCL broadcast trainer → SGLang without Megatron.

Replaces the heavy Megatron ``Policy`` with a single-GPU Ray actor
(``MockTrainer``) that mimics only the trainer-side half of
``refit_sglang_distributed``:

  1. Initialize a 1-rank ``gloo`` default torch process group, so that the
     cross-process ``init_process_group`` (the helper we're testing) runs
     *after* a default PG is already up — exactly the condition that holds
     during a real Megatron refit.
  2. Call ``connect_rollout_engines_from_distributed`` to bring up the
     trainer ↔ engine NCCL group.
  3. Run ``broadcast_hf_buckets_via_distributed_impl`` for one tiny
     bucket containing one fake tensor.

A pass here isolates the trainer ↔ engine NCCL transport from any
Megatron-specific machinery (AutoBridge export, refit-buffer sizing,
mcore TP/PP collectives, etc.). A failure points at either the in-tree
``init_process_group`` helper or the cross-process NCCL channel
establishment (P2P/IPC vs SHM transport).
"""

from __future__ import annotations

import gc
import os
import socket

import pytest
import ray
import torch
from _megatron_helpers import SGLANG_TP1, make_sglang_cfg
from _nemotron_slicer import ensure_sliced_model

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration

pytestmark = pytest.mark.sglang


# ---------------------------------------------------------------------------
# Sliced-model fixture (mirror of test_megatron_sglang_weight_update.py)
# ---------------------------------------------------------------------------
_NEMOTRON_TEST_MODEL_PATH_ENV = "NEMOTRON_TEST_MODEL_PATH"


@pytest.fixture(scope="session")
def sliced_model_path() -> str:
    override = os.environ.get(_NEMOTRON_TEST_MODEL_PATH_ENV)
    if override:
        return override
    return str(ensure_sliced_model())


# ---------------------------------------------------------------------------
# MockTrainer — one Ray actor on one GPU, mirrors the trainer-side ops
# ---------------------------------------------------------------------------
@ray.remote(num_gpus=1)
class MockTrainer:
    """Single-GPU mock of a Megatron rank-0 trainer.

    Holds the ``model_update_group`` returned by ``connect_rollout_engines_
    from_distributed`` in the actor's process so it does not need to cross
    Ray's serialization boundary.
    """

    def __init__(self) -> None:
        self._model_update_group = None

    def setup_default_pg(self) -> dict:
        """Initialize a 1-rank gloo default PG (mimics Megatron's startup)."""
        import torch.distributed as dist

        if not dist.is_initialized():
            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(port))
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{port}",
                world_size=1,
                rank=0,
            )
        torch.cuda.set_device(0)
        return {
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "device": str(torch.cuda.current_device()),
            "device_name": torch.cuda.get_device_name(0),
        }

    def connect_to_engine(
        self,
        rollout_engines: list,
        engine_gpu_counts: list,
        group_name: str,
    ) -> None:
        """Bring up the trainer ↔ engine NCCL group via the production helper."""
        from nemo_rl.models.policy.utils import (
            connect_rollout_engines_from_distributed,
        )

        self._model_update_group = connect_rollout_engines_from_distributed(
            group_name=group_name,
            rollout_engines=rollout_engines,
            engine_gpu_counts=engine_gpu_counts,
        )

    def broadcast_one_bucket(
        self,
        rollout_engines: list,
        rollout_engine_lock,
        group_name: str,
        weight_version: int,
        param_name: str,
        shape: tuple,
        dtype_str: str,
    ) -> None:
        """Drive the production broadcast helper for a single fake tensor."""
        from nemo_rl.models.policy.utils import (
            broadcast_hf_buckets_via_distributed_impl,
        )

        target_dtype = getattr(torch, dtype_str)
        tensor = torch.empty(shape, dtype=target_dtype, device="cuda:0")
        tensor.fill_(1.0)

        bucket_iter = iter([[(param_name, tensor)]])
        broadcast_hf_buckets_via_distributed_impl(
            bucket_iterator=bucket_iter,
            rollout_engines=rollout_engines,
            rollout_engine_lock=rollout_engine_lock,
            group_name=group_name,
            model_update_group=self._model_update_group,
            weight_version=weight_version,
        )

    def shutdown(self) -> None:
        import torch.distributed as dist

        if self._model_update_group is not None:
            try:
                dist.destroy_process_group(self._model_update_group)
            except Exception:
                pass
            self._model_update_group = None
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fixture: real SGLangGeneration on its own RayVirtualCluster
# ---------------------------------------------------------------------------
@pytest.fixture
def smoke_env(ray_cluster, sliced_model_path):
    """Build a single-engine ``SGLangGeneration`` (sgl_tp1) on its own cluster."""
    sglang_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,
        name="smoke-sglang",
    )
    sglang_cfg = make_sglang_cfg(
        model_path=sliced_model_path,
        sglang=SGLANG_TP1,
        colocated=False,
    )
    sglang_gen = SGLangGeneration(sglang_cluster, sglang_cfg)
    yield sglang_gen, sglang_cluster
    try:
        sglang_gen.shutdown()
    except Exception:
        pass
    try:
        sglang_cluster.shutdown()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------
def test_disag_broadcast_smoke(smoke_env):
    """Single-bucket NCCL broadcast: MockTrainer → real sglang engine.

    Sequencing:
        1. MockTrainer.setup_default_pg     — 1-rank gloo default PG
        2. fetch_updatable_engines_with_recover (engine + lock + gpu counts)
        3. MockTrainer.connect_to_engine    — cross-process NCCL group
        4. MockTrainer.broadcast_one_bucket — single-tensor bucket via prod helper

    The fake tensor uses an unrecognized parameter name (``_smoke_test_param``)
    so SGLang's ``model.load_weights`` will return ``(False, "...")`` *after*
    the broadcast itself completes. We assert only that the broadcast succeeds
    (no NCCL transport error) — verifying the load-weights body is the real
    test's job.
    """
    from nemo_rl.models.policy.utils import fetch_updatable_engines_with_recover

    sglang_gen, _sglang_cluster = smoke_env

    # 1. Pull engine + lock + per-engine GPU count from the same path the
    # production refit uses, so any future changes to the engine-discovery
    # API stay in sync.
    (
        rollout_engines,
        rollout_engine_lock,
        _num_new,
        engine_gpu_counts,
        _engine_gpu_offsets,
    ) = fetch_updatable_engines_with_recover(sglang_gen)
    rollout_engines = [e for e in rollout_engines if e is not None]
    assert len(rollout_engines) == 1, (
        f"smoke test expects one engine, got {len(rollout_engines)}"
    )
    assert rollout_engine_lock is not None, "rollout_engine_lock not set"
    print(f"[smoke] engines={len(rollout_engines)} gpu_counts={engine_gpu_counts}")

    # 2. Spawn MockTrainer on a separate GPU. Ray's pool has the GPUs the
    # sglang engine didn't claim.
    mock_trainer = MockTrainer.remote()
    try:
        info = ray.get(mock_trainer.setup_default_pg.remote())
        print(f"[smoke] MockTrainer default PG up: {info}")

        # 3. Stand up the trainer ↔ engine NCCL group.
        ray.get(
            mock_trainer.connect_to_engine.remote(
                rollout_engines=rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
                group_name="smoke-group",
            )
        )
        print("[smoke] connect_rollout_engines_from_distributed succeeded")

        # 4. Drive the production broadcast helper for one tiny bucket.
        # Failure here is the same NCCL ``invalid argument`` we'd see in the
        # full Megatron→SGLang refit if the cross-process NCCL transport is
        # broken (e.g. P2P/IPC channel mapping).
        ray.get(
            mock_trainer.broadcast_one_bucket.remote(
                rollout_engines=rollout_engines,
                rollout_engine_lock=rollout_engine_lock,
                group_name="smoke-group",
                weight_version=1,
                param_name="_smoke_test_param.weight",
                shape=(8,),
                dtype_str="bfloat16",
            )
        )
        print("[smoke] broadcast_hf_buckets_via_distributed_impl succeeded")
    finally:
        try:
            ray.get(mock_trainer.shutdown.remote())
        except Exception:
            pass
        try:
            ray.kill(mock_trainer)
        except Exception:
            pass
