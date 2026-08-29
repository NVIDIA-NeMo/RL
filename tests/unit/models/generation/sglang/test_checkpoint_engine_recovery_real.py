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

"""4-GPU end-to-end restart-recovery test for checkpoint-engine SGLang refit.

Topology (NIXL requires ``train_world_size >= rollout_world_size``):
2 policy sender actors (the real ``PolicyCheckpointEngineMixin`` over a real
``NIXLCheckpointEngine``, streaming BF16 weights straight from the HF
checkpoint) + 2 real SGLang engines (TP=1, ``use_fault_tolerance: true``),
driven by the real ``CheckpointEngineWeightSynchronizer``.

Flow (the design's functional contract):
  1. init communicator + baseline refit (consumes the startup engine count);
  2. record engine 0's actor identity and its receivers' NIXL agent_names;
  3. crash engine 0 via ``_simulate_crash``; wait for the health monitor to
     mark its slot ``None``;
  4. recover; assert a NEW actor identity;
  5. weight oracle ON THE REPLACEMENT: snapshot -> reset_tensors (garbage the
     weights) -> ``sync_weights()`` (probe -> rebind -> transfer) ->
     compare (proves real, current bytes arrived — not zeros, not the reset
     values). The harness applies no policy update, so the senders' checkpoint
     weights are exactly what the replacement must converge back to;
  6. assert the replacement's agent_name changed while the survivor's did not,
     and that a request routed DIRECTLY to the replacement's server generates.
"""

import gc
import time

import pytest
import ray
import requests
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration
from nemo_rl.models.policy.workers.checkpoint_engine import (
    PolicyCheckpointEngineMixin,
)
from nemo_rl.weight_sync.checkpoint_engine_weight_synchronizer import (
    CheckpointEngineWeightSynchronizer,
)

from .helpers import MODEL_PATH

pytestmark = pytest.mark.sglang

PAD_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645
CHECK_INTERVAL = 5
CHECK_TIMEOUT = 30
DETECT_TIMEOUT = 300


def _make_recovery_cfg(pad_token_id):
    return {
        "backend": "sglang",
        "model_name": MODEL_PATH,
        "model_path": MODEL_PATH,
        "tokenizer": {"name": MODEL_PATH},
        "dtype": "bfloat16",
        "max_new_tokens": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": [EOS_TOKEN_ID],
        "stop_strings": None,
        "_pad_token_id": pad_token_id,
        "sglang_cfg": {
            "model_path": MODEL_PATH,
            "dtype": "bfloat16",
            "random_seed": 42,
            "context_length": 1024,
            "log_level": "info",
            "skip_server_warmup": True,
            "tp_size": 1,
            "dp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "disable_cuda_graph": True,
            "mem_fraction_static": 0.3,
            "sglang_server_config": {
                "num_gpus": 2,
                "num_gpus_per_engine": 1,
                "needs_offload": False,
                "cpu_weight_backup": False,
                "sglang_server_concurrency": 64,
                "pause_generation_mode": "retract",
            },
            "sglang_router_config": {
                "sglang_router_ip": None,
                "sglang_router_port": None,
            },
            "use_fault_tolerance": True,
            "rollout_health_check_interval": CHECK_INTERVAL,
            "rollout_health_check_timeout": CHECK_TIMEOUT,
            "rollout_health_check_first_wait": 0,
        },
        "sglang_kwargs": {},
    }


@ray.remote(num_gpus=1)  # pragma: no cover
class _CheckpointSenderWorker(PolicyCheckpointEngineMixin):
    """Minimal real sender: the production mixin over checkpoint safetensors.

    Streams the HF checkpoint's tensors (exactly the names SGLang's live
    loader expects) through a real NIXL engine — everything downstream of the
    weight iterator is production code.
    """

    def __init__(self, rank: int, model_path: str):
        self.rank = rank
        self._model_path = model_path
        self.checkpoint_engine = None

    def _checkpoint_engine_weight_iterator(self, kv_scales=None):
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        local_dir = snapshot_download(
            self._model_path, allow_patterns=["*.safetensors", "*.json"]
        )
        import glob
        import os

        for shard in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
            with safe_open(shard, framework="pt", device="cpu") as tensors:
                for name in tensors.keys():
                    yield (
                        name,
                        tensors.get_tensor(name).to(torch.bfloat16).cuda(),
                    )


class _SenderWorkerGroup:
    def __init__(self, actors):
        self.workers = actors

    def run_all_workers_single_data(self, method_name, **kwargs):
        assert method_name == "checkpoint_engine_rpc"
        return [
            actor.checkpoint_engine_rpc.remote(
                checkpoint_method=kwargs["checkpoint_method"],
                method_kwargs=kwargs.get("method_kwargs"),
            )
            for actor in self.workers
        ]


class _SenderPolicy:
    """Driver-side policy facade: worker group + the refit-info hook."""

    def __init__(self, actors):
        self.worker_group = _SenderWorkerGroup(actors)

    def prepare_refit_info(self):
        # SGLangGeneration.prepare_refit_info is a no-op; nothing to describe.
        return {}


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@pytest.fixture(scope="module")
def recovery_stack(ray_cluster, tokenizer):
    """2 sender actors + 2 real SGLang engines + the real synchronizer."""
    senders = [_CheckpointSenderWorker.remote(rank, MODEL_PATH) for rank in range(2)]
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=2,
        name="ckpt-engine-recovery-test",
    )
    gen = SGLangGeneration(cluster, _make_recovery_cfg(tokenizer.pad_token_id))
    sync = CheckpointEngineWeightSynchronizer(
        _SenderPolicy(senders),
        gen,
        {
            "backend": "nixl",
            "update_weights_bucket_memory_ratio": 0.05,
            "engine_kwargs": {"nixl": {"device": "cuda", "release_after_refit": False}},
        },
    )
    yield gen, sync
    try:
        gen.shutdown()
    except Exception:
        pass
    try:
        cluster.shutdown()
    except Exception:
        pass
    for sender in senders:
        try:
            ray.kill(sender)
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()


def _make_input(tokenizer, prompt):
    token_ids = tokenizer.encode(prompt)
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "input_lengths": torch.tensor([len(token_ids)], dtype=torch.long),
        }
    )


def _wait_for_dead_slot(gen, index, timeout=DETECT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gen.all_engines[index] is None:
            return True
        time.sleep(1)
    return False


def _receiver_agent_names(gen):
    """agent_name per engine, via the (idempotent) prepare metadata."""
    metadata = ray.get(gen.run_checkpoint_engine_method("prepare_checkpoint_engine"))
    names = []
    for engine_metadata in metadata:
        entries = (
            engine_metadata if isinstance(engine_metadata, list) else [engine_metadata]
        )
        names.append(tuple(entry["agent_name"] for entry in entries))
    return names


def test_crashed_engine_is_rebound_and_receives_current_weights(
    recovery_stack, tokenizer
):
    gen, sync = recovery_stack

    # --- Baseline: communicator up, one full refit through the envelope. ---
    sync.init_communicator()
    assert gen.num_new_engines == 0, (
        "initial communicator setup must consume the startup engine count"
    )
    gen.prepare_for_generation()
    sync.sync_weights()
    assert not sync.is_stale
    result = gen.generate(_make_input(tokenizer, "The capital of France is"))
    assert result["generation_lengths"][0].item() > 0

    old_actor_id = gen.all_engines[0]._actor_id.hex()
    old_agent_names = _receiver_agent_names(gen)

    # --- Crash engine 0; the health monitor must notice on its own. ---
    ray.get(gen.all_engines[0]._simulate_crash.remote())
    assert _wait_for_dead_slot(gen, 0), "health monitor did not kill the crashed engine"

    # --- Recover: fresh actor, then garbage its weights (the oracle). ---
    gen.recover_updatable_engines()
    replacement = gen.all_engines[0]
    assert replacement is not None
    assert replacement._actor_id.hex() != old_actor_id
    assert gen.num_new_engines == 1

    ray.get(replacement.check_weights.remote(action="snapshot"))
    ray.get(replacement.check_weights.remote(action="reset_tensors"))

    # --- The recovery refit: probe -> rebind -> transfer, one sync call. ---
    sync.sync_weights()
    assert not sync.is_stale
    assert gen.num_new_engines == 0, (
        "a successful rebind must consume the recovery engine count"
    )
    assert sync._terminal_error is None

    # --- Oracle: the replacement holds real, current bytes again. ---
    compare = ray.get(replacement.check_weights.remote(action="compare"))
    assert compare is not None
    if isinstance(compare, dict):
        assert compare.get("success", True), compare
        for key in ("unwritten", "corrupted"):
            if key in compare:
                assert int(compare[key]) == 0, compare

    # --- Paired identity: only the replaced engine's incarnation changed. ---
    new_agent_names = _receiver_agent_names(gen)
    assert new_agent_names[0] != old_agent_names[0], (
        "replacement receivers must carry fresh NIXL agent_names"
    )
    assert new_agent_names[1] == old_agent_names[1], (
        "survivor receivers must be reused, not rebuilt"
    )

    # --- Direct request to the replacement's own server (not the router). ---
    base_url = ray.get(replacement.get_base_url.remote())
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 8, "temperature": 0.0},
        },
        timeout=60,
    )
    assert response.status_code == 200, response.text
    assert response.json().get("text"), response.json()

    # --- Fleet still serves through the normal path. ---
    result = gen.generate(_make_input(tokenizer, "The capital of Italy is"))
    assert result["generation_lengths"][0].item() > 0
    gen.finish_generation()
