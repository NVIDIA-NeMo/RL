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

"""End-to-end weight update tests using SGLangGeneration + mock FSDP trainer.

Verifies the full weight-streaming path:
  1. SGLangGeneration.check_weights("snapshot")  — save original weights
  2. SGLangGeneration.check_weights("reset_tensors") — randomize weights
  3. Mock FSDP trainer streams Qwen3-1.7B weights via stream_weights_via_http_impl
  4. SGLangGeneration.check_weights("compare")  — verify restored weights

Parametrised over two configurations (both require 4 GPUs):
  • 1 server  × TP=4  — single-server high-TP
  • 2 servers × TP=2  — multi-server routing

Model: Qwen/Qwen3-1.7B
"""

import gc
import os

import pytest
import ray
import torch
import torch.distributed as dist

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.redesign.ray_utils import find_available_port, get_host_info
from nemo_rl.models.generation.redesign.sglang_generation import SGLangGeneration

from helpers import make_actor_env_vars, post_and_assert_200

pytestmark = pytest.mark.sglang

MODEL_PATH = "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# SGLang config builder
# ---------------------------------------------------------------------------
def _make_sglang_cfg(tp_size):
    return {
        "sglang_cfg": {
            "model_path": MODEL_PATH,
            "dtype": "bfloat16",
            "random_seed": 42,
            "context_length": 1024,
            "log_level": "warning",
            "skip_server_warmup": True,
            "dp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "disable_piecewise_cuda_graph": True,
        },
        "sglang_server": {
            "num_gpus": 4,
            "num_gpus_per_engine": tp_size,
            "needs_offload": True,
            "cpu_weight_backup": False,
            "sglang_server_concurrency": 64,
        },
        "sglang_router": {
            "sglang_router_ip": None,
            "sglang_router_port": None,
        },
    }


# ---------------------------------------------------------------------------
# Mock FSDP trainer worker
# ---------------------------------------------------------------------------
@ray.remote(num_cpus=0.1)
class MockFSDPWorker:
    """Simulates one FSDP rank for weight streaming.

    Loads the full model on a single GPU and calls the real
    ``stream_weights_via_http_impl`` to send weights to SGLang servers.
    """

    def init(self, rank, world_size, master_addr, master_port, model_path, gpu_index):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(gpu_index)
        os.environ["WORLD_SIZE"] = str(world_size)

        self.rank = rank
        self.gpu_index = gpu_index

        torch.cuda.set_device(gpu_index)
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        from transformers import AutoModelForCausalLM

        device = torch.device(f"cuda:{gpu_index}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device)

        from nemo_rl.utils.nvml import get_device_uuid

        self.device_uuid = get_device_uuid(gpu_index)

    def get_device_uuid(self):
        return self.device_uuid

    def stream_weights(self, rollout_engines, num_gpus_per_engine):
        from nemo_rl.models.policy.utils import stream_weights_via_http_impl

        if not hasattr(self, "_ipc_worker_state"):
            self._ipc_worker_state = {}

        stream_weights_via_http_impl(
            model=self.model,
            rollout_engines=rollout_engines,
            num_gpus_per_engine=num_gpus_per_engine,
            rank=self.rank,
            world_size=dist.get_world_size(),
            worker_name=f"MockFSDPWorker-{self.rank}",
            buffer_size_bytes=512 * 1024 * 1024,
            worker_state=self._ipc_worker_state,
        )

    def shutdown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(
    params=[
        pytest.param({"tp_size": 4, "num_servers": 1}, id="tp4_1server"),
        pytest.param({"tp_size": 2, "num_servers": 2}, id="tp2_2servers"),
    ]
)
def sglang_gen(request, ray_cluster):
    """Real SGLangGeneration: RayVirtualCluster → router → engines."""
    cfg = request.param
    tp_size = cfg["tp_size"]

    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[4],
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=4,
        name="weight-update-test",
    )
    cluster_cfg = {"gpus_per_node": 4, "num_nodes": 1}
    sglang_cfg = _make_sglang_cfg(tp_size)

    gen = SGLangGeneration(cluster, cluster_cfg, sglang_cfg)
    yield gen

    try:
        gen.shutdown()
    except Exception:
        pass
    try:
        cluster.shutdown()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture
def mock_trainer(ray_cluster, sglang_gen):
    """4 MockFSDPWorker actors with torch.distributed (gloo), each loading Qwen3-1.7B.

    Actors are launched into the SGLang cluster's placement group using
    PlacementGroupSchedulingStrategy with fractional GPU (num_gpus=0.2), so
    they co-reside with the SGLang worker (which also takes 0.2) on the same
    bundles. This matches the nemo_rl colocated-mode and miles patterns; the
    PG's bundles have ``CPU: max_colocated_worker_groups`` capacity (=2) to
    fit both worker groups.
    """
    host_ip = get_host_info()[1]
    master_port = find_available_port(29500)
    env_vars = make_actor_env_vars()

    pg = sglang_gen.cluster.get_placement_groups()[0]

    workers = []
    for rank in range(4):
        w = MockFSDPWorker.options(
            num_cpus=0.2,
            num_gpus=0.2,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=rank,
            ),
            runtime_env={"env_vars": env_vars},
        ).remote()
        workers.append(w)

    # All workers must init simultaneously (gloo rendezvous).
    ray.get([
        w.init.remote(
            rank=rank,
            world_size=4,
            master_addr=host_ip,
            master_port=master_port,
            model_path=MODEL_PATH,
            gpu_index=rank,
        )
        for rank, w in enumerate(workers)
    ])

    yield workers

    for w in workers:
        try:
            ray.get(w.shutdown.remote())
        except Exception:
            pass
        ray.kill(w)
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_weight_update_roundtrip(sglang_gen, mock_trainer):
    """Snapshot -> reset -> offload -> update -> compare -> onload_kv.

    Exercises the full colocated-refit memory dance:
      snapshot -> reset_tensors -> offload_weights -> offload_kv ->
      onload_weights -> update_weights (stream) -> check compare ->
      onload_kv.
    """
    # 1. Snapshot original Qwen3-1.7B weights.
    print("[STEP 1/7] Snapshotting original weights...", flush=True)
    sglang_gen.check_weights("snapshot")
    print("[STEP 1/7] Snapshot complete.", flush=True)

    # 2. Randomize all model weights on the SGLang servers.
    print("[STEP 2/7] Randomizing (reset_tensors) model weights...", flush=True)
    sglang_gen.check_weights("reset_tensors")
    print("[STEP 2/7] Reset complete.", flush=True)

    # 3. Offload weights and KV cache to CPU (refit prelude).
    print("[STEP 3/7] Offloading weights and KV cache to CPU...", flush=True)
    sglang_gen.offload_weights()
    sglang_gen.offload_kv()
    print("[STEP 3/7] Offload complete.", flush=True)

    # 4. Onload weight buffers back to GPU so IPC handles can target them.
    print("[STEP 4/7] Onloading weight buffers back to GPU...", flush=True)
    sglang_gen.onload_weights()
    print("[STEP 4/7] Onload weights complete.", flush=True)

    # 5. All 4 mock FSDP workers stream weights simultaneously via CUDA IPC over HTTP.
    print("[STEP 5/7] Streaming weights from mock FSDP workers via CUDA IPC...", flush=True)
    rollout_engines = sglang_gen.rollout_engines
    num_gpus_per_engine = sglang_gen.num_gpus_per_engine
    ray.get([
        w.stream_weights.remote(rollout_engines, num_gpus_per_engine)
        for w in mock_trainer
    ])
    print("[STEP 5/7] Weight streaming complete.", flush=True)

    # 6. Compare current weights against snapshot - raises on mismatch.
    print("[STEP 6/7] Comparing current weights against snapshot...", flush=True)
    sglang_gen.check_weights("compare")
    print("[STEP 6/7] Compare passed.", flush=True)

    # 7. Onload KV cache to finish the refit cycle.
    print("[STEP 7/7] Onloading KV cache to finish refit cycle...", flush=True)
    sglang_gen.onload_kv()
    print("[STEP 7/7] Roundtrip complete.", flush=True)


# ---------------------------------------------------------------------------
# Test: roundtrip + router-based generate() + greedy before/after comparison
# ---------------------------------------------------------------------------
def test_weight_update_roundtrip_with_router_generation(sglang_gen, mock_trainer):
    """Full refit roundtrip with generation via router and per-worker HTTP 200 checks.

    Differs from ``test_weight_update_roundtrip`` in two ways:

    1. *Generation through the router.* Both the pre-snapshot and post-onload_kv
       generations go through ``sglang_gen.generate(..., greedy=True)`` which
       calls ``generate_one_sample(router_ip, router_port, ...)`` — i.e. an
       HTTP POST to ``http://{router_ip}:{router_port}/generate``, not to
       any individual server. (``sglang_gen.generate`` internally calls
       ``resp.raise_for_status()`` so a successful return implies HTTP 200.)
       Parametrised over ``tp4_1server`` and ``tp2_2servers``; both configs
       share the same router, so the same generation path is exercised in
       both.
    2. *Per-worker HTTP 200 checks for the refit cycle.* Instead of calling
       ``sglang_gen.check_weights(...)`` / ``offload_weights`` / ``offload_kv`` /
       ``onload_weights`` / ``onload_kv``, this test iterates
       ``sglang_gen.engines`` and drives the equivalent HTTP endpoints on
       **every worker** directly via ``post_and_assert_200`` (same pattern as
       ``tests/unit/models/generation/redesign/test_sglang_worker_memory.py``).
       That way every single memory/weights transition is verified to return
       ``resp.status_code == 200`` — ``_make_request`` would hide the status
       behind ``raise_for_status()``.

    Strict outer check: with ``temperature=0.0`` the mock FSDP trainer streams
    the original Qwen3-1.7B weights back, so pre- and post-roundtrip greedy
    token sequences must match exactly.
    """
    from transformers import AutoTokenizer

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Sanity: router endpoint is set so sglang_gen.generate actually routes.
    assert sglang_gen.router_ip is not None and sglang_gen.router_port is not None, (
        "router_ip/router_port not set on sglang_gen — generate() would not route"
    )
    print(
        f"[setup] Router endpoint: http://{sglang_gen.router_ip}:{sglang_gen.router_port}",
        flush=True,
    )

    # All logical-engine node-0 actors (one per SGLang server).
    engines = [e for e in sglang_gen.engines if e is not None]
    assert len(engines) >= 1, "sglang_gen has no engines"
    base_urls = ray.get([e.get_base_url.remote() for e in engines])
    assert all(u is not None for u in base_urls), f"missing base_url in {base_urls}"
    print(f"[setup] {len(engines)} worker(s); base_urls={base_urls}", flush=True)

    # --- Per-worker HTTP helpers -----------------------------------------------
    def _http_check_weights_all(action: str):
        """POST /weights_checker on every worker, asserting 200 each time."""
        for url in base_urls:
            post_and_assert_200(url, "weights_checker", {"action": action})

    def _http_release_weights_all():
        """Flush cache + POST /release_memory_occupation(tags=[weights]) per worker."""
        for engine, url in zip(engines, base_urls):
            ray.get(engine.flush_cache.remote())
            post_and_assert_200(
                url, "release_memory_occupation", {"tags": ["weights"]}
            )

    def _http_release_kv_all():
        """Flush cache + POST /release_memory_occupation(tags=[kv_cache, cuda_graph])."""
        for engine, url in zip(engines, base_urls):
            ray.get(engine.flush_cache.remote())
            post_and_assert_200(
                url,
                "release_memory_occupation",
                {"tags": ["kv_cache", "cuda_graph"]},
            )

    def _http_resume_weights_all():
        """POST /resume_memory_occupation(tags=[weights]) per worker."""
        for url in base_urls:
            post_and_assert_200(
                url, "resume_memory_occupation", {"tags": ["weights"]}
            )

    def _http_resume_kv_all():
        """POST /resume_memory_occupation(tags=[kv_cache, cuda_graph]) per worker."""
        for url in base_urls:
            post_and_assert_200(
                url,
                "resume_memory_occupation",
                {"tags": ["kv_cache", "cuda_graph"]},
            )

    # --- Router-based greedy generation ---------------------------------------
    test_prompt = "The capital of France is"
    input_ids = tokenizer.encode(test_prompt, add_special_tokens=True)
    input_len = len(input_ids)

    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "input_lengths": torch.tensor([input_len], dtype=torch.long),
        }
    )

    def _generate(tag):
        result = sglang_gen.generate(data, greedy=True)
        for key in ("output_ids", "generation_lengths", "unpadded_sequence_lengths", "logprobs"):
            assert key in result, f"[{tag}] generate() output missing key: {key}"
        gen_len = int(result["generation_lengths"][0].item())
        assert gen_len > 0, f"[{tag}] generate() returned 0 tokens (no new tokens generated)"
        tokens = result["output_ids"][0, input_len : input_len + gen_len].tolist()
        assert all(isinstance(t, int) for t in tokens), (
            f"[{tag}] output tokens should be ints, got {tokens!r}"
        )
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        assert len(text) > 0, f"[{tag}] decoded generated text is empty"
        print(f"[{tag}] gen_len={gen_len} tokens={tokens} text={text!r}", flush=True)
        return tokens

    # --- Generation BEFORE snapshot (via router) -------------------------------
    print("[PRE] Router greedy generate() before snapshot...", flush=True)
    tokens_before = _generate("PRE")

    # --- Steps 1-7, every HTTP call asserted 200 ------------------------------
    print("[STEP 1/7] Snapshotting original weights (HTTP weights_checker×workers)...", flush=True)
    _http_check_weights_all("snapshot")
    print("[STEP 1/7] Snapshot complete.", flush=True)

    print("[STEP 2/7] Randomizing weights (HTTP weights_checker reset_tensors×workers)...", flush=True)
    _http_check_weights_all("reset_tensors")
    print("[STEP 2/7] Reset complete.", flush=True)

    print("[STEP 3/7] Offloading weights + KV (HTTP release_memory_occupation×workers)...", flush=True)
    _http_release_weights_all()
    _http_release_kv_all()
    print("[STEP 3/7] Offload complete.", flush=True)

    print("[STEP 4/7] Onloading weights (HTTP resume_memory_occupation weights×workers)...", flush=True)
    _http_resume_weights_all()
    print("[STEP 4/7] Onload weights complete.", flush=True)

    print("[STEP 5/7] Streaming weights from mock FSDP workers via CUDA IPC...", flush=True)
    rollout_engines = sglang_gen.rollout_engines
    num_gpus_per_engine = sglang_gen.num_gpus_per_engine
    ray.get([
        w.stream_weights.remote(rollout_engines, num_gpus_per_engine)
        for w in mock_trainer
    ])
    print("[STEP 5/7] Weight streaming complete.", flush=True)

    print("[STEP 6/7] Compare vs snapshot (HTTP weights_checker compare×workers)...", flush=True)
    _http_check_weights_all("compare")
    print("[STEP 6/7] Compare passed.", flush=True)

    print("[STEP 7/7] Onloading KV (HTTP resume_memory_occupation kv×workers)...", flush=True)
    _http_resume_kv_all()
    print("[STEP 7/7] Roundtrip complete.", flush=True)

    # --- Generation AFTER onload_kv (via router) -------------------------------
    print("[POST] Router greedy generate() after onload_kv...", flush=True)
    tokens_after = _generate("POST")

    # --- Sanity: generate() actually produced new tokens on BOTH runs ----------
    assert len(tokens_before) > 0, "generate() returned no tokens before roundtrip"
    assert len(tokens_after) > 0, "generate() returned no tokens after roundtrip"
    assert len(tokens_before) == len(tokens_after), (
        f"Different number of generated tokens before vs. after: "
        f"before={len(tokens_before)}, after={len(tokens_after)}"
    )

    # --- Strict equality (greedy => deterministic across roundtrip) ------------
    assert tokens_before == tokens_after, (
        "Greedy tokens changed across the refit roundtrip:\n"
        f"  before (pre-snapshot):  {tokens_before}\n"
        f"  after  (post-onload_kv): {tokens_after}"
    )
    print(
        f"[ASSERT] Greedy tokens match before vs. after roundtrip "
        f"(n={len(tokens_before)} tokens, both non-empty, both via router).",
        flush=True,
    )
