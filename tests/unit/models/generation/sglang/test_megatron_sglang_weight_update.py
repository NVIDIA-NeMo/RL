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

"""End-to-end weight update tests: real Megatron Policy → real SGLangGeneration.

Mirrors ``tests/unit/models/generation/sglang/test_weight_update_real.py`` but
replaces the ``MockFSDPWorker`` (FSDP / DTensor) trainer with a *real* Megatron
``Policy`` using ``nemo_rl.models.policy.workers.megatron_policy_worker``.

Cross-product the user asked for:

  • Mode:    ``colocate`` (IPC) and ``disaggregate`` (NCCL broadcast)
  • Megatron parallelism:
        - ep2 pp2          (TP=1, PP=2, EP=2, DP=2 → 4 GPUs)
        - tp2 pp2          (TP=2, PP=2, EP=1, DP=1 → 4 GPUs)
        - tp2 ep2 pp2      (TP=2, PP=2, EP=2, DP=2 → 8 GPUs)
  • SGLang shape:
        - tp=4 ep=4 dp=4 --enable-dp-attention   (4 engine GPUs)
        - tp=4 ep=2 dp=4 --enable-dp-attention   (4 engine GPUs)
        - tp=2 ep=2 pp=2                          (4 engine GPUs)

Model: ``nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`` sliced to 14 layers
(``MEMEM*EMEMEM*E``; sliced once per host into ``HF_HOME``). 14 is the
smallest layer count that simultaneously (a) gives every sglang PP rank
at least one attention layer — necessary to avoid a NemotronH-implementation
corner case where the weight checker's CPU copy of the rank's
``embed_tokens.weight`` fails with ``CUDA error: invalid argument`` when
the rank has zero attention layers — and (b) divides evenly by PP=2 so
megatron is happy with our ``pp=2`` configs. See ``_nemotron_slicer.py``.

Tests:

  * ``test_weight_update_roundtrip`` — snapshot → reset → offload → onload
    weights → refit (Megatron streams to SGLang) → compare → onload kv.
  * ``test_weight_update_roundtrip_with_router_generation`` — same flow,
    but bracketed by router-driven greedy ``generate()`` calls; every
    HTTP call is asserted 200 by reaching into the per-worker endpoints.
"""

from __future__ import annotations

import gc
import os

import pytest
import ray
import torch
from _megatron_helpers import (
    EOS_TOKEN_ID,
    MEGATRON_CFGS,
    MEGATRON_DP1,
    PAD_TOKEN_ID,
    SGLANG_CFGS,
    SGLANG_TP1,
    TestTriple,
    make_policy_config,
    make_sglang_cfg,
    megatron_world_size,
    required_world_size,
)
from _nemotron_slicer import ensure_sliced_model
from helpers import post_and_assert_200

from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.sglang.sglang_generation import SGLangGeneration

pytestmark = pytest.mark.sglang

# ---------------------------------------------------------------------------
# Per-host GPU budget. The host's Ray cluster decides what is actually
# available; ``cluster.world_size()`` checks happen inside
# ``RayVirtualCluster``. We use this constant only to pre-skip variants whose
# parallelism layout is impossible on this host (avoid ResourceInsufficient).
# ---------------------------------------------------------------------------
TOTAL_AVAILABLE_GPUS = 8


# ---------------------------------------------------------------------------
# Cartesian product → pytest params
# ---------------------------------------------------------------------------
def _build_params() -> list[pytest.param]:
    out: list[pytest.param] = []
    # Cartesian product over the multi-GPU shapes.
    pairs: list[tuple] = [(m, s) for m in MEGATRON_CFGS for s in SGLANG_CFGS]
    # Plus one explicit single-GPU pairing: Megatron DP=1 (TP=PP=EP=1) →
    # SGLang TP=1. Adding it as a standalone pair (rather than expanding
    # MEGATRON_CFGS / SGLANG_CFGS) keeps the matrix from blowing up with
    # combinations no one asked for.
    pairs.append((MEGATRON_DP1, SGLANG_TP1))
    for colocated in (True, False):
        for m, s in pairs:
            triple = TestTriple(megatron=m, sglang=s, colocated=colocated)
            marks: list = []
            need = required_world_size(
                megatron=m, sglang=s, colocated=colocated
            )
            if need > TOTAL_AVAILABLE_GPUS:
                marks.append(
                    pytest.mark.skip(
                        reason=(
                            f"{triple.id} needs {need} GPUs but only "
                            f"{TOTAL_AVAILABLE_GPUS} are available"
                        )
                    )
                )
            out.append(pytest.param(triple, id=triple.id, marks=marks))
    return out


# ---------------------------------------------------------------------------
# Sliced-model fixture
# ---------------------------------------------------------------------------
# Allow tests to run against an arbitrary HuggingFace local snapshot path via
# ``NEMOTRON_TEST_MODEL_PATH``; this is mainly useful for diagnostics where we
# want to compare slice vs. full-model behaviour without re-slicing.
_NEMOTRON_TEST_MODEL_PATH_ENV = "NEMOTRON_TEST_MODEL_PATH"


@pytest.fixture(scope="session")
def sliced_model_path() -> str:
    """Materialize the sliced Nemotron-3-Nano checkpoint once per session.

    Honours the ``NEMOTRON_TEST_MODEL_PATH`` env var as an override so we can
    point the suite at the full upstream checkpoint when debugging
    slice-specific issues.
    """
    override = os.environ.get(_NEMOTRON_TEST_MODEL_PATH_ENV)
    if override:
        return override
    return str(ensure_sliced_model())


# ---------------------------------------------------------------------------
# Cluster + SGLang fixture (parametrised)
#
# Note: Megatron ``Policy`` is **not** built in the fixture — see the test
# bodies. The required sequence per the GRPO colocate refit is:
#
#     snapshot → reset → offload sglang weights+kv+cuda_graph
#       → create megatron Policy → onload sglang weights → refit
#       → compare → onload sglang kv+cuda_graph
#
# Disaggregate is similar but skips the offload/onload dance because trainer
# and inference live on different GPUs:
#
#     snapshot → reset → create megatron Policy → refit → compare
#
# Putting the Policy creation in the test body lets us observe each step in
# the order GRPO actually exercises in production.
# ---------------------------------------------------------------------------
@pytest.fixture(params=_build_params())
def env(request, ray_cluster, sliced_model_path):
    """Materialize ``(triple, sglang_gen, train_cluster, sliced_model_path)``.

    Colocate: a single ``RayVirtualCluster`` with ``max_colocated_worker_groups
    = 2`` so trainer + SGLang share the same placement-group bundles.

    Disaggregate: two clusters — one for the Megatron trainer, one for the
    SGLang engines. They live in separate placement groups but in the same
    Ray runtime.
    """
    triple: TestTriple = request.param
    m, s, colocated = triple.megatron, triple.sglang, triple.colocated

    train_world = megatron_world_size(m)
    sglang_world = s.num_gpus_per_engine

    # --- build clusters ----------------------------------------------------
    if colocated:
        # Single shared cluster. Bundles equal max(train, sglang) GPUs because
        # both worker groups need to fit on the same placement group.
        bundle_count = max(train_world, sglang_world)
        train_cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[bundle_count],
            use_gpus=True,
            max_colocated_worker_groups=2,
            num_gpus_per_node=bundle_count,
            name=f"colo-{triple.id}",
        )
        sglang_cluster = train_cluster
    else:
        train_cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[train_world],
            use_gpus=True,
            max_colocated_worker_groups=1,
            num_gpus_per_node=train_world,
            name=f"disag-train-{triple.id}",
        )
        sglang_cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[sglang_world],
            use_gpus=True,
            max_colocated_worker_groups=1,
            num_gpus_per_node=sglang_world,
            name=f"disag-infer-{triple.id}",
        )

    # --- build SGLangGeneration; weights stay live on GPU until the test
    # explicitly offloads them (avoids a NemotronH × multi-rank ×
    # released-storage corner case in sglang's weight_checker.snapshot).
    sglang_cfg = make_sglang_cfg(
        model_path=sliced_model_path,
        sglang=s,
        colocated=colocated,
    )
    sglang_gen = SGLangGeneration(sglang_cluster, sglang_cfg)

    yield triple, sglang_gen, train_cluster, sliced_model_path

    # --- teardown ----------------------------------------------------------
    try:
        sglang_gen.shutdown()
    except Exception:
        pass
    try:
        train_cluster.shutdown()
    except Exception:
        pass
    if not colocated:
        try:
            sglang_cluster.shutdown()
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _refit_buffer_bytes() -> int:
    """Per-bucket buffer size for the streaming refit. 256 MiB is plenty for
    the sliced 14-layer model and matches the order-of-magnitude used in the
    DTensor mock test."""
    return 256 * 1024 * 1024


def _refit_megatron_to_sglang(*, policy, policy_generation, colocated: bool) -> None:
    """Drive one full Megatron → SGLang refit using the production helpers.

    Mirrors the dispatch logic in
    ``nemo_rl.algorithms.grpo._refit_sglang_dispatch`` so we exercise exactly
    the code path GRPO uses, parametrised on ``colocated``.
    """
    from nemo_rl.models.policy.workers import megatron_policy_worker as _backend

    helper = (
        _backend.refit_sglang_colocated
        if colocated
        else _backend.refit_sglang_distributed
    )
    helper(
        policy=policy,
        policy_generation=policy_generation,
        buffer_size_bytes=_refit_buffer_bytes(),
    )


def _build_policy(*, train_cluster, megatron_shape, model_path, colocated: bool):
    """Construct a real Megatron ``Policy`` for the test body.

    Kept out of the fixture because the colocate flow requires sglang's
    weights to be offloaded *first* (so megatron can claim GPU memory
    without OOM); that ordering is most readable inside the test body.
    """
    from transformers import AutoTokenizer

    from nemo_rl.models.policy.lm_policy import Policy

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    policy_cfg = make_policy_config(
        model_path=model_path,
        megatron=megatron_shape,
        colocated=colocated,
    )
    return Policy(
        cluster=train_cluster,
        config=policy_cfg,
        tokenizer=tokenizer,
        init_optimizer=False,
        init_reference_model=False,
    )


def _shutdown_policy(policy) -> None:
    try:
        policy.shutdown()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_weight_update_roundtrip(env):
    """Full Megatron→SGLang refit roundtrip with snapshot/reset/compare.

    Sequencing per the GRPO colocate refit contract:

    Colocate (sglang and megatron share GPUs):
        1. snapshot         — capture sglang's freshly-loaded weights
        2. reset_tensors    — overwrite weights with random data
        3. offload sglang weights + kv + cuda_graph to CPU
        4. create megatron Policy        ← claims the now-free GPU mem
        5. onload sglang weights         ← weights are random/invalid here
        6. refit (megatron → sglang)     ← restores them
        7. compare against snapshot      ← passes ⇔ refit landed correctly
        8. onload sglang kv + cuda_graph

    Disaggregate (sglang and megatron on disjoint GPUs):
        1. snapshot
        2. reset_tensors
        3. create megatron Policy        (no offload needed; different GPUs)
        4. refit (megatron → sglang via NCCL broadcast)
        5. compare

    Uses the production refit helpers
    (``megatron_policy_worker.refit_sglang_{colocated,distributed}``) — no
    naive reimplementation.
    """
    triple, sglang_gen, train_cluster, model_path = env
    colocated = triple.colocated

    # 1. Snapshot sglang's freshly-loaded weights.
    print("[STEP 1] Snapshotting original sglang weights...", flush=True)
    sglang_gen.check_weights("snapshot")
    print("[STEP 1] Snapshot complete.", flush=True)

    # 2. Randomize sglang's weights — refit MUST overwrite them. Without
    # this step ``compare`` in step 7 trivially passes regardless of
    # whether refit actually copied anything.
    print("[STEP 2] Randomizing (reset_tensors) sglang weights...", flush=True)
    sglang_gen.check_weights("reset_tensors")
    print("[STEP 2] Reset complete.", flush=True)

    if colocated:
        # 3. Offload sglang weights + kv + cuda_graph to CPU so megatron
        # can claim GPU memory at construction time.
        print("[STEP 3] Offloading sglang weights+kv+cuda_graph...", flush=True)
        sglang_gen.offload_weights()
        sglang_gen.offload_kv()
        print("[STEP 3] Offload complete.", flush=True)

    # 4. Create the megatron trainer.
    print(
        f"[STEP 4] Creating Megatron Policy ({triple.megatron.id})...",
        flush=True,
    )
    policy = _build_policy(
        train_cluster=train_cluster,
        megatron_shape=triple.megatron,
        model_path=model_path,
        colocated=colocated,
    )
    try:
        # The refit drivers expect both sides to have exchanged the
        # state-dict shape via ``prepare_refit_info`` once at startup.
        state_dict_info = policy.prepare_refit_info()
        sglang_gen.prepare_refit_info(state_dict_info)
        print("[STEP 4] Megatron Policy ready.", flush=True)

        if colocated:
            # 5. Onload sglang weights so refit (CUDA IPC) can target them.
            print("[STEP 5] Onloading sglang weight buffers...", flush=True)
            sglang_gen.onload_weights()
            print("[STEP 5] Onload weights complete.", flush=True)

        # 6. Refit weights through the production helper.
        print(
            f"[STEP 6] Refitting Megatron → SGLang via "
            f"{'IPC (colocated)' if colocated else 'NCCL broadcast (disaggregate)'}...",
            flush=True,
        )
        _refit_megatron_to_sglang(
            policy=policy, policy_generation=sglang_gen, colocated=colocated
        )
        print("[STEP 6] Refit complete.", flush=True)

        # 7. Compare current sglang weights against the step-1 snapshot.
        # Passes ⇔ megatron streamed the correct values back.
        print("[STEP 7] Comparing current weights against snapshot...", flush=True)
        sglang_gen.check_weights("compare")
        print("[STEP 7] Compare passed.", flush=True)

        if colocated:
            # 8. Onload sglang kv + cuda_graph for next-time inference.
            print("[STEP 8] Onloading sglang kv+cuda_graph...", flush=True)
            sglang_gen.onload_kv()
            print("[STEP 8] Roundtrip complete.", flush=True)
    finally:
        _shutdown_policy(policy)


def test_weight_update_roundtrip_with_router_generation(env):
    """Full refit roundtrip with router-driven generation and per-worker 200 checks.

    Same colocate / disaggregate sequencing as ``test_weight_update_roundtrip``
    (snapshot → reset → [offload] → create trainer → [onload weights] → refit
    → compare → [onload kv]), but additionally:

    1. *Generation through the router.* Both the pre-snapshot and
       post-onload_kv generations go through ``sglang_gen.generate(...,
       greedy=True)`` which routes via the SGLang router. With
       ``greedy=True`` the two token sequences must be identical token-by-
       token across the roundtrip.
    2. *Per-worker HTTP 200 checks for the refit cycle.* Instead of calling
       ``sglang_gen.check_weights / offload_* / onload_*`` (which use Ray
       actor methods that consume the status code), this test iterates
       ``sglang_gen.engines`` and drives the equivalent HTTP endpoints on
       every worker directly via ``post_and_assert_200``.
    """
    from transformers import AutoTokenizer

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    triple, sglang_gen, train_cluster, model_path = env
    colocated = triple.colocated

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Sanity: router must be set so ``generate`` actually routes.
    assert sglang_gen.router_ip is not None and sglang_gen.router_port is not None, (
        "router_ip/router_port not set on sglang_gen — generate() would not route"
    )
    print(
        f"[setup] {triple.id} router=http://{sglang_gen.router_ip}:{sglang_gen.router_port}",
        flush=True,
    )

    engines = [e for e in sglang_gen.engines if e is not None]
    assert len(engines) >= 1, "sglang_gen has no engines"
    base_urls = ray.get([e.get_base_url.remote() for e in engines])
    assert all(u is not None for u in base_urls), f"missing base_url in {base_urls}"
    print(f"[setup] {len(engines)} worker(s); base_urls={base_urls}", flush=True)

    # --- Per-worker HTTP helpers -----------------------------------------------
    def _http_check_weights_all(action: str) -> None:
        for url in base_urls:
            post_and_assert_200(url, "weights_checker", {"action": action})

    def _http_release_weights_all() -> None:
        for engine, url in zip(engines, base_urls):
            ray.get(engine.flush_cache.remote())
            post_and_assert_200(
                url, "release_memory_occupation", {"tags": ["weights"]}
            )

    def _http_release_kv_all() -> None:
        for engine, url in zip(engines, base_urls):
            ray.get(engine.flush_cache.remote())
            post_and_assert_200(
                url,
                "release_memory_occupation",
                {"tags": ["kv_cache", "cuda_graph"]},
            )

    def _http_resume_weights_all() -> None:
        for url in base_urls:
            post_and_assert_200(url, "resume_memory_occupation", {"tags": ["weights"]})

    def _http_resume_kv_all() -> None:
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

    def _generate(tag: str) -> list[int]:
        result = sglang_gen.generate(data, greedy=True)
        for key in (
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ):
            assert key in result, f"[{tag}] generate() output missing key: {key}"
        gen_len = int(result["generation_lengths"][0].item())
        assert gen_len > 0, (
            f"[{tag}] generate() returned 0 tokens (no new tokens generated)"
        )
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

    # 1. snapshot via per-worker HTTP weights_checker
    print("[STEP 1] Snapshotting weights (HTTP weights_checker×workers)...", flush=True)
    _http_check_weights_all("snapshot")
    print("[STEP 1] Snapshot complete.", flush=True)

    # 2. reset_tensors via per-worker HTTP weights_checker — refit MUST
    # overwrite these random values for ``compare`` and the post-greedy
    # token sequence to match.
    print("[STEP 2] Randomizing weights (HTTP reset_tensors×workers)...", flush=True)
    _http_check_weights_all("reset_tensors")
    print("[STEP 2] Reset complete.", flush=True)

    if colocated:
        # 3. offload sglang weights + kv + cuda_graph
        print(
            "[STEP 3] Offloading weights+kv+cuda_graph (HTTP release_memory_occupation×workers)...",
            flush=True,
        )
        _http_release_weights_all()
        _http_release_kv_all()
        print("[STEP 3] Offload complete.", flush=True)

    # 4. create the megatron trainer
    print(f"[STEP 4] Creating Megatron Policy ({triple.megatron.id})...", flush=True)
    policy = _build_policy(
        train_cluster=train_cluster,
        megatron_shape=triple.megatron,
        model_path=model_path,
        colocated=colocated,
    )
    try:
        state_dict_info = policy.prepare_refit_info()
        sglang_gen.prepare_refit_info(state_dict_info)
        print("[STEP 4] Megatron Policy ready.", flush=True)

        if colocated:
            # 5. onload sglang weight buffers
            print(
                "[STEP 5] Onloading weights (HTTP resume_memory_occupation×workers)...",
                flush=True,
            )
            _http_resume_weights_all()
            print("[STEP 5] Onload weights complete.", flush=True)

        # 6. refit through production helper
        print(
            f"[STEP 6] Refitting Megatron → SGLang via "
            f"{'IPC (colocated)' if colocated else 'NCCL broadcast (disaggregate)'}...",
            flush=True,
        )
        _refit_megatron_to_sglang(
            policy=policy, policy_generation=sglang_gen, colocated=colocated
        )
        print("[STEP 6] Refit complete.", flush=True)

        # 7. compare vs snapshot
        print(
            "[STEP 7] Compare vs snapshot (HTTP weights_checker×workers)...",
            flush=True,
        )
        _http_check_weights_all("compare")
        print("[STEP 7] Compare passed.", flush=True)

        if colocated:
            # 8. onload sglang kv + cuda_graph
            print(
                "[STEP 8] Onloading kv+cuda_graph (HTTP resume_memory_occupation×workers)...",
                flush=True,
            )
            _http_resume_kv_all()
            print("[STEP 8] Onload kv complete.", flush=True)
    finally:
        _shutdown_policy(policy)

    # --- Generation AFTER onload_kv (via router) -------------------------------
    print("[POST] Router greedy generate() after onload_kv...", flush=True)
    tokens_after = _generate("POST")

    # --- Sanity & strict equality ---------------------------------------------
    assert len(tokens_before) > 0, "generate() returned no tokens before roundtrip"
    assert len(tokens_after) > 0, "generate() returned no tokens after roundtrip"
    assert len(tokens_before) == len(tokens_after), (
        f"Different number of generated tokens before vs. after: "
        f"before={len(tokens_before)}, after={len(tokens_after)}"
    )
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


# Surface the constants so callers (or rerun tooling) can sanity-check that
# ``PAD_TOKEN_ID`` / ``EOS_TOKEN_ID`` line up with what the sliced tokenizer
# reports. Kept at module scope rather than inside the test body so it's
# visible in ``pytest --collect-only``.
__all__ = [
    "PAD_TOKEN_ID",
    "EOS_TOKEN_ID",
    "test_weight_update_roundtrip",
    "test_weight_update_roundtrip_with_router_generation",
]
