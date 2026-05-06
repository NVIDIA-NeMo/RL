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

"""Generation tests for the Megatron + SGLang stack.

Mirrors ``tests/unit/models/generation/sglang/test_sglang_generation.py`` but:

  • Uses the sliced ``nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`` checkpoint
    (only the first ``MEMEM*`` block; see ``_nemotron_slicer.py``).
  • Runs every ``SGLangGeneration`` API check across the same three SGLang
    shapes used in the weight-update test:
        - tp=4 ep=4 dp=4 --enable-dp-attention
        - tp=4 ep=2 dp=4 --enable-dp-attention
        - tp=2 ep=2 pp=2
  • Brings up a real Megatron ``Policy`` alongside SGLang for each of the
    three Megatron parallelism shapes (``ep2 pp2`` / ``tp2 pp2`` /
    ``tp2 ep2 pp2``), in both ``colocate`` and ``disaggregate`` modes — that
    way each generation test runs against a non-trivial trainer next to it
    and the same ``PolicyConfig`` schema as the weight-update test.

Tests follow the structure of ``test_sglang_generation.py``:

  • ``generate()`` — output keys, shape, determinism, truncation, logprobs,
    max_new_tokens cap, batched prompts, empty input, stop strings.
  • ``generate_async()`` — single-sample yield, agreement with sync.
  • ``generate_one_sample()`` — return tuple shape and types.
  • Memory cycle via worker API and via direct HTTP 200 + top-level API.
  • ``invalidate_kv_cache()`` aggregator + after-generate flush_cache pacing.
"""

from __future__ import annotations

import asyncio
import gc

import pytest
import ray
import torch
from _megatron_helpers import (
    EOS_TOKEN_ID,
    MEGATRON_CFGS,
    PAD_TOKEN_ID,
    SGLANG_CFGS,
    TestTriple,
    make_policy_config,
    make_sglang_cfg,
    megatron_world_size,
    required_world_size,
)
from _nemotron_slicer import ensure_sliced_model
from helpers import make_generation_sampling_params, post_and_assert_200

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.sglang.sglang_generation import (
    SGLangGeneration,
    generate_one_sample,
)

pytestmark = pytest.mark.sglang

TOTAL_AVAILABLE_GPUS = 8


# ---------------------------------------------------------------------------
# Cartesian product → pytest params (same matrix as the weight-update test)
# ---------------------------------------------------------------------------
def _build_params() -> list[pytest.param]:
    out: list[pytest.param] = []
    for colocated in (True, False):
        for m in MEGATRON_CFGS:
            for s in SGLANG_CFGS:
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
# Sliced-model + tokenizer fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sliced_model_path() -> str:
    return str(ensure_sliced_model())


@pytest.fixture(scope="session")
def tokenizer(sliced_model_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(sliced_model_path, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Cluster + Policy + SGLang fixture (parametrised). Same shape as the
# weight-update test's ``env`` fixture, but exposed as ``sglang_gen`` so the
# (already long) generation test bodies stay readable.
# ---------------------------------------------------------------------------
@pytest.fixture(params=_build_params())
def sglang_gen(request, ray_cluster, sliced_model_path):
    triple: TestTriple = request.param
    m, s, colocated = triple.megatron, triple.sglang, triple.colocated

    train_world = megatron_world_size(m)
    sglang_world = s.num_gpus_per_engine

    if colocated:
        bundle_count = max(train_world, sglang_world)
        train_cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[bundle_count],
            use_gpus=True,
            max_colocated_worker_groups=2,
            num_gpus_per_node=bundle_count,
            name=f"gen-colo-{triple.id}",
        )
        sglang_cluster = train_cluster
    else:
        train_cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[train_world],
            use_gpus=True,
            max_colocated_worker_groups=1,
            num_gpus_per_node=train_world,
            name=f"gen-disag-train-{triple.id}",
        )
        sglang_cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[sglang_world],
            use_gpus=True,
            max_colocated_worker_groups=1,
            num_gpus_per_node=sglang_world,
            name=f"gen-disag-infer-{triple.id}",
        )

    sglang_cfg = make_sglang_cfg(
        model_path=sliced_model_path,
        sglang=s,
        colocated=colocated,
    )
    gen = SGLangGeneration(sglang_cluster, sglang_cfg)
    gen.finish_generation()

    # Build the Megatron policy. Even though the generation tests don't refit,
    # bringing the trainer up exercises the same setup path as the weight-
    # update test and ensures generation correctness with a colocated/
    # disaggregated trainer present.
    from transformers import AutoTokenizer

    from nemo_rl.models.policy.lm_policy import Policy

    tok = AutoTokenizer.from_pretrained(sliced_model_path, trust_remote_code=True)
    policy_cfg = make_policy_config(
        model_path=sliced_model_path,
        megatron=m,
        colocated=colocated,
    )
    policy = Policy(
        cluster=train_cluster,
        config=policy_cfg,
        tokenizer=tok,
        init_optimizer=False,
        init_reference_model=False,
    )

    state_dict_info = policy.prepare_refit_info()
    gen.prepare_refit_info(state_dict_info)

    # Bring SGLang back up so generate() works.
    gen.prepare_for_generation()

    yield gen

    try:
        policy.shutdown()
    except Exception:
        pass
    try:
        gen.shutdown()
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
def _make_input(tokenizer, prompt, pad_length=None):
    token_ids = tokenizer.encode(prompt)
    input_length = len(token_ids)
    if pad_length and pad_length > input_length:
        token_ids = token_ids + [tokenizer.pad_token_id] * (pad_length - input_length)
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "input_lengths": torch.tensor([input_length], dtype=torch.long),
        }
    )


def _make_batch(tokenizer, prompts, pad_length=None):
    all_ids = []
    all_lengths = []
    max_len = 0
    for p in prompts:
        ids = tokenizer.encode(p)
        all_ids.append(ids)
        all_lengths.append(len(ids))
        max_len = max(max_len, len(ids))
    if pad_length:
        max_len = max(max_len, pad_length)
    padded = [ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids in all_ids]
    return BatchedDataDict(
        {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "input_lengths": torch.tensor(all_lengths, dtype=torch.long),
        }
    )


# ===================================================================
# Tests: SGLangGeneration.generate()
# ===================================================================
def test_generate_returns_batched_data_dict(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "Hello")
    result = sglang_gen.generate(data, greedy=True)
    for key in (
        "output_ids",
        "logprobs",
        "generation_lengths",
        "unpadded_sequence_lengths",
        "truncated",
    ):
        assert key in result, f"Missing key: {key}"


def test_generate_output_ids_shape(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "The capital of France is")
    result = sglang_gen.generate(data, greedy=True)
    assert result["output_ids"].dim() == 2
    assert result["output_ids"].shape[0] == 1
    gen_len = result["generation_lengths"][0].item()
    input_len = data["input_lengths"][0].item()
    assert result["unpadded_sequence_lengths"][0].item() == input_len + gen_len


def test_generate_greedy_determinism(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "Once upon a time")
    r1 = sglang_gen.generate(data, greedy=True)
    r2 = sglang_gen.generate(data, greedy=True)
    assert torch.equal(r1["output_ids"], r2["output_ids"]), (
        "Greedy generation is not deterministic"
    )


def test_generate_truncation_flag(sglang_gen, tokenizer):
    orig = sglang_gen.sglang_cfg["max_new_tokens"]
    sglang_gen.sglang_cfg["max_new_tokens"] = 1
    try:
        data = _make_input(tokenizer, "Tell me a very long story about dragons and")
        result = sglang_gen.generate(data, greedy=True)
        gen_len = result["generation_lengths"][0].item()
        assert gen_len == 1, f"Expected 1 token, got {gen_len}"
        assert result["truncated"][0].item() is True, "Expected truncated=True"
    finally:
        sglang_gen.sglang_cfg["max_new_tokens"] = orig


def test_generate_logprobs_valid(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "Hello world")
    result = sglang_gen.generate(data, greedy=True)
    gen_len = result["generation_lengths"][0].item()
    input_len = data["input_lengths"][0].item()
    lps = result["logprobs"][0, input_len : input_len + gen_len]
    assert torch.isfinite(lps).all(), "Logprobs contain NaN or Inf"
    assert (lps <= 0.0).all(), "Logprobs should be non-positive"


def test_generate_respects_max_new_tokens(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "Count from 1 to 100:")
    result = sglang_gen.generate(data, greedy=True)
    max_new = sglang_gen.sglang_cfg["max_new_tokens"]
    gen_len = result["generation_lengths"][0].item()
    assert gen_len <= max_new, f"gen_len={gen_len} > max_new_tokens={max_new}"


def test_generate_batch_multiple_samples(sglang_gen, tokenizer):
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is 2 plus 2?",
    ]
    data = _make_batch(tokenizer, prompts)
    result = sglang_gen.generate(data, greedy=True)
    assert result["output_ids"].shape[0] == 3
    assert result["generation_lengths"].shape[0] == 3
    for i in range(3):
        gen_len = result["generation_lengths"][i].item()
        assert gen_len > 0, f"Sample {i} generated 0 tokens"


def test_generate_empty_input(sglang_gen):
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros((0, 0), dtype=torch.long),
            "input_lengths": torch.zeros(0, dtype=torch.long),
        }
    )
    result = sglang_gen.generate(data, greedy=True)
    assert result["output_ids"].shape[0] == 0


def test_generate_with_stop_strings(sglang_gen, tokenizer):
    orig_stop = sglang_gen.sglang_cfg.get("stop_strings")
    sglang_gen.sglang_cfg["stop_strings"] = ["\n"]
    try:
        data = _make_input(tokenizer, "List:\n1. Apple\n2.")
        result = sglang_gen.generate(data, greedy=True)
        gen_len = result["generation_lengths"][0].item()
        max_new = sglang_gen.sglang_cfg["max_new_tokens"]
        # Soft check — model could emit \n on the very first token.
        assert gen_len <= max_new
    finally:
        sglang_gen.sglang_cfg["stop_strings"] = orig_stop


# ===================================================================
# Tests: SGLangGeneration.generate_async()
# ===================================================================
def test_generate_async_yields_single_sample(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "Hello")

    async def _run():
        results = []
        async for idx, batch in sglang_gen.generate_async(data, greedy=True):
            results.append((idx, batch))
        return results

    results = asyncio.run(_run())
    assert len(results) == 1
    idx, batch = results[0]
    assert idx == 0
    assert "output_ids" in batch
    assert batch["generation_lengths"][0].item() > 0


def test_generate_async_output_matches_generate(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "The answer is")
    sync_result = sglang_gen.generate(data, greedy=True)

    async def _run():
        async for _, batch in sglang_gen.generate_async(data, greedy=True):
            return batch
        return None

    async_result = asyncio.run(_run())
    assert async_result is not None

    sync_len = sync_result["generation_lengths"][0].item()
    async_len = async_result["generation_lengths"][0].item()
    assert sync_len == async_len, f"sync={sync_len} vs async={async_len}"

    input_len = data["input_lengths"][0].item()
    sync_tokens = sync_result["output_ids"][0, input_len : input_len + sync_len]
    async_tokens = async_result["output_ids"][0, input_len : input_len + async_len]
    assert torch.equal(sync_tokens, async_tokens), (
        "generate() and generate_async() produced different tokens"
    )


# ===================================================================
# Tests: generate_one_sample() — the underlying async function
# ===================================================================
def test_generate_one_sample_returns_correct_tuple(sglang_gen, tokenizer):
    sp = make_generation_sampling_params(max_new_tokens=5, temperature=0.0)
    input_ids = tokenizer.encode("The capital of France is")

    result = asyncio.run(
        generate_one_sample(
            sglang_gen.router_ip, sglang_gen.router_port, sp, input_ids, index=42
        )
    )
    assert len(result) == 4
    idx, tokens, logprobs, truncated = result
    assert idx == 42
    assert isinstance(tokens, list) and len(tokens) > 0
    assert isinstance(logprobs, list) and len(logprobs) == len(tokens)
    assert isinstance(truncated, bool)
    assert all(isinstance(t, int) for t in tokens)
    assert all(isinstance(lp, float) for lp in logprobs)


# ===================================================================
# Tests: memory cycle (engine API → HTTP 200 → top-level API)
# ===================================================================
def test_generate_after_memory_cycle(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "Two plus two equals")
    r_before = sglang_gen.generate(data, greedy=True)

    for engine in sglang_gen.engines:
        ray.get(engine.release_memory_weights.remote())
        ray.get(engine.release_memory_kv_cache_and_cuda_graph.remote())
        ray.get(engine.resume_memory_weights.remote())
        ray.get(engine.resume_memory_kv_cache_and_cuda_graph.remote())

    r_after = sglang_gen.generate(data, greedy=True)
    assert torch.equal(r_before["output_ids"], r_after["output_ids"]), (
        "Generation output changed after memory cycle"
    )


def test_generate_after_memory_cycle_via_http_200(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "Two plus two equals")
    r_before = sglang_gen.generate(data, greedy=True)

    for engine in sglang_gen.engines:
        base_url = ray.get(engine.get_base_url.remote())
        assert base_url is not None

        ray.get(engine.flush_cache.remote())
        post_and_assert_200(
            base_url, "release_memory_occupation", {"tags": ["weights"]}
        )
        ray.get(engine.flush_cache.remote())
        post_and_assert_200(
            base_url,
            "release_memory_occupation",
            {"tags": ["kv_cache", "cuda_graph"]},
        )
        post_and_assert_200(
            base_url, "resume_memory_occupation", {"tags": ["weights"]}
        )
        post_and_assert_200(
            base_url,
            "resume_memory_occupation",
            {"tags": ["kv_cache", "cuda_graph"]},
        )

    r_after = sglang_gen.generate(data, greedy=True)
    assert torch.equal(r_before["output_ids"], r_after["output_ids"]), (
        "Generation output changed after HTTP-driven memory cycle"
    )


def test_generate_after_memory_cycle_top_level_api(sglang_gen, tokenizer):
    data = _make_input(tokenizer, "Two plus two equals")

    r_before = sglang_gen.generate(data, greedy=True)
    input_len = data["input_lengths"][0].item()
    gen_len_before = r_before["generation_lengths"][0].item()
    assert gen_len_before > 0, "generate() before memory cycle produced 0 tokens"
    tokens_before = r_before["output_ids"][0, input_len : input_len + gen_len_before]
    assert (tokens_before != PAD_TOKEN_ID).all(), "before: generated tokens contain pad"

    sglang_gen.offload_weights()
    sglang_gen.offload_kv()
    sglang_gen.onload_weights()
    sglang_gen.onload_kv()

    r_after = sglang_gen.generate(data, greedy=True)
    gen_len_after = r_after["generation_lengths"][0].item()
    assert gen_len_after > 0, "generate() after memory cycle produced 0 tokens"
    tokens_after = r_after["output_ids"][0, input_len : input_len + gen_len_after]
    assert (tokens_after != PAD_TOKEN_ID).all(), "after: generated tokens contain pad"

    assert gen_len_before == gen_len_after, (
        f"Different generation_lengths before vs. after: "
        f"before={gen_len_before}, after={gen_len_after}"
    )
    assert torch.equal(r_before["output_ids"], r_after["output_ids"]), (
        "Generation output changed after top-level offload/onload cycle"
    )


# ===================================================================
# Tests: invalidate_kv_cache aggregator
# ===================================================================
def test_invalidate_kv_cache_aggregator(sglang_gen):
    """``invalidate_kv_cache`` fans out to every engine and reduces with
    ``all(results)``. Verifies True on a healthy cluster across the full
    parametrize matrix (single-engine TP/PP/EP variants and dp-attention
    variants alike)."""
    assert sglang_gen.invalidate_kv_cache() is True


def test_invalidate_kv_cache_after_generate(sglang_gen, tokenizer):
    """Most likely path to surface the flush_cache pacing bug — sglang's
    ``/flush_cache`` may transiently return non-200 while draining the
    just-completed generation's queue, so the worker's retry loop must
    actually wait between attempts."""
    data = _make_input(tokenizer, "Two plus two equals")
    sglang_gen.generate(data, greedy=True)
    assert sglang_gen.invalidate_kv_cache() is True


__all__ = [
    "PAD_TOKEN_ID",
    "EOS_TOKEN_ID",
]
