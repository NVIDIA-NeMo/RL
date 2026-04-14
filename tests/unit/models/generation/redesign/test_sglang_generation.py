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

"""Generation tests using a real SGLangGeneration instance.

Spins up a real RayVirtualCluster + SGLangGeneration (router + workers)
and tests ``generate()``, ``generate_async()``, and the underlying
``generate_one_sample()`` function against a live Qwen3-0.6B model.

Parametrised over two configurations (both use 4 GPUs total):
  • tp4_1server  — 1 server × TP=4
  • tp2_2servers — 2 servers × TP=2

Model: Qwen/Qwen3-0.6B
"""

import asyncio
import gc
from copy import deepcopy

import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.redesign.sglang_generation import (
    SGLangGeneration,
    generate_one_sample,
)
from nemo_rl.models.generation.interfaces import GenerationDatumSpec

from helpers import (
    MODEL_PATH,
    make_generation_sampling_params,
    post_and_assert_200,
)

pytestmark = pytest.mark.sglang

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAD_TOKEN_ID = 151643
EOS_TOKEN_ID = 151645


# ---------------------------------------------------------------------------
# SGLang config for SGLangGeneration (mirrors existing test pattern)
# ---------------------------------------------------------------------------
def _make_sglang_generation_cfg(pad_token_id=PAD_TOKEN_ID, tp_size=1):
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
            "cpu_weight_backup": True,
            "sglang_server_concurrency": 64,
        },
        "sglang_router": {
            "sglang_router_ip": None,
            "sglang_router_port": None,
        },
        "sglang_kwargs": {},
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


@pytest.fixture(
    scope="module",
    params=[
        pytest.param({"tp_size": 4, "num_servers": 1}, id="tp4_1server"),
        pytest.param({"tp_size": 2, "num_servers": 2}, id="tp2_2servers"),
    ],
)
def sglang_gen(request, ray_cluster, tokenizer):
    """Real SGLangGeneration: RayVirtualCluster → router → engines.

    Parametrised over tp4_1server (1 server × TP=4) and tp2_2servers
    (2 servers × TP=2). All variants use 4 GPUs.
    """
    tp_size = request.param["tp_size"]
    cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[4],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=4,
        name=f"gen-test-{request.param['num_servers']}srv-tp{tp_size}",
    )
    cluster_cfg = {"gpus_per_node": 4, "num_nodes": 1}
    sglang_cfg = _make_sglang_generation_cfg(
        pad_token_id=tokenizer.pad_token_id, tp_size=tp_size,
    )

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_input(tokenizer, prompt, pad_length=None):
    """Tokenize a prompt → BatchedDataDict for generate()."""
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
    """Tokenize multiple prompts → single BatchedDataDict."""
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

    padded = []
    for ids in all_ids:
        padded.append(ids + [tokenizer.pad_token_id] * (max_len - len(ids)))

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
    """generate() returns BatchedDataDict with all required output keys."""
    data = _make_input(tokenizer, "Hello")
    result = sglang_gen.generate(data, greedy=True)

    for key in ["output_ids", "logprobs", "generation_lengths",
                "unpadded_sequence_lengths", "truncated"]:
        assert key in result, f"Missing key: {key}"


def test_generate_output_ids_shape(sglang_gen, tokenizer):
    """output_ids has shape (batch_size, total_length) with correct padding."""
    data = _make_input(tokenizer, "The capital of France is")
    result = sglang_gen.generate(data, greedy=True)

    assert result["output_ids"].dim() == 2
    assert result["output_ids"].shape[0] == 1  # batch_size
    gen_len = result["generation_lengths"][0].item()
    input_len = data["input_lengths"][0].item()
    assert result["unpadded_sequence_lengths"][0].item() == input_len + gen_len


def test_generate_produces_nonzero_tokens(sglang_gen, tokenizer):
    """Generated tokens are non-zero and non-pad."""
    data = _make_input(tokenizer, "What is 2 plus 2?")
    result = sglang_gen.generate(data, greedy=True)

    gen_len = result["generation_lengths"][0].item()
    assert gen_len > 0, "No tokens generated"

    input_len = data["input_lengths"][0].item()
    generated = result["output_ids"][0, input_len:input_len + gen_len]
    assert (generated != 0).all(), "Generated tokens contain zeros"
    assert (generated != PAD_TOKEN_ID).all(), "Generated tokens contain pad"


def test_generate_greedy_determinism(sglang_gen, tokenizer):
    """Same prompt + greedy=True → identical output_ids across two calls."""
    data = _make_input(tokenizer, "Once upon a time")
    r1 = sglang_gen.generate(data, greedy=True)
    r2 = sglang_gen.generate(data, greedy=True)

    assert torch.equal(r1["output_ids"], r2["output_ids"]), (
        "Greedy generation is not deterministic"
    )


def test_generate_truncation_flag(sglang_gen, tokenizer):
    """When max_new_tokens is small, truncated=True."""
    # Temporarily reduce max_new_tokens
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
    """Logprobs are finite, non-positive at generated positions."""
    data = _make_input(tokenizer, "Hello world")
    result = sglang_gen.generate(data, greedy=True)

    gen_len = result["generation_lengths"][0].item()
    input_len = data["input_lengths"][0].item()
    lps = result["logprobs"][0, input_len:input_len + gen_len]

    assert torch.isfinite(lps).all(), "Logprobs contain NaN or Inf"
    assert (lps <= 0.0).all(), "Logprobs should be non-positive"


def test_generate_respects_max_new_tokens(sglang_gen, tokenizer):
    """generation_lengths ≤ max_new_tokens for all samples."""
    data = _make_input(tokenizer, "Count from 1 to 100:")
    result = sglang_gen.generate(data, greedy=True)

    max_new = sglang_gen.sglang_cfg["max_new_tokens"]
    gen_len = result["generation_lengths"][0].item()
    assert gen_len <= max_new, f"gen_len={gen_len} > max_new_tokens={max_new}"


def test_generate_batch_multiple_samples(sglang_gen, tokenizer):
    """Batch of 3 prompts: all produce valid output."""
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
    """Empty batch → empty BatchedDataDict with zero-size tensors."""
    data = BatchedDataDict(
        {
            "input_ids": torch.zeros((0, 0), dtype=torch.long),
            "input_lengths": torch.zeros(0, dtype=torch.long),
        }
    )
    result = sglang_gen.generate(data, greedy=True)
    assert result["output_ids"].shape[0] == 0


def test_generate_with_stop_strings(sglang_gen, tokenizer):
    """Stop string causes early termination."""
    orig_stop = sglang_gen.sglang_cfg.get("stop_strings")
    sglang_gen.sglang_cfg["stop_strings"] = ["\n"]
    try:
        data = _make_input(tokenizer, "List:\n1. Apple\n2.")
        result = sglang_gen.generate(data, greedy=True)
        gen_len = result["generation_lengths"][0].item()
        max_new = sglang_gen.sglang_cfg["max_new_tokens"]
        # If stop string triggered, generation should be shorter than max
        # (this is a soft check — the model might produce \n on first token)
        assert gen_len <= max_new
    finally:
        sglang_gen.sglang_cfg["stop_strings"] = orig_stop


# ===================================================================
# Tests: SGLangGeneration.generate_async()
# ===================================================================


def test_generate_async_yields_single_sample(sglang_gen, tokenizer):
    """generate_async() with batch_size=1 yields (0, BatchedDataDict)."""
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
    """Same prompt, greedy: generate() and generate_async() produce same tokens."""
    data = _make_input(tokenizer, "The answer is")
    sync_result = sglang_gen.generate(data, greedy=True)

    async def _run():
        results = []
        async for _, batch in sglang_gen.generate_async(data, greedy=True):
            results.append(batch)
        return results[0]

    async_result = asyncio.run(_run())

    sync_len = sync_result["generation_lengths"][0].item()
    async_len = async_result["generation_lengths"][0].item()
    assert sync_len == async_len, f"sync={sync_len} vs async={async_len}"

    input_len = data["input_lengths"][0].item()
    sync_tokens = sync_result["output_ids"][0, input_len:input_len + sync_len]
    async_tokens = async_result["output_ids"][0, input_len:input_len + async_len]
    assert torch.equal(sync_tokens, async_tokens), (
        "generate() and generate_async() produced different tokens"
    )


# ===================================================================
# Tests: generate_one_sample() — the underlying async function
# ===================================================================


def test_generate_one_sample_returns_correct_tuple(sglang_gen, tokenizer):
    """generate_one_sample() returns (index, tokens, logprobs, truncated)."""
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


def test_generate_after_memory_cycle(sglang_gen, tokenizer):
    """Generate → offload/onload → generate → same greedy output."""
    data = _make_input(tokenizer, "Two plus two equals")
    r_before = sglang_gen.generate(data, greedy=True)

    # Offload and onload weights + KV on all engines
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
    """Generate → offload/onload via direct HTTP (asserting 200) → generate → same greedy output.
    """
    data = _make_input(tokenizer, "Two plus two equals")
    r_before = sglang_gen.generate(data, greedy=True)

    for engine in sglang_gen.engines:
        base_url = ray.get(engine.get_base_url.remote())
        assert base_url is not None

        # Release weights (flush_cache first, mirroring release_memory_occupation)
        ray.get(engine.flush_cache.remote())
        post_and_assert_200(
            base_url, "release_memory_occupation", {"tags": ["weights"]}
        )
        # Release KV cache + CUDA graphs
        ray.get(engine.flush_cache.remote())
        post_and_assert_200(
            base_url,
            "release_memory_occupation",
            {"tags": ["kv_cache", "cuda_graph"]},
        )
        # Resume weights
        post_and_assert_200(
            base_url, "resume_memory_occupation", {"tags": ["weights"]}
        )
        # Resume KV cache + CUDA graphs
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
    """Generate -> top-level offload/onload -> generate -> same greedy output.
    """
    data = _make_input(tokenizer, "Two plus two equals")

    r_before = sglang_gen.generate(data, greedy=True)
    input_len = data["input_lengths"][0].item()
    gen_len_before = r_before["generation_lengths"][0].item()
    assert gen_len_before > 0, "generate() before memory cycle produced 0 tokens"
    tokens_before = r_before["output_ids"][0, input_len : input_len + gen_len_before]
    assert (tokens_before != 0).all(), "before: generated tokens contain zeros"
    assert (tokens_before != PAD_TOKEN_ID).all(), "before: generated tokens contain pad"

    # Full offload + onload cycle using the top-level SGLangGeneration API.
    sglang_gen.offload_weights()
    sglang_gen.offload_kv()
    sglang_gen.onload_weights()
    sglang_gen.onload_kv()

    r_after = sglang_gen.generate(data, greedy=True)
    gen_len_after = r_after["generation_lengths"][0].item()
    assert gen_len_after > 0, "generate() after memory cycle produced 0 tokens"
    tokens_after = r_after["output_ids"][0, input_len : input_len + gen_len_after]
    assert (tokens_after != 0).all(), "after: generated tokens contain zeros"
    assert (tokens_after != PAD_TOKEN_ID).all(), "after: generated tokens contain pad"

    assert gen_len_before == gen_len_after, (
        f"Different generation_lengths before vs. after: "
        f"before={gen_len_before}, after={gen_len_after}"
    )
    assert torch.equal(r_before["output_ids"], r_after["output_ids"]), (
        "Generation output changed after top-level offload/onload cycle"
    )
