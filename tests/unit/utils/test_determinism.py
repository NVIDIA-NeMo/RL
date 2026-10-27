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

import asyncio
import os
import random
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo_rl.utils.determinism import (
    DETERMINISTIC_ENV,
    capture_rng_state,
    configure_grpo_determinism,
    keyed_seed,
    restore_rng_state,
)


@pytest.fixture(autouse=True)
def preserve_process_state(monkeypatch):
    state = capture_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    for key in (*DETERMINISTIC_ENV, "PYTHONHASHSEED"):
        if key in os.environ:
            monkeypatch.setenv(key, os.environ[key])
        else:
            monkeypatch.delenv(key, raising=False)
    yield
    restore_rng_state(state)
    torch.use_deterministic_algorithms(deterministic)


def test_rng_round_trip():
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(10))
    for _ in range(13):
        random.random()
        np.random.rand()
        torch.rand(17)
    restore_rng_state(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(10), expected[2])


def test_seed_identity_and_domain_separation():
    keys = [
        keyed_seed(42, phase, step, sample, turn)
        for phase in ("train", "validation")
        for step in range(3)
        for sample in range(100)
        for turn in range(4)
    ]
    assert len(set(keys)) == len(keys)
    assert all(0 <= key < 2**63 for key in keys)
    assert keyed_seed(42, "train", 1, 2) == keyed_seed(42, "train", 1, 2)
    assert keyed_seed(42, "12", 3) != keyed_seed(42, "1", 23)


def test_disabled_does_not_change_rng_or_environment(monkeypatch):
    monkeypatch.setenv("NCCL_ALGO", "Tree")
    state = torch.get_rng_state().clone()
    configure_grpo_determinism(
        SimpleNamespace(grpo=SimpleNamespace(deterministic=False))
    )
    assert os.environ["NCCL_ALGO"] == "Tree"
    assert torch.equal(state, torch.get_rng_state())


@pytest.fixture
def recipe():
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    return MasterConfig(
        **OmegaConf.to_container(
            load_config(
                "examples/configs/recipes/llm/grpo-qwen2.5-0.5b-1n4g-megatron-deterministic.yaml"
            ),
            resolve=True,
        )
    )


def test_recipe_resolves_before_actor_creation(recipe, monkeypatch):
    monkeypatch.setattr(
        "nemo_rl.utils.determinism.configure_torch_determinism", lambda: None
    )
    configure_grpo_determinism(recipe)
    mcfg = recipe.policy["megatron_cfg"]
    assert mcfg["deterministic_mode"] and mcfg["batch_invariant_mode"]
    assert mcfg["seed"] == recipe.grpo.seed
    assert mcfg["env_vars"]["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    assert mcfg["batch_invariant_backend"] == "te_native"
    assert recipe.policy["make_sequence_length_divisible_by"] % 64 == 0
    for key in (
        "MAMBA_DETERMINISTIC",
        "CAUSAL_CONV1D_DETERMINISTIC",
        "TRITON_CACHE_AUTOTUNING",
    ):
        assert mcfg["env_vars"][key] == DETERMINISTIC_ENV[key]
        assert (
            recipe.policy["generation"]["mcore_generation_config"]["env_vars"][key]
            == DETERMINISTIC_ENV[key]
        )
    assert recipe.env["math"]["deterministic"]
    assert recipe.policy["generation"]["mcore_generation_config"][
        "request_seeded_sampling"
    ]
    assert (
        recipe.policy["generation"]["mcore_generation_config"]["logprobs_mode"]
        == "raw_logprobs"
    )
    configure_grpo_determinism(recipe)  # setup repeats the driver normalization.


@pytest.mark.parametrize("graphs", [False, True])
@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize("chunked_prefill", [False, True])
def test_determinism_preserves_inference_optimizations(
    recipe, graphs, prefix_caching, chunked_prefill
):
    inference = recipe.policy["generation"]["mcore_generation_config"]
    choices = {
        "cuda_graph_impl": "local" if graphs else "none",
        "num_cuda_graphs": 4 if graphs else None,
        "inference_cuda_graph_scope": "block" if graphs else "none",
        "use_cuda_graphs_for_non_decode_steps": graphs,
        "enable_prefix_caching": prefix_caching,
        "enable_chunked_prefill": chunked_prefill,
    }
    inference.update(choices)
    configure_grpo_determinism(recipe)
    assert {key: inference[key] for key in choices} == choices
    assert inference["batch_invariant_mode"]
    assert inference["request_seeded_sampling"]


def test_speculative_generation_remains_unsupported(recipe):
    recipe.policy["generation"]["mcore_generation_config"]["num_speculative_tokens"] = 1
    with pytest.raises(ValueError, match="speculative generation"):
        configure_grpo_determinism(recipe)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "environments",
    [
        {"gym": {"num_workers": 2}},
        {"math": {"math_verify_impl": "dapo_math_verify"}},
        {"math": {"math_verify_impl": "hf_math_verify", "verifier_type": "other"}},
    ],
)
def test_unverified_environments_warn_without_disabling_determinism(
    recipe, asynchronous, environments
):
    recipe.env = deepcopy(environments)
    recipe.grpo.async_grpo.enabled = asynchronous
    recipe.policy["generation"]["colocated"]["enabled"] = False
    with pytest.warns(
        UserWarning, match="end-to-end reproducibility cannot be guaranteed"
    ):
        configure_grpo_determinism(recipe)
    assert recipe.env == environments
    assert recipe.policy["megatron_cfg"]["deterministic_mode"]
    assert recipe.policy["generation"]["mcore_generation_config"][
        "request_seeded_sampling"
    ]


def test_mixed_environments_keep_math_determinism(recipe):
    recipe.env["gym"] = {"num_workers": 2}
    with pytest.warns(UserWarning, match="environments: gym"):
        configure_grpo_determinism(recipe)
    assert recipe.env["math"]["deterministic"]
    assert recipe.env["gym"] == {"num_workers": 2}


def test_native_preserves_router_precision_and_parallel_alignment(recipe):
    recipe.policy["megatron_cfg"]["moe_router_dtype"] = "fp64"
    recipe.policy["make_sequence_length_divisible_by"] = 3
    configure_grpo_determinism(recipe)
    assert recipe.policy["megatron_cfg"]["moe_router_dtype"] == "fp64"
    assert recipe.policy["make_sequence_length_divisible_by"] == 192


def test_triton_resolves_supported_router_precision(recipe):
    recipe.policy["megatron_cfg"]["batch_invariant_backend"] = "triton"
    recipe.policy["megatron_cfg"]["moe_router_dtype"] = "fp64"
    configure_grpo_determinism(recipe)
    assert recipe.policy["megatron_cfg"]["moe_router_dtype"] == "fp32"


def test_native_grouped_tensor_rejects_unsupported_probability_padding(recipe):
    recipe.policy["megatron_cfg"]["model_overrides"] = {"moe_use_grouped_tensor": True}
    recipe.policy["megatron_cfg"]["moe_router_dtype"] = "fp64"
    with pytest.raises(ValueError, match="probability padding"):
        configure_grpo_determinism(recipe)


def test_async_uses_fixed_lag(recipe):
    recipe.grpo.async_grpo.enabled = True
    recipe.policy["generation"]["colocated"]["enabled"] = False
    configure_grpo_determinism(recipe)
    from nemo_rl.models.generation.interfaces import should_use_async_rollouts

    assert should_use_async_rollouts(recipe.policy["generation"])


@pytest.mark.parametrize(
    "setting,value,message",
    [
        ("max_trajectory_age_steps", 2, "one-step lag"),
        ("in_flight_weight_updates", True, "one-step lag"),
        ("max_generation_failures", 1, "drop failed"),
    ],
)
def test_async_rejects_nondeterministic_scheduling(recipe, setting, value, message):
    recipe.grpo.async_grpo.enabled = True
    setattr(recipe.grpo.async_grpo, setting, value)
    with pytest.raises(ValueError, match=message):
        configure_grpo_determinism(recipe)


def test_async_checkpointing_is_explicitly_unsupported(recipe):
    recipe.grpo.async_grpo.enabled = True
    recipe.policy["generation"]["colocated"]["enabled"] = False
    recipe.checkpointing["enabled"] = True
    with pytest.raises(ValueError, match="checkpoint/resume"):
        configure_grpo_determinism(recipe)


def test_cpu_optimizer_is_explicitly_unsupported(recipe):
    recipe.policy["megatron_cfg"]["optimizer"]["optimizer_cpu_offload"] = True
    with pytest.raises(ValueError, match="GPU-only"):
        configure_grpo_determinism(recipe)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_rejects_seeds_invalid_for_hash_or_numpy(recipe, seed):
    recipe.grpo.seed = seed
    with pytest.raises(ValueError, match="seed in"):
        configure_grpo_determinism(recipe)


def test_seeded_async_rollout_propagates_generation_failure(monkeypatch):
    from nemo_rl.experience import rollouts

    async def unavailable(*args, **kwargs):
        assert kwargs["sampling_seed"] == keyed_seed(42, "turn", 0)
        raise RuntimeError("generation unavailable")

    monkeypatch.setattr(
        rollouts, "async_generate_response_for_sample_turn", unavailable
    )
    with pytest.raises(RuntimeError, match="generation unavailable"):
        asyncio.run(
            rollouts.run_sample_multi_turn_rollout(
                sample_idx=0,
                initial_sample_state={
                    "message_log": [],
                    "extra_env_info": {},
                    "task_name": "math",
                    "sampling_seed": 42,
                },
                policy_generation=None,
                tokenizer=None,
                task_to_env={},
                max_seq_len=16,
                max_rollout_turns=1,
            )
        )


@pytest.mark.parametrize(
    "max_turns,truncated,release_before_reward",
    [(1, False, True), (3, True, True), (3, False, False)],
)
def test_generation_release_obeys_multi_turn_lifetime(
    monkeypatch, max_turns, truncated, release_before_reward
):
    from nemo_rl.experience import rollouts

    released = []
    calls = []

    async def generate(*args, **kwargs):
        calls.append(kwargs["sampling_seed"])
        return [], torch.tensor([7]), 1, {"_response_truncated": [truncated]}

    def reward(*args):
        assert bool(released) == release_before_reward
        return SimpleNamespace(
            rewards=torch.tensor([1.0]),
            terminateds=torch.tensor([True]),
            observations=[{"content": "", "role": "environment"}],
            next_stop_strings=[None],
            metadata=[None],
        )

    monkeypatch.setattr(rollouts, "async_generate_response_for_sample_turn", generate)
    monkeypatch.setattr(rollouts, "calculate_rewards", reward)
    asyncio.run(
        rollouts.run_sample_multi_turn_rollout(
            sample_idx=5,
            initial_sample_state={
                "message_log": [],
                "extra_env_info": {},
                "task_name": "math",
                "sampling_seed": 42,
            },
            policy_generation=None,
            tokenizer=lambda *args, **kwargs: SimpleNamespace(
                input_ids=torch.empty((1, 0), dtype=torch.long)
            ),
            task_to_env={},
            max_seq_len=16,
            max_rollout_turns=max_turns,
            on_generation_complete=released.append,
        )
    )
    assert released == [5]
    assert calls == [keyed_seed(42, "turn", 0)]


@pytest.mark.parametrize(
    "recipe_name",
    [
        "grpo-qwen2.5-0.5b-1n4g-megatron-deterministic.yaml",
        "grpo-qwen2.5-0.5b-1n4g-megatron-deterministic-async.yaml",
    ],
)
def test_recipe_matches_main_batch_invariant_contract(recipe_name):
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.models.megatron.zero_train_gen_mismatch import (
        validate_batch_invariant_mode,
    )
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

    register_omegaconf_resolvers()
    config = MasterConfig(
        **OmegaConf.to_container(
            load_config("examples/configs/recipes/llm/" + recipe_name), resolve=True
        )
    )
    configure_grpo_determinism(config)
    assert validate_batch_invariant_mode(config.policy).violations == []
