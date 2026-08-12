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

"""Tests for dLLM policy config resolution and the setup-time guards."""

import pytest

from nemo_rl.models.policy.dllm import (
    DllmConfig,
    dllm_config_from_policy,
    validate_dllm_policy,
)

VALID_LOSS = {
    "sequence_level_importance_ratios": True,
    "token_level_loss": False,
    "use_importance_sampling_correction": False,
    "position_aligned_logprobs": True,
}


def make_policy(dllm=True, **overrides):
    cfg = {
        "dtensor_cfg": {"enabled": True, "context_parallel_size": 1},
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "megatron_cfg": {"enabled": False},
        "router_replay": {"enabled": False},
    }
    if dllm:
        cfg["dllm"] = {"enabled": True, "mask_id": 126336}
    cfg.update(overrides)
    return cfg


def test_absent_block_resolves_to_none():
    assert dllm_config_from_policy(make_policy(dllm=False)) is None


def test_disabled_block_resolves_to_none():
    """Present-but-disabled must behave exactly like absent."""
    policy = make_policy()
    policy["dllm"]["enabled"] = False
    assert dllm_config_from_policy(policy) is None


def test_raw_dict_is_coerced_with_basemodel_defaults():
    """Defaults come from DllmConfig, not from the call site."""
    cfg = dllm_config_from_policy(make_policy())
    assert isinstance(cfg, DllmConfig)
    assert cfg.quadrature == "gauss-2"
    assert cfg.mc_samples == 1
    assert cfg.shift_targets is False


def test_already_validated_config_passes_through():
    policy = make_policy()
    policy["dllm"] = DllmConfig(enabled=True, mask_id=1, quadrature="gauss-5")
    assert dllm_config_from_policy(policy).quadrature == "gauss-5"


def test_valid_config_passes_validation():
    validate_dllm_policy(make_policy(), VALID_LOSS)


def test_non_dllm_policy_is_not_constrained():
    """None of the dLLM restrictions apply to an ordinary autoregressive run."""
    policy = make_policy(dllm=False, sequence_packing={"enabled": True})
    validate_dllm_policy(policy, {"token_level_loss": True})


@pytest.mark.parametrize(
    "section", ["sequence_packing", "dynamic_batching", "megatron_cfg", "router_replay"]
)
def test_causal_only_features_are_rejected(section):
    policy = make_policy(**{section: {"enabled": True}})
    with pytest.raises(ValueError, match=f"policy.{section} is not supported"):
        validate_dllm_policy(policy, VALID_LOSS)


def test_context_parallelism_is_rejected():
    policy = make_policy(dtensor_cfg={"enabled": True, "context_parallel_size": 2})
    with pytest.raises(ValueError, match="context_parallel_size"):
        validate_dllm_policy(policy, VALID_LOSS)


def test_context_parallel_size_one_is_allowed():
    policy = make_policy(dtensor_cfg={"enabled": True, "context_parallel_size": 1})
    validate_dllm_policy(policy, VALID_LOSS)


def test_token_level_ratios_are_rejected():
    loss = dict(VALID_LOSS, sequence_level_importance_ratios=False)
    with pytest.raises(ValueError, match="sequence_level_importance_ratios=true"):
        validate_dllm_policy(make_policy(), loss)


def test_token_level_loss_is_rejected():
    loss = dict(VALID_LOSS, token_level_loss=True)
    with pytest.raises(ValueError, match="token_level_loss=false"):
        validate_dllm_policy(make_policy(), loss)


def test_importance_sampling_correction_is_rejected():
    """Denoising produces no per-token sampling logprobs to correct against."""
    loss = dict(VALID_LOSS, use_importance_sampling_correction=True)
    with pytest.raises(ValueError, match="use_importance_sampling_correction"):
        validate_dllm_policy(make_policy(), loss)


def test_loss_config_may_be_an_object_not_a_mapping():
    """loss_fn arrives as a ClippedPGLossConfig BaseModel in the real pipeline."""

    class LossObj:
        sequence_level_importance_ratios = True
        token_level_loss = False
        use_importance_sampling_correction = False
        position_aligned_logprobs = True

    validate_dllm_policy(make_policy(), LossObj())


def test_mismatched_microbatch_sizes_are_rejected():
    """Different microbatch shapes give the two passes different masks."""
    policy = make_policy(logprob_batch_size=2, train_micro_batch_size=4)
    with pytest.raises(ValueError, match="must equal"):
        validate_dllm_policy(policy, VALID_LOSS)


def test_matching_microbatch_sizes_are_allowed():
    policy = make_policy(logprob_batch_size=4, train_micro_batch_size=4)
    validate_dllm_policy(policy, VALID_LOSS)


def test_microbatch_guard_does_not_apply_without_dllm():
    policy = make_policy(dllm=False, logprob_batch_size=2, train_micro_batch_size=8)
    validate_dllm_policy(policy, VALID_LOSS)


def test_loss_must_not_shift_when_policy_is_position_aligned():
    """The default loss shift is an off-by-one against dLLM logprobs."""
    loss = dict(VALID_LOSS, position_aligned_logprobs=False)
    with pytest.raises(ValueError, match="position_aligned_logprobs"):
        validate_dllm_policy(make_policy(), loss)


def test_loss_must_shift_when_policy_emits_next_token_logprobs():
    """The mirror case: a shifting policy needs the loss to shift too."""
    policy = make_policy()
    policy["dllm"]["shift_targets"] = True
    with pytest.raises(ValueError, match="position_aligned_logprobs"):
        validate_dllm_policy(policy, VALID_LOSS)


def test_consistent_shifting_pair_is_allowed():
    policy = make_policy()
    policy["dllm"]["shift_targets"] = True
    validate_dllm_policy(policy, dict(VALID_LOSS, position_aligned_logprobs=False))


VALID_GENERATION = {
    "backend": "dllm",
    "colocated": {"enabled": True},
    "max_new_tokens": 128,
}


def make_generation(**overrides):
    cfg = dict(VALID_GENERATION)
    cfg["colocated"] = dict(VALID_GENERATION["colocated"])
    cfg.update(overrides)
    return cfg


def test_a_dllm_generation_block_passes_validation():
    validate_dllm_policy(make_policy(generation=make_generation()), VALID_LOSS)


def test_an_absent_generation_block_is_not_validated():
    """Logprob-only entrypoints (SFT, evaluation) configure no rollouts."""
    policy = make_policy()
    policy["generation"] = None
    validate_dllm_policy(policy, VALID_LOSS)


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm", "megatron"])
def test_autoregressive_backends_are_rejected(backend):
    policy = make_policy(generation=make_generation(backend=backend))
    with pytest.raises(ValueError, match="have no KV cache"):
        validate_dllm_policy(policy, VALID_LOSS)


def test_non_colocated_generation_is_rejected():
    policy = make_policy(generation=make_generation())
    policy["generation"]["colocated"]["enabled"] = False
    with pytest.raises(ValueError, match="colocated.enabled must be true"):
        validate_dllm_policy(policy, VALID_LOSS)


def test_max_new_tokens_must_be_a_multiple_of_the_block_length():
    policy = make_policy(generation=make_generation(max_new_tokens=100))
    with pytest.raises(ValueError, match="multiple of policy.dllm.block_length"):
        validate_dllm_policy(policy, VALID_LOSS)


def test_a_matching_custom_block_length_is_accepted():
    policy = make_policy(generation=make_generation(max_new_tokens=96))
    policy["dllm"]["block_length"] = 48
    validate_dllm_policy(policy, VALID_LOSS)


def test_generation_is_not_validated_when_dllm_is_disabled():
    """A non-dLLM policy keeps whatever generation backend it configured."""
    policy = make_policy(dllm=False, generation=make_generation(backend="vllm"))
    validate_dllm_policy(policy, VALID_LOSS)


class _Async:
    def __init__(self, enabled):
        self.enabled = enabled


class _Grpo:
    def __init__(self, enabled):
        self.async_grpo = _Async(enabled)


def test_async_grpo_is_rejected_for_dllm():
    """Async rollouts would silently fall back to synchronous ones."""
    policy = make_policy(generation=make_generation())
    with pytest.raises(ValueError, match="async_grpo.enabled is not supported"):
        validate_dllm_policy(policy, VALID_LOSS, _Grpo(True))


def test_sync_grpo_is_accepted_for_dllm():
    validate_dllm_policy(
        make_policy(generation=make_generation()), VALID_LOSS, _Grpo(False)
    )


def test_async_grpo_as_a_plain_mapping_is_also_rejected():
    policy = make_policy(generation=make_generation())
    with pytest.raises(ValueError, match="async_grpo.enabled is not supported"):
        validate_dllm_policy(policy, VALID_LOSS, {"async_grpo": {"enabled": True}})


def test_an_omitted_grpo_config_skips_the_async_check():
    """Callers without a grpo section (SFT, eval) must still validate."""
    validate_dllm_policy(make_policy(generation=make_generation()), VALID_LOSS)


def test_async_grpo_is_ignored_when_dllm_is_disabled():
    policy = make_policy(dllm=False, generation=make_generation(backend="vllm"))
    validate_dllm_policy(policy, VALID_LOSS, _Grpo(True))
