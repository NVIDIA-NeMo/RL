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

"""Tests for masked-diffusion config resolution and setup-time guards."""

import copy

import pytest
from pydantic import ValidationError

from gdpo import (
    DenoiseConfig,
    MaskedDiffusionConfig,
    SdmcLikelihoodConfig,
    masked_diffusion_config_from_policy,
    validate_gdpo_config,
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
        cfg["masked_diffusion"] = {"enabled": True, "mask_id": 126336}
    cfg.update(overrides)
    return cfg


def test_absent_block_resolves_to_none():
    assert masked_diffusion_config_from_policy(make_policy(dllm=False)) is None


def test_disabled_block_resolves_to_none():
    """Present-but-disabled must behave exactly like absent."""
    policy = make_policy()
    policy["masked_diffusion"]["enabled"] = False
    assert masked_diffusion_config_from_policy(policy) is None


def test_raw_dict_is_coerced_with_basemodel_defaults():
    """Defaults come from MaskedDiffusionConfig, not from the call site."""
    cfg = masked_diffusion_config_from_policy(make_policy())
    assert isinstance(cfg, MaskedDiffusionConfig)
    assert cfg.likelihood.quadrature == "gauss-2"
    assert cfg.likelihood.mc_samples == 1
    assert cfg.shift_targets is False


def test_already_validated_config_passes_through():
    policy = make_policy()
    policy["masked_diffusion"] = MaskedDiffusionConfig(
        enabled=True,
        mask_id=1,
        likelihood={"quadrature": "gauss-5"},
    )
    assert (
        masked_diffusion_config_from_policy(policy).likelihood.quadrature == "gauss-5"
    )


def test_valid_config_passes_validation():
    validate_gdpo_config(make_policy(), VALID_LOSS)


def test_non_dllm_policy_is_not_constrained():
    """None of the dLLM restrictions apply to an ordinary autoregressive run."""
    policy = make_policy(dllm=False, sequence_packing={"enabled": True})
    validate_gdpo_config(policy, {"token_level_loss": True})


@pytest.mark.parametrize(
    "section", ["sequence_packing", "dynamic_batching", "megatron_cfg", "router_replay"]
)
def test_causal_only_features_are_rejected(section):
    policy = make_policy(**{section: {"enabled": True}})
    with pytest.raises(ValueError, match=f"policy.{section} is not supported"):
        validate_gdpo_config(policy, VALID_LOSS)


def test_context_parallelism_is_rejected():
    policy = make_policy(dtensor_cfg={"enabled": True, "context_parallel_size": 2})
    with pytest.raises(ValueError, match="context_parallel_size"):
        validate_gdpo_config(policy, VALID_LOSS)


def test_context_parallel_size_one_is_allowed():
    policy = make_policy(dtensor_cfg={"enabled": True, "context_parallel_size": 1})
    validate_gdpo_config(policy, VALID_LOSS)


def test_token_level_ratios_are_rejected():
    loss = dict(VALID_LOSS, sequence_level_importance_ratios=False)
    with pytest.raises(ValueError, match="sequence_level_importance_ratios=true"):
        validate_gdpo_config(make_policy(), loss)


def test_token_level_loss_is_rejected():
    loss = dict(VALID_LOSS, token_level_loss=True)
    with pytest.raises(ValueError, match="token_level_loss=false"):
        validate_gdpo_config(make_policy(), loss)


@pytest.mark.parametrize(
    "setting,value,error",
    [
        (
            "use_importance_sampling_correction",
            True,
            "use_importance_sampling_correction",
        ),
        ("reference_policy_kl_penalty", 0.1, "reference_policy_kl_penalty must be 0"),
        ("use_kl_in_reward", True, "use_kl_in_reward is not supported"),
    ],
)
def test_token_logprob_dependent_loss_options_are_rejected(setting, value, error):
    loss = dict(VALID_LOSS, **{setting: value})
    with pytest.raises(ValueError, match=error):
        validate_gdpo_config(make_policy(), loss)


def test_loss_config_may_be_an_object_not_a_mapping():
    """loss_fn arrives as a ClippedPGLossConfig BaseModel in the real pipeline."""

    class LossObj:
        sequence_level_importance_ratios = True
        token_level_loss = False
        use_importance_sampling_correction = False
        position_aligned_logprobs = True

    validate_gdpo_config(make_policy(), LossObj())


def test_loss_must_not_shift_when_policy_is_position_aligned():
    """The default loss shift is an off-by-one against dLLM logprobs."""
    loss = dict(VALID_LOSS, position_aligned_logprobs=False)
    with pytest.raises(ValueError, match="position_aligned_logprobs"):
        validate_gdpo_config(make_policy(), loss)


def test_loss_must_shift_when_policy_emits_next_token_logprobs():
    """The mirror case: a shifting policy needs the loss to shift too."""
    policy = make_policy()
    policy["masked_diffusion"]["shift_targets"] = True
    with pytest.raises(ValueError, match="position_aligned_logprobs"):
        validate_gdpo_config(policy, VALID_LOSS)


def test_consistent_shifting_pair_is_allowed():
    policy = make_policy()
    policy["masked_diffusion"]["shift_targets"] = True
    validate_gdpo_config(policy, dict(VALID_LOSS, position_aligned_logprobs=False))


VALID_GENERATION = {
    "backend": "automodel",
    "colocated": {"enabled": True},
    "max_new_tokens": 128,
    "denoise_cfg": {"type": "block", "block_length": 32},
}


def make_generation(**overrides):
    cfg = copy.deepcopy(VALID_GENERATION)
    cfg.update(overrides)
    return cfg


def test_a_dllm_generation_block_passes_validation():
    validate_gdpo_config(make_policy(generation=make_generation()), VALID_LOSS)


def test_an_absent_generation_block_is_not_validated():
    """Logprob-only entrypoints (SFT, evaluation) configure no rollouts."""
    policy = make_policy()
    policy["generation"] = None
    validate_gdpo_config(policy, VALID_LOSS)


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm", "megatron"])
def test_autoregressive_backends_are_rejected(backend):
    policy = make_policy(generation=make_generation(backend=backend))
    with pytest.raises(ValueError, match="have no KV cache"):
        validate_gdpo_config(policy, VALID_LOSS)


def test_non_colocated_generation_is_rejected():
    policy = make_policy(generation=make_generation())
    policy["generation"]["colocated"]["enabled"] = False
    with pytest.raises(ValueError, match="colocated.enabled must be true"):
        validate_gdpo_config(policy, VALID_LOSS)


def test_max_new_tokens_must_be_a_multiple_of_the_block_length():
    policy = make_policy(generation=make_generation(max_new_tokens=100))
    with pytest.raises(ValueError, match="multiple of policy.generation.denoise_cfg"):
        validate_gdpo_config(policy, VALID_LOSS)


def test_stop_strings_are_rejected():
    policy = make_policy(generation=make_generation(stop_strings=["</answer>"]))
    with pytest.raises(ValueError, match="stop_strings is not supported"):
        validate_gdpo_config(policy, VALID_LOSS)


def test_a_matching_custom_block_length_is_accepted():
    policy = make_policy(generation=make_generation(max_new_tokens=96))
    policy["generation"]["denoise_cfg"]["block_length"] = 48
    validate_gdpo_config(policy, VALID_LOSS)


def test_generation_is_not_validated_when_dllm_is_disabled():
    """A non-dLLM policy keeps whatever generation backend it configured."""
    policy = make_policy(dllm=False, generation=make_generation(backend="vllm"))
    validate_gdpo_config(policy, VALID_LOSS)


class _Async:
    def __init__(self, enabled):
        self.enabled = enabled


class _Grpo:
    def __init__(self, enabled, seq_logprob_error_threshold=None):
        self.async_grpo = _Async(enabled)
        self.seq_logprob_error_threshold = seq_logprob_error_threshold


def test_async_grpo_is_rejected_for_dllm():
    """Async rollouts would silently fall back to synchronous ones."""
    policy = make_policy(generation=make_generation())
    with pytest.raises(ValueError, match="async_grpo.enabled is not supported"):
        validate_gdpo_config(policy, VALID_LOSS, _Grpo(True))


def test_sync_grpo_is_accepted_for_dllm():
    validate_gdpo_config(
        make_policy(generation=make_generation()), VALID_LOSS, _Grpo(False)
    )


def test_seq_logprob_error_threshold_is_rejected_for_dllm():
    policy = make_policy(generation=make_generation())
    with pytest.raises(ValueError, match="seq_logprob_error_threshold"):
        validate_gdpo_config(policy, VALID_LOSS, _Grpo(False, 1.05))


def test_async_grpo_as_a_plain_mapping_is_also_rejected():
    policy = make_policy(generation=make_generation())
    with pytest.raises(ValueError, match="async_grpo.enabled is not supported"):
        validate_gdpo_config(policy, VALID_LOSS, {"async_grpo": {"enabled": True}})


def test_an_omitted_grpo_config_skips_the_async_check():
    """Callers without a grpo section (SFT, eval) must still validate."""
    validate_gdpo_config(make_policy(generation=make_generation()), VALID_LOSS)


@pytest.mark.parametrize(
    "model,field,value",
    [
        (MaskedDiffusionConfig, "mask_id", -1),
        (SdmcLikelihoodConfig, "quadrature", "gauss-99"),
        (SdmcLikelihoodConfig, "mc_samples", 0),
        (SdmcLikelihoodConfig, "p_mask_prompt", -0.1),
        (SdmcLikelihoodConfig, "p_mask_prompt", 1.1),
        (DenoiseConfig, "block_length", 0),
        (DenoiseConfig, "diffusion_steps", 0),
        (DenoiseConfig, "cfg_scale", -0.1),
    ],
)
def test_invalid_diffusion_config_values_are_rejected(model, field, value):
    with pytest.raises(ValidationError):
        model(**{field: value})


def test_async_grpo_is_ignored_when_dllm_is_disabled():
    policy = make_policy(dllm=False, generation=make_generation(backend="vllm"))
    validate_gdpo_config(policy, VALID_LOSS, _Grpo(True))
