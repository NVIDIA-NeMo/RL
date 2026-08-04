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

    validate_dllm_policy(make_policy(), LossObj())
