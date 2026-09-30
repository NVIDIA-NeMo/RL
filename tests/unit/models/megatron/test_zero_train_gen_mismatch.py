"""Unit tests for the zero train/generation KL preset (`zero_train_gen_mismatch`).

Everything here is CPU-only. Megatron-Core, Transformer Engine and FlashAttention
are replaced by small fake modules so the resolve / validate / enable logic runs
in any environment, and the shipped zero-KL recipes are loaded from disk and
pushed through the same gates the workers run at startup.
"""

import copy
import sys
import types
import warnings
from dataclasses import make_dataclass
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf
from packaging.version import Version

from nemo_rl.models.megatron import zero_train_gen_mismatch as zgm
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

CONFIGS_DIR = Path(__file__).resolve().parents[4] / "examples" / "configs"

ZERO_KL_EXEMPLARS = [
    "grpo_qwen3_30ba3b_megatron_zero_train_gen_kl.yaml",
    "grpo_qwen3_30ba3b_megatron_zero_train_gen_kl_colocated.yaml",
]

# Installed-package versions that satisfy every gate in `_validate_packages`.
GOOD_VERSIONS = {
    "transformer_engine": Version("2.18.0"),
    "flash_attn": Version("2.8.1"),
    "flash-attn-4": Version("4.0.0b19"),
    "nvidia-cutlass-dsl": Version("4.6.2"),
}


def _policy_config() -> dict[str, Any]:
    """Smallest policy config that passes every zero-KL gate."""
    return {
        "precision": "bfloat16",
        "sequence_packing": {"enabled": False},
        "megatron_cfg": {
            "zero_train_gen_mismatch": True,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 4,
            "sequence_parallel": True,
        },
        "generation": {
            "backend": "megatron",
            "mcore_generation_config": {"transformer_impl": "inference_optimized"},
        },
    }


def _resolved_config() -> dict[str, Any]:
    config = _policy_config()
    zgm.resolve_zero_train_gen_mismatch(config)
    return config


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch, dotted_name: str, leaf: types.ModuleType
) -> None:
    """Register `leaf` as `dotted_name`, with fake parent packages linked by attribute."""
    parts = dotted_name.split(".")
    parent: types.ModuleType | None = None
    for index in range(len(parts)):
        name = ".".join(parts[: index + 1])
        if index == len(parts) - 1:
            module = leaf
        else:
            module = types.ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
        if parent is not None:
            setattr(parent, parts[index], module)
        parent = module


def _set_versions(
    monkeypatch: pytest.MonkeyPatch, versions: dict[str, Version]
) -> None:
    monkeypatch.setattr(zgm, "_package_version", lambda name: versions.get(name))
    monkeypatch.setattr(zgm, "_validate_megatron_core_features", lambda out: None)


# --------------------------------------------------------------------------- #
# ZeroTrainGenValidation
# --------------------------------------------------------------------------- #


def test_raise_if_invalid_is_silent_without_violations():
    zgm.ZeroTrainGenValidation(warnings=["only a warning"]).raise_if_invalid("hdr")


def test_raise_if_invalid_lists_every_violation():
    result = zgm.ZeroTrainGenValidation(violations=["first", "second"])
    with pytest.raises(ValueError, match="hdr") as excinfo:
        result.raise_if_invalid("hdr")
    assert "  - first" in str(excinfo.value)
    assert "  - second" in str(excinfo.value)


def test_emit_warnings_raises_user_warnings():
    result = zgm.ZeroTrainGenValidation(warnings=["careful"])
    with pytest.warns(UserWarning, match="careful"):
        result.emit_warnings()


# --------------------------------------------------------------------------- #
# resolve_zero_train_gen_mismatch
# --------------------------------------------------------------------------- #


def test_resolve_applies_defaults_on_both_sides():
    config = _resolved_config()
    megatron_cfg = config["megatron_cfg"]
    generation_cfg = config["generation"]["mcore_generation_config"]

    assert megatron_cfg["batch_invariant_mode"] is True
    assert megatron_cfg["moe_permute_fusion"] is False
    assert megatron_cfg["attention_backend"] == "flash"
    assert megatron_cfg["flash_attention_version"] == 4
    assert megatron_cfg["batch_invariant_backend"] == "te_native"
    assert megatron_cfg["batch_invariant_collective"] == "ordered"
    assert generation_cfg["logprobs_mode"] == "raw_logprobs"
    assert generation_cfg["enable_chunked_prefill"] is False


def test_resolve_does_not_warn_when_recipe_already_matches():
    config = _policy_config()
    config["megatron_cfg"]["moe_permute_fusion"] = False
    config["megatron_cfg"]["flash_attention_version"] = 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        zgm.resolve_zero_train_gen_mismatch(config)


def test_resolve_warns_and_overrides_a_conflicting_value():
    config = _policy_config()
    config["megatron_cfg"]["flash_attention_version"] = 3
    with pytest.warns(UserWarning, match="flash_attention_version"):
        zgm.resolve_zero_train_gen_mismatch(config)
    assert config["megatron_cfg"]["flash_attention_version"] == 4


def test_resolve_is_a_noop_when_the_preset_is_off():
    config = _policy_config()
    config["megatron_cfg"]["zero_train_gen_mismatch"] = False
    before = copy.deepcopy(config)
    zgm.resolve_zero_train_gen_mismatch(config)
    assert config == before


def test_resolve_without_generation_only_touches_megatron_cfg():
    config = _policy_config()
    del config["generation"]
    zgm.resolve_zero_train_gen_mismatch(config)
    assert config["megatron_cfg"]["batch_invariant_mode"] is True


# --------------------------------------------------------------------------- #
# validate_batch_invariant_mode
# --------------------------------------------------------------------------- #


def test_batch_invariant_valid_config_has_no_findings():
    result = zgm.validate_batch_invariant_mode(_resolved_config())
    assert result.violations == []
    assert result.warnings == []


def test_batch_invariant_off_reports_nothing():
    config = _policy_config()
    config["megatron_cfg"]["context_parallel_size"] = 4
    assert zgm.validate_batch_invariant_mode(config).violations == []


def test_batch_invariant_rejects_tensor_parallel():
    config = _resolved_config()
    config["megatron_cfg"]["tensor_model_parallel_size"] = 2
    result = zgm.validate_batch_invariant_mode(config)
    assert any("tensor_model_parallel_size=1" in v for v in result.violations)


def test_batch_invariant_reports_a_train_generation_tp_mismatch():
    config = _resolved_config()
    config["megatron_cfg"]["tensor_model_parallel_size"] = 2
    config["generation"]["mcore_generation_config"]["tensor_model_parallel_size"] = 1
    result = zgm.validate_batch_invariant_mode(config)
    assert any(
        "same Megatron settings" in v and "tensor_model_parallel_size" in v
        for v in result.violations
    )


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda c: c["megatron_cfg"].update(context_parallel_size=2), "context"),
        (
            lambda c: c.__setitem__("sequence_packing", {"enabled": True}),
            "sequence_packing.enabled=False",
        ),
        (
            lambda c: c["megatron_cfg"].update(use_fused_linear_logprobs=True),
            "use_fused_linear_logprobs",
        ),
        (
            lambda c: c["megatron_cfg"].update(attention_backend="fused"),
            "attention_backend='flash'",
        ),
        (
            lambda c: c["megatron_cfg"].update(flash_attention_version=2),
            "flash_attention_version to be 3 or 4",
        ),
        (
            lambda c: c["generation"].update(backend="vllm"),
            "generation.backend='megatron'",
        ),
        (
            lambda c: c.__setitem__("precision", "float16"),
            "precision='bfloat16'",
        ),
    ],
    ids=["cp", "packing", "fused-logprobs", "attention", "fa-version", "gen", "dtype"],
)
def test_batch_invariant_rejects_unsupported_settings(mutate, expected):
    config = _resolved_config()
    mutate(config)
    result = zgm.validate_batch_invariant_mode(config)
    assert any(expected in v for v in result.violations), result.violations


def test_batch_invariant_requires_the_resolved_fields():
    config = _resolved_config()
    del config["megatron_cfg"]["batch_invariant_backend"]
    result = zgm.validate_batch_invariant_mode(config)
    assert any("batch_invariant_backend" in v for v in result.violations)


# --------------------------------------------------------------------------- #
# validate_zero_train_gen_mismatch
# --------------------------------------------------------------------------- #


def _validate(config: dict[str, Any], **kwargs: Any) -> zgm.ZeroTrainGenValidation:
    return zgm.validate_zero_train_gen_mismatch(
        config, check_packages=False, check_platform=False, **kwargs
    )


def test_validate_valid_config_has_no_findings():
    result = _validate(_resolved_config())
    assert result.violations == []
    assert result.warnings == []


def test_validate_is_a_noop_when_the_preset_is_off():
    config = _policy_config()
    config["megatron_cfg"]["zero_train_gen_mismatch"] = False
    config["precision"] = "float16"
    assert _validate(config).violations == []


def test_validate_rejects_a_non_megatron_generation_backend():
    config = _resolved_config()
    config["generation"]["backend"] = "vllm"
    result = _validate(config)
    assert any("generation.backend must be 'megatron'" in v for v in result.violations)


def test_validate_rejects_missing_generation_block():
    config = _resolved_config()
    del config["generation"]
    result = _validate(config)
    assert any("generation.backend must be 'megatron'" in v for v in result.violations)


def test_validate_rejects_an_unknown_generation_transformer_impl():
    config = _resolved_config()
    config["generation"]["mcore_generation_config"]["transformer_impl"] = "local"
    result = _validate(config)
    assert any("transformer_impl must be" in v for v in result.violations)


def test_validate_only_warns_for_transformer_engine_generation():
    config = _resolved_config()
    config["generation"]["mcore_generation_config"]["transformer_impl"] = (
        "transformer_engine"
    )
    result = _validate(config)
    assert result.violations == []
    assert any("inference_optimized" in w for w in result.warnings)


@pytest.mark.parametrize("version", [None, 3, 2])
def test_validate_requires_flash_attention_4(version):
    config = _resolved_config()
    if version is None:
        del config["megatron_cfg"]["flash_attention_version"]
    else:
        config["megatron_cfg"]["flash_attention_version"] = version
    result = _validate(config)
    assert any("flash_attention_version=4" in v for v in result.violations)


@pytest.mark.parametrize("key", ["multi_latent_attention", "use_mla"])
def test_validate_rejects_mla(key):
    config = _resolved_config()
    config["megatron_cfg"][key] = True
    assert any("MLA" in v for v in _validate(config).violations)


@pytest.mark.parametrize("key", ["hybrid_attention_ratio", "mamba_num_heads"])
def test_validate_rejects_linear_attention_hybrids(key):
    config = _resolved_config()
    config["megatron_cfg"][key] = 0.5
    assert any(key in v for v in _validate(config).violations)


def test_validate_rejects_non_bf16_precision():
    config = _resolved_config()
    config["precision"] = "float16"
    assert any("precision='bfloat16'" in v for v in _validate(config).violations)


@pytest.mark.parametrize("side", ["train", "generation"])
def test_validate_rejects_fp8_on_either_side(side):
    config = _resolved_config()
    if side == "train":
        config["megatron_cfg"]["fp8_cfg"] = {"enabled": True}
    else:
        config["generation"]["mcore_generation_config"]["fp8_cfg"] = {"enabled": True}
    assert any("FP8" in v for v in _validate(config).violations)


# --------------------------------------------------------------------------- #
# Package and platform gates
# --------------------------------------------------------------------------- #


def test_packages_pass_with_the_pinned_stack(monkeypatch):
    _set_versions(monkeypatch, GOOD_VERSIONS)
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_packages(out)
    assert out.violations == []


@pytest.mark.parametrize(
    ("name", "replacement", "expected"),
    [
        ("transformer_engine", None, "transformer_engine is not installed"),
        ("transformer_engine", Version("2.17"), "transformer_engine>=2.18"),
        ("flash_attn", None, "flash_attn>="),
        ("flash_attn", Version("2.7"), "flash_attn>="),
        ("flash-attn-4", None, "flash-attn-4 is not installed"),
        ("nvidia-cutlass-dsl", None, "nvidia-cutlass-dsl>="),
        ("nvidia-cutlass-dsl", Version("4.5.2"), "nvidia-cutlass-dsl>="),
    ],
)
def test_packages_report_each_missing_or_old_dependency(
    monkeypatch, name, replacement, expected
):
    versions = dict(GOOD_VERSIONS)
    if replacement is None:
        del versions[name]
    else:
        versions[name] = replacement
    _set_versions(monkeypatch, versions)
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_packages(out)
    assert any(expected in v for v in out.violations), out.violations


def test_packages_accept_flash_attn_4_b19_below_the_mcore_gate(monkeypatch):
    """MCore's own gate is b20; the zero-KL preset must not repeat it."""
    versions = dict(GOOD_VERSIONS)
    versions["flash-attn-4"] = Version("4.0.0b19")
    _set_versions(monkeypatch, versions)
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_packages(out)
    assert not any("flash-attn-4" in v for v in out.violations)


def test_packages_accept_the_underscore_distribution_name(monkeypatch):
    versions = dict(GOOD_VERSIONS)
    del versions["flash-attn-4"]
    versions["flash_attn_4"] = Version("4.0.0b19")
    _set_versions(monkeypatch, versions)
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_packages(out)
    assert out.violations == []


def test_megatron_core_feature_probe_passes_with_all_fields(monkeypatch):
    config_cls = make_dataclass(
        "TransformerConfig",
        [(name, bool, False) for name in zgm.MEGATRON_CORE_REQUIRED_CONFIG_FIELDS],
    )
    module = types.ModuleType("megatron.core.transformer.transformer_config")
    module.TransformerConfig = config_cls  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, module.__name__, module)
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_megatron_core_features(out)
    assert out.violations == []


def test_megatron_core_feature_probe_names_the_missing_fields(monkeypatch):
    present = zgm.MEGATRON_CORE_REQUIRED_CONFIG_FIELDS[:1]
    config_cls = make_dataclass(
        "TransformerConfig", [(n, bool, False) for n in present]
    )
    module = types.ModuleType("megatron.core.transformer.transformer_config")
    module.TransformerConfig = config_cls  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, module.__name__, module)
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_megatron_core_features(out)
    assert len(out.violations) == 1
    for missing in zgm.MEGATRON_CORE_REQUIRED_CONFIG_FIELDS[1:]:
        assert missing in out.violations[0]
    assert present[0] not in out.violations[0]


def test_megatron_core_feature_probe_reports_a_missing_install(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "megatron.core.transformer.transformer_config", None
    )
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_megatron_core_features(out)
    assert any("not importable" in v for v in out.violations)


def test_platform_device_check_is_skipped_without_check_device():
    config = _resolved_config()
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_platform(config, out, check_device=False)
    assert out.violations == []


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((9, 0), "Hopper"), ((8, 0), "Blackwell"), ((10, 0), None), ((10, 3), None)],
)
def test_platform_requires_blackwell(monkeypatch, capability, expected):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_platform(_resolved_config(), out, check_device=True)
    if expected is None:
        assert out.violations == []
    else:
        assert any(expected in v for v in out.violations)


# --------------------------------------------------------------------------- #
# allow_installed_flash_attn_4
# --------------------------------------------------------------------------- #


def _fake_mcore_attention(have_fa4: bool) -> types.ModuleType:
    module = types.ModuleType("megatron.core.transformer.attention")
    module.HAVE_FA4 = have_fa4  # type: ignore[attr-defined]
    module.flash_attn4_varlen_func = None  # type: ignore[attr-defined]
    return module


def _fake_flash_attn_cute(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("flash_attn.cute")

    def flash_attn_varlen_func(*args, **kwargs):
        return "fa4"

    module.flash_attn_varlen_func = flash_attn_varlen_func  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, "flash_attn.cute", module)
    return module


def test_allow_installed_flash_attn_4_opens_mcores_version_gate(monkeypatch):
    attention = _fake_mcore_attention(have_fa4=False)
    _install_fake_module(monkeypatch, attention.__name__, attention)
    cute = _fake_flash_attn_cute(monkeypatch)
    monkeypatch.setattr(
        zgm, "_first_package_version", lambda names: Version("4.0.0b19")
    )

    zgm.allow_installed_flash_attn_4()

    assert attention.HAVE_FA4 is True
    assert attention.flash_attn4_varlen_func is cute.flash_attn_varlen_func


def test_allow_installed_flash_attn_4_leaves_an_open_gate_alone(monkeypatch):
    attention = _fake_mcore_attention(have_fa4=True)
    sentinel = object()
    attention.flash_attn4_varlen_func = sentinel  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, attention.__name__, attention)
    _fake_flash_attn_cute(monkeypatch)

    def unexpected(names):
        raise AssertionError("package lookup must not run when FA4 is already enabled")

    monkeypatch.setattr(zgm, "_first_package_version", unexpected)

    zgm.allow_installed_flash_attn_4()

    assert attention.flash_attn4_varlen_func is sentinel


def test_allow_installed_flash_attn_4_does_nothing_when_not_installed(monkeypatch):
    attention = _fake_mcore_attention(have_fa4=False)
    _install_fake_module(monkeypatch, attention.__name__, attention)
    monkeypatch.setattr(zgm, "_first_package_version", lambda names: None)

    zgm.allow_installed_flash_attn_4()

    assert attention.HAVE_FA4 is False
    assert attention.flash_attn4_varlen_func is None


# --------------------------------------------------------------------------- #
# enable_batch_invariant_kernels / configure_zero_train_gen_mismatch
# --------------------------------------------------------------------------- #


def _fake_batch_invariant_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, tuple, dict]]:
    calls: list[tuple[str, tuple, dict]] = []
    module = types.ModuleType(
        "megatron.core.transformer.custom_layers.batch_invariant_kernels"
    )
    module.assert_te_supports_batch_invariant_attention = lambda: calls.append(  # type: ignore[attr-defined]
        ("assert_te", (), {})
    )
    module.enable_batch_invariant_mode = lambda *a, **k: calls.append(  # type: ignore[attr-defined]
        ("enable", a, k)
    )
    _install_fake_module(monkeypatch, module.__name__, module)
    return calls


def test_enable_batch_invariant_kernels_passes_the_resolved_settings(monkeypatch):
    calls = _fake_batch_invariant_kernels(monkeypatch)
    opened: list[bool] = []
    monkeypatch.setattr(
        zgm, "allow_installed_flash_attn_4", lambda: opened.append(True)
    )

    zgm.enable_batch_invariant_kernels(_resolved_config())

    assert opened == [True]
    assert calls == [
        ("assert_te", (), {}),
        ("enable", (), {"backend": "te_native", "collective": "ordered"}),
    ]


def test_configure_is_a_noop_when_the_preset_is_off(monkeypatch):
    config = _policy_config()
    config["megatron_cfg"]["zero_train_gen_mismatch"] = False
    monkeypatch.setattr(
        zgm,
        "enable_batch_invariant_kernels",
        lambda cfg: pytest.fail("kernels must not be enabled"),
    )
    zgm.configure_zero_train_gen_mismatch(config, apply_kernels=True)
    assert "batch_invariant_mode" not in config["megatron_cfg"]


def test_configure_without_kernels_resolves_and_validates(monkeypatch):
    monkeypatch.setattr(
        zgm,
        "enable_batch_invariant_kernels",
        lambda cfg: pytest.fail("kernels must not be enabled"),
    )
    config = _policy_config()
    zgm.configure_zero_train_gen_mismatch(config, apply_kernels=False)
    assert config["megatron_cfg"]["batch_invariant_mode"] is True


def test_configure_raises_on_an_invalid_config(monkeypatch):
    monkeypatch.setattr(zgm, "enable_batch_invariant_kernels", lambda cfg: None)
    config = _policy_config()
    config["precision"] = "float16"
    with pytest.raises(ValueError, match="zero_train_gen_mismatch=true failed"):
        zgm.configure_zero_train_gen_mismatch(config, apply_kernels=False)


def test_configure_raises_when_batch_invariant_checks_fail(monkeypatch):
    monkeypatch.setattr(zgm, "enable_batch_invariant_kernels", lambda cfg: None)
    config = _policy_config()
    config["megatron_cfg"]["tensor_model_parallel_size"] = 2
    with pytest.raises(ValueError, match="batch_invariant_mode=True failed"):
        zgm.configure_zero_train_gen_mismatch(config, apply_kernels=False)


def test_configure_enables_kernels_after_validation(monkeypatch):
    enabled: list[dict[str, Any]] = []
    monkeypatch.setattr(zgm, "enable_batch_invariant_kernels", enabled.append)
    _set_versions(monkeypatch, GOOD_VERSIONS)
    config = _policy_config()
    # No GPU is needed: the platform gate skips when CUDA is unavailable.
    monkeypatch.setattr(zgm, "_validate_platform", lambda *a, **k: None)
    zgm.configure_zero_train_gen_mismatch(config, apply_kernels=True)
    assert enabled == [config]


# --------------------------------------------------------------------------- #
# Shipped recipes
# --------------------------------------------------------------------------- #


def _load_policy(name: str) -> dict[str, Any]:
    register_omegaconf_resolvers()
    cfg = load_config(CONFIGS_DIR / name)
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    assert isinstance(container, dict)
    policy = container["policy"]
    assert isinstance(policy, dict)
    return policy


@pytest.mark.parametrize("name", ZERO_KL_EXEMPLARS)
def test_shipped_recipes_pass_the_zero_kl_gates(name):
    policy = _load_policy(name)
    zgm.configure_zero_train_gen_mismatch(policy, apply_kernels=False)
    assert policy["megatron_cfg"]["batch_invariant_mode"] is True


@pytest.mark.parametrize("name", ZERO_KL_EXEMPLARS)
def test_shipped_recipes_train_and_generate_with_the_same_tp(name):
    policy = _load_policy(name)
    zgm.configure_zero_train_gen_mismatch(policy, apply_kernels=False)
    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

    inference = merged_inference_megatron_cfg(policy)
    assert (
        inference["tensor_model_parallel_size"]
        == policy["megatron_cfg"]["tensor_model_parallel_size"]
    )
    assert inference["context_parallel_size"] == 1


@pytest.mark.parametrize(
    "name", [n for n in ZERO_KL_EXEMPLARS if n.endswith("_colocated.yaml")]
)
def test_colocated_recipes_share_the_training_gpus(name):
    policy = _load_policy(name)
    assert policy["generation"]["colocated"]["enabled"] is True
    assert policy["generation"]["refit_transport"] == "mcore"
    assert policy["generation"]["mcore_generation_config"]["refit_backend"] == "nccl"
    assert policy["megatron_cfg"]["optimizer"]["optimizer_cpu_offload"] is True
