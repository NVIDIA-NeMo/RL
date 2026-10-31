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

"""Unit tests for the zero train/generation KL preset (`zero_train_gen_mismatch`).

Everything here is CPU-only. Megatron-Core, Transformer Engine and FlashAttention
are replaced by small fake modules so the validate / enable logic runs
in any environment, and the shipped zero-KL recipes are loaded from disk and
pushed through the same gates the workers run at startup.
"""

import copy
import sys
import types
from dataclasses import make_dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.version import Version

from nemo_rl.models.megatron import zero_train_gen_mismatch as zgm

CONFIGS_DIR = Path(__file__).resolve().parents[4] / "examples" / "configs"

# Installed-package versions that satisfy every gate in `_validate_packages`.
GOOD_VERSIONS = {
    "transformer_engine": Version("2.18.0"),
    "flash_attn": Version("2.8.1"),
    "flash-attn-4": Version("4.0.0b19"),
    "nvidia-cutlass-dsl": Version("4.6.2"),
}


def _policy_config() -> dict[str, Any]:
    """Smallest policy config that passes every zero-KL gate (recipe-complete)."""
    return {
        "precision": "bfloat16",
        "make_sequence_length_divisible_by": 64,
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False, "sequence_length_round": 64},
        "megatron_cfg": {
            "zero_train_gen_mismatch": True,
            "batch_invariant_mode": True,
            "batch_invariant_backend": "te_native",
            "batch_invariant_collective": "ordered",
            "attention_backend": "flash",
            "flash_attention_version": 4,
            "moe_permute_fusion": False,
            "env_vars": {
                "CUBLASLT_WORKSPACE_SIZE": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":0:0",
            },
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 4,
            "sequence_parallel": True,
        },
        "generation": {
            "backend": "megatron",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "colocated": {"enabled": False},
            "mcore_generation_config": {
                "transformer_impl": "inference_optimized",
                "logprobs_mode": "raw_logprobs",
                "enable_chunked_prefill": False,
            },
        },
    }


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
# Required values: the recipe sets them, the validator checks them
# --------------------------------------------------------------------------- #


def test_valid_recipe_values_are_accepted_without_modification():
    config = _policy_config()
    before = copy.deepcopy(config)
    result = _validate(config)
    assert result.violations == []
    assert config == before  # the validator never rewrites the config


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("batch_invariant_mode", False),
        ("moe_permute_fusion", True),
        ("attention_backend", "fused"),
        ("flash_attention_version", 2),
        ("batch_invariant_backend", "triton"),
        ("batch_invariant_collective", "multimem"),
    ],
)
def test_validate_rejects_a_wrong_megatron_value_without_rewriting_it(key, bad):
    config = _policy_config()
    config["megatron_cfg"][key] = bad
    result = _validate(config)
    assert any(f"policy.megatron_cfg.{key} must be" in v for v in result.violations)
    assert config["megatron_cfg"][key] == bad


def test_validate_rejects_a_missing_required_megatron_value():
    config = _policy_config()
    del config["megatron_cfg"]["moe_permute_fusion"]
    result = _validate(config)
    assert any("moe_permute_fusion must be False" in v for v in result.violations)


def test_validate_allows_chunked_prefill():
    config = _policy_config()
    config["generation"]["mcore_generation_config"]["enable_chunked_prefill"] = True
    assert _validate(config).violations == []


# --------------------------------------------------------------------------- #
# validate_batch_invariant_mode
# --------------------------------------------------------------------------- #


def test_batch_invariant_valid_config_has_no_findings():
    result = zgm.validate_batch_invariant_mode(_policy_config())
    assert result.violations == []
    assert result.warnings == []


def test_batch_invariant_off_reports_nothing():
    config = _policy_config()
    config["megatron_cfg"]["batch_invariant_mode"] = False
    config["megatron_cfg"]["context_parallel_size"] = 4
    assert zgm.validate_batch_invariant_mode(config).violations == []


def test_batch_invariant_allows_matched_tensor_parallel():
    config = _policy_config()
    config["megatron_cfg"]["tensor_model_parallel_size"] = 2
    config["generation"]["mcore_generation_config"]["tensor_model_parallel_size"] = 2
    assert zgm.validate_batch_invariant_mode(config).violations == []


def test_batch_invariant_rejects_tensor_parallel_above_two():
    config = _policy_config()
    config["megatron_cfg"]["tensor_model_parallel_size"] = 4
    config["generation"]["mcore_generation_config"]["tensor_model_parallel_size"] = 4
    result = zgm.validate_batch_invariant_mode(config)
    assert any("tensor_model_parallel_size <= 2" in v for v in result.violations)


def test_batch_invariant_reports_a_train_generation_tp_mismatch():
    config = _policy_config()
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
    ids=["cp", "fused-logprobs", "attention", "fa-version", "gen", "dtype"],
)
def test_batch_invariant_rejects_unsupported_settings(mutate, expected):
    config = _policy_config()
    mutate(config)
    result = zgm.validate_batch_invariant_mode(config)
    assert any(expected in v for v in result.violations), result.violations


def test_batch_invariant_accepts_sequence_packing():
    config = _policy_config()
    config["sequence_packing"] = {"enabled": True}
    assert zgm.validate_batch_invariant_mode(config).violations == []


def test_batch_invariant_requires_the_batch_invariant_fields():
    config = _policy_config()
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
    result = _validate(_policy_config())
    assert result.violations == []
    assert result.warnings == []


def test_validate_is_a_noop_when_the_preset_is_off():
    config = _policy_config()
    config["megatron_cfg"]["zero_train_gen_mismatch"] = False
    config["precision"] = "float16"
    assert _validate(config).violations == []


def test_validate_rejects_a_non_megatron_generation_backend():
    config = _policy_config()
    config["generation"]["backend"] = "vllm"
    result = _validate(config)
    assert any("generation.backend must be 'megatron'" in v for v in result.violations)


def test_validate_rejects_colocated_generation():
    config = _policy_config()
    config["generation"]["colocated"] = {"enabled": True}
    result = _validate(config)
    assert any("does not support colocated generation" in v for v in result.violations)


def test_validate_rejects_missing_generation_block():
    config = _policy_config()
    del config["generation"]
    result = _validate(config)
    assert any("generation.backend must be 'megatron'" in v for v in result.violations)


def test_validate_rejects_a_non_inference_optimized_generation_transformer_impl():
    config = _policy_config()
    config["generation"]["mcore_generation_config"]["transformer_impl"] = (
        "transformer_engine"
    )
    result = _validate(config)
    assert any("transformer_impl must be" in v for v in result.violations)


@pytest.mark.parametrize("version", [None, 3, 2])
def test_validate_requires_flash_attention_4(version):
    config = _policy_config()
    if version is None:
        del config["megatron_cfg"]["flash_attention_version"]
    else:
        config["megatron_cfg"]["flash_attention_version"] = version
    result = _validate(config)
    assert any("flash_attention_version must be 4" in v for v in result.violations)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("multi_latent_attention", True),
        ("experimental_attention_variant", "some_variant"),
    ],
)
def test_validate_provider_rejects_unsupported_architecture(key, value):
    config = _policy_config()
    model_cfg = types.SimpleNamespace(**{key: value})

    with pytest.raises(ValueError, match=key):
        zgm.validate_zero_train_gen_model_provider(config, model_cfg)


_HYBRID_MAMBA_ENV_VARS = {"MAMBA_DETERMINISTIC": "1", "TRITON_CACHE_AUTOTUNING": "1"}


@pytest.fixture
def _clean_mamba_env(monkeypatch):
    for key in _HYBRID_MAMBA_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    "model_cfg",
    [
        types.SimpleNamespace(hybrid_layer_pattern="M*-"),
        types.SimpleNamespace(is_hybrid_model=True),
    ],
)
def test_validate_provider_accepts_hybrid_mamba(_clean_mamba_env, model_cfg):
    config = _policy_config()
    config["megatron_cfg"]["env_vars"].update(_HYBRID_MAMBA_ENV_VARS)

    zgm.validate_zero_train_gen_model_provider(config, model_cfg)


def test_validate_provider_accepts_hybrid_mamba_env_from_launch(
    _clean_mamba_env, monkeypatch
):
    for key, value in _HYBRID_MAMBA_ENV_VARS.items():
        monkeypatch.setenv(key, value)
    config = _policy_config()
    model_cfg = types.SimpleNamespace(hybrid_layer_pattern="M*-")

    zgm.validate_zero_train_gen_model_provider(config, model_cfg)


@pytest.mark.parametrize("missing", sorted(_HYBRID_MAMBA_ENV_VARS))
def test_validate_provider_requires_hybrid_mamba_env_pair(_clean_mamba_env, missing):
    config = _policy_config()
    config["megatron_cfg"]["env_vars"].update(
        {k: v for k, v in _HYBRID_MAMBA_ENV_VARS.items() if k != missing}
    )
    model_cfg = types.SimpleNamespace(hybrid_layer_pattern="M*-")

    with pytest.raises(ValueError, match=f"got {missing}=None"):
        zgm.validate_zero_train_gen_model_provider(config, model_cfg)


def test_validate_provider_skips_mamba_env_for_non_hybrid(_clean_mamba_env):
    config = _policy_config()
    model_cfg = types.SimpleNamespace(hybrid_layer_pattern=None)

    zgm.validate_zero_train_gen_model_provider(config, model_cfg)


@pytest.mark.parametrize(
    ("packing", "divisible_by", "tp", "accepted"),
    [
        (True, 64, 1, False),
        (True, 128, 1, True),
        (True, 256, 2, True),
        (True, 192, 2, False),
        (False, 64, 1, True),
    ],
    ids=["packed-64", "packed-128", "packed-256-tp2", "packed-192-tp2", "unpacked-64"],
)
def test_validate_provider_requires_mamba_chunk_aligned_packing(
    _clean_mamba_env, packing, divisible_by, tp, accepted
):
    config = _policy_config()
    config["megatron_cfg"]["env_vars"].update(_HYBRID_MAMBA_ENV_VARS)
    config["megatron_cfg"]["tensor_model_parallel_size"] = tp
    config["sequence_packing"] = {"enabled": packing}
    config["make_sequence_length_divisible_by"] = divisible_by
    model_cfg = types.SimpleNamespace(hybrid_layer_pattern="M*-", mamba_chunk_size=128)

    if accepted:
        zgm.validate_zero_train_gen_model_provider(config, model_cfg)
    else:
        with pytest.raises(
            ValueError, match="multiple of 128 \\(mamba_chunk_size=128\\)"
        ):
            zgm.validate_zero_train_gen_model_provider(config, model_cfg)


def test_validate_provider_allows_64_aligned_packing_for_non_hybrid(_clean_mamba_env):
    config = _policy_config()
    config["sequence_packing"] = {"enabled": True}
    model_cfg = types.SimpleNamespace(hybrid_layer_pattern=None)

    zgm.validate_zero_train_gen_model_provider(config, model_cfg)


def test_validate_provider_is_noop_when_the_preset_is_off():
    config = _policy_config()
    config["megatron_cfg"]["zero_train_gen_mismatch"] = False
    model_cfg = types.SimpleNamespace(multi_latent_attention=True)

    zgm.validate_zero_train_gen_model_provider(config, model_cfg)


def test_validate_rejects_non_bf16_precision():
    config = _policy_config()
    config["precision"] = "float16"
    assert any("precision='bfloat16'" in v for v in _validate(config).violations)


_MXFP8 = {"enabled": True, "fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": True}


def test_validate_accepts_mxfp8_on_both_sides():
    # The generation side inherits policy.megatron_cfg.fp8_cfg.
    config = _policy_config()
    config["megatron_cfg"]["fp8_cfg"] = dict(_MXFP8)
    config["generation"]["mcore_generation_config"][
        "inference_grouped_gemm_backend"
    ] = "torch"
    assert _validate(config).violations == []


@pytest.mark.parametrize("side", ["train", "generation"])
def test_validate_rejects_mxfp8_on_one_side_only(side):
    config = _policy_config()
    if side == "train":
        config["megatron_cfg"]["fp8_cfg"] = dict(_MXFP8)
        config["generation"]["mcore_generation_config"]["fp8_cfg"] = {"enabled": False}
    else:
        config["generation"]["mcore_generation_config"]["fp8_cfg"] = dict(_MXFP8)
    assert any("same precision" in v for v in _validate(config).violations)


def test_validate_rejects_a_non_mxfp8_fp8_recipe():
    config = _policy_config()
    config["megatron_cfg"]["fp8_cfg"] = {**_MXFP8, "fp8_recipe": "blockwise"}
    assert any("fp8_recipe='mxfp8'" in v for v in _validate(config).violations)


def test_validate_rejects_flashinfer_grouped_gemm_with_mxfp8():
    config = _policy_config()
    config["megatron_cfg"]["fp8_cfg"] = dict(_MXFP8)
    config["generation"]["mcore_generation_config"][
        "inference_grouped_gemm_backend"
    ] = "flashinfer"
    assert any("flashinfer" in v for v in _validate(config).violations)


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
    out = zgm.ZeroTrainGenValidation()
    zgm._validate_platform(out, check_device=False)
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
    zgm._validate_platform(out, check_device=True)
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
# enable_batch_invariant_kernels / validate_zero_train_gen_kl
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


def test_enable_batch_invariant_kernels_passes_the_recipe_settings(monkeypatch):
    calls = _fake_batch_invariant_kernels(monkeypatch)
    opened: list[bool] = []
    shimmed: list[bool] = []
    monkeypatch.setattr(
        zgm, "allow_installed_flash_attn_4", lambda: opened.append(True)
    )
    monkeypatch.setattr(
        zgm, "_use_full_vocab_log_softmax", lambda: shimmed.append(True)
    )
    tuned: list[bool] = []
    monkeypatch.setattr(
        zgm, "_enable_mcore_deterministic_tuning", lambda: tuned.append(True)
    )

    zgm.enable_batch_invariant_kernels(_policy_config())

    assert opened == [True]
    assert shimmed == [True]
    assert tuned == [True]
    assert calls == [
        ("assert_te", (), {}),
        ("enable", (), {"backend": "te_native", "collective": "ordered"}),
    ]


def _fake_tuning(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []
    tuning = types.ModuleType("megatron.core.tuning")
    tuning.set_deterministic_mode = lambda value: calls.append(("set", value))  # type: ignore[attr-defined]
    tuning.install = lambda *a, **k: calls.append(("install", a, k))  # type: ignore[attr-defined]
    _install_fake_module(monkeypatch, tuning.__name__, tuning)
    return calls


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", [("set", True), ("install", (), {"deterministic": True})]),
        ("0", []),
        (None, []),
    ],
)
def test_mamba_deterministic_turns_on_mcore_deterministic_tuning(
    monkeypatch, value, expected
):
    calls = _fake_tuning(monkeypatch)
    if value is None:
        monkeypatch.delenv("MAMBA_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("MAMBA_DETERMINISTIC", value)

    zgm._enable_mcore_deterministic_tuning()

    assert calls == expected


def test_mcore_without_tuning_relies_on_mamba_deterministic(monkeypatch):
    monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
    monkeypatch.setitem(sys.modules, "megatron.core.tuning", None)

    zgm._enable_mcore_deterministic_tuning()


def test_validate_kl_is_a_noop_when_everything_is_off():
    config = _policy_config()
    config["megatron_cfg"]["zero_train_gen_mismatch"] = False
    config["megatron_cfg"]["batch_invariant_mode"] = False
    config["precision"] = "float16"
    zgm.validate_zero_train_gen_kl(config, check_environment=False)


@pytest.mark.parametrize("policy", [{}, {"megatron_cfg": {"enabled": False}}])
def test_validate_kl_ignores_non_megatron_policies(policy):
    zgm.validate_zero_train_gen_kl(policy, check_environment=False)


def test_validate_kl_accepts_a_valid_config_without_touching_the_environment(
    monkeypatch,
):
    # No package or GPU lookups on the driver.
    monkeypatch.setattr(
        zgm, "_validate_packages", lambda out: pytest.fail("driver checks packages")
    )

    def _platform(out, *, check_device):
        assert not check_device, "driver checks GPU"

    monkeypatch.setattr(zgm, "_validate_platform", _platform)
    config = _policy_config()
    before = copy.deepcopy(config)
    zgm.validate_zero_train_gen_kl(config, check_environment=False)
    assert config == before


def test_validate_kl_checks_the_environment_on_workers(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(zgm, "_validate_packages", lambda out: seen.append("packages"))
    monkeypatch.setattr(
        zgm,
        "_validate_platform",
        lambda out, *, check_device: seen.append(f"platform={check_device}"),
    )
    zgm.validate_zero_train_gen_kl(_policy_config(), check_environment=True)
    assert seen == ["platform=True", "packages"]


def test_validate_kl_raises_on_an_invalid_preset():
    config = _policy_config()
    config["precision"] = "float16"
    with pytest.raises(ValueError, match="zero_train_gen_mismatch=true failed"):
        zgm.validate_zero_train_gen_kl(config, check_environment=False)


def test_validate_kl_raises_when_batch_invariant_checks_fail():
    config = _policy_config()
    config["megatron_cfg"]["context_parallel_size"] = 2
    with pytest.raises(ValueError, match="batch_invariant_mode=True failed"):
        zgm.validate_zero_train_gen_kl(config, check_environment=False)


def test_validate_kl_checks_batch_invariant_mode_without_the_preset():
    config = _policy_config()
    config["megatron_cfg"]["zero_train_gen_mismatch"] = False
    config["generation"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="top_p=1.0"):
        zgm.validate_zero_train_gen_kl(config, check_environment=False)


# --------------------------------------------------------------------------- #
# Batch-invariant checks that used to be silent config rewrites
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda c: c["generation"].update(top_p=0.9), "top_p=1.0"),
        (lambda c: c["generation"].update(top_k=50), "top_k=null"),
        (lambda c: c["generation"].update(temperature=0.7), "temperature=1.0"),
        (
            lambda c: c["generation"]["mcore_generation_config"].update(
                logprobs_mode="processed_logprobs"
            ),
            "logprobs_mode='raw_logprobs'",
        ),
        (
            lambda c: c.update(make_sequence_length_divisible_by=1),
            "make_sequence_length_divisible_by",
        ),
        (
            lambda c: c.update(
                dynamic_batching={"enabled": True, "sequence_length_round": 32}
            ),
            "dynamic_batching.sequence_length_round",
        ),
        (
            lambda c: c["megatron_cfg"].update(env_vars=None),
            "env_vars.CUBLASLT_WORKSPACE_SIZE",
        ),
        (
            lambda c: c["megatron_cfg"]["env_vars"].update(
                CUBLAS_WORKSPACE_CONFIG=":4096:8"
            ),
            "env_vars.CUBLAS_WORKSPACE_CONFIG",
        ),
    ],
    ids=[
        "top_p",
        "top_k",
        "temperature",
        "logprobs_mode",
        "divisible_by",
        "dynamic_round",
        "no_env_vars",
        "bad_env_var",
    ],
)
def test_batch_invariant_requires_recipe_values_instead_of_rewriting(mutate, expected):
    config = _policy_config()
    mutate(config)
    before = copy.deepcopy(config)
    result = zgm.validate_batch_invariant_mode(config)
    assert any(expected in v for v in result.violations), result.violations
    assert config == before


def test_batch_invariant_does_not_require_cublas_env_for_other_backends():
    config = _policy_config()
    config["megatron_cfg"]["batch_invariant_backend"] = "triton"
    config["megatron_cfg"]["env_vars"] = None
    result = zgm.validate_batch_invariant_mode(config)
    assert not any("env_vars" in v for v in result.violations)


def test_batch_invariant_token_multiple_accepts_larger_multiples():
    config = _policy_config()
    config["make_sequence_length_divisible_by"] = 128
    assert zgm.validate_batch_invariant_mode(config).violations == []


# --------------------------------------------------------------------------- #
# Shipped recipes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("tp_size", "expected"), [(1, 64), (2, 64), (3, 66), (128, 128)]
)
def test_batch_invariant_token_multiple(tp_size, expected):
    from nemo_rl.models.megatron.batch_invariant import batch_invariant_token_multiple

    assert batch_invariant_token_multiple(tp_size) == expected


def _standalone_megatron_exemplars() -> list[Path]:
    """Top-level exemplars that define a full (enabled) Megatron block themselves."""
    found = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or "defaults" in raw:
            continue
        for section in raw.values():
            megatron_cfg = (
                section.get("megatron_cfg") if isinstance(section, dict) else None
            )
            if isinstance(megatron_cfg, dict) and "apply_rope_fusion" in megatron_cfg:
                found.append(path)
                break
    return found


@pytest.mark.parametrize("path", _standalone_megatron_exemplars(), ids=lambda p: p.name)
def test_base_exemplars_pin_flash_attention_to_fa2(path):
    # flash-attn-4 is installed in the mcore extra; without an explicit pin TE
    # would pick FA4 for every Megatron run. Only batch-invariant recipes use 4.
    raw = yaml.safe_load(path.read_text())
    for section in raw.values():
        megatron_cfg = (
            section.get("megatron_cfg") if isinstance(section, dict) else None
        )
        if isinstance(megatron_cfg, dict) and "apply_rope_fusion" in megatron_cfg:
            assert megatron_cfg.get("flash_attention_version") == 2, path.name


def test_token_rounder_matches_megatron_core():
    batch_dimensions_utils = pytest.importorskip(
        "megatron.core.inference.batch_dimensions_utils"
    )
    from nemo_rl.models.megatron.batch_invariant import MCORE_TOKEN_ROUNDER

    assert MCORE_TOKEN_ROUNDER == batch_dimensions_utils.TOKEN_ROUNDER


def test_logprob_shim_uses_full_vocab_log_softmax_at_any_tp(monkeypatch):
    import torch

    from nemo_rl.distributed import model_utils

    for name in ("DistributedLogprob", "ChunkedDistributedLogprob"):
        monkeypatch.setattr(model_utils, name, getattr(model_utils, name))
    zgm._use_full_vocab_log_softmax()
    zgm._use_full_vocab_log_softmax()  # idempotent: must not wrap twice

    # TP 2, rank 1: rank 0's half is a function of rank 1's, so any sequence
    # slice of the shard gathers to the matching slice of the full row. MCore's
    # gather returns its input unchanged at TP 1.
    def full_row(shard):
        return torch.cat([shard.flip(-1) * 0.5, shard], dim=-1)

    gathered = []

    def fake_gather(shard, group):
        gathered.append((shard.shape[1], group))
        return shard if group == "tp1" else full_row(shard)

    mappings = types.SimpleNamespace(
        gather_from_tensor_model_parallel_region=fake_gather
    )
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel.mappings", mappings)

    logits = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    target = torch.randint(0, 16, (2, 5))
    expected = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, target.unsqueeze(-1))
        .squeeze(-1)
    )

    shim = model_utils.DistributedLogprob
    assert isinstance(shim, zgm._FullVocabLogprob)
    assert not isinstance(shim._original, zgm._FullVocabLogprob)
    assert torch.equal(shim.apply(logits, target, 0, 16, "tp1", True), expected)

    chunked = model_utils.ChunkedDistributedLogprob
    assert torch.equal(chunked.apply(logits, target, 0, 16, 2, "tp1", True), expected)
    gathered.clear()

    shard = logits[..., 8:]
    expected_tp2 = (
        torch.log_softmax(full_row(shard).float(), dim=-1)
        .gather(-1, target.unsqueeze(-1))
        .squeeze(-1)
    )
    assert torch.equal(shim.apply(shard, target, 8, 16, "tp2", True), expected_tp2)
    assert gathered == [(5, "tp2")]
    gathered.clear()
    assert torch.equal(
        chunked.apply(shard, target, 8, 16, 2, "tp2", True), expected_tp2
    )
    assert gathered == [(2, "tp2"), (2, "tp2"), (1, "tp2")]

    original = types.SimpleNamespace(apply=lambda *a: "orig")
    monkeypatch.setattr(shim, "_original", original)
    assert shim.apply(logits, target, 0, 16, None, True) == "orig"


@pytest.mark.parametrize("chunked", [False, True])
def test_logprob_shim_backward_is_the_replaced_functions_backward(monkeypatch, chunked):
    import torch

    def full_row(shard):
        return torch.cat([shard.flip(-1) * 0.5, shard], dim=-1)

    mappings = types.SimpleNamespace(
        gather_from_tensor_model_parallel_region=lambda shard, group: full_row(shard)
    )
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel.mappings", mappings)

    calls = []

    class Original(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, target, *rest):
            calls.append(rest)
            ctx.save_for_backward(logits)
            return logits.float().sin().sum(-1)

        @staticmethod
        def backward(ctx, grad_output):
            (logits,) = ctx.saved_tensors
            grad = logits.float().cos() * grad_output.unsqueeze(-1) * 3.0
            return (grad,) + (None,) * (len(calls[-1]) + 1)

    original_args = (8, 16, 2, "tp2") if chunked else (8, 16, "tp2")
    shard = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    target = torch.randint(0, 16, (2, 5))
    grad_output = torch.randn(2, 5)

    reference_logits = shard.clone().requires_grad_()
    Original.apply(reference_logits, target, *original_args, False).backward(
        grad_output
    )

    calls.clear()
    logits = shard.clone().requires_grad_()
    shim = zgm._FullVocabLogprob(Original, chunked=chunked)
    out = shim.apply(logits, target, *original_args, False)
    assert calls == []  # the forward does not call the replaced function
    out.backward(grad_output)

    assert calls == [(*original_args, False)]
    assert logits.grad.dtype == torch.bfloat16
    assert torch.equal(logits.grad, reference_logits.grad)
