# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production config helper without importing GPU-only workers."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

source = (
    Path(__file__).resolve().parents[4]
    / "nemo_rl/models/generation/vllm/vllm_worker.py"
)
tree = ast.parse(source.read_text())
function = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_strip_training_engram_override"
)
namespace = {"Any": Any, "copy": copy}
exec(
    compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
    namespace,
)
strip_override = namespace[function.name]
complete_function = next(
    n for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "_complete_deepseek_v41_text_override"
)
exec(
    compile(ast.Module(body=[complete_function], type_ignores=[]), str(source), "exec"),
    namespace,
)
complete_override = namespace[complete_function.name]


@pytest.mark.parametrize("extra", [{}, {"num_attention_heads": 64}])
def test_strip_host_path_preserves_policy_and_other_rollout_overrides(extra):
    policy = {
        "hf_overrides": {
            "text_config": {"engram_host_checkpoint": "/checkpoint", **extra},
            "expert_dtype": "bf16",
        }
    }
    kwargs = copy.deepcopy(policy)
    strip_override(kwargs)
    assert (
        policy["hf_overrides"]["text_config"]["engram_host_checkpoint"] == "/checkpoint"
    )
    assert kwargs["hf_overrides"] == {
        "expert_dtype": "bf16",
        **({"text_config": extra} if extra else {}),
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"hf_overrides": None},
        {"hf_overrides": {"text_config": {"num_attention_heads": 64}}},
    ],
)
def test_unrelated_configuration_unchanged(kwargs):
    original = copy.deepcopy(kwargs)
    strip_override(kwargs)
    assert kwargs == original


@pytest.mark.parametrize("as_dict", [False, True])
def test_dapo_partial_override_preserves_checkpoint_structure(as_dict):
    base = {"num_attention_heads": 64, "hidden_size": 4096,
            "num_hidden_layers": 61, "compressed_kv_format": "original",
            "rope_parameters": {"rope_theta": 10000}}
    cfg = SimpleNamespace(model_type="deepseek_v41", text_config=(
        base if as_dict else SimpleNamespace(to_dict=lambda: base)))
    policy = {"hf_overrides": {"text_config": {
        "engram_host_checkpoint": "/checkpoint", "compressed_kv_format": "mxfp8"},
        "quantization_config": None, "expert_dtype": "bf16"}}
    kwargs = copy.deepcopy(policy)
    strip_override(kwargs)
    complete_override(kwargs, cfg)
    text = kwargs["hf_overrides"]["text_config"]
    # vLLM replaces the text config: the replacement must be self-contained.
    assert text == {**base, "compressed_kv_format": "mxfp8"}
    assert "engram_host_checkpoint" not in text
    assert kwargs["hf_overrides"]["quantization_config"] is None
    assert kwargs["hf_overrides"]["expert_dtype"] == "bf16"
    text["rope_parameters"]["rope_theta"] = 42
    assert base["rope_parameters"]["rope_theta"] == 10000
    assert policy["hf_overrides"]["text_config"]["engram_host_checkpoint"] == "/checkpoint"


def test_explicit_rollout_field_wins_and_unrelated_models_are_unchanged():
    cfg = SimpleNamespace(model_type="deepseek_v41", text_config={"num_attention_heads": 64})
    kwargs = {"hf_overrides": {"text_config": {"num_attention_heads": 32}}}
    complete_override(kwargs, cfg)
    assert kwargs["hf_overrides"]["text_config"]["num_attention_heads"] == 32
    other = copy.deepcopy(kwargs)
    complete_override(other, SimpleNamespace(model_type="other"))
    assert other == kwargs


@pytest.mark.parametrize("kwargs", [{}, {"hf_overrides": None},
    {"hf_overrides": {"quantization_config": None}}])
def test_no_text_override_does_not_add_one(kwargs):
    before = copy.deepcopy(kwargs)
    complete_override(kwargs, SimpleNamespace(model_type="deepseek_v41"))
    assert kwargs == before


@pytest.mark.parametrize("explicit", [False, True])
def test_flat_vllm_config_promotes_text_overrides_without_shadowing(explicit):
    from transformers import PretrainedConfig

    class FlatV41Config(PretrainedConfig):
        model_type = "deepseek_v41"

    cfg = FlatV41Config(num_attention_heads=64, num_hidden_layers=40)
    kwargs = {"hf_overrides": {"text_config": {
        "engram_host_checkpoint": "/checkpoint", "compressed_kv_format": "mxfp8"}}}
    if explicit:
        kwargs["hf_overrides"]["compressed_kv_format"] = "explicit-rollout"
    strip_override(kwargs)
    complete_override(kwargs, cfg)
    cfg.update(kwargs["hf_overrides"])
    assert not hasattr(cfg, "text_config")
    assert cfg.get_text_config() is cfg
    assert cfg.get_text_config().num_attention_heads == 64
    assert cfg.num_hidden_layers == 40
    assert cfg.compressed_kv_format == ("explicit-rollout" if explicit else "mxfp8")
    assert not hasattr(cfg, "engram_host_checkpoint")
