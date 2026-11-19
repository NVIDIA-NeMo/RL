# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check native initialization after the RL training defaults are applied."""

import ast
import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

# Exercise the production normalization without importing GPU worker dependencies.
source = Path(__file__).resolve().parents[4] / "nemo_rl/models/generation/__init__.py"
function = next(
    node
    for node in ast.parse(source.read_text()).body
    if isinstance(node, ast.FunctionDef) and node.name == "configure_generation_config"
)
namespace = {
    "GenerationConfig": dict,
    "TokenizerType": object,
    "VllmConfig": dict,
    "TrtllmConfig": dict,
    "cast": cast,
    "warnings": warnings,
    "VLLM_SPARSE_REFIT_TRANSPORTS": (),
}
exec(
    compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
    namespace,
)
configure_generation_config = namespace[function.name]


@pytest.mark.parametrize("initial_format", ["auto", "dummy"])
@pytest.mark.parametrize("is_eval", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_native_checkpoint_loading_survives_training_defaults(
    initial_format, is_eval, native
):
    config = {
        "backend": "vllm",
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {"load_format": initial_format},
        "vllm_kwargs": {
            "hf_overrides": {
                "expert_dtype": "fp4" if native else "bf16",
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [32, 32],
                    "scale_fmt": "ue8m0",
                    "expert_dtype": "fp4",
                }
                if native
                else None,
            }
        },
    }
    overrides = deepcopy(config["vllm_kwargs"])
    result = configure_generation_config(
        config, SimpleNamespace(pad_token_id=0, eos_token_id=1), is_eval=is_eval
    )
    assert result["vllm_cfg"]["load_format"] == (
        "auto" if native or is_eval else "dummy"
    )
    assert result["vllm_kwargs"] == overrides
