#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

MATCHING_RECIPE="$TMP_DIR/matching_mxfp8.yaml"
MISMATCH_RECIPE="$TMP_DIR/mismatch_blockwise.yaml"
FP8_PARAM_RECIPE="$TMP_DIR/fp8_param.yaml"
DISABLED_FP8_PARAM_RECIPE="$TMP_DIR/disabled_fp8_param.yaml"
TEST_PY="$TMP_DIR/test_te_precision_config.py"

cat > "$MATCHING_RECIPE" <<'YAML'
configs:
  mxfp8:
    transformer_engine_config_type: TEQuantizationParams
    training_recipe: {fp8_quantization_recipe: mxfp8}
    evaluation_recipe: {}
matchers:
  all: {config: mxfp8, type: glob, pattern: "*", enabled: true}
YAML

cat > "$MISMATCH_RECIPE" <<'YAML'
configs:
  blockwise:
    transformer_engine_config_type: TEQuantizationParams
    training_recipe: {fp8_quantization_recipe: blockwise}
matchers:
  all: {config: blockwise, type: glob, pattern: "*", enabled: true}
YAML

cat > "$FP8_PARAM_RECIPE" <<'YAML'
configs:
  mxfp8_params:
    transformer_engine_config_type: TEQuantizationParams
    training_recipe: {fp8_quantization_recipe: mxfp8, fp8_param: true}
matchers:
  all: {config: mxfp8_params, type: glob, pattern: "*", enabled: true}
YAML

cat > "$DISABLED_FP8_PARAM_RECIPE" <<'YAML'
configs:
  mxfp8_params:
    transformer_engine_config_type: TEQuantizationParams
    training_recipe: {fp8_quantization_recipe: mxfp8, fp8_param: true}
matchers:
  all: {config: mxfp8_params, type: glob, pattern: "*"}
YAML

cat > "$TEST_PY" <<'PY'
import sys
import warnings
from types import SimpleNamespace

import torch

from nemo_rl.models.megatron.setup import _apply_precision_config


def apply_precision_config(recipe_file: str):
    model_cfg = SimpleNamespace(bf16=False, fp16=False)
    config = {
        "megatron_cfg": {
            "pipeline_dtype": "bfloat16",
            "te_precision_config_file": recipe_file,
            "fp8_cfg": {"enabled": True, "fp8_recipe": "mxfp8"},
        }
    }
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        _apply_precision_config(model_cfg, config, torch.bfloat16)
    assert any("fp8_cfg" in str(w.message) for w in warning_records)
    return model_cfg


def expect_value_error(recipe_file: str, message_fragment: str) -> None:
    try:
        apply_precision_config(recipe_file)
    except ValueError as exc:
        assert message_fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"{recipe_file} did not raise ValueError")


(
    matching_recipe,
    mismatch_recipe,
    fp8_param_recipe,
    disabled_fp8_param_recipe,
) = sys.argv[1:5]

model_cfg = apply_precision_config(matching_recipe)
assert model_cfg.quant_recipe.matchers

expect_value_error(mismatch_recipe, "mixed FP8 precision recipes")
expect_value_error(fp8_param_recipe, "fp8_param or fp4_param")

model_cfg = apply_precision_config(disabled_fp8_param_recipe)
assert model_cfg.quant_recipe.matchers == []

print("TE precision config validation functional test passed")
PY

cd "$PROJECT_ROOT"
uv run --extra mcore coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    "$TEST_PY" \
    "$MATCHING_RECIPE" \
    "$MISMATCH_RECIPE" \
    "$FP8_PARAM_RECIPE" \
    "$DISABLED_FP8_PARAM_RECIPE"
