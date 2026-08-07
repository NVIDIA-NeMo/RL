#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../..")

cd "$PROJECT_ROOT"

# ── Block 1: library-level roundtrip test (unchanged) ────────────────────────
uv run --extra mcore coverage run -a --data-file="$PROJECT_ROOT/tests/.coverage" --source="$PROJECT_ROOT/nemo_rl" \
    tests/functional/test_converter_roundtrip.py

# ── Block 2: CLI entry-point tests ───────────────────────────────────────────
MODEL="Qwen/Qwen2-0.5B"
TMPDIR=$(mktemp -d -t nrl-converter-cli-XXXXXX)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

echo "================================================================"
echo "CLI Entry Point Tests — temp dir: $TMPDIR"
echo "================================================================"

# Create all checkpoints and write their paths to $TMPDIR/paths.env.
# Uses converter_test_utils helpers so checkpoint creation is not duplicated.
uv run --extra mcore python \
    tests/functional/_cli_test_setup.py "$TMPDIR" "$MODEL"
source "$TMPDIR/paths.env"

# ── convert_dcp_to_hf.py ─────────────────────────────────────────────────────
# Exercises: parse_args(); config.yaml read; tokenizer fallback branch
# (no ../tokenizer dir beside DCP path → falls back to config["policy"]["tokenizer"]["name"]);
# hf_overrides extraction; convert_dcp_to_hf() called with those values.
echo "--- convert_dcp_to_hf.py CLI ---"
uv run --extra mcore python examples/converters/convert_dcp_to_hf.py \
    --config        "$CONFIG_YAML" \
    --dcp-ckpt-path "$DCP_PATH" \
    --hf-ckpt-path  "$TMPDIR/cli_dcp_to_hf"

uv run --extra mcore python \
    tests/functional/_cli_test_verify_dcp.py "$MODEL" "$TMPDIR/cli_dcp_to_hf"

# ── convert_megatron_to_hf.py ────────────────────────────────────────────────
# Exercises: parse_args(); config.yaml read; --hf-model-name override branch
# (args.hf_model_name is set → used instead of config["policy"]["model_name"]);
# hf_overrides extraction; strict=True default (--no-strict not passed).
echo "--- convert_megatron_to_hf.py CLI ---"
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
    --config             "$CONFIG_YAML" \
    --hf-model-name      "$MODEL" \
    --megatron-ckpt-path "$MEG_PATH" \
    --hf-ckpt-path       "$TMPDIR/cli_meg_to_hf"

uv run --extra mcore python \
    tests/functional/_cli_test_verify_meg.py "$MODEL" "$TMPDIR/cli_meg_to_hf"

# ── convert_lora_to_hf.py — merge path ──────────────────────────────────────
# Exercises: parse_args(); the else-branch of main() → merge_lora_to_hf().
echo "--- convert_lora_to_hf.py CLI (merge) ---"
uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
    --base-ckpt     "$MEG_PATH" \
    --adapter-ckpt  "$LORA_PATH" \
    --hf-model-name "$MODEL" \
    --hf-ckpt-path  "$TMPDIR/cli_lora_merged"

uv run --extra mcore python \
    tests/functional/_cli_test_verify_lora_merge.py "$MODEL" "$TMPDIR/cli_lora_merged"

# ── convert_lora_to_hf.py — adapter-only path ───────────────────────────────
# Exercises: the if-branch of main() (args.adapter_only=True) →
# export_lora_adapter_to_hf(). This branch was never tested through the CLI.
echo "--- convert_lora_to_hf.py CLI (--adapter-only) ---"
uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
    --base-ckpt     "$MEG_PATH" \
    --adapter-ckpt  "$LORA_PATH" \
    --hf-model-name "$MODEL" \
    --hf-ckpt-path  "$TMPDIR/cli_lora_adapter_only" \
    --adapter-only

# File-existence check only — weight content was already verified in the
# roundtrip test (steps 7d + adapter_only_merged_hf assertions).
[[ -f "$TMPDIR/cli_lora_adapter_only/adapter_config.json" ]] || \
    { echo "FAIL: adapter_config.json missing from adapter-only export"; exit 1; }
{ [[ -f "$TMPDIR/cli_lora_adapter_only/adapter_model.safetensors" ]] || \
  [[ -f "$TMPDIR/cli_lora_adapter_only/adapter_model.bin" ]]; } || \
    { echo "FAIL: no adapter weight file in adapter-only export"; exit 1; }
echo "✓ convert_lora_to_hf CLI (--adapter-only) produced expected PEFT directory"

echo "================================================================"
echo "ALL CLI ENTRY POINT TESTS PASSED"
echo "================================================================"
