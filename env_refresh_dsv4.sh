#!/bin/bash
# Wrapper around env_refresh.sh that additionally bakes two patches over
# the freshly-built vLLM worker venv before --container-save fires:
#
#   1. tools/vllm_deepseek_v4_config_patch.py over
#      vllm/transformers_utils/configs/deepseek_v4.py — defensive superset
#      of upstream main's narrow fix (which now declares
#      max_position_embeddings/rope_* as kwargs but not compress_*/etc.).
#      Track upstream: https://github.com/vllm-project/vllm/pull/40860
#
#   2. tools/patch_vllm_dsv4_base_fp8_quick.sh applied to the venv's vllm
#      module files. Adds env-gated DSV4-Flash-Base FP8-block-quant support
#      (routing experts through Fp8MoEMethod, .scale -> .weight_scale_inv
#      mapping, k_norm -> Identity, use_mega_moe forced False). Requires
#      VLLM_DSV4_BASE_FP8=1 at runtime to activate; the patched code is a
#      no-op when the flag is unset, so Flash workflows are unaffected.
set -e

REPO=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/src/NeMo-RL/nemo-rl-deepseek-v4
cd "$REPO"

bash env_refresh.sh

VLLM_VENV=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker

echo ""
echo "=== [1/2] Applying vLLM DSV4 config patch over the venv ==="
TARGET=$VLLM_VENV/lib/python3.13/site-packages/vllm/transformers_utils/configs/deepseek_v4.py
cp -v "$REPO/tools/vllm_deepseek_v4_config_patch.py" "$TARGET"
md5sum "$TARGET"

echo ""
echo "=== [2/2] Applying DSV4-Base FP8 quick-patch (idempotent) ==="
bash "$REPO/tools/patch_vllm_dsv4_base_fp8_quick.sh" "$VLLM_VENV/bin/python"

echo ""
echo "=== Both patches baked in. Set VLLM_DSV4_BASE_FP8=1 at runtime to enable Base FP8 path. ==="
