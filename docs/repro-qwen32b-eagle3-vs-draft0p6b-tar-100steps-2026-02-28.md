# Repro: Qwen3-32B TAR Comparison (0.6B Draft vs Eagle3), 100 Steps

Date: 2026-02-28

## Goal and Context

This runbook captures an apples-to-apples GRPO comparison for the hypothesis:

- verifier: `Qwen3-32B`
- arm A: speculative method `draft_model` with `Qwen3-0.6B`
- arm B: speculative method `eagle3` with `RedHatAI/Qwen3-32B-speculator.eagle3`
- all other training and decoding hyperparameters held constant
- speculative window: `num_speculative_tokens=4`
- checkpointing: enabled, save every 10 steps

The completed 0.6B baseline this compares against:

- `/home/scratch.shaunakj_other/logs/grpo-32b-spec0p6b-temp0p6-s4-seed124-train-normal100-2026-02-25-235814`

## Important Runtime Caveat (Why `HOME` Must Be Scratch)

`nemo_rl/models/generation/vllm/vllm_worker.py` sets per-worker:

- `VLLM_CACHE_ROOT=~/.cache/vllm_<seed>`

So if `HOME` is not on scratch, vLLM compile cache may go to `/home/...` and fail with:

- `OSError: [Errno 28] No space left on device`

To force all vLLM compile caches onto scratch, set:

- `HOME=/home/scratch.shaunakj_other`

## 1. Shared Setup (Both Arms)

```bash
cd /home/scratch.shaunakj_other/Development/RL

export SCR=/home/scratch.shaunakj_other
export BASE_CFG=/home/scratch.shaunakj_other/tmp/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.2026-02-25-235814.seed124.normal100.local.yaml

export VERIFIER=/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
export DRAFT06B=/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
export EAGLE3=/home/scratch.shaunakj_other/.cache/hf_user/hub/models--RedHatAI--Qwen3-32B-speculator.eagle3/snapshots/e5756763c9b3bef3cc260cab70b76008fb42a81b

# scratch-backed runtime/caches (short TMP path to avoid AF_UNIX path-length issues)
export HOME="$SCR"
export TMPDIR="$SCR/t"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export RAY_TMPDIR="$TMPDIR"

export HF_HOME="$SCR/.cache/hf_user"
export HF_HUB_CACHE="$SCR/.cache/hf_user/hub"
export HUGGINGFACE_HUB_CACHE="$SCR/.cache/hf_user/hub"
export TRANSFORMERS_CACHE="$SCR/.cache/hf_user/hub"

export XDG_CACHE_HOME="$SCR/c/xdg"
export TORCHINDUCTOR_CACHE_DIR="$SCR/c/ti"
export TRITON_CACHE_DIR="$SCR/c/triton"
export VLLM_CONFIG_ROOT="$SCR/c/vllm"
export VLLM_CACHE_ROOT="$SCR/c/vllm"
export CUDA_CACHE_PATH="$SCR/c/cuda"

mkdir -p "$TMPDIR" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" "$VLLM_CONFIG_ROOT" "$CUDA_CACHE_PATH" "$SCR/logs" "$SCR/results"
```

## 2. Launch Helper (Keeps Params Fixed; Swaps Only Spec Method/Model)

```bash
run_arm () {
  local ARM_NAME="$1"         # e.g. draft0p6b or eagle3
  local SPEC_METHOD="$2"      # draft_model or eagle3
  local SPEC_MODEL="$3"       # model path
  local TS
  TS="$(date +%Y-%m-%d-%H%M%S)"
  export TAG="grpo-32b-${ARM_NAME}-temp0p6-s4-seed124-train-normal100-ckpt10-${TS}"
  export LOGDIR="$SCR/logs/$TAG"
  export RESROOT="$SCR/results/$TAG"
  mkdir -p "$LOGDIR" "$RESROOT"

  uv run ./examples/run_grpo.py --config "$BASE_CFG" \
    ++grpo.max_num_steps=100 \
    ++grpo.max_num_epochs=1 \
    ++grpo.val_period=0 \
    ++grpo.val_at_start=false \
    ++grpo.val_at_end=false \
    ++grpo.seed=124 \
    ++grpo.num_prompts_per_step=2 \
    ++grpo.num_generations_per_prompt=4 \
    ++policy.dtensor_cfg.activation_checkpointing=true \
    ++policy.model_name="$VERIFIER" \
    ++policy.tokenizer.name="$VERIFIER" \
    ++policy.max_total_sequence_length=5120 \
    ++policy.generation.max_new_tokens=4096 \
    ++policy.generation.temperature=0.6 \
    ++policy.generation.top_p=1.0 \
    ++policy.generation.top_k=null \
    ++policy.generation.vllm_cfg.load_format=auto \
    ++policy.generation.vllm_cfg.max_model_len=5120 \
    ++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN \
    ++policy.generation.vllm_kwargs.speculative_config.model="$SPEC_MODEL" \
    ++policy.generation.vllm_kwargs.speculative_config.method="$SPEC_METHOD" \
    ++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=4 \
    ++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1 \
    ++data.train.dataset_name=ResponseDataset \
    ++data.train.data_path=/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl \
    ++data.train.split_validation_size=0.0 \
    ++data.validation=null \
    ++data.default.dataset_name=ResponseDataset \
    ++data.default.input_key=input \
    ++data.default.output_key=output \
    ++env.math.num_workers=1 \
    ++logger.log_dir="$LOGDIR" \
    ++checkpointing.enabled=true \
    ++checkpointing.save_period=10 \
    ++checkpointing.checkpoint_dir="$RESROOT" \
    ++policy.optimizer.kwargs.lr=5e-6 \
    ++loss_fn.ratio_clip_min=0.2 \
    ++loss_fn.ratio_clip_max=0.2 \
    ++loss_fn.reference_policy_kl_penalty=0.01 \
    ++policy.max_grad_norm=1.0
}
```

## 3. Run Each Arm

### Arm A: 0.6B Draft

```bash
run_arm draft0p6b draft_model "$DRAFT06B"
```

### Arm B: Eagle3

```bash
run_arm eagle3 eagle3 "$EAGLE3"
```

## 4. Poll Every 10 Minutes

```bash
# set LOGFILE to your run's stdout log if using tee/nohup.
# if launching in foreground, use the same terminal output instead.
while true; do
  date
  rg -n "SETUP COMPLETE|Step [0-9]+/100|Saving checkpoint for step|Token Acceptance Rate|No space left on device|Traceback|ERROR" "$LOGFILE" | tail -n 30
  sleep 600
done
```

## 5. Pick Up Latest Checkpoint

```bash
latest_ckpt () {
  local RESROOT="$1"
  find "$RESROOT" -maxdepth 1 -type d -name 'step_*' | sort -V | tail -n 1
}

LATEST="$(latest_ckpt "$RESROOT")"
echo "LATEST=$LATEST"
cat "$LATEST/training_info.json"
```

Checkpoint layout:

- `$RESROOT/step_<N>/training_info.json`
- `$RESROOT/step_<N>/policy/weights`
- `$RESROOT/step_<N>/policy/optimizer`
- `$RESROOT/step_<N>/policy/tokenizer`

## 6. Resume From Last Checkpoint (Same Run, Continue Steps)

GRPO auto-resumes from the latest `step_*` in `checkpointing.checkpoint_dir`.
So to continue, rerun with:

- same `checkpointing.checkpoint_dir`
- larger `grpo.max_num_steps`

Example (continue to 200):

```bash
uv run ./examples/run_grpo.py --config "$BASE_CFG" \
  ...same overrides as launch... \
  ++grpo.max_num_steps=200 \
  ++checkpointing.enabled=true \
  ++checkpointing.save_period=10 \
  ++checkpointing.checkpoint_dir="$RESROOT"
```

If `max_num_steps` is already reached, it will exit quickly after loading state.

