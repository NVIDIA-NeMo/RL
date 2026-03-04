# Reproduce vLLM 0.16.0 Spec-Decoding Acceptance Rerun (Non-Zero Acceptance)

This guide reproduces the single-step run that verifies speculative-decoding acceptance is non-zero after preserving `load_format=auto`.

## 1. Preconditions

- Repo and Python environment already exist at:
  - `/home/scratch.shaunakj_other/Development/RL`
  - `/home/scratch.shaunakj_other/Development/RL/.venv`
- Cached model snapshot exists:
  - `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18`
- Cached data file exists:
  - `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`

## 2. Use scratch-backed cache/temp paths

```bash
export HOME=/home/scratch.shaunakj_other
export TMPDIR=/home/scratch.shaunakj_other/tmp
export UV_CACHE_DIR=/home/scratch.shaunakj_other/.cache/uv
export XDG_CACHE_HOME=/home/scratch.shaunakj_other/.cache
export VLLM_CACHE_ROOT=/home/scratch.shaunakj_other/.cache/vllm
export HF_HOME=/home/scratch.shaunakj_other/.cache/huggingface
export HF_DATASETS_CACHE=/home/scratch.shaunakj_other/.cache/hf_json_cache
export RAY_TMPDIR=/home/scratch.shaunakj_other/raytmp
export NRL_SKIP_LOCAL_VENV=true
export PYTHONUNBUFFERED=1

mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE" "$RAY_TMPDIR"
```

## 3. Run the one-step acceptance rerun

```bash
cd /home/scratch.shaunakj_other/Development/RL

MODEL=/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
LOGROOT=/home/scratch.shaunakj_other/logs/specdecode-1step-greedy-autoformat-fixed-skipvenv-14b
RESROOT=/home/scratch.shaunakj_other/results/specdecode-1step-greedy-autoformat-fixed-skipvenv-14b

mkdir -p "$LOGROOT" "$RESROOT"
.venv/bin/ray stop --force || true

stdbuf -oL -eL .venv/bin/python examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen2.5-7b-spec-decode-1n8g.yaml \
  ++grpo.max_num_steps=1 \
  ++grpo.val_period=0 \
  ++cluster.gpus_per_node=1 \
  ++logger.log_dir=$LOGROOT \
  ++checkpointing.checkpoint_dir=$RESROOT \
  ++policy.model_name=$MODEL \
  ++policy.tokenizer.name=$MODEL \
  ++policy.generation.vllm_kwargs.speculative_config.model=$MODEL \
  ++grpo.num_prompts_per_step=1 \
  ++grpo.num_generations_per_prompt=1 \
  ++policy.train_global_batch_size=1 \
  ++policy.train_micro_batch_size=1 \
  ++policy.logprob_batch_size=1 \
  ++policy.generation.max_new_tokens=64 \
  ++policy.max_total_sequence_length=512 \
  ++policy.generation.vllm_cfg.max_model_len=512 \
  ++policy.generation.vllm_cfg.gpu_memory_utilization=0.5 \
  ++policy.generation.vllm_cfg.load_format=auto \
  ++data.train.dataset_name=ResponseDataset \
  ++data.train.data_path=/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl \
  ++data.train.split_validation_size=0.0 \
  ++data.validation=null \
  ++data.default.dataset_name=ResponseDataset \
  ++data.default.input_key=input \
  ++data.default.output_key=output \
  ++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN \
  ++policy.generation.temperature=0 \
  ++policy.generation.top_p=1.0 \
  ++policy.generation.top_k=null \
  2>&1 | tee $LOGROOT/run.log
```

## 4. Verify acceptance and counters

```bash
rg -n "Selected spec-decode counters|Early Token Acceptance Rate|load_format=auto|OutOfMemoryError" \
  /home/scratch.shaunakj_other/logs/specdecode-1step-greedy-autoformat-fixed-skipvenv-14b/run.log
```

Expected lines include:

- `Selected spec-decode counters: accepted=vllm:spec_decode_num_accepted_tokens, proposed=vllm:spec_decode_num_draft_tokens`
- `Early Token Acceptance Rate: 1.0000 (55/55)`
- `load_format=auto`

## 5. Notes

- Draft and verifier are intentionally the same model path in this rerun.
- The run may still fail later in policy training with CUDA OOM. That does not invalidate the speculative-decoding acceptance check because acceptance is emitted before the training failure.
