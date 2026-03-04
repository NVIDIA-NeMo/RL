# Reproduce vLLM 0.16.0 Spec-Decoding Acceptance Debug Run (Scratch)

This guide reproduces the single-step speculative-decoding debug run that shows:

- selected counters are found, and
- accepted draft-token delta is `0` while proposed draft-token delta is positive.

For the validated rerun with non-zero acceptance (`55/55`), see:
- `docs/repro-spec-decode-vllm-0.16.0-acceptance-rerun.md`
- `docs/specdecode-zero-acceptance-memory-2026-02-17.md`

It starts from pulling code into scratch, building the local environment, and running the job.

## 1. Pull Code Into Scratch

```bash
mkdir -p /home/scratch.shaunakj_other/Development
cd /home/scratch.shaunakj_other/Development

git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL

git submodule update --init --depth 1
```

## 2. Install uv and Build Python Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env"

cd /home/scratch.shaunakj_other/Development/RL
uv venv .venv

# Install locked deps with required extras for this run
uv sync --locked --extra vllm --extra fsdp --extra mcore --extra automodel
```

## 3. Confirm vLLM 0.16.0 Is Active

```bash
cd /home/scratch.shaunakj_other/Development/RL
.venv/bin/python - <<'PY'
import vllm
print(vllm.__version__)
assert vllm.__version__.startswith("0.16."), vllm.__version__
PY
```

## 4. Ensure Cached Inputs Exist

This run uses only local cached assets.

```bash
# Cached 14B snapshot used for policy + draft model
ls -d /home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*

# Cached local JSONL used as training data
ls -l /home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl
```

## 5. Use Scratch-Only Cache and Temp Paths

```bash
export HOME=/home/scratch.shaunakj_other
export TMPDIR=/home/scratch.shaunakj_other/tmp
export XDG_CACHE_HOME=/home/scratch.shaunakj_other/.cache
export VLLM_CACHE_ROOT=/home/scratch.shaunakj_other/.cache/vllm
export HF_HOME=/home/scratch.shaunakj_other/.cache/huggingface
export HF_DATASETS_CACHE=/home/scratch.shaunakj_other/.cache/hf_json_cache
export RAY_TMPDIR=/home/scratch.shaunakj_other/raytmp
export NRL_SKIP_LOCAL_VENV=true
export PYTHONUNBUFFERED=1

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE" "$RAY_TMPDIR"
```

## 6. Run the Single-Step Repro

```bash
cd /home/scratch.shaunakj_other/Development/RL

MODEL=/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18
LOGROOT=/home/scratch.shaunakj_other/logs/specdecode-1step-earlydebug-14b
RESROOT=/home/scratch.shaunakj_other/results/specdecode-1step-earlydebug-14b

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
  ++data.train.dataset_name=ResponseDataset \
  ++data.train.data_path=/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl \
  ++data.train.split_validation_size=0.0 \
  ++data.validation=null \
  ++data.default.dataset_name=ResponseDataset \
  ++data.default.input_key=input \
  ++data.default.output_key=output \
  ++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN \
  2>&1 | tee $LOGROOT/run.log
```

## 7. Verify Debug Output

```bash
rg -n "Using cache directory|Selected spec-decode counters|Early Token Acceptance|OutOfMemoryError" \
  /home/scratch.shaunakj_other/logs/specdecode-1step-earlydebug-14b/run.log
```

Expected key lines include:

- `Selected spec-decode counters: accepted=vllm:spec_decode_num_accepted_tokens, proposed=vllm:spec_decode_num_draft_tokens`
- `Early Token Acceptance Rate: 0.0000 (0/315)`
- `Early Token Acceptance Debug: ... accepted_deltas={0: 0.0}, proposed_deltas={0: 315.0} ...`
- The run may then fail during policy training with CUDA OOM; this is acceptable for this debug repro because the acceptance diagnostics are already captured.

## 8. Notes

- This repro intentionally reuses cached model/data and does not build a Docker image.
- vLLM cache should stay in scratch (`/home/scratch.shaunakj_other/.cache/vllm/...`).
- If you need full image build instructions, see `docs/repro-spec-decode-build.md`.
