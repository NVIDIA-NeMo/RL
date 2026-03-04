# Reproduce 32B vs 32B+14B Spec-Decoding Comparison (Greedy, Realistic Prompts)

This document reproduces the comparison run that applies all three fixes:

1. realistic prompts (OpenMath JSONL),
2. lower-entropy decoding (`temperature=0`), and
3. fewer speculative tokens (`num_speculative_tokens=2`).

It compares:

- Baseline: `Qwen3-32B` (no speculative decoding)
- Speculative: `Qwen3-32B` target + `Qwen3-14B` draft

## 1. Fresh Pull + Environment Build

```bash
mkdir -p /home/scratch.shaunakj_other/Development
cd /home/scratch.shaunakj_other/Development

git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL

git submodule update --init --depth 1

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.cargo/env"

uv venv .venv
uv sync --locked --extra vllm --extra fsdp --extra mcore --extra automodel
```

## 2. Preconditions

This repro expects:

- 8 visible GPUs
- vLLM `0.16.x`
- local model snapshots at:
  - `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
  - `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18`
- realistic prompt file:
  - `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
  - format: JSONL rows with at least an `input` field

Quick checks:

```bash
cd /home/scratch.shaunakj_other/Development/RL
.venv/bin/python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), vllm.__version__)"
nvidia-smi -L

ls -d /home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*
ls -d /home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*
ls -l /home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl
```

## 3. Export Stable Scratch Paths

```bash
export HOME=/home/scratch.shaunakj_other
export TMPDIR=/home/scratch.shaunakj_other/tmp
export UV_CACHE_DIR=/home/scratch.shaunakj_other/.cache/uv
export XDG_CACHE_HOME=/home/scratch.shaunakj_other/.cache
export VLLM_CACHE_ROOT=/home/scratch.shaunakj_other/.cache/vllm
export HF_HOME=/home/scratch.shaunakj_other/.cache/huggingface
export HF_DATASETS_CACHE=/home/scratch.shaunakj_other/.cache/hf_json_cache
export RAY_TMPDIR=/home/scratch.shaunakj_other/raytmp
export VLLM_LOG_STATS_INTERVAL=1
export PYTHONUNBUFFERED=1

mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE" "$RAY_TMPDIR"
mkdir -p /home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18
```

`VLLM_LOG_STATS_INTERVAL=1` is important; without it, short runs may finish before `SpecDecoding metrics` lines (with TAR) are logged.

## 4. Run Baseline (32B, Greedy)

```bash
cd /home/scratch.shaunakj_other/Development/RL

.venv/bin/python - <<'PY' 2>&1 | tee /home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18/rerun3b-baseline-32b-greedy-openmath64-log1s.log
import json
import time
from vllm import LLM, SamplingParams

MODEL = "/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
DATA = "/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl"
NUM_PROMPTS = 64
MAX_NEW_TOKENS = 128

prompts = []
with open(DATA, "r") as f:
    for line in f:
        obj = json.loads(line)
        prompts.append(obj["input"])
        if len(prompts) >= NUM_PROMPTS:
            break

llm = LLM(
    model=MODEL,
    tokenizer=MODEL,
    tensor_parallel_size=8,
    max_model_len=1024,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
    attention_backend="FLASH_ATTN",
    async_scheduling=False,
    disable_log_stats=False,
    seed=123,
)

sampling_params = SamplingParams(
    n=1,
    temperature=0.0,
    top_p=1.0,
    max_tokens=MAX_NEW_TOKENS,
    ignore_eos=True,
)

start = time.perf_counter()
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
elapsed = time.perf_counter() - start

tok = llm.get_tokenizer()
prompt_tokens = sum(len(tok.encode(p)) for p in prompts)
output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
total_tokens = prompt_tokens + output_tokens

print("=== RERUN3B BASELINE SUMMARY ===")
print(f"elapsed_s={elapsed:.4f}")
print(f"num_prompts={len(prompts)}")
print(f"prompt_tokens={prompt_tokens}")
print(f"output_tokens={output_tokens}")
print(f"total_tokens={total_tokens}")
print(f"requests_per_s={len(prompts)/elapsed:.4f}")
print(f"output_tokens_per_s={output_tokens/elapsed:.4f}")
print(f"total_tokens_per_s={total_tokens/elapsed:.4f}")
PY
```

## 5. Run Speculative (32B + 14B, Greedy, `num_speculative_tokens=2`)

```bash
cd /home/scratch.shaunakj_other/Development/RL

.venv/bin/python - <<'PY' 2>&1 | tee /home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18/rerun3b-spec-32b-draft14b-greedy-openmath64-spec2-log1s.log
import json
import time
from vllm import LLM, SamplingParams

TARGET = "/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"
DRAFT = "/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18"
DATA = "/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl"
NUM_PROMPTS = 64
MAX_NEW_TOKENS = 128
NUM_SPEC_TOKENS = 2

prompts = []
with open(DATA, "r") as f:
    for line in f:
        obj = json.loads(line)
        prompts.append(obj["input"])
        if len(prompts) >= NUM_PROMPTS:
            break

llm = LLM(
    model=TARGET,
    tokenizer=TARGET,
    tensor_parallel_size=8,
    max_model_len=1024,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
    attention_backend="FLASH_ATTN",
    async_scheduling=False,
    disable_log_stats=False,
    seed=123,
    speculative_config={
        "method": "draft_model",
        "model": DRAFT,
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "draft_tensor_parallel_size": 8,
    },
)

sampling_params = SamplingParams(
    n=1,
    temperature=0.0,
    top_p=1.0,
    max_tokens=MAX_NEW_TOKENS,
    ignore_eos=True,
)

start = time.perf_counter()
outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
elapsed = time.perf_counter() - start

tok = llm.get_tokenizer()
prompt_tokens = sum(len(tok.encode(p)) for p in prompts)
output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
total_tokens = prompt_tokens + output_tokens

print("=== RERUN3B SPEC SUMMARY ===")
print(f"elapsed_s={elapsed:.4f}")
print(f"num_prompts={len(prompts)}")
print(f"num_speculative_tokens={NUM_SPEC_TOKENS}")
print(f"prompt_tokens={prompt_tokens}")
print(f"output_tokens={output_tokens}")
print(f"total_tokens={total_tokens}")
print(f"requests_per_s={len(prompts)/elapsed:.4f}")
print(f"output_tokens_per_s={output_tokens/elapsed:.4f}")
print(f"total_tokens_per_s={total_tokens/elapsed:.4f}")
PY
```

## 6. Extract Throughput + TAR

### 6.1 Summaries and SpecDec lines

```bash
rg -n "RERUN3B|elapsed_s|requests_per_s|output_tokens_per_s|total_tokens_per_s|SpecDecoding metrics|Avg Draft acceptance rate|Per-position acceptance rate" \
  /home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18/rerun3b-baseline-32b-greedy-openmath64-log1s.log \
  /home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18/rerun3b-spec-32b-draft14b-greedy-openmath64-spec2-log1s.log
```

### 6.2 Speedup ratio (`spec / baseline`)

```bash
b_total=$(rg -n "^total_tokens_per_s=" /home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18/rerun3b-baseline-32b-greedy-openmath64-log1s.log | tail -n1 | sed 's/.*=//')
s_total=$(rg -n "^total_tokens_per_s=" /home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18/rerun3b-spec-32b-draft14b-greedy-openmath64-spec2-log1s.log | tail -n1 | sed 's/.*=//')
b_out=$(rg -n "^output_tokens_per_s=" /home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18/rerun3b-baseline-32b-greedy-openmath64-log1s.log | tail -n1 | sed 's/.*=//')
s_out=$(rg -n "^output_tokens_per_s=" /home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18/rerun3b-spec-32b-draft14b-greedy-openmath64-spec2-log1s.log | tail -n1 | sed 's/.*=//')

awk -v bt="$b_total" -v st="$s_total" -v bo="$b_out" -v so="$s_out" 'BEGIN{
  printf("baseline_total_tps=%s\nspec_total_tps=%s\nspec_over_base_total=%.4f\n", bt, st, st/bt);
  printf("baseline_output_tps=%s\nspec_output_tps=%s\nspec_over_base_output=%.4f\n", bo, so, so/bo);
}'
```

### 6.3 Aggregate TAR across logged intervals

```bash
.venv/bin/python - <<'PY'
import re
log = "/home/scratch.shaunakj_other/logs/specdecode-speedup-2026-02-18/rerun3b-spec-32b-draft14b-greedy-openmath64-spec2-log1s.log"
acc = dr = 0
with open(log) as f:
    for line in f:
        m = re.search(r"Accepted: (\d+) tokens, Drafted: (\d+) tokens, .*Avg Draft acceptance rate: ([0-9.]+)%", line)
        if m:
            acc += int(m.group(1))
            dr += int(m.group(2))
print(f"aggregate_accepted={acc}")
print(f"aggregate_drafted={dr}")
print(f"aggregate_tar_pct={100.0 * acc / dr:.2f}")
PY
```

## 7. Reference Output (2026-02-17 run)

From this exact setup:

- Baseline:
  - `total_tokens_per_s=2510.7412`
  - `output_tokens_per_s=1735.8420`
- Speculative:
  - `total_tokens_per_s=2523.2874`
  - `output_tokens_per_s=1744.5160`
  - interval TAR lines: `83.8%`, `87.4%`, `91.4%`
  - aggregate TAR from logged accepted/drafted totals: `87.99%` (`4507/5122`)
- Speedup:
  - `spec_over_base_total=1.0050`
  - `spec_over_base_output=1.0050`

## 8. Notes

- Do not use `vllm bench throughput` for this particular comparison if you need strict greedy equivalence; in this vLLM version it builds `SamplingParams(temperature=1.0, top_p=1.0)` in the benchmark path.
- `attention_backend="FLASH_ATTN"` and `enforce_eager=True` were used for stability in this environment.
- For this setup, `draft_tensor_parallel_size` must match target `tensor_parallel_size` (`8`).

## 9. Automated Sweep Runner

Script:

- `tools/specdecode_sweep_32b14b.py`

### 9.1 Sweep `num_speculative_tokens`

```bash
cd /home/scratch.shaunakj_other/Development/RL

export HOME=/home/scratch.shaunakj_other
export TMPDIR=/home/scratch.shaunakj_other/tmp
export UV_CACHE_DIR=/home/scratch.shaunakj_other/.cache/uv
export XDG_CACHE_HOME=/home/scratch.shaunakj_other/.cache
export VLLM_CACHE_ROOT=/home/scratch.shaunakj_other/.cache/vllm
export HF_HOME=/home/scratch.shaunakj_other/.cache/huggingface
export HF_DATASETS_CACHE=/home/scratch.shaunakj_other/.cache/hf_json_cache
export RAY_TMPDIR=/home/scratch.shaunakj_other/raytmp
export VLLM_LOG_STATS_INTERVAL=1
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE" "$RAY_TMPDIR"

.venv/bin/python tools/specdecode_sweep_32b14b.py \
  --sweep-spec-tokens 1 2 3 4 5 \
  --run-tag improving-throughput
```

Observed in this environment:

- `spec_tokens=2` was the best speculative point for `max_new_tokens=128`.
- TAR stayed high (`~78%` to `~93%`) across the sweep, but throughput dropped for larger speculative chunks.

### 9.2 Sweep decode length (find crossover)

```bash
cd /home/scratch.shaunakj_other/Development/RL

for t in 64 128 256 512; do
  .venv/bin/python tools/specdecode_sweep_32b14b.py \
    --sweep-spec-tokens 2 \
    --max-new-tokens "$t" \
    --run-tag "improving-throughput-len$t"
done
```

Observed crossover (spec vs baseline output throughput):

- `max_new_tokens=64`: `0.8109x`
- `max_new_tokens=128`: `0.9432x`
- `max_new_tokens=256`: `1.0842x`
- `max_new_tokens=512`: `1.0518x`

So this setup starts to show net throughput improvement at longer decode lengths (`256+`).
