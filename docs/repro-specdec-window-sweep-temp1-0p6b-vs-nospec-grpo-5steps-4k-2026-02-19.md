# Repro Guide: GRPO 5-step 4k Temp=1 Spec-Window Sweep (Qwen3-0.6B Draft) vs No-Spec

This guide reproduces the completed sweep and matching baseline methodology.

## 1) Preconditions

- Repo path: `/home/scratch.shaunakj_other/Development/RL`
- Dataset exists:
  `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
- Target snapshot exists:
  `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
- Draft snapshot exists:
  `/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`

## 2) Environment setup

```bash
cd /home/scratch.shaunakj_other/Development/RL

uv venv .venv
uv sync --locked --extra vllm --extra fsdp --extra mcore --extra automodel
```

## 3) Shared exports

```bash
cd /home/scratch.shaunakj_other/Development/RL

export TARGET=/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
export DRAFT=/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca

export HOME=/home/scratch.shaunakj_other
export TMPDIR=/home/scratch.shaunakj_other/tmp
export UV_CACHE_DIR=/home/scratch.shaunakj_other/.cache/uv
export XDG_CACHE_HOME=/home/scratch.shaunakj_other/.cache
export VLLM_CACHE_ROOT=/home/scratch.shaunakj_other/.cache/vllm
export HF_HOME=/home/scratch.shaunakj_other/.cache/huggingface
export HF_DATASETS_CACHE=/home/scratch.shaunakj_other/.cache/hf_json_cache

export RAY_TMPDIR=/tmp/ray
export VLLM_LOG_STATS_INTERVAL=1
export PYTHONUNBUFFERED=1
export NRL_FORCE_LOCAL_RAY=true
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export POLICY_PY=/home/scratch.shaunakj_other/Development/RL/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python

# Important for this sweep: avoid stale compile-cache shape reuse.
export VLLM_DISABLE_COMPILE_CACHE=1

mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE" "$RAY_TMPDIR"
```

## 4) Local config path fix

```bash
cd /home/scratch.shaunakj_other/Development/RL

export LOCAL_CFG=/home/scratch.shaunakj_other/tmp/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.local.yaml
cp examples/configs/recipes/llm/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.yaml "$LOCAL_CFG"
sed -i 's#^defaults: /opt/nemo-rl/examples/configs/grpo_math_1B.yaml#defaults: /home/scratch.shaunakj_other/Development/RL/examples/configs/grpo_math_1B.yaml#' "$LOCAL_CFG"
```

## 5) Run sweep (`s in 1 2 4 7 10`)

```bash
cd /home/scratch.shaunakj_other/Development/RL

SWEEP_STATUS=/home/scratch.shaunakj_other/logs/spec-window-sweep-temp1-0p6b-1-2-4-7-10-nocache-$(date +%F-%H%M%S).status.txt
: > "$SWEEP_STATUS"

echo "summary_file=$SWEEP_STATUS"

for SPEC_TOKENS in 1 2 4 7 10; do
  export SPEC_TOKENS

  # Unique inductor cache per point to avoid graph-shape contamination
  export TORCHINDUCTOR_CACHE_DIR=/home/scratch.shaunakj_other/tmp/torchinductor_0p6b_s${SPEC_TOKENS}_$(date +%s)
  mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

  export RUN_TAG=grpo-32b-spec0p6b-temp1-s${SPEC_TOKENS}-l4096-actckpt-loadfmtauto-sweep-nocache-$(date +%F-%H%M%S)
  export LOGROOT=/home/scratch.shaunakj_other/logs/$RUN_TAG
  export RESROOT=/home/scratch.shaunakj_other/results/$RUN_TAG
  export RUN_LOG=$LOGROOT/run-5steps-spec0p6b-temp1-s${SPEC_TOKENS}-l4096-actckpt-flashattn-loadfmtauto-nocache.log
  mkdir -p "$LOGROOT" "$RESROOT"

  echo "=== START SPEC_TOKENS=${SPEC_TOKENS} ==="
  echo "RUN_LOG=$RUN_LOG"

  .venv/bin/ray stop --force || true

  time stdbuf -oL -eL .venv/bin/python - <<'PY' 2>&1 | tee "$RUN_LOG"
import os
import runpy
import sys
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY

policy_py = os.environ['POLICY_PY']
ACTOR_ENVIRONMENT_REGISTRY[
    'nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2'
] = policy_py
ACTOR_ENVIRONMENT_REGISTRY[
    'nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker'
] = policy_py

sys.argv = [
    'examples/run_grpo.py',
    '--config',
    os.environ['LOCAL_CFG'],
    '++grpo.max_num_steps=5',
    '++grpo.val_period=0',
    '++grpo.val_at_start=false',
    '++grpo.val_at_end=false',
    '++policy.dtensor_cfg.activation_checkpointing=true',
    f"++policy.model_name={os.environ['TARGET']}",
    f"++policy.tokenizer.name={os.environ['TARGET']}",
    '++policy.generation.max_new_tokens=4096',
    '++policy.generation.temperature=1.0',
    '++policy.generation.top_p=1.0',
    '++policy.generation.top_k=null',
    '++policy.generation.vllm_cfg.load_format=auto',
    '++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN',
    f"++policy.generation.vllm_kwargs.speculative_config.model={os.environ['DRAFT']}",
    f"++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens={os.environ['SPEC_TOKENS']}",
    '++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1',
    '++data.train.dataset_name=ResponseDataset',
    '++data.train.data_path=/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl',
    '++data.train.split_validation_size=0.0',
    '++data.validation=null',
    '++data.default.dataset_name=ResponseDataset',
    '++data.default.input_key=input',
    '++data.default.output_key=output',
    f"++logger.log_dir={os.environ['LOGROOT']}",
    f"++checkpointing.checkpoint_dir={os.environ['RESROOT']}",
]
runpy.run_path('examples/run_grpo.py', run_name='__main__')
PY

  rc=$?
  echo "spec_tokens=${SPEC_TOKENS} rc=${rc} run_log=${RUN_LOG}" | tee -a "$SWEEP_STATUS"
  echo "=== DONE SPEC_TOKENS=${SPEC_TOKENS} rc=${rc} ==="
done

echo "SWEEP_DONE"
cat "$SWEEP_STATUS"
```

## 6) No-spec baseline

Use matched temp=1/4k no-spec run from:

`/home/scratch.shaunakj_other/logs/grpo-32b-nospec-temp1-l4096-actckpt-loadfmtauto-2026-02-18-202012/run-5steps-nospec-temp1-l4096-actckpt-flashattn-loadfmtauto.log`

If you need to rerun no-spec instead of reusing that artifact, use the same command as above with:

- `++policy.generation.vllm_kwargs.speculative_config=null`
- and identical other overrides.

## 7) Parse steady-state metrics (steps 2-5)

```bash
# Example metric extraction helper for one run log
LOG=/path/to/run.log
awk '
  match($0, /Step ([0-9]+)\/5/, m) {step=m[1]}
  match($0, /Total step time: ([0-9.]+)s/, m) {tot[step]=m[1]}
  match($0, /  • generation: ([0-9.]+)s/, m) {gen[step]=m[1]}
  match($0, /- E2E \(Tokens\/sec\): ([0-9.]+)/, m) {e2e[step]=m[1]}
  match($0, /Spec Decode Token Acceptance Rate: ([0-9.]+) \(([0-9]+)\/([0-9]+)\)/, m) {acc[step]=m[2]; prop[step]=m[3]}
  END {
    for (i=2;i<=5;i++) {gs+=gen[i]; ts+=tot[i]; es+=e2e[i]; as+=acc[i]; ps+=prop[i]}
    printf("avg_gen=%.2f avg_step=%.2f avg_e2e=%.2f tar=%.4f\n", gs/4, ts/4, es/4, as/ps)
  }
' "$LOG"
```

## 8) Completion criteria

- Status file contains 5 lines, each `spec_tokens=<...> rc=0 ...`
- Final loop prints `SWEEP_DONE`.
- Expected non-fatal teardown noise may include `.nfs*` cleanup `OSError: [Errno 16] Device or resource busy`.
