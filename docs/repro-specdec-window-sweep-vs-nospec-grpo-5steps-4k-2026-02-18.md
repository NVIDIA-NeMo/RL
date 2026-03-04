# Repro Guide: GRPO 5-step 4k No-Spec vs Spec-Window Sweep (2026-02-18)

This is a standalone end-to-end runbook to reproduce:

1. Spec decode sweep with 32B target + 1.7B draft at speculative windows `2`, `7`, `10`
2. Non-spec baseline at the same 4k decode settings

## 1) Preconditions

- Repo path: `/home/scratch.shaunakj_other/Development/RL`
- Dataset exists:
  `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
- Target snapshot exists:
  `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
- Draft snapshot exists:
  `/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

## 2) Fresh code sync

```bash
cd /home/scratch.shaunakj_other/Development/RL

git fetch --all --prune
git pull --ff-only
git submodule update --init --depth 1
```

## 3) Environment preparation

```bash
cd /home/scratch.shaunakj_other/Development/RL

uv venv .venv
uv sync --locked --extra vllm --extra fsdp --extra mcore --extra automodel

.venv/bin/python - <<'PY'
import torch, vllm
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
print('vllm', vllm.__version__)
PY
```

## 4) Verify train-time `load_format` behavior

This experiment assumes the train path respects explicit `vllm_cfg.load_format`.

Expected snippet in `nemo_rl/models/generation/__init__.py`:

```python
if "load_format" not in config["vllm_cfg"]:
    config["vllm_cfg"]["load_format"] = "auto" if is_eval else "dummy"
```

Check:

```bash
cd /home/scratch.shaunakj_other/Development/RL
rg -n 'if "load_format" not in config\["vllm_cfg"\]' nemo_rl/models/generation/__init__.py
```

## 5) One-time policy-worker numpy compatibility fix

```bash
POLICY_PY=/home/scratch.shaunakj_other/Development/RL/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python

"$POLICY_PY" -m pip install --upgrade numpy==2.4.2

"$POLICY_PY" - <<'PY'
import numpy
import numpy._core.numeric
from torch.distributed.tensor import register_op_strategy
print('numpy', numpy.__version__)
print('dtensor register_op_strategy import OK')
PY
```

## 6) Local config with local defaults path

```bash
cd /home/scratch.shaunakj_other/Development/RL

export LOCAL_CFG=/home/scratch.shaunakj_other/tmp/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.local.yaml
mkdir -p /home/scratch.shaunakj_other/tmp

cp examples/configs/recipes/llm/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.yaml "$LOCAL_CFG"
sed -i 's#^defaults: /opt/nemo-rl/examples/configs/grpo_math_1B.yaml#defaults: /home/scratch.shaunakj_other/Development/RL/examples/configs/grpo_math_1B.yaml#' "$LOCAL_CFG"
```

## 7) Shared environment exports

```bash
cd /home/scratch.shaunakj_other/Development/RL

export TARGET=/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
export DRAFT=/home/scratch.shaunakj_other/.cache/huggingface/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e

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

mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE" "$RAY_TMPDIR"
```

## 8) Run A: spec decode sweep (`2 7 10`)

```bash
cd /home/scratch.shaunakj_other/Development/RL

SWEEP_STATUS=/home/scratch.shaunakj_other/logs/spec-window-sweep-2-7-10-$(date +%F-%H%M%S).status.txt
: > "$SWEEP_STATUS"

echo "summary_file=$SWEEP_STATUS"

for SPEC_TOKENS in 2 7 10; do
  export RUN_TAG=grpo-32b-spec1p7b-s${SPEC_TOKENS}-l4096-actckpt-loadfmtauto-sweep-$(date +%F-%H%M%S)
  export LOGROOT=/home/scratch.shaunakj_other/logs/$RUN_TAG
  export RESROOT=/home/scratch.shaunakj_other/results/$RUN_TAG
  export RUN_LOG=$LOGROOT/run-5steps-spec1p7b-s${SPEC_TOKENS}-l4096-actckpt-flashattn-loadfmtauto.log
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
    '++policy.generation.temperature=0.0',
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

## 9) Run B: no-spec baseline (`speculative_config=null`)

```bash
cd /home/scratch.shaunakj_other/Development/RL

export RUN_TAG=grpo-32b-nospec-l4096-actckpt-loadfmtauto-$(date +%F-%H%M%S)
export LOGROOT=/home/scratch.shaunakj_other/logs/$RUN_TAG
export RESROOT=/home/scratch.shaunakj_other/results/$RUN_TAG
export RUN_LOG=$LOGROOT/run-5steps-nospec-l4096-actckpt-flashattn-loadfmtauto.log
mkdir -p "$LOGROOT" "$RESROOT"

echo "RUN_TAG=$RUN_TAG"
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
    '++policy.generation.temperature=0.0',
    '++policy.generation.top_p=1.0',
    '++policy.generation.top_k=null',
    '++policy.generation.vllm_cfg.load_format=auto',
    '++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN',
    '++policy.generation.vllm_kwargs.speculative_config=null',
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
```

## 10) Verify completion

```bash
# Sweep completion summary
cat /home/scratch.shaunakj_other/logs/spec-window-sweep-2-7-10-*.status.txt

# Per-run sanity checks
rg -n "SETUP COMPLETE|Step [1-5]/5|Max number of steps has been reached" /home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-s2-l4096-actckpt-loadfmtauto-sweep-*/run-5steps-spec1p7b-s2-l4096-actckpt-flashattn-loadfmtauto.log
rg -n "SETUP COMPLETE|Step [1-5]/5|Max number of steps has been reached" /home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-s7-l4096-actckpt-loadfmtauto-sweep-*/run-5steps-spec1p7b-s7-l4096-actckpt-flashattn-loadfmtauto.log
rg -n "SETUP COMPLETE|Step [1-5]/5|Max number of steps has been reached" /home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-s10-l4096-actckpt-loadfmtauto-sweep-*/run-5steps-spec1p7b-s10-l4096-actckpt-flashattn-loadfmtauto.log
rg -n "SETUP COMPLETE|Step [1-5]/5|Max number of steps has been reached" /home/scratch.shaunakj_other/logs/grpo-32b-nospec-l4096-actckpt-loadfmtauto-*/run-5steps-nospec-l4096-actckpt-flashattn-loadfmtauto.log
```

## 11) Extract comparable metrics (steps 2-5)

```bash
cd /home/scratch.shaunakj_other/Development/RL

.venv/bin/python - <<'PY'
import re, json

logs = {
  'nospec': '/home/scratch.shaunakj_other/logs/grpo-32b-nospec-l4096-actckpt-loadfmtauto-2026-02-18-183726/run-5steps-nospec-l4096-actckpt-flashattn-loadfmtauto.log',
  'spec2': '/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-s2-l4096-actckpt-loadfmtauto-sweep-2026-02-18-175434/run-5steps-spec1p7b-s2-l4096-actckpt-flashattn-loadfmtauto.log',
  'spec7': '/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-s7-l4096-actckpt-loadfmtauto-sweep-2026-02-18-180546/run-5steps-spec1p7b-s7-l4096-actckpt-flashattn-loadfmtauto.log',
  'spec10': '/home/scratch.shaunakj_other/logs/grpo-32b-spec1p7b-s10-l4096-actckpt-loadfmtauto-sweep-2026-02-18-181751/run-5steps-spec1p7b-s10-l4096-actckpt-flashattn-loadfmtauto.log',
}

pat_step = re.compile(r"=+ Step (\\d+)/(\\d+) =+")
pat_total = re.compile(r"Total step time:\\s*([0-9.]+)s")
pat_gen = re.compile(r"generation:\\s*([0-9.]+)s")
pat_e2e = re.compile(r"E2E \\(Tokens/sec\\):\\s*([0-9.]+)")
pat_acc = re.compile(r"Token Acceptance Rate:\\s*([0-9.]+)\\s*\\((\\d+)/(\\d+)\\)")

out = {}
for name, path in logs.items():
    text = open(path).read()
    marks = [(int(m.group(1)), m.start()) for m in pat_step.finditer(text)]
    marks.append((None, len(text)))
    rows = []
    for i in range(len(marks)-1):
        s = marks[i][1]
        e = marks[i+1][1]
        b = text[s:e]
        step = marks[i][0]
        row = {
            'step': step,
            'total_s': float(pat_total.search(b).group(1)),
            'gen_s': float(pat_gen.search(b).group(1)),
            'e2e_toks_s': float(pat_e2e.search(b).group(1)),
        }
        am = pat_acc.search(b)
        if am:
            row['tar'] = float(am.group(1))
            row['acc'] = int(am.group(2))
            row['prop'] = int(am.group(3))
        rows.append(row)
    s25 = [r for r in rows if r['step'] >= 2]
    out[name] = {
        'avg_total_s_2_5': sum(r['total_s'] for r in s25)/len(s25),
        'avg_gen_s_2_5': sum(r['gen_s'] for r in s25)/len(s25),
        'avg_e2e_toks_s_2_5': sum(r['e2e_toks_s'] for r in s25)/len(s25),
        'tar_2_5': (sum(r['acc'] for r in s25 if 'acc' in r)/sum(r['prop'] for r in s25 if 'prop' in r)) if any('acc' in r for r in s25) else None,
    }

base = out['nospec']
for name in ['spec2', 'spec7', 'spec10']:
    d = out[name]
    out[name]['gen_speedup_vs_nospec'] = base['avg_gen_s_2_5'] / d['avg_gen_s_2_5']
    out[name]['e2e_speedup_vs_nospec'] = d['avg_e2e_toks_s_2_5'] / base['avg_e2e_toks_s_2_5']
    out[name]['step_time_speedup_vs_nospec'] = base['avg_total_s_2_5'] / d['avg_total_s_2_5']

print(json.dumps(out, indent=2))
PY
```

