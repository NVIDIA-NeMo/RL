# Repro Guide: GRPO 5-step SpecDec vs Non-Spec (2026-02-18)

This is a standalone end-to-end runbook from fresh code sync through both benchmark runs.

- Spec run: 32B target + 14B draft, `num_speculative_tokens=2`
- Non-spec run: same setup, `speculative_config=null`

## 1) Preconditions

- Repo path: `/home/scratch.shaunakj_other/Development/RL`
- Dataset exists: `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`
- 32B target snapshot exists:
  `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
- 14B draft snapshot exists:
  `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18`

## 2) Fresh code sync (pull latest)

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

## 4) Ensure train-time `load_format` override works

This comparison assumes the patch that respects explicit `vllm_cfg.load_format` on train path.

Expected snippet in `nemo_rl/models/generation/__init__.py`:

```python
if "load_format" not in config["vllm_cfg"]:
    config["vllm_cfg"]["load_format"] = "auto" if is_eval else "dummy"
```

Check it:

```bash
cd /home/scratch.shaunakj_other/Development/RL
rg -n "if \"load_format\" not in config\[\"vllm_cfg\"\]" nemo_rl/models/generation/__init__.py
```

If no match, patch before running benchmarks.

## 5) One-time policy-worker venv compatibility fix

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

## 6) Create local config with local defaults path

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
export DRAFT=/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18

export HOME=/home/scratch.shaunakj_other
export TMPDIR=/home/scratch.shaunakj_other/tmp
export UV_CACHE_DIR=/home/scratch.shaunakj_other/.cache/uv
export XDG_CACHE_HOME=/home/scratch.shaunakj_other/.cache
export VLLM_CACHE_ROOT=/home/scratch.shaunakj_other/.cache/vllm
export HF_HOME=/home/scratch.shaunakj_other/.cache/huggingface
export HF_DATASETS_CACHE=/home/scratch.shaunakj_other/.cache/hf_json_cache

# Important on this host: keep Ray internals in /tmp/ray
export RAY_TMPDIR=/tmp/ray

export VLLM_LOG_STATS_INTERVAL=1
export PYTHONUNBUFFERED=1
export NRL_FORCE_LOCAL_RAY=true
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export POLICY_PY=/home/scratch.shaunakj_other/Development/RL/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python

mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE" "$RAY_TMPDIR"
```

## 8) Run A: SpecDec benchmark (32B + 14B draft)

```bash
cd /home/scratch.shaunakj_other/Development/RL

export RUN_TAG=grpo-32b-spec14b-s2-l256-loadfmtauto-$(date +%F)
export LOGROOT=/home/scratch.shaunakj_other/logs/$RUN_TAG
export RESROOT=/home/scratch.shaunakj_other/results/$RUN_TAG
export RUN_LOG=$LOGROOT/run-5steps-spec14b-flashattn-scratchpaths-policyvenv-loadfmtauto.log
mkdir -p "$LOGROOT" "$RESROOT"

.venv/bin/ray stop --force || true

stdbuf -oL -eL .venv/bin/python - <<'PY' 2>&1 | tee "$RUN_LOG"
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
    f"++policy.model_name={os.environ['TARGET']}",
    f"++policy.tokenizer.name={os.environ['TARGET']}",
    '++policy.generation.max_new_tokens=256',
    '++policy.generation.temperature=0.0',
    '++policy.generation.top_p=1.0',
    '++policy.generation.top_k=null',
    '++policy.generation.vllm_cfg.load_format=auto',
    '++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN',
    f"++policy.generation.vllm_kwargs.speculative_config.model={os.environ['DRAFT']}",
    '++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=2',
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
```

## 9) Run B: Non-spec baseline benchmark (32B only)

```bash
cd /home/scratch.shaunakj_other/Development/RL

export RUN_TAG=grpo-32b-nospec-l256-loadfmtauto-$(date +%F)
export LOGROOT=/home/scratch.shaunakj_other/logs/$RUN_TAG
export RESROOT=/home/scratch.shaunakj_other/results/$RUN_TAG
export RUN_LOG=$LOGROOT/run-5steps-nospec-flashattn-scratchpaths-policyvenv-loadfmtauto.log
mkdir -p "$LOGROOT" "$RESROOT"

.venv/bin/ray stop --force || true

stdbuf -oL -eL .venv/bin/python - <<'PY' 2>&1 | tee "$RUN_LOG"
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
    f"++policy.model_name={os.environ['TARGET']}",
    f"++policy.tokenizer.name={os.environ['TARGET']}",
    '++policy.generation.max_new_tokens=256',
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

## 10) Verify completion for each run

```bash
rg -n "SETUP COMPLETE|Step [1-5]/5|Max number of steps has been reached" "$RUN_LOG"
ls -l "$LOGROOT"/exp_*/train_data_step*.jsonl
```

## 11) Extract per-step timings + throughput + TAR

```bash
.venv/bin/python - <<'PY'
import re, os
log = os.environ['RUN_LOG']
text = open(log).read()

marks = [(int(m.group(1)), m.start()) for m in re.finditer(r"=+ Step (\d+)/(\d+) =+", text)]
marks.append((None, len(text)))

for i in range(len(marks)-1):
    step, s = marks[i]
    e = marks[i+1][1]
    b = text[s:e]

    total = re.search(r"Total step time:\s+([0-9.]+)s", b)
    gen = re.search(r"\n\s*• generation:\s+([0-9.]+)s", b)
    e2e = re.search(r"E2E \(Tokens/sec\):\s+([0-9.]+)", b)
    gen_w = re.search(r"Generation Worker Group \(Tokens/sec\):\s+([0-9.]+)", b)
    tar = re.search(r"Spec Decode Token Acceptance Rate:\s+([0-9.]+) \((\d+)/(\d+)\)", b)

    print(
        f"step={step} "
        f"total_s={total.group(1) if total else 'NA'} "
        f"generation_s={gen.group(1) if gen else 'NA'} "
        f"e2e_tok_s={e2e.group(1) if e2e else 'NA'} "
        f"gen_worker_tok_s={gen_w.group(1) if gen_w else 'NA'} "
        f"tar={tar.group(1) if tar else 'NA'} "
        f"accepted={tar.group(2) if tar else 'NA'} "
        f"drafted={tar.group(3) if tar else 'NA'}"
    )
PY
```

## 12) Notes

- The `.nfs... resource busy` traceback may appear at teardown after successful completion.
- For this host, keeping `RAY_TMPDIR=/tmp/ray` avoided initialization stalls.
