# Reproduce GRPO + Spec-Decoding Run (5 steps, 32B target + 14B draft, scratch paths)

This runbook captures the exact setup that successfully completed the 5-step benchmark run.

## 1. Preconditions

- Repo is freshly pulled at `/home/scratch.shaunakj_other/Development/RL`
- 8 GPUs are visible
- Local model snapshots exist:
  - target: `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137`
  - draft: `/home/scratch.shaunakj_other/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18`
- Prompt dataset exists: `/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl`

## 2. Fresh environment setup

```bash
cd /home/scratch.shaunakj_other/Development/RL

# if needed after fresh pull
# git submodule update --init --depth 1

uv venv .venv
uv sync --locked --extra vllm --extra fsdp --extra mcore --extra automodel

.venv/bin/python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), vllm.__version__)"
```

## 3. One-time policy-worker venv compatibility fix

This avoids the `numpy._core.numeric` import failure when policy workers run in the automodel venv.

```bash
POLICY_PY=/home/scratch.shaunakj_other/Development/RL/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python

"$POLICY_PY" -m pip install --upgrade numpy==2.4.2

"$POLICY_PY" - <<'PY'
import numpy
import numpy._core.numeric
from torch.distributed.tensor import register_op_strategy
print("numpy", numpy.__version__)
print("dtensor register_op_strategy import OK")
PY
```

## 4. Create local config copy with local defaults path

The recipe in tree points at `/opt/nemo-rl/...`; patch it to your local checkout path.

```bash
cd /home/scratch.shaunakj_other/Development/RL

LOCAL_CFG=/home/scratch.shaunakj_other/tmp/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.local.yaml
mkdir -p /home/scratch.shaunakj_other/tmp
cp examples/configs/recipes/llm/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.yaml "$LOCAL_CFG"
sed -i 's#^defaults: /opt/nemo-rl/examples/configs/grpo_math_1B.yaml#defaults: /home/scratch.shaunakj_other/Development/RL/examples/configs/grpo_math_1B.yaml#' "$LOCAL_CFG"
```

## 5. Run command (scratch writes + policy-worker mapping)

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

# Important: Ray init hung on this host when RAY_TMPDIR was in /home/scratch.
# Keep Ray internals in /tmp/ray for this specific run.
export RAY_TMPDIR=/tmp/ray

export VLLM_LOG_STATS_INTERVAL=1
export PYTHONUNBUFFERED=1
export NRL_FORCE_LOCAL_RAY=true
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
export POLICY_PY=/home/scratch.shaunakj_other/Development/RL/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python

RUN_TAG=grpo-32b-spec14b-s2-l256-$(date +%F)
export LOGROOT=/home/scratch.shaunakj_other/logs/$RUN_TAG
export RESROOT=/home/scratch.shaunakj_other/results/$RUN_TAG
export RUN_LOG=$LOGROOT/run-5steps-spec14b-flashattn-scratchpaths-policyvenv.log
export LOCAL_CFG=/home/scratch.shaunakj_other/tmp/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.local.yaml

mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$HF_DATASETS_CACHE" "$RAY_TMPDIR" "$LOGROOT" "$RESROOT"

.venv/bin/ray stop --force || true

stdbuf -oL -eL .venv/bin/python - <<'PY' 2>&1 | tee "$RUN_LOG"
import os
import runpy
import sys
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY

policy_py = os.environ["POLICY_PY"]
ACTOR_ENVIRONMENT_REGISTRY[
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
] = policy_py
ACTOR_ENVIRONMENT_REGISTRY[
    "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"
] = policy_py

sys.argv = [
    "examples/run_grpo.py",
    "--config",
    os.environ["LOCAL_CFG"],
    "++grpo.max_num_steps=5",
    "++grpo.val_period=0",
    "++grpo.val_at_start=false",
    "++grpo.val_at_end=false",
    f"++policy.model_name={os.environ['TARGET']}",
    f"++policy.tokenizer.name={os.environ['TARGET']}",
    "++policy.generation.max_new_tokens=256",
    "++policy.generation.temperature=0.0",
    "++policy.generation.top_p=1.0",
    "++policy.generation.top_k=null",
    "++policy.generation.vllm_kwargs.attention_backend=FLASH_ATTN",
    f"++policy.generation.vllm_kwargs.speculative_config.model={os.environ['DRAFT']}",
    "++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=2",
    "++policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size=1",
    "++data.train.dataset_name=ResponseDataset",
    "++data.train.data_path=/home/scratch.shaunakj_other/openmath-cached-sample-256.jsonl",
    "++data.train.split_validation_size=0.0",
    "++data.validation=null",
    "++data.default.dataset_name=ResponseDataset",
    "++data.default.input_key=input",
    "++data.default.output_key=output",
    f"++logger.log_dir={os.environ['LOGROOT']}",
    f"++checkpointing.checkpoint_dir={os.environ['RESROOT']}",
]
runpy.run_path("examples/run_grpo.py", run_name="__main__")
PY
```

## 6. Verify run completion

```bash
rg -n "SETUP COMPLETE|Step [1-5]/5|Max number of steps has been reached" "$RUN_LOG"
ls -l "$LOGROOT"/exp_*/train_data_step*.jsonl
```

## 7. Extract step-level throughput/TAR from run log

```bash
.venv/bin/python - <<'PY'
import re, os
log = os.environ["RUN_LOG"]
text = open(log).read()

step_marks = [(int(m.group(1)), m.start()) for m in re.finditer(r"=+ Step (\d+)/(\d+) =+", text)]
step_marks.append((None, len(text)))

for i in range(len(step_marks)-1):
    step, start = step_marks[i]
    end = step_marks[i+1][1]
    block = text[start:end]

    e2e_group = re.search(r"E2E \(Tokens/sec\):\s+([0-9.]+)", block)
    e2e_gpu = re.search(r"E2E \(Tokens/sec/gpu\):\s+([0-9.]+)", block)
    tar = re.search(r"Token Acceptance Rate:\s+([0-9.]+) \((\d+)/(\d+)\)", block)
    step_time = re.search(r"Total step time:\s+([0-9.]+)s", block)

    print(
        f"step={step} "
        f"e2e_tok_s_group={e2e_group.group(1) if e2e_group else 'NA'} "
        f"e2e_tok_s_gpu={e2e_gpu.group(1) if e2e_gpu else 'NA'} "
        f"tar={tar.group(1) if tar else 'NA'} "
        f"accepted={tar.group(2) if tar else 'NA'} "
        f"drafted={tar.group(3) if tar else 'NA'} "
        f"step_time_s={step_time.group(1) if step_time else 'NA'}"
    )
PY
```

## 8. Expected behavior from the validated run

- Run reaches `Step 5/5` and exits after `Max number of steps has been reached`.
- Output files are under `$LOGROOT` and `$RESROOT`.
- In the validated run, training completed but showed `Loss: nan` and `Token Acceptance Rate: 0.0000` per step.
- A final `.nfs... resource busy` traceback may appear during process cleanup after completion.
