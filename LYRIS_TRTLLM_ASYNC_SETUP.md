# NeMo-RL + TRT-LLM async GRPO on Lyris — setup & troubleshooting log

Running async (1-step off-policy) GRPO for Qwen3-8B with **Megatron training (TP=4)** +
**TRT-LLM generation (TP=4)** across 2× GB200 nodes on the **lyris** cluster, reusing
Shiki's prebaked container.

---

## 1. Key paths

| Thing | Path |
|---|---|
| RL repo (this one; branch `trtllm`) | `/lustre/fsw/coreai_comparch_trtllm/erinh/RL` |
| Launch bundle | `/lustre/fsw/coreai_comparch_trtllm/erinh/trtllm_verl_scripts/nemo-rl-scripts/` |
| Container (Shiki's sqsh) | `/lustre/fsw/coreai_comparch_trtllm/shikiw/images/nemo-rl-py313-trtllm.sqsh` |
| Local model (no HF download) | `/lustre/fsw/coreai_comparch_trtllm/erinh/llm-models/Qwen3/Qwen3-8B-Base` |
| Secrets/HF cache env | `/lustre/fsw/coreai_comparch_trtllm/erinh/launch_scripts/env.sh` |
| Config | `examples/configs/recipes/llm/grpo-qwen3-8b-2n4g-megatron-trtllm-async-1off.yaml` |
| In-container venv (has `tensorrt_llm`) | `/opt/nemo_rl_venv` |
| Repo mount point in container | `/opt/nemo-rl` |

Related repos: `RL_shiki` (Shiki's sync repro, `shikicloud/RL.git`), `RL_shuyi`
(`shuyix/asyncrl_trtllm`, the async source branch). This RL repo already contains the
async work; the only extra commit on the async branch is "Initial AsyncRL support".

---

## 2. The launch bundle (`nemo-rl-scripts/`)

Cluster-aware launcher. Files:

- `cluster_config.yaml` — per-cluster values keyed by `$WS_CLUSTER_NAME` (only values that
  DIFFER between clusters: `container`, `rl_repo`, `partition`, `env_script`).
- `cluster_env.sh` — pure-awk parser; `source` it to export `WS_CONTAINER / WS_RL_REPO /
  WS_PARTITION / WS_ENV_SCRIPT`. Defaults `WS_CLUSTER_NAME=lyris`.
- `start_container.sh` — parses `--nodes/--segment`, resolves cluster values, `sbatch ray.sub`.
- `ray.sub` — brings up the 2-node Ray cluster; sources `node_init_script.sh` on **both**
  nodes at bringup, then idles + writes `<jobid>-attach.sh`.
- `node_init_script.sh` — per-node env: PATH/CPATH/CUDA, NCCL pin, sources `env.sh`
  (`WS_ENV_SCRIPT`) for HF/WANDB tokens + cache, and `pip install -e $WS_RL_REPO`
  (re-points `nemo_rl` at the repo).

`~/.bashrc` has `export WS_CLUSTER_NAME=lyris` and the `gnt` aliases (point at this bundle).

---

## 3. Quick start (lyris)

```bash
source ~/.bashrc                       # WS_CLUSTER_NAME=lyris + gnt aliases
g2nt                                   # submit 2-node job  (g1nt = 1 node)
#   -> sbatch'd; writes nemo-rl-scripts/<jobid>-attach.sh, brings up Ray, idles

# once head is up (watch <jobid>-logs/ray-head.log):
cd /lustre/fsw/coreai_comparch_trtllm/erinh/trtllm_verl_scripts/nemo-rl-scripts
bash <jobid>-attach.sh                  # interactive shell on the head node

# inside the head container:
cd /lustre/fsw/coreai_comparch_trtllm/erinh/RL
source /lustre/fsw/coreai_comparch_trtllm/erinh/trtllm_verl_scripts/nemo-rl-scripts/node_init_script.sh
/opt/nemo_rl_venv/bin/python examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3-8b-2n4g-megatron-trtllm-async-1off.yaml \
  2>&1 | tee /lustre/fsw/coreai_comparch_trtllm/erinh/RL/8b-trtllm-async-$(date +%m%d).log
```

### Non-negotiables
- **Use `/opt/nemo_rl_venv/bin/python`, NOT `uv run`.** `uv run` re-syncs/wipes the venv (see bump #1).
- **`cd` into this RL repo before launching** — so the entry script and the `nemo_rl`
  library are the same repo.
- Source `node_init_script.sh` **only on the head** at launch — workers got it at bringup.
- Model path is already hard-coded local in the yaml; no `policy.model_name=` override needed.

---

## 4. Bumps we hit (and fixes)

### #1 — `uv run` wipes the baked venv (loses `tensorrt_llm`)
`uv run python examples/run_grpo.py` printed `Removed virtual environment at:
/opt/nemo_rl_venv` then rebuilt from the repo's lock and failed. `uv run` manages the venv
from the repo's `uv.lock`; the baked venv (which holds an editable `tensorrt_llm`) doesn't
match, so it nukes it.
**Fix:** launch with `/opt/nemo_rl_venv/bin/python` directly. Never `uv run` against this container.

### #2 — repo mount vs `nemo_rl` import path
`/opt/nemo-rl` is a **bind mount** (fixed at container start; can't be changed live). The
*imported* `nemo_rl` is whatever the editable install points at. To run a different repo on a
live job, re-point the install instead of remounting:
```bash
/opt/nemo_rl_venv/bin/pip install -e /lustre/fsw/coreai_comparch_trtllm/erinh/RL \
  --no-deps --ignore-requires-python
```
`--no-deps` keeps the baked env intact; `--ignore-requires-python` bypasses the repo's
py312 pin on this py313 image. `node_init_script.sh` does this automatically via `$WS_RL_REPO`.
Verify: `findmnt -no SOURCE --target /opt/nemo-rl` (mount) vs
`python -c "import nemo_rl,os;print(os.path.dirname(nemo_rl.__file__))"` (import).

### #3 — submodule init: `fatal: transport 'file' not allowed`
git's CVE-2022-39253 blocks `file://` submodule fetches (Megatron-LM appears twice in the tree).
**Fix:**
```bash
cd /lustre/fsw/coreai_comparch_trtllm/erinh/RL
git -c protocol.file.allow=always submodule update --init --recursive
```

### #4 — submodule fetch tries remote `shuyi`: `'shuyi' does not appear to be a git repository`
This repo's branch `trtllm` tracks a remote named `shuyi` (`shuyixiong/RL.git`). When a pinned
submodule commit isn't on the submodule's `origin`, git's fallback fetches from a remote named
after the **superproject branch's remote** (`shuyi`), which doesn't exist inside the submodule.
The pinned Megatron-LM commit `d30c3ae…` was absent from the top-level clone but present in
Megatron-Bridge's nested Megatron-LM.
**Fix — fetch the commit from the local sibling clone:**
```bash
cd /lustre/fsw/coreai_comparch_trtllm/erinh/RL/3rdparty/Megatron-LM-workspace/Megatron-LM
git -c protocol.file.allow=always fetch \
  ../../Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  d30c3ae5469fe3f6a64d4fd2e63b6e7f7844ea81
git checkout d30c3ae5469fe3f6a64d4fd2e63b6e7f7844ea81
```
Verify all clean: `git submodule status --recursive | grep '^[-+]'` (no output = clean).

### #5 — `node_init_nemorl.sh` sourced a non-existent `env.sh`
Line was `source /lustre/.../erinh/env.sh` (doesn't exist). The real one is
`/lustre/.../erinh/launch_scripts/env.sh` (sets `HF_TOKEN/HF_HOME/HF_DATASETS_CACHE/WANDB_*`).
With the wrong path, none of those got set → run dies on HF/wandb auth.
**Fix:** point at `launch_scripts/env.sh`. The bundle's `node_init_script.sh` now resolves it
via `WS_ENV_SCRIPT` from `cluster_config.yaml`.

### #6 — two parallel `node_init` files
- `node_init_script.sh` (bundle) — sourced by `nemo-rl-scripts/ray.sub`. **Canonical; what `g2nt` uses.**
- `node_init_nemorl.sh` (repo) — sourced only by the older `trtllm_verl_scripts/ray_nemorl.sub`
  (the `--async` parent flow). **Obsolete for the bundle flow.**
Use `node_init_script.sh` at launch. Don't mix them.

### #7 — wrong-cluster paths (oci-hsg vs lyris)
The bundle was authored for **oci-hsg** (`/lustre/fsw/portfolios/coreai/users/erinh/...`,
`--partition=batch`), which don't exist on lyris. lyris uses
`/lustre/fsw/coreai_comparch_trtllm/erinh/...` and `--partition=gb200`.
**Fix:** all such values live in `cluster_config.yaml` now, selected by `WS_CLUSTER_NAME=lyris`.
Also `ray.sub` was changed to `source "$SLURM_SUBMIT_DIR/node_init_script.sh"` (was a hardcoded
oci-hsg path), and the sbatch `--gpus-per-node` flag was dropped (gb200 reports no GRES;
`--exclusive` claims the node).

### #8 — model downloaded from HF instead of using the local copy
`model_name: Qwen/Qwen3-8B-Base` is an HF id → downloads. Hard-coded to the local path
`/lustre/.../llm-models/Qwen3/Qwen3-8B-Base` in the yaml (tokenizer follows `model_name`).

### #9 — dataset cache incompatible: `TypeError: must be called with a dataclass type or instance`
`load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")` read a **prepared cache written by a newer
`datasets`** (feature `"_type": "List"`). The container ships `datasets 3.1.0`, which has no
`List` feature → `fields(class_type)` blows up.
**Fix — delete the prepared cache; it rebuilds from the local parquet (no download needed):**
```bash
rm -rf /lustre/fsw/coreai_comparch_trtllm/erinh/.hf_data_cache/BytedTsinghua-SIA___dapo-math-17k
rm -rf /lustre/fsw/coreai_comparch_trtllm/erinh/.hf_data_cache/BytedTsinghua-SIA___aime-2024
```
The raw parquet stays in `.hf_model_cache/hub/datasets--BytedTsinghua-SIA--DAPO-Math-17k`;
`datasets 3.1.0` re-infers a compatible `Sequence` schema. Recurs only if a tool with newer
`datasets` rewrites the cache — clear `.hf_data_cache/<dataset>` again.

### #10 — async weight update fails: `PyExecutor.control_action() got an unexpected keyword argument 'drain'`
The async recipe (`in_flight_weight_updates: true`) makes NeMo-RL call TRT-LLM's
`control_action(drain=…)`, but the container's `tensorrt_llm` (v1.3.0rc15) doesn't have the
in-flight-weight-update support. The refit raises, then the trajectory collector hangs
(`⏸️ Pausing collection ... Waiting for weight update...`). The run otherwise gets all the way
through model load + dataset + TRT-LLM generation + "Starting async GRPO training" first.

**Cause:** two commits from Shuyi's TRT-LLM fork were never applied to the container's
editable `tensorrt_llm` at `/workspace/TensorRT-LLM`:
- `ce008ed870909b951f75741b3d120f1ea51ade59` ("Support inflight weight update")
- `def8e9da2a7e6c70eb9a9e58459c8a4eddb7b826` ("Remove preempt_all_inflight_requests …")
from `https://github.com/shuyixiong/TensorRT-LLM` branch `user/shuyix/inflight_weight_update`.

**Fix — cherry-pick them into `/workspace/TensorRT-LLM` on EVERY node** (it's editable-installed,
so the `.py` change is live on next import; GitHub is reachable from the container):
```bash
cd /workspace/TensorRT-LLM
git fetch https://github.com/shuyixiong/TensorRT-LLM.git user/shuyix/inflight_weight_update
git -c user.email=erinh@nvidia.com -c user.name=erinh cherry-pick \
  ce008ed870909b951f75741b3d120f1ea51ade59 \
  def8e9da2a7e6c70eb9a9e58459c8a4eddb7b826
# verify:
grep -n "def control_action" tensorrt_llm/_torch/pyexecutor/py_executor.py
#   -> def control_action(self, *, drain: bool = True):
```
Applies cleanly onto v1.3.0rc15 (auto-merges `py_executor.py`, no conflicts). On a live job,
run it inside each node's container via `bash <jobid>-attach.sh` (head) and
`bash <jobid>-attach.sh 1` (worker).

**⚠️ Ephemeral:** this lives in the per-node container overlay — gone when the job ends; a fresh
`g2nt` job comes up unpatched. To persist, add an idempotent fetch+cherry-pick block to
`node_init_script.sh` (runs on both nodes at bringup), guarded by
`grep -q 'def control_action(self, \*, drain' …/py_executor.py`. *(Not yet added — TODO.)*

---

## 5. Handy verification commands

```bash
# which repo is mounted vs which nemo_rl imports (run inside container)
findmnt -no SOURCE --target /opt/nemo-rl
/opt/nemo_rl_venv/bin/python -c "import nemo_rl,os;print(os.path.dirname(nemo_rl.__file__))"

# cluster values resolve correctly
cd .../nemo-rl-scripts && WS_CLUSTER_NAME=lyris bash -c 'source ./cluster_env.sh && env | grep ^WS_'

# tokens load from env.sh
source /lustre/fsw/coreai_comparch_trtllm/erinh/launch_scripts/env.sh && echo "${HF_TOKEN:+HF set} ${WANDB_API_KEY:+WANDB set}"

# datasets version (List support)
/opt/nemo_rl_venv/bin/python -c "import datasets,datasets.features as f;print(datasets.__version__, hasattr(f,'List'))"

# jobs / attach
squeue -u erinh
bash nemo-rl-scripts/<jobid>-attach.sh      # head;  add '1' for worker 1
```

---

## 6. Environment facts (lyris)
- Cluster `datasets`: **3.1.0** (no `List` feature).
- Container Python: **3.13**; repo `requires-python` is `>=3.12,<3.13` → editable install uses
  `--ignore-requires-python`.
- `HF_HOME=/lustre/.../erinh/.hf_model_cache`, `HF_DATASETS_CACHE=/lustre/.../erinh/.hf_data_cache`
  (from `env.sh`; on shared `/lustre` so all nodes share).
- Partitions: `gb200`, `gb200-backfill`, `gb300`, `gb300-backfill` (no GRES → no `--gpus-per-node`).
