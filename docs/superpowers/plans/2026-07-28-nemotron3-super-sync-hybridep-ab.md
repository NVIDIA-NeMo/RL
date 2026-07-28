# Nemotron3 Super Sync HybridEP A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible baseline and HybridEP profiles for the existing
Nemotron3 Super synchronous 32n4g recipe, run a clean dispatcher A/B on
OCI-HSG, and publish the performance and short-run numerical results.

**Architecture:** Preserve the existing sync recipe as the all-to-all baseline.
Add one inherited YAML overlay whose only resolved changes are the three
HybridEP dispatcher fields, plus one reusable model profile per arm. Submit
both profiles through the existing launcher with `DISPATCHER_MODE=recipe` so
the YAML files are the sole dispatcher-control surface.

**Tech Stack:** YAML, OmegaConf, Python 3.13, pytest, Bash, Git, SLURM, Ray,
OCI-HSG GB200, static HTML.

## Global Constraints

- Baseline recipe remains unmodified:
  `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml`.
- NeMo-RL branch:
  `sna/qwen30-pr2964-f725-pin-oci-20260727`.
- Megatron-Bridge:
  `483749cb773415f7608525838607dcefc62e4307`.
- Megatron-LM:
  `4d04e7625c5e84f984a9f01aef58cb006b0aa7ac`.
- DeepEP:
  `f725d29699f5bda9ba789456bb9579af69844685`.
- Container SHA256:
  `5e9f6066897057d8701e0722a5023c08a997f10f4eff61340c249ed73f7c33fc`.
- Use 32 nodes, 4 GPUs per node, segment size 8, 20 steps, four-hour wall
  time, and `NCCL_NVLS_ENABLE=0`.
- Disable W&B, checkpointing, and padding telemetry for both performance arms.
- Run `sbatch --test-only`, select the highest current user-level FairShare,
  and monitor both jobs for at least five minutes.

---

### Task 1: Add the Sync Config Contract Test

**Files:**
- Create: `tests/unit/tools/test_nemotron3_super_sync_hybridep_config.py`

**Interfaces:**
- Consumes:
  `nemo_rl.utils.config.load_config(config_path: str | Path) -> DictConfig`.
- Produces: a regression contract proving the HybridEP config differs from
  the baseline only in the three dispatcher fields.

- [ ] **Step 1: Write the failing test**

```python
from copy import deepcopy
from pathlib import Path

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config


def test_super_sync_hybridep_only_changes_dispatcher() -> None:
    project_root = Path(__file__).resolve().parents[3]
    config_dir = project_root / "examples" / "configs" / "recipes" / "llm" / "performance"
    baseline_path = config_dir / "grpo-nemotron3-super-120BA12B-32n4g.yaml"
    hybridep_path = config_dir / "grpo-nemotron3-super-120BA12B-32n4g-hybridep.yaml"

    baseline = OmegaConf.to_container(load_config(baseline_path), resolve=True)
    hybridep = OmegaConf.to_container(load_config(hybridep_path), resolve=True)
    expected = deepcopy(baseline)
    dispatcher = expected["policy"]["megatron_cfg"]
    dispatcher["moe_token_dispatcher_type"] = "flex"
    dispatcher["moe_flex_dispatcher_backend"] = "hybridep"
    dispatcher["moe_hybridep_num_sms"] = 32

    assert baseline["policy"]["megatron_cfg"]["moe_token_dispatcher_type"] == "alltoall"
    assert hybridep == expected
```

- [ ] **Step 2: Run the test and verify RED**

```bash
python3 -m pytest -q --confcutdir=tests/unit/tools \
  tests/unit/tools/test_nemotron3_super_sync_hybridep_config.py
```

Expected: FAIL because
`grpo-nemotron3-super-120BA12B-32n4g-hybridep.yaml` does not exist.

### Task 2: Add the HybridEP Config and Reusable Profiles

**Files:**
- Create:
  `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-hybridep.yaml`
- Create:
  `scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync.env`
- Create:
  `scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync-hybridep.env`
- Modify: `scripts/experiments/oci-hsg/hybridep/README.md`

**Interfaces:**
- Consumes: the unchanged baseline sync recipe and existing
  `submit_grpo.sh` model-profile contract.
- Produces: two explicit config/profile inputs accepted by the existing
  launcher.

- [ ] **Step 1: Add the minimal HybridEP overlay**

```yaml
defaults: grpo-nemotron3-super-120BA12B-32n4g.yaml

policy:
  megatron_cfg:
    moe_token_dispatcher_type: flex
    moe_flex_dispatcher_backend: hybridep
    moe_hybridep_num_sms: 32
```

- [ ] **Step 2: Add the baseline sync profile**

```bash
export NCCL_NVLS_ENABLE=0

MODEL_ID=nemotron3-super-120ba12b-32n4g-sync
CONFIG_PATH=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml
NUM_ACTOR_NODES=32
GPUS_PER_NODE=4
SEGMENT_SIZE=8
MAX_STEPS=20
TIME_LIMIT=04:00:00
DEFAULT_DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685
```

- [ ] **Step 3: Add the HybridEP sync profile**

```bash
export NCCL_NVLS_ENABLE=0

MODEL_ID=nemotron3-super-120ba12b-32n4g-sync-hybridep
CONFIG_PATH=examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-hybridep.yaml
NUM_ACTOR_NODES=32
GPUS_PER_NODE=4
SEGMENT_SIZE=8
MAX_STEPS=20
TIME_LIMIT=04:00:00
DEFAULT_DEEPEP_COMMIT=f725d29699f5bda9ba789456bb9579af69844685
```

- [ ] **Step 4: Document the two commands**

Add baseline and HybridEP command examples to the launcher README. Both
commands set `DISPATCHER_MODE=recipe`, `WANDB_ENABLED=False`, and
`NEMO_RL_HYBRIDEP_LOG_PACKING=0`.

- [ ] **Step 5: Run the focused test and verify GREEN**

```bash
python3 -m pytest -q --confcutdir=tests/unit/tools \
  tests/unit/tools/test_nemotron3_super_sync_hybridep_config.py
```

Expected: 1 passed.

- [ ] **Step 6: Commit the config contract**

```bash
git add \
  tests/unit/tools/test_nemotron3_super_sync_hybridep_config.py \
  examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-hybridep.yaml \
  scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync.env \
  scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync-hybridep.env \
  scripts/experiments/oci-hsg/hybridep/README.md
git commit -s -m "test: add Super sync HybridEP profiles"
```

### Task 3: Verify and Publish the Source

**Files:**
- Read: all files created in Tasks 1–2.
- Read: `uv.lock`.

**Interfaces:**
- Consumes: the complete config/profile change.
- Produces: a clean pushed commit safe for cluster submission.

- [ ] **Step 1: Run focused and launcher tests**

```bash
python3 -m pytest -q --confcutdir=tests/unit/tools \
  tests/unit/tools/test_nemotron3_super_sync_hybridep_config.py \
  tests/unit/tools/test_hybridep_submit_grpo.py
```

Expected: 6 passed.

- [ ] **Step 2: Run config validation**

```bash
python3 -m pytest -q \
  tests/unit/test_config_validation.py \
  -k 'grpo-nemotron3-super-120BA12B-32n4g-hybridep'
```

Expected: the HybridEP config validation cases pass.

- [ ] **Step 3: Run static verification**

```bash
bash -n scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
uv lock --check
git diff --check
git status --short
```

Expected: shell syntax and lock checks exit zero, diff check is clean, and
only the intended commit plus this plan follow the pushed branch.

- [ ] **Step 4: Push the branch**

```bash
git push origin sna/qwen30-pr2964-f725-pin-oci-20260727
```

### Task 4: Submit and Monitor the OCI-HSG A/B

**Files:**
- Read:
  `scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync.env`
- Read:
  `scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync-hybridep.env`
- Create remotely: per-run `submission.env`, SLURM output, Ray logs, and
  TensorBoard logs under `exp_logs/hybridep/`.

**Interfaces:**
- Consumes: the pushed source commit and two reusable profiles.
- Produces: one canonical baseline job and one canonical HybridEP job.

- [ ] **Step 1: Submit the baseline**

```bash
DISPATCHER_MODE=recipe \
WANDB_ENABLED=False \
NEMO_RL_HYBRIDEP_LOG_PACKING=0 \
RUN_NAME=nemotron3-super-120ba12b-32n4g-sync-alltoall-f725-clean-ab \
RUN_SUFFIX=sync-clean-ab \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync.env
```

- [ ] **Step 2: Submit the HybridEP arm**

```bash
DISPATCHER_MODE=recipe \
WANDB_ENABLED=False \
NEMO_RL_HYBRIDEP_LOG_PACKING=0 \
RUN_NAME=nemotron3-super-120ba12b-32n4g-sync-hybridep-f725-clean-ab \
RUN_SUFFIX=sync-clean-ab \
scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
  scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-sync-hybridep.env
```

The launcher must pull the pushed branch, initialize recursive submodules,
choose the highest FairShare account, pass `sbatch --test-only`, and record
exact source/image/config provenance.

- [ ] **Step 3: Monitor both jobs for five minutes**

```bash
squeue -j "${BASELINE_JOB_ID},${HYBRIDEP_JOB_ID}" \
  -o "%i|%j|%T|%M|%r|%N"
```

Scan bounded log tails for `Traceback`, actor loss, timeout, NCCL, CUDA,
illegal address, OOM, NaN, Inf, and model-config failures.

- [ ] **Step 4: Verify the terminal gate**

```bash
sacct -j "${BASELINE_JOB_ID},${HYBRIDEP_JOB_ID}" -n -X \
  -o JobIDRaw,State,ExitCode,Elapsed,Start,End,NodeList
```

Expected: both jobs report `COMPLETED 0:0`, 20 complete training steps, and no
fatal signature.

### Task 5: Analyze and Publish the Result

**Files:**
- Modify:
  `/Users/sna/nemo-rl_release_perf_investigator/public/qwen3-30ba3b-hybridep-oci-20260727/index.html`

**Interfaces:**
- Consumes: complete Ray driver logs from both Task 4 jobs.
- Produces: a secret-free Super sync A/B result table in the existing report.

- [ ] **Step 1: Parse the matched window**

Extract steps 5–20 from both driver logs. Compute arithmetic mean/median phase
times and ratio-of-sums E2E, policy, and LogProb tokens/second/GPU.

- [ ] **Step 2: Compute deltas**

```text
throughput_change_pct = (HybridEP / alltoall - 1) × 100
time_change_pct = (HybridEP / alltoall - 1) × 100
```

Also report reward, generation KL error, validation accuracy, and response
length at steps 10 and 20 without making a convergence claim.

- [ ] **Step 3: Update and validate HTML**

```bash
python3 -c 'from html.parser import HTMLParser; from pathlib import Path; p=Path("public/qwen3-30ba3b-hybridep-oci-20260727/index.html"); HTMLParser().feed(p.read_text())'
rg -n -i '(api[_-]?key|secret|password|token)[[:space:]]*[:=][[:space:]]*[A-Za-z0-9_./+\-]{8,}' \
  public/qwen3-30ba3b-hybridep-oci-20260727/index.html
git diff --check
```

Expected: HTML parses, no credential-shaped value is found, and the diff is
clean.

- [ ] **Step 4: Commit and push the report**

```bash
git add public/qwen3-30ba3b-hybridep-oci-20260727/index.html
git commit -s -m "report: add Super sync HybridEP A/B"
git push origin main
```
