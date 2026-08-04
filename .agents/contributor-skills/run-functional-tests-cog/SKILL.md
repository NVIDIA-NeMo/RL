---
name: run-functional-tests-cog
description: Run NeMo-RL L1/functional test shell scripts on a Slurm GPU cluster (e.g. the OCI HSG GB200 cluster) using the `cog` CLI, reusing a prebuilt NeMo-RL container image instead of building one. Use to reproduce/debug CI functional-test failures (e.g. the Megatron generation "illegal memory access") on real cluster hardware.
when_to_use: Reproducing a CI functional-test failure locally on cluster GPUs; running tests/functional/*.sh on hsg/oci; debugging a Megatron/vLLM generation crash that only shows up in CI; "run L1 functional test on the cluster", "replicate the CI illegal memory access", "run the megatron functional test on GB200".
---

# Running NeMo-RL functional tests on a cluster with `cog`

`cog` (checkout at `~/cog`, installed as `cog`) is a local control plane that
runs Slurm jobs inside an enroot/pyxis container on a registered cluster. This
skill uses it to run NeMo-RL's `tests/functional/*.sh` scripts on the **OCI HSG
GB200 cluster** (registered in cog as `oci-hsg`, SSH alias `oci` /
`oci-hsg-cs-001-login-01.nvidia.com`), reusing the prebuilt nightly image
`nvcr.io/nvidian/nemo-rl:nightly` (NO image build).

The CI failure this was first built to reproduce: `L1_Functional_Tests_Megatron_4.sh`
→ `grpo_megatron_generation.sh` crashes during Megatron `DynamicInferenceEngine`
CUDA-graph warmup with `torch.AcceleratorError: CUDA error: an illegal memory
access was encountered` (in `fused_bias_dropout`). It reproduces on GB200
(aarch64), matching the CI runner arch.

## How it is wired (one-time setup — already done)

cog only knew about Megatron-LM repos, so a **NeMo-RL repo profile** was added
to the cog checkout. If `cog` is reinstalled or the `~/cog` checkout is reset,
re-apply these (then `cd ~/cog && uv tool install --editable . --force`):

1. `~/cog/src/cog/repo_profiles/nemo_rl.py` — `NeMoRLProfile`:
   - validates the repo via `pyproject.toml` + `uv.lock` + `nemo_rl/`.
   - default base image = `nvcr.io/nvidian/nemo-rl:nightly`.
   - `default_exec_env` points HF + compile caches at shared scratch so
     models/tokenizers/datasets download **once** and are reused:
     `HF_HOME=<scratch>/nemo-rl-cache/hf` (+ `HF_HUB_CACHE`, `HF_DATASETS_CACHE`,
     `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`).
   - `uv_sync_commands` does NOT rebuild the venv. NeMo-RL's image ships a
     fully-resolved venv at `/opt/nemo_rl_venv` (with editable installs of
     `nemo_rl` and the `3rdparty` Megatron-Bridge/Megatron-LM checkouts). cog
     expects a venv at `<env_root>/.venv`, so the "env build" just symlinks it:
     `ln -sfn /opt/nemo_rl_venv "${UV_PROJECT_ENVIRONMENT}"`.
2. `~/cog/src/cog/repo_profiles/__init__.py` — `NeMoRLProfile` added to
   `resolve_profile`'s candidate list.
3. `~/cog/src/cog/worktree/snapshot.py` — `_list_files` skips entries that are
   not regular files/symlinks. NeMo-RL has 3 git **submodule** gitlinks
   (`3rdparty/*`); without this, workspace sync raised
   `Unsupported worktree entry type`.
4. `~/cog/src/cog/images/enroot.py` — `plan_enroot_import` now passes
   `-t 04:00:00` (override with `COG_IMPORT_TIME`). The nightly image is ~50 GB;
   the `cpu` partition's `DefaultTime` is `00:31:00`, so a no-`-t` import gets
   SIGTERM'd mid-extraction.

The `oci-hsg` cluster is already registered (`cog cluster ls`):
scratch `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space`,
accounts `coreai_dlalgo_llm`, import partition `cpu`. NGC enroot creds exist at
`~/.config/enroot/.credentials` on the login node (needed to pull the image).

## Cluster facts that matter (OCI HSG)

- Nodes are **GB200 NVL72, aarch64, 4 GPUs/node** (`gpu:4`). Login + compute
  nodes have internet and `enroot` (so workers download HF models directly).
- **QOS requires whole-node GPU jobs**: `normal` QOS `MinTRESPerJob=gres/gpu=4`.
  Requesting fewer GPUs fails with `QOSMinGRES` / "Job violates accounting/QOS
  policy". **Always request `--gpus 4`** even though the test pins
  `cluster.gpus_per_node=2` (the extra GPUs sit idle; this still matches CI's
  2-GPU behaviour).
- `batch` partition: default + MaxTime checks; use `--time 02:00:00`.

## Secrets / tokens

All credentials live in a single local file: **`/Users/shanmugamr@nvidia.com/tokens`**
(one `KEY=value` per line). It currently provides `HF_TOKEN`, `WANDB_API_KEY`,
`GITLAB_TOKEN`, and `GITHUB_TOKEN`. Never hard-code or paste token values; always
read them from this file.

Some functional tests require a token — e.g. `grpo_megatron_generation_async_gym.sh`
aborts immediately with `[ERROR] HF_TOKEN is not set` because it downloads the
`workplace_assistant` dataset from Hugging Face.

`cog submit` has **no `--env` flag**, so inject secrets by expanding them locally
into the remote `--command` string (write a wrapper script so the value never
appears in your shell history):

```bash
set -a; source /Users/shanmugamr@nvidia.com/tokens; set +a   # load into env
: "${HF_TOKEN:?not found in tokens file}"
# then, inside the here-doc that builds the remote command, emit:
#   export HF_TOKEN='${HF_TOKEN}'
# (single-quoted, locally-expanded) as the first line before running the test.
```

Note the token becomes visible in the remote process/`squeue` command line — fine
for this internal cluster, but do not log it to shared files.

## Run it

```bash
# 0. Sanity: profile resolves to nemo_rl + image = nightly
cog profile --repo ~/RL --run-name nemo-rl-l1-mcore4 --cluster-name oci-hsg

# 1. One-time per image: import the nightly sqsh (~50 GB, ~60 min on cpu).
#    Cached afterwards (cache_hit: true returns instantly).
cog prepare-image --repo ~/RL --cluster-name oci-hsg

# 2. Submit the functional test. Run it from the image's /opt/nemo-rl, NOT the
#    synced workspace. Why: NeMo-RL's uv workspace includes member `nemo-gym`,
#    which lives in the `3rdparty/Gym` git SUBMODULE. cog (correctly) does not
#    sync submodule contents, so the synced repo has empty 3rdparty dirs and
#    `uv run` fails with: "`nemo-gym` references a workspace ... but is not a
#    workspace member". The image's /opt/nemo-rl has the full submodules + a
#    consistent venv. But /opt/nemo-rl/tests/functional ships EMPTY, so copy the
#    synced functional scripts in first, then run from /opt/nemo-rl (the test
#    scripts set PROJECT_ROOT from their own location, so they must live under
#    /opt/nemo-rl for uv to resolve the full workspace). The container rootfs is
#    a writable ephemeral overlay, so the copy is fine.
cog submit \
  --repo ~/RL \
  --cluster-name oci-hsg \
  --run-name nemo-rl-l1-mcore4 \
  --command 'mkdir -p /opt/nemo-rl/tests/functional && cp -rf tests/functional/. /opt/nemo-rl/tests/functional/ && cp -f tests/*.py /opt/nemo-rl/tests/ 2>/dev/null || true; cd /opt/nemo-rl && bash tests/functional/L1_Functional_Tests_Megatron_4.sh' \
  --gpus 4 --nodes 1 --ntasks-per-node 1 \
  --partition batch --time 02:00:00 \
  --job-name nemo-rl-l1mcore4
```

`cog submit` blocks until the job finishes, so run it backgrounded and poll.
First-run gotcha: cog recomputes the local workspace hash (reads every tracked
+ untracked file, ~5–7 min) before it submits, then the job may sit `PENDING`
in the `batch` queue for a while.

### Run a single sub-test instead of the whole L1 suite

The illegal-memory-access is the **first** sub-test, so the L1 script aborts
there anyway. To iterate faster, target just that script:

```bash
cog submit --repo ~/RL --cluster-name oci-hsg --run-name nemo-rl-mcore-gen \
  --command 'cp -rf tests/functional/. /opt/nemo-rl/tests/functional/; cd /opt/nemo-rl && uv run --no-sync bash ./tests/functional/grpo_megatron_generation.sh' \
  --gpus 4 --nodes 1 --ntasks-per-node 1 --partition batch --time 01:00:00
```

## Watch / debug

```bash
# Find the job id (also in the cog submit JSON: .job.job_id)
ssh oci 'squeue -u $USER -o "%.10i %.16j %.8T %.10M %R"'

# Slurm logs live at <run-root>/slurm/<jobid>.{out,err}
cog logs slurm --run-name nemo-rl-l1-mcore4 --job-id <JOBID> --stream both
ssh oci 'tail -f /lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/runs/nemo-rl-l1-mcore4/slurm/<JOBID>.err'
```

Expected reproduction signature near the end of stderr:

```
File ".../megatron/core/inference/engines/dynamic_engine.py", line ... create_cuda_graphs
...
File ".../megatron/core/fusions/fused_bias_dropout.py", line 56, in _bias_dropout_add_func
    out.add_(residual)
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

## Iterating on a fix

To test local `nemo_rl/` edits while still using the image's full uv workspace,
overlay your synced `nemo_rl/` (and edited scripts) onto the image before
running, e.g. extend the command with
`cp -rf nemo_rl/. /opt/nemo-rl/nemo_rl/`. Megatron-LM/vLLM and the
`3rdparty` submodules stay from the image. For many quick iterations prefer a
persistent allocation (avoids re-queuing each time):

```bash
cog session start --repo ~/RL --session-handle nrl-dbg --gpus 4 --time 04:00:00 --partition batch
cog session exec --session-handle nrl-dbg --repo ~/RL \
  --command 'cp -rf tests/functional/. /opt/nemo-rl/tests/functional/ && cp -rf nemo_rl/. /opt/nemo-rl/nemo_rl/; cd /opt/nemo-rl && bash tests/functional/L1_Functional_Tests_Megatron_4.sh' \
  --wait-timeout 3600
cog session stop --session-handle nrl-dbg
```

## Gotchas checklist

- `QOSMinGRES` error → request `--gpus 4` (whole node).
- Image import SIGTERM'd → ensure cog passes `-t` (see setup #4) or set
  `COG_IMPORT_TIME=06:00:00`.
- `tests/functional/...: No such file or directory` → the nightly image ships
  an empty `tests/functional`; copy the synced scripts in first (see the
  `cp -rf tests/functional/. /opt/nemo-rl/tests/functional/` step).
- `` `nemo-gym` references a workspace ... but is not a workspace member `` →
  you ran from the synced workspace, whose `3rdparty` submodules are empty. Run
  from `/opt/nemo-rl` instead (full submodules + venv).
- Workspace sync `Unsupported worktree entry type` → snapshot fix (setup #3)
  missing; reinstall cog from `~/cog`.
- Model/dataset re-downloading every run → HF cache env not exported; confirm
  the `nemo_rl` profile's `default_exec_env` (setup #1) and that the run mounts
  scratch (cog mounts `<scratch>:<scratch>` automatically).
