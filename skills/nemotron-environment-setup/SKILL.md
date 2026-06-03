---
name: nemotron-environment-setup
description: Set up or audit the dfw NeMo-RL/Nemotron diffusion training environment, including the Nemo-RL fork/worktree, Git submodules for Megatron-Bridge and Megatron-LM, local editable SGLang fork, UV driver and worker environments, and sanity checks before training or evaluation.
---

# Nemotron Environment Setup

Use this skill when setting up a fresh dfw environment, auditing dependency paths, or debugging why a run is using the wrong NeMo-RL, Megatron-Bridge, Megatron-LM, or SGLang code.

## New User Variables

The concrete paths in this skill were written from the current `snorouzi` deployment. A new user should first define their own roots and replace hardcoded `/home/snorouzi`, `/lustre/.../snorouzi`, and `sajadn` fork URLs unless they intentionally want to reuse those shared checkouts.

Use this mapping for a fresh user-owned setup:

```bash
export USER_NAME=<your_dfw_user>
export GITHUB_USER=<your_github_user>

export HOME_DIR=/home/${USER_NAME}
export CODE_DIR=${HOME_DIR}/code
export REPO_DIR=${HOME_DIR}/diffusion_RL/RL

export LUSTRE_USER_ROOT=/lustre/fsw/portfolios/coreai/users/${USER_NAME}
export DRIVER_ENV_ROOT=${LUSTRE_USER_ROOT}/nemorl_uv_driver_envs
export WORKER_ENV_ROOT=${LUSTRE_USER_ROOT}/nemo_rl_worker_venvs
export UV_CACHE_ROOT=${LUSTRE_USER_ROOT}/uv_cache
```

Then create the base directories:

```bash
mkdir -p "${HOME_DIR}/diffusion_RL" "${CODE_DIR}" "${LUSTRE_USER_ROOT}"
```

If your workspace uses a different Lustre portfolio or project path, set `LUSTRE_USER_ROOT` to that location and keep the driver envs, worker envs, and UV cache underneath it.

After cloning, audit scripts for user-specific defaults before submitting jobs:

```bash
grep -RIn "/home/snorouzi\|users/snorouzi\|sajadn" \
  pyproject.toml .gitmodules tools/nemotron_diffusion examples/configs skills
```

Replace those values only where the path should be user-owned. Keep `sajadn` fork URLs only when that fork is intentionally the canonical dependency source.

## Canonical Working Tree

For the current reference deployment, the main repo is:

```bash
cd /home/snorouzi/diffusion_RL/RL
```

For a new user, clone your NeMo-RL fork into `REPO_DIR`:

```bash
git clone git@github.com:${GITHUB_USER}/RL.git "${REPO_DIR}"
cd "${REPO_DIR}"
git checkout <branch>
```

Check the current branch and commit before changing dependencies:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

For training submission details after setup, use:

```text
${REPO_DIR}/skills/nemotron-experiment-submit/SKILL.md
```

For checkpoint conversion and evaluation, use:

```text
${REPO_DIR}/skills/gsm8k-checkpoint-eval/SKILL.md
```

## Forks and Submodules

The repo uses local editable proxy packages in `pyproject.toml`, backed by Git submodules:

```toml
megatron-bridge = { path = "3rdparty/Megatron-Bridge-workspace", editable = true }
megatron-core = { workspace = true }
```

The source repos live one level deeper:

```text
3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
3rdparty/Megatron-LM-workspace/Megatron-LM
```

Initialize or repair submodules with:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Expected `.gitmodules` intent:

```text
Megatron-Bridge: git@github.com:sajadn/Megatron-Bridge.git, branch nemotron-diffusion
Megatron-LM:     https://github.com/NVIDIA/Megatron-LM.git
```

For a new user, either keep the shared `sajadn/Megatron-Bridge` fork if that is the canonical fork for the experiment, or change `.gitmodules` to your fork and commit the `.gitmodules` update plus the submodule gitlink:

```bash
git config -f .gitmodules submodule.3rdparty/Megatron-Bridge-workspace/Megatron-Bridge.url \
  git@github.com:${GITHUB_USER}/Megatron-Bridge.git
git config -f .gitmodules submodule.3rdparty/Megatron-Bridge-workspace/Megatron-Bridge.branch \
  nemotron-diffusion
git submodule sync --recursive
git submodule update --init --recursive
```

Verify the actual checked-out sources:

```bash
git submodule status --recursive

git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge remote -v
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge branch --show-current
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD

git -C 3rdparty/Megatron-LM-workspace/Megatron-LM remote -v
git -C 3rdparty/Megatron-LM-workspace/Megatron-LM rev-parse --abbrev-ref HEAD
git -C 3rdparty/Megatron-LM-workspace/Megatron-LM rev-parse HEAD
```

If committing submodule changes in NeMo-RL, first commit/push the nested repo change in the submodule repo, then commit the updated gitlink in the NeMo-RL repo. Git records only the submodule commit hash, not the whole nested repository contents.

## SGLang Fork

Current setup uses a local editable SGLang checkout in `pyproject.toml`:

```toml
sglang = { path = "/home/snorouzi/code/sglang-nemotron-dllm-a652eb48/python", editable = true }
```

That local checkout should track the fork:

```text
git@github.com:sajadn/slgang-nemotron-diffusion.git
```

Set up or refresh it with:

```bash
mkdir -p "${CODE_DIR}"
git clone git@github.com:sajadn/slgang-nemotron-diffusion.git \
  "${CODE_DIR}/sglang-nemotron-dllm-a652eb48"
cd "${CODE_DIR}/sglang-nemotron-dllm-a652eb48"
git checkout <branch-or-commit>
```

For a new user fork, replace `sajadn` with `${GITHUB_USER}` or the fork owner:

```bash
git clone git@github.com:${GITHUB_USER}/slgang-nemotron-diffusion.git \
  "${CODE_DIR}/sglang-nemotron-dllm-a652eb48"
```

Then update `pyproject.toml` to point at the local checkout:

```toml
sglang = { path = "/home/<your_dfw_user>/code/sglang-nemotron-dllm-a652eb48/python", editable = true }
```

Verify:

```bash
git -C "${CODE_DIR}/sglang-nemotron-dllm-a652eb48" remote -v
git -C "${CODE_DIR}/sglang-nemotron-dllm-a652eb48" branch --show-current
git -C "${CODE_DIR}/sglang-nemotron-dllm-a652eb48" rev-parse HEAD
git -C "${CODE_DIR}/sglang-nemotron-dllm-a652eb48" status --short
```

Editable path dependencies pick up normal Python source edits in new driver/worker processes. Dependency metadata, path changes, kernels, compiled extensions, or package layout changes still require reinstall/rebuild.

## UV Environments

UV is responsible for resolving and creating:

- driver envs under `${DRIVER_ENV_ROOT}/...`
- Ray worker envs under `${WORKER_ENV_ROOT}...`
- package cache under `${UV_CACHE_ROOT}...`

Use one stable `ENV_TAG` for a fixed dependency path layout:

```bash
ENV_TAG=mb_3rdparty_sglagn_local_fork
```

Change `ENV_TAG` only when dependency paths change, such as a new local SGLang path or different Megatron workspace path. Normal Python edits inside the existing editable NeMo-RL, SGLang, Megatron-Bridge, or Megatron-LM trees do not need a new tag.

For a new user, make sure the submit script defaults or exported environment variables point at the user-owned roots:

```bash
export UV_PROJECT_ENVIRONMENT=${DRIVER_ENV_ROOT}/diffusion_RL_RL_${ENV_TAG}
export UV_CACHE_DIR=${UV_CACHE_ROOT}_${ENV_TAG}
export NEMO_RL_WORKER_VENV_DIR=${WORKER_ENV_ROOT}_${ENV_TAG}
```

If the script derives these internally, update the script defaults or pass the variables explicitly in the submission command.

## Rebuild Flags

For normal reruns after envs exist:

```bash
NRL_FORCE_REBUILD_VENVS=false
FORCE_REINSTALL_PACKAGES=false
FORCE_REINSTALL_SGLANG=false
```

After changing the SGLang dependency path in `pyproject.toml`, use a new `ENV_TAG` and force SGLang reinstall once:

```bash
ENV_TAG=<new_dependency_layout_tag>
FORCE_REINSTALL_SGLANG=true
```

After changing dependency sources or when worker envs are stale, rebuild worker venvs once:

```bash
NRL_FORCE_REBUILD_VENVS=true
```

Avoid broad rebuilds unless needed; they can trigger slow builds of packages such as TransformerEngine.

## Runtime Provenance

Before submitting a new family of experiments, log or print these:

```bash
cd "${REPO_DIR}"
git rev-parse HEAD
git status --short | wc -l

git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge status --short | wc -l

git -C 3rdparty/Megatron-LM-workspace/Megatron-LM rev-parse HEAD
git -C 3rdparty/Megatron-LM-workspace/Megatron-LM status --short | wc -l

git -C "${CODE_DIR}/sglang-nemotron-dllm-a652eb48" rev-parse HEAD
git -C "${CODE_DIR}/sglang-nemotron-dllm-a652eb48" status --short | wc -l
```

Training scripts should write equivalent provenance into W&B/runtime config. If a run unexpectedly uses a different codebase, compare these paths and commits first.

## Sanity Checks

Check dependency declarations:

```bash
grep -nE "sglang =|megatron-(bridge|core)|Megatron-(Bridge|LM)" pyproject.toml uv.lock
```

Check imports from the active UV env only after the env is built:

```bash
UV_PROJECT_ENVIRONMENT=${DRIVER_ENV_ROOT}/diffusion_RL_RL_${ENV_TAG} \
UV_CACHE_DIR=${UV_CACHE_ROOT}_${ENV_TAG} \
uv run --locked python - <<'PY'
import nemo_rl, sglang
import megatron.bridge
print("nemo_rl", nemo_rl.__file__)
print("sglang", sglang.__file__)
print("megatron.bridge", megatron.bridge.__file__)
PY
```

If this fails because the env has not been built yet, submit a small smoke run or run the intended submit command once with the appropriate rebuild/reinstall flags.
