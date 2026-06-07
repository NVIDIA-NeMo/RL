---
name: nemotron-environment-setup
description: Set up or audit the dfw NeMo-RL/Nemotron diffusion training environment, including the Nemo-RL fork/worktree, Git submodules for Megatron-Bridge and Megatron-LM, local editable SGLang fork, UV driver and worker environments, and sanity checks before training or evaluation.
---

# Nemotron Environment Setup

Use this skill when setting up a fresh dfw environment, auditing dependency paths, or debugging why a run is using the wrong NeMo-RL, Megatron-Bridge, Megatron-LM, or SGLang code.

## New User Variables

The concrete paths in this skill are parameterized through the variables below.  Use this mapping for a fresh setup:

```bash
export USER_NAME=<your_dfw_user>
export GITHUB_USER=<your_github_user>

export HOME_DIR=/home/${USER_NAME}
export REPO_DIR=${HOME_DIR}/diffusion_RL/RL

export LUSTRE_USER_ROOT=/lustre/fsw/portfolios/coreai/users/${USER_NAME}
export DRIVER_ENV_ROOT=${LUSTRE_USER_ROOT}/nemorl_uv_driver_envs
export WORKER_ENV_ROOT=${LUSTRE_USER_ROOT}/nemo_rl_worker_venvs
export UV_CACHE_ROOT=${LUSTRE_USER_ROOT}/uv_cache
```

Then create the base directories:

```bash
mkdir -p "${HOME_DIR}/diffusion_RL" "${LUSTRE_USER_ROOT}"
```

If your workspace uses a different Lustre portfolio or project path, set `LUSTRE_USER_ROOT` to that location and keep the driver envs, worker envs, and UV cache underneath it.

## Canonical Working Tree

For the current reference deployment, the main repo is:

```bash
cd "${REPO_DIR}"
```

For a new user, clone the NeMo-RL fork used by this workflow into `REPO_DIR` and check out `dllm_clean`:

```bash
git clone --branch dllm_clean git@github.com:sajadn/RL.git "${REPO_DIR}"
cd "${REPO_DIR}"
```

Check the current branch and commit before changing dependencies:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
```

## Forks Submodules (Megatron-Bridge and Megatron-LM)

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
sglang = { path = "${HOME_DIR}/diffusion_RL/sglang-nemotron-dllm-a652eb48/python", editable = true }
```

That local checkout should track the fork:

```text
git@github.com:sajadn/slgang-nemotron-diffusion.git
```

Set up or refresh it with:

```bash
git clone git@github.com:sajadn/slgang-nemotron-diffusion.git \
  "${HOME_DIR}/diffusion_RL/sglang-nemotron-dllm-a652eb48"
cd "${HOME_DIR}/diffusion_RL/sglang-nemotron-dllm-a652eb48"
git checkout restore/a652eb48-good-run
```

For the current reference setup, use SGLang branch `restore/a652eb48-good-run`.

Editable path dependencies pick up normal Python source edits in new driver/worker processes. For dependency metadata, path changes, kernels, compiled extensions, or package layout changes, follow `${REPO_DIR}/skills/nemotron-experiment-submit/SKILL.md`.

## UV Environments and Submit Flags

To understand `ENV_TAG`, UV driver/worker/cache paths, rebuild/reinstall flags, and how to submit a job, please take a look at the skill below:

```text
${REPO_DIR}/skills/nemotron-experiment-submit/SKILL.md
```

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

git -C "${HOME_DIR}/diffusion_RL/sglang-nemotron-dllm-a652eb48" rev-parse HEAD
git -C "${HOME_DIR}/diffusion_RL/sglang-nemotron-dllm-a652eb48" status --short | wc -l
```

Training scripts should write equivalent provenance into W&B/runtime config. If a run unexpectedly uses a different codebase, compare these paths and commits first.

## Sanity Checks

Check dependency declarations:

```bash
grep -nE "sglang =|megatron-(bridge|core)|Megatron-(Bridge|LM)" pyproject.toml uv.lock
```

Check imports from the active UV env only after the env is built. Get the current `ENV_TAG`, `UV_PROJECT_ENVIRONMENT`, and `UV_CACHE_DIR` from `${REPO_DIR}/skills/nemotron-experiment-submit/SKILL.md`:

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

If this fails because the env has not been built yet, submit a small smoke run using the guidance in `${REPO_DIR}/skills/nemotron-experiment-submit/SKILL.md`.
