---
name: reproduce-recipe
description: Procedure for reproducing another user's NeMo-RL training recipe (SWE/GRPO SLURM runs) under your own account. Covers copying the config, adapting the launch script to your dirs/credentials, and the critical enroot-image-vs-vLLM-pin check that prevents silent wrong-version runs.
when_to_use: Reproducing or rerunning someone else's recipe; adapting another user's run script to your own account; 'reproduce X's run', 'rerun ruit/bihu's recipe', 'copy this recipe to my dirs', 'repoint paths', 'why am I getting zero reward', 'which container/vLLM does this use', 'is my enroot image correct'.
allowed-tools: Bash Read Grep Glob Edit Write
---

# Reproduce Another User's NeMo-RL Recipe

A recipe = a **config YAML** + a **SLURM launch script** (`run_*.sh` that builds `COMMAND`/`SETUP_COMMAND` and `sbatch ray.sub`). Goal: rerun it faithfully under *your* account without changing the science.

**Golden rule:** repoint only your *environment* (dirs, credentials, account). Keep everything that defines the *experiment* (container, model, data, parallelism, hyperparameters) byte-for-byte. The #1 silent failure is an enroot image whose baked vLLM does not match the code — verify it (see below).

## 1. Pull the source artifacts first

Read the source YAML and run script before changing anything:

```bash
SRC=/lustre/fsw/.../<user>/<repo>/test_assets/SWE
Read "$SRC/<config>.yaml" "$SRC/<run>.sh"
```

Note from the run script: `CONTAINER`, `MODEL_PATH`, data paths, `SBATCH_ACCOUNT`, env-source line, `SETUP_COMMAND` (enabled?), and the entry config.

## 2. Copy the config verbatim

The config is usually environment-agnostic. Copy it; confirm identical. Do **not** rewrite data/model paths inside the YAML unless they are unreadable from your account (`ls -l` them).

```bash
diff <local>/test_assets/SWE/<config>.yaml "$SRC/<config>.yaml" && echo "(identical)"
```

## 3. Adapt the launch script — your env, their science

**Repoint to YOUR writable dirs / credentials:**

| Knob | Set to |
|------|--------|
| `REPO_ROOT`, `SNAPSHOT_DIR` | your repo checkout |
| `HF_HOME`, `HF_DATASETS_CACHE` | your writable HF dirs |
| `UV_CACHE_DIR` | **persistent path on Lustre** (not `/tmp` — node-local, wiped between jobs → cold start). `UV_LINK_MODE=copy` if cache and venv are on different filesystems |
| vllm/inductor/triton caches, `BASE_LOG_DIR`, `CHECKPOINT_DIR` | your dirs |
| tokens | inherit `HF_TOKEN`/`WANDB_API_KEY`/`GITHUB_TOKEN` from your shell; **do NOT** `source` the other user's `export_env_vars.sh`. Fail-fast: `: "${WANDB_API_KEY:?}"` |
| `SBATCH_ACCOUNT`, partition | yours |
| W&B project/name | yours, unless intentionally sharing a project for comparison |

`chmod 700 "$BASE_LOG_DIR"` — `ray.sub` runs `set -x` and traces tokens from `COMMAND` into `slurm-%j.out`.

**KEEP unchanged (these define the experiment):** `CONTAINER`, `MODEL_PATH`, train/val data, `TP/EP/CP/PP/VLLM_TP`, all `grpo.*`/`loss_fn.*` hyperparameters, and `SETUP_COMMAND`.

## 4. CRITICAL — verify the enroot image matches the code

The image bakes prefetched venvs. With `NRL_FORCE_REBUILD_VENVS=false` (and the config key `env.nemo_gym.skip_venv_if_present: true`) those venvs are **reused**, so the *image's* vLLM is what actually runs — not necessarily what `uv.lock` pins. A mismatch is silent and catastrophic (e.g. vLLM 0.20.0's hermes tool parser emits zero `function_call`s → agent never acts → 0.0 reward).

```bash
# What the code pins:
grep -iE 'vllm' pyproject.toml | grep -E 'releases/download|vllm=='
awk '/name = "vllm"/{f=1} f&&/version =/{print; exit}' uv.lock

# What the image actually bakes:
unsquashfs -l "$CONTAINER" 2>/dev/null | grep -iaoE 'vllm-0\.[0-9]+\.[0-9]+[^/]*\.dist-info' | sort -u

# If SETUP_COMMAND is disabled, the image MUST already bake the sandbox runtime:
unsquashfs -l "$CONTAINER" 2>/dev/null | grep -aiE '(bin|libexec)/(apptainer|singularity|starter)$'
```

- Image vLLM **must equal** the `uv.lock` pin. If it doesn't, use an image built from the matching commit, or enable a `SETUP_COMMAND` that runs `uv sync --frozen` to rebuild the venv.
- If apptainer/singularity is **not** baked, enable the `SETUP_COMMAND` that installs it (the apptainer-install block is idempotent — it skips when already present).
- Image naming often encodes the commit (e.g. `…-swe_bench-<short-sha>-…`); prefer the image built from the exact commit your branch is on.

## 5. Verify env-var consistency — don't blindly copy the COMMAND env

The source script's `COMMAND`/`SETUP_COMMAND` env vars were chosen for *its* commit and *its* image. Do **not** paste them in unchecked — for every env var you copy, verify it is (a) actually read by the code **at your checked-out commit (incl. the Gym submodule)** and (b) still needed. Env-var semantics drift between commits.

```bash
# Is this env var read by anything at your commit?
grep -rn "MY_ENV_VAR" . --include=*.py --include=*.sh --include=*.yaml --include=*.toml | grep -v '\.git/'
# (only your own run script matches => the var is dead; find the real knob, often a config key)
```

Verified gotchas (HEAD ~`6de99f772`):
- **`NEMO_GYM_SKIP_VENV_IF_PRESENT=1` is a no-op** — read by no code. The real control is the config key `env.nemo_gym.skip_venv_if_present` (set it in the YAML, not as an env var).
- **`NRL_IGNORE_VERSION_MISMATCH=1` does not prevent a crash** — `_check_container_fingerprint()` only *warns* on mismatch and proceeds (no `raise`). It runs only when `NRL_CONTAINER=1` and `NRL_FORCE_REBUILD_VENVS!=true`. Using a cross-commit image guarantees the mismatch warning; the flag just silences it. Keep it only as warning-hygiene, and **manually verify the deps you care about (vLLM) match `uv.lock`** — the flag does not make them match.

When you reuse an image built from a *different* commit than your HEAD, the fingerprint mismatch is expected; it does not fix dependency drift, it only reports it. Confirm the critical packages by hand (§4).

## 6. Match code version + clean submodules

```bash
git log -1 --oneline                          # your HEAD
git rev-list --left-right --count origin/<their-branch>...HEAD   # 0 0 = same code
git submodule update --init --recursive       # bring nested submodules to recorded SHAs
git submodule status --recursive              # no leading +/- = at recorded commit
```

A recipe may depend on backported fixes (the commit message often says so, e.g. "stays on vLLM 0.17.1"). Reproduce on the commit the recipe targets. To make submodules clean, remove untracked cruft (editor files, build artifacts like `helpers_cpp`) with `git submodule foreach --recursive git clean -fd` — inspect first, never blind `-x`.

## 7. Smoke test, then launch

Validate the container/tool-parser end-to-end cheaply before committing GPUs:

```bash
# add to COMMAND for a short smoke test, with a short SBATCH_TIME
grpo.max_num_steps=3
```

**Use a fresh `CHECKPOINT_DIR` for the smoke test** (a distinct `EXP_SUFFIX`). If a checkpoint already exists in `CHECKPOINT_DIR`, the run **auto-resumes** it — and if your smoke `max_num_steps` differs from the run that wrote it, Megatron aborts in `MegatronPolicyWorker.__init__` with `OptimizerParamScheduler: class input value N and checkpointvalue M ... weight decay iterations do not match` (the optimizer schedule is sized `steps × train_global_batch_size`). A fresh dir starts from `model_name` and sidesteps it without touching the prior checkpoint.

After `sbatch`, capture the job id and find the W&B URL in the driver log:

```bash
grep -rhoE 'https://wandb[^ ]*' "${BASE_LOG_DIR}/${JOB_ID}-logs/ray-driver.log" | sort -u
```

## Checklist

- [ ] Config copied (identical or only unreadable paths repointed)
- [ ] All dirs/caches/logs/tokens/account repointed to yours; tokens fail-fast; log dir `chmod 700`
- [ ] Container, model, data, parallelism, hyperparameters unchanged
- [ ] Every copied env var verified read-by-code at your commit + still needed (no dead/no-op vars)
- [ ] **Image vLLM == `uv.lock` pin**; sandbox runtime present (or `SETUP_COMMAND` enabled)
- [ ] HEAD on the recipe's target commit; submodules updated + clean
- [ ] 5-step smoke test passes before full run
