---
name: build-and-dependency
description: Build, dependency, and smart-cache management for NeMo-RL. Use when setting up an environment, building or running a container, using uv or venvs, diagnosing uv errors, adding or removing dependencies, or launching any test, benchmark, rollout, or training run whose package, model, dataset, or compiler caches must be configured.
---

# Build and Dependency Guide

---

## Docker Images

Build the release image (includes all dependencies and pre-fetched venvs):

```bash
# Build from local source
docker buildx build -f docker/Dockerfile --tag nemo-rl:latest .

# Build from a specific git ref (no local clone needed)
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=main \
    --tag nemo-rl:latest \
    https://github.com/NVIDIA-NeMo/RL.git
```

Skip optional backends to reduce build time:

```bash
# Skip vLLM and SGLang
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_VLLM_BUILD=1 \
    --build-arg SKIP_SGLANG_BUILD=1 \
    --tag nemo-rl:latest .
```

See @docs/docker.md for full options.

---

## Always Use uv

**Never use `pip install` directly** — always go through `uv`.

```bash
# Run a script
uv run examples/run_grpo.py

# Run tests
uv run --group test bash tests/run_unit.sh

# Install all deps from lockfile
uv sync --locked
```

Exception: `Dockerfile.ngc_pytorch` is exempt from this rule.

## Mandatory smart cache policy

Enable smart caches for every build, test, benchmark, rollout, and training
run unless the user explicitly requests a cold-cache experiment. This is a
mandatory launch gate, not an optional optimization: every launcher created
or modified for NeMo-RL must enable it by default, and an `srun`, `sbatch`, or
`ray.sub` submission must not proceed until all applicable package, model,
dataset, compiler/JIT, and container-build caches have explicit safe roots.
Log those roots before measured work begins.

Treat this as an opt-out policy: all applicable caches are enabled by default,
including for short diagnostics. "Smart" means selecting cache placement and
namespaces from the workload rather than blindly sharing one directory:

1. Detect which package manager, model/data loader, backend, and JIT compiler
   the command will actually use.
2. Derive persistent namespaces from compatibility inputs and give each job a
   node-local writable working cache where concurrency makes that safer.
3. Seed node-local caches from compatible persistent artifacts before the
   workload, then write back only complete reusable artifacts.
4. Log cache roots, namespace keys, and cold/seeded/warm state before starting
   measured work. If an applicable cache is unconfigured, fix the launch
   instead of proceeding.
5. A user-requested cold-cache run disables only the cache layer being tested;
   keep unrelated caches enabled and label the cold layer explicitly.

- **Packages:** use uv's cache, not pip's. Make `UV_CACHE_DIR` or
  `UV_CACHE_DIR_OVERRIDE` explicit. A direct single-node job may use a mounted
  persistent cache; a multi-node job should seed node-local `/tmp/uv_cache`
  from a persistent cache to avoid shared-filesystem contention. Set
  `PIP_CACHE_DIR` only for an unavoidable third-party subprocess that really
  invokes pip; never replace uv with pip.
- **Models and datasets:** require a mounted persistent `HF_HOME` and derive
  `HF_DATASETS_CACHE` from it. A local model path does not remove this
  requirement because tokenizers, datasets, or transitive components may
  still consult the Hugging Face cache.
- **Compiler/JIT caches:** keep the hot vLLM, DeepGEMM, TorchInductor, and
  Triton caches on node-local storage. Seed them from persistent storage before
  startup and safely write back newly completed artifacts after the run when
  they will be reused. Do not put a highly concurrent mutable JIT cache
  directly on Lustre merely to make it persistent.
  NeMo-RL's `BaseVllmGenerationWorker.configure_worker` assigns each data-
  parallel worker `${HOME}/.cache/vllm_<seed>` and overrides an inherited
  `VLLM_CACHE_ROOT`. Therefore, verify the cache path printed by vLLM; for
  Slurm jobs, set `NRL_VLLM_CACHE_ROOT_BASE` to a node-local absolute path and
  `NRL_VLLM_CACHE_WRITEBACK_DIR` to the compatible persistent seed, seed the
  actual `vllm_<seed>` directories before launch, and let normal worker
  shutdown merge them back under a per-seed lock. For older branches without
  those variables, use a node-local `HOME` and an explicit write-back hook.
  Setting only a driver-level `VLLM_CACHE_ROOT` is not proof that vLLM's
  compile cache is persistent.
- **Compatibility:** namespace persistent caches by the inputs that can make
  artifacts incompatible: at least `uv.lock` fingerprint, container/backend
  version, GPU architecture, and model family/config where relevant. Never
  reuse a cache across an unknown compatibility boundary.
- **Concurrency and integrity:** do not delete or rewrite a shared cache while
  another job may use it. Ignore partial/temp entries when seeding, use
  atomic/age-guarded write-back, and keep per-job node-local working caches.
- **Measurement:** record whether each cache was cold, seeded, or warm. Keep
  dependency sync, model loading, and first-time compilation outside measured
  steady-state latency. When cold-start performance matters, report it as a
  separate metric rather than mixing it with warm model-call timing.

Before submission, audit all applicable variables: `UV_CACHE_DIR` or
`UV_CACHE_DIR_OVERRIDE`, `PIP_CACHE_DIR` (only if a subprocess really invokes
pip), `HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE` when required by the
installed Transformers version, `VLLM_CACHE_ROOT`, `DG_JIT_CACHE_DIR`,
`INDUCTOR_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and `TRITON_CACHE_DIR`.
Container builds must also use persistent, compatibility-keyed BuildKit layer
caches when the launcher supports them. Missing applicable cache configuration
is a launch defect, not an acceptable default.

## Slurm and container uv safety

- In a Pyxis/Enroot `srun` command, invoke environment assignment with the
  absolute `/usr/bin/env`. Do not use bare `env`: this cluster's inherited
  `PATH` can resolve it to a non-executable cache path such as `.../uv/env`.
- Choose the uv mode deliberately:
  - Use `uv run --frozen ...` for NeMo-RL entrypoints that import the current
    repository and must honor `uv.lock`.
  - Use `uv run --no-project python ...` only for standalone scripts that do
    not import NeMo-RL or its project dependencies.
  - Never switch a repository-importing script to `--no-project` merely to
    avoid a build; that can silently run the container's stale installed code.
- `ray.sub` intentionally unsets a host `UV_CACHE_DIR`. Do not propagate an
  arbitrary host cache into the container. If a persistent cache is required,
  set `UV_CACHE_DIR_OVERRIDE` so `ray.sub` mounts it explicitly.
- Treat an image-baked `/root/.cache/uv` as a read-mostly seed, not as the
  writable cache for a large environment refresh. New downloads and archive
  extraction can exhaust the container overlay. Put the writable cache on
  node-local `/tmp/uv_cache` or a mounted persistent path; keep a copy-mode
  venv on node-local storage when cached package symlinks may disappear.
- Keep `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` for NeMo-RL Ray jobs. Otherwise Ray
  may create a large uv environment per task, causing severe shared-cache
  contention.
- Before a long direct `srun`, preserve stderr with a Slurm output file and
  verify the resolved executable inside the same container. Never print the
  full environment or `COMMAND`; old logs may contain credentials.
- Before starting an expensive model benchmark, run a cheap import preflight
  in the exact same container, uv mode, extras, and virtual environment. At a
  minimum, import the repository module and backend (`transformers` and vLLM
  for a vLLM benchmark), and record package paths and versions without dumping
  the environment.

### Known failures and classification

| Evidence | Symptom | Cause | Prevention |
| --- | --- | --- | --- |
| Slurm `14011110` | `execve(): .../uv/env: Permission denied`, exit 13 | Bare `env` resolved through an inherited cache-prefixed `PATH` | Use `/usr/bin/env` |
| Slurm `14011010` | Allocation running with only `.extern`, no compute step or GPU process | The client-side command timeout killed attached `srun` and left an orphan allocation | Keep the client timeout longer than the job or use `sbatch`; audit and cancel orphans |
| Slurm `14011179` | `/usr/bin/env` launched successfully, then the program exited 1 | Not the earlier executable-resolution failure; stderr was lost when the attached client was interrupted | Persist stderr and diagnose the Python/vLLM traceback before blaming uv |
| Slurm `14011490` | `uv run --frozen` synchronized two packages, then `from transformers import PreTrainedTokenizerBase` failed with `transformers` from an unknown location | The selected project environment was not import-compatible with the benchmark; the exact package/extra mismatch remains to be diagnosed | Gate expensive launches on an identical-environment import preflight; do not treat a successful uv sync as proof that imports work |
| Slurm `14011963` | Syncing an isolated vLLM environment failed while extracting `wandb-core` with `No space left on device` | `/root/.cache/uv` was used as a writable download/extraction cache and exhausted the container overlay | Use a node-local or mounted persistent writable UV cache; do not refresh a large environment into `/root/.cache/uv` |
| Slurm `14022103`, `14022104` | Both nominally warm arms compiled for about 103 seconds under `/root/.cache/vllm_3` | `BaseVllmGenerationWorker.configure_worker` overrode the launcher's persistent `VLLM_CACHE_ROOT`; the advertised seed directory was never used or written back | Use the NeMo-RL node-local cache-base/write-back variables above and confirm the actual vLLM path in logs |
| Slurm `14025830` | A short unit test spent more than four minutes copying a large uv seed before it could print its first test log | The launch blindly seeded all of a shared uv cache into `/tmp` even though this was one single-node consumer | For a single-node diagnostic with no competing writers, use the mounted persistent uv cache directly; reserve node-local seeding for multi-node or contention-sensitive work and log before copying |
| Slurm `14026004`, `14026069`, `14026206`, `14026418`, `14026519`, `14026592` | Repository Ruff/pytest checks repeatedly failed in the SWE image: missing console scripts, packages without `__main__`, then missing Ray/Transformers or the source tree in isolated `uvx` environments | A container was added even though these unit tests only needed the repository's existing compute-node uv environment; the image's prebuilt `/opt/nemo_rl_venv` was not a complete development/test toolchain | For ordinary repository unit tests, reuse a compatibility-keyed local uv cache and the known-good compute-node command (`14026728`: `uv run --frozen --group test pytest ...`; add `--group dev` for Ruff). Use `/usr/bin/env` in `srun`. Add a container only when the test actually requires it, and then preflight the exact tool entry points and imports first |
| Slurm `14028866` | A direct compute-node vLLM test failed before preflight with `No interpreter found for Python 3.13.13` | An old launcher forced `UV_PROJECT_ENVIRONMENT` to a new empty `/tmp` venv, so uv abandoned the compatible repository environment and required an unavailable managed interpreter | For direct repository tests, do not set `UV_PROJECT_ENVIRONMENT` or a per-job venv unless isolation is required and the exact Python interpreter is pre-seeded; reuse the repository environment with the lock-keyed `UV_CACHE_DIR` |
| Slurm `14028904` | `uv run --extra vllm` fetched DeepGEMM and failed building its wheel with `CUDA_HOME environment variable is not set` | The test launcher enabled an optional backend extra that the compatible repository environment did not need, turning a test preflight into an unnecessary native dependency rebuild | Do not add extras speculatively. Use the minimal locked groups that already satisfy the identical-environment import preflight; when a native extra is genuinely required, preflight its compiler/CUDA environment and cache its build artifacts before the expensive run |
| Slurm `14054851` | Repository pytest in the SWE image synchronized successfully, then `/opt/nemo_rl_venv/bin/pytest` failed to import `console_main` | The image's inherited virtual environment supplied a broken pytest console entry point; package synchronization did not validate the executable actually selected by `PATH` | Run ordinary repository tests on the compute-node uv environment with `uv run --frozen --group test python -m pytest ...`; clear inherited `VIRTUAL_ENV`, `UV_PROJECT_ENVIRONMENT`, and `PYTHONPATH`, and preflight the resolved interpreter |
| Slurm `14054891` | A fresh 219-package temporary project environment was created, but the same `/opt/nemo_rl_venv/bin/pytest` entry point ran and failed | Forcing `UV_PROJECT_ENVIRONMENT` changed the synchronized environment but did not remove the container's inherited executable from `PATH`; the isolated refresh was both expensive and ineffective | Do not create a fresh per-job environment to repair an unrelated console-script selection problem. Prefer `python -m pytest` through the known-good locked uv environment and reuse the compatibility-keyed package cache |
| Slurm `14064725`, `14064836` | `uv 0.8.17` warned that `pyproject.toml` field `exclude-dependencies` was unknown, then continued syncing while ignoring that setting | The compute-node uv executable predates the project configuration schema used by current `main`; a successful sync does not mean all dependency exclusions were honored | Preflight and record `uv --version` before sync, and use a repository-supported uv version that recognizes every `[tool.uv]` field. Until the minimum version is declared, treat this warning as unsafe for native/vLLM builds because excluded overlapping packages may be installed |
| Slurm `14068972` | Building DeepEP under the new nightly failed in build isolation with `ModuleNotFoundError: No module named 'torch'` | The image's `uv 0.8.17` did not understand current `[tool.uv]` settings; after settings discovery failed, the native build did not receive the project's torch build dependency/no-isolation behavior | Namespace and prewarm a current uv executable with the image/model/lock cache (`uv 0.11.29` in `14069009`), fail fast on the uv version/API gate, and only then build the shared vLLM runtime environment |
| Slurm `14069608` | `uv run --no-sync ruff` created an empty `nemo-rl-cp3.13.13-*` environment, rewired the repository `.venv` symlink into the runtime seed, then failed to spawn Ruff | A persistent uv *seed* directory was passed as the live `UV_CACHE_DIR`; current uv stores managed project environments below that cache and may update `.venv` even with `--no-sync` | Never use a seed-only cache as the live `UV_CACHE_DIR`. Seed a separate writable cache or use the established project cache, verify `.venv` and the requested executable before the test, and restore any repository symlink immediately if a preflight changes it |
| Slurm `14069783` | A seeded Nano 3 vLLM cache failed on every TP rank with `FileNotFoundError` for `/tmp/nemo_rl_background_prefill_14069657/.../artifact_compile_range_*` | TorchInductor's persisted metadata contained the prior job's absolute node-local path, but the next launcher seeded it under a new job-ID path | Use a compatibility-keyed but path-stable node-local root across jobs, include the cache-layout version in the persistent namespace, and never seed absolute-path compiler artifacts into a different local path |
| Slurm `14084604` | A submit-wrapper manifest check ran bare `uv run` through uv `0.8.17`, warned that `exclude-dependencies` was unknown, synchronized project dependencies, and failed building `fast-hadamard-transform` with `ModuleNotFoundError: torch` | A standard-library-only audit accidentally invoked project environment resolution with an obsolete uv before any rollout was submitted | Run standard-library submission checks with an existing explicit Python interpreter. If a repository import truly requires uv, gate its version and use the compatible executable with `--frozen --no-sync`; never use bare `uv run` in a submit wrapper |
| Slurm `14085053` | A supposedly warm Nano 3 launch still rebuilt Transformer Engine, fast-hadamard-transform, mamba-ssm, and causal-conv1d; the old completion marker also prevented the newly complete cache from being written back | `.seed-complete` described an earlier partial prewarm, not the current `uv sync --frozen --extra mcore` contract | Version completion markers by the exact sync contract (for example `.seed-complete-mcore-v1`) and create them only after that command succeeds; a directory being nonempty is not a valid completeness test |
| Slurm `14091119`, `14091120` | A strict sequential off/on pair rebuilt Transformer Engine, fast-hadamard-transform, mamba-ssm, causal-conv1d, and DeepEP in both driver starts even though both arms used the same model/container/lock namespace | The setup-only `.seed-complete-mcore-v1` marker suppressed every later merge, while native wheels added by the driver were never written back from node-local `/tmp/uv_cache` | Never use a completion marker as a permanent merge skip. Merge complete entries after each successful locked sync, and add a lock-protected post-driver writeback that excludes temporary entries and preserves the workload exit code |
| Slurm `14087197` | Direct Pyxis preflight could not execute repository `.venv/bin/python` because its symlink target under `/home/joyang/.local/share/uv/python/...` did not exist in the container | The project venv depended on a uv-managed interpreter outside the mounted `/lustre` tree | Before using a host-created venv in a container, resolve its interpreter symlink and mount that interpreter tree at the same absolute path, or use the launcher's known-good uv environment |
| Slurm `14087398` | Nightly image Python reported `No module named pytest.__main__; 'pytest' is a package and cannot be directly executed` | The image's base environment is not a complete repository test environment even though a `pytest` package path exists | Do not infer a usable test runner from importability. Preflight `python -m pytest --version` in the exact environment, and use the repository's locked test environment for actual tests |
| Slurm `14087660` | Gym remained at 0/3 servers ready while three `uv pip install` processes spent more than 30 minutes with up to 116 threads each in Lustre `do_renameat2`; train and validation both wrote the same SWE-agent venv | Gym evaluated `skip_venv_if_present` while constructing all server commands, before either shared venv existed, so both commands independently scheduled a full install into one path | Do not configure an unused train server for rollout-only validation collection. More generally, serialize setup by resolved venv path and publish a completion marker only after dependency installation succeeds; checking only `bin/python` before concurrent launch is not a completeness contract |

An exit code alone does not establish a uv failure. Classify it as uv-related
only when stderr identifies uv dependency resolution, locking, cache, or
environment creation. Errors before Python `execve` are launcher/PATH errors;
tracebacks after Python starts belong to the executed program until proven
otherwise.

---

## Adding Dependencies

```bash
# Add a runtime dependency
uv add <package>

# Add an optional dependency
uv add --optional --extra <group> <package>

# Regenerate the lockfile after changes
uv lock
```

Commit both `pyproject.toml` and `uv.lock` together:

```bash
git add pyproject.toml uv.lock
git commit -s -m "build: add <package> dependency"
```

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `uv sync --locked` fails | Dependency conflict or stale lockfile | Re-run `uv lock` and commit updated lock |
| `ModuleNotFoundError` after pip install | pip installed outside uv-managed venv | Use `uv add` + `uv sync`, never bare `pip install` |
| Docker build fails at vLLM | vLLM build time overhead | Pass `--build-arg SKIP_VLLM_BUILD=1` |
