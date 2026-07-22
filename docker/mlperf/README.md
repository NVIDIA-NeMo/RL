# 1. Problem

Reinforcement-learning post-training (GRPO) of Qwen3.5-397B-A17B on agentic
software-engineering tasks: the policy solves R2E-Gym SWE instances through
the OpenHands agent framework inside sandboxed apptainer containers, driven by
[nemo-rl](https://github.com/NVIDIA-NeMo/RL) with NeMo Gym environments.

Unlike the pretraining/fine-tuning benchmarks in this repo, a run is a Ray
cluster: Megatron policy training and vLLM generation run on DISAGGREGATED
node pools (`DGXNNODES = TRAIN_NODES + GEN_NODES`), and NeMo-Gym schedules SWE
sandbox rollouts against the generation pool.

## Sources and provenance

The Dockerfile defaults to the public `NVIDIA-NeMo/RL` repository and requires
`NEMO_RL_REVISION` to name an immutable commit. Production builds pass the
qualified commit explicitly rather than relying on a moving branch name.
Everything benchmark-owned — recipes (`conf/`), the driver
(`run_grpo_nemo_gym.py`), MLPerf logging (`mlperf_grpo_logging.py`), narrow
Gym/NeMo-RL/R2E patches, launchers, and data scripts — lives in this
repository and is baked over that clone.

This directory was synchronized from
`optimized@cf2151a3cdc80b4a6f2213fc992963077acd3c31`. The intentional RL-side
extensions are bounded Gym disconnect retries and delayed periodic validation
for the legacy synchronous, TransferQueue synchronous, and asynchronous
trainers.

The carryover audit compared exact Git trees and blobs, not branch names:

| Tree | Revision | Audit result |
|---|---|---|
| `main` Qwen integration | target branch at review time | Baseline Dockerfile, patch stack, recipes, and launcher |
| Historical enablement base | `41352f74610762cd2f3014533be27f4c30f8524e` | Identical commit object across the audited public mirrors; contains the pause-race and every-refit prefix-cache fixes |
| Docker source tree | required `NEMO_RL_REVISION` | Current canonical Qwen branch commit; carried patches are applied only when absent |
| NVIDIA-NeMo/RL `main` | `bc382dc455c7da39d2c4ec0715e70028f050844e` | Live 2026-07-22 comparison; still lacks grouped validation, delayed validation, and the qualified Qwen3.5 suffix-prefix repair |
| Qualified Qwen revision | required `NEMO_RL_REVISION` | Source of the qualified profile and remaining fixes; use its immutable commit rather than a branch name |

The optimized-owned driver, MLPerf logger, recipes, and former runtime-fix
files at the baseline hash-identically matched their counterparts in
`41352f746`. That historical commit was pushed unchanged between all three
repositories above. Programmatic diffs against the exact dependency pins then reduced
all seven broad file overlays to narrow build-time patches for TE SM103, R2E
environment isolation, bounded Gym disconnect retries, and the fla PR #1000
GDN restriction. The global FLA
diagnostic and redundant R2E utility overlays were removed. The NeMo-RL stack
adds only grouped pass@k and delayed async validation; the source pin already
contains pause-race commit `9402b7de8` and prefix-cache invalidation commit
`3a643e753`. The atomic Gym metrics patch remains because it is exercised by
trajectory writes and is absent from pinned Gym. The former Qwen-specific Gym
tool-call patch was removed: pinned Gym already normalizes reasoning for
structured tool-call history, and the exact Qwen3.5 template renders the
normalized and tagged-content forms identically. Pinned Gym's
`responses_api_models/vllm_model/tests/test_app.py::TestApp::test_responses_reasoning_parser_reasoning`
covers that multi-turn structured-tool-call normalization. Missing or invalid
per-instance rollouts remain loud, masked zero-reward trajectories through the
immutable source pin, preserving batch shape without creating reward. This RL
branch also retains a 120-second bound for repeated Gym disconnect retries so a
dead endpoint cannot wedge a run indefinitely. See `patches/nemo-rl/README.md`
for the reverse-check decisions.

The image records the resolved recursive submodule manifest, optimized and
NeMo-RL revisions, downstream patch hashes, locked DeepEP/vLLM/Transformer
Engine artifacts, the effective policy-only HybridEP override, base image
reference, and lockfile hash in `/NEMO_RL_PROVENANCE.txt`. NVIDIA-internal
telemetry is excluded by default and recorded only when explicitly enabled.
The launcher records the submitted `CONT` image/SquashFS reference as
`source_image.name` when JET is enabled.

| Component | Immutable revision/version |
|---|---|
| NeMo-RL source | required `NEMO_RL_REVISION`; recorded in the image provenance |
| NeMo Gym | `610a08ab5fe9f8f5fb5fff36b170429ea67f0f92` |
| Megatron Bridge | `554c7b9324225aa863eee52e8b8fdde7abced2b1` |
| Megatron Core | `002255075c3728fded9a2e435677840b08560d55` |
| Automodel | `24b47e856263d313b942f0ed666c63fff83306b4` |
| DeepEP, Megatron policy (arm64) | clean upstream `e0a5b1d9848ab3e7b4a67842bf06f067bfac67f8`, replacing locked `a48493600c4886c1b297aaa78db0e1ebc2d8dd6c` only in the policy actor venv; SM100+SM103 |
| vLLM | `0.20.0`; wheel SHA256 `29a135ca0d70650f057f15c7c0b560d24659524c771f70fbddc24597c861c118` (arm64), `24d28892e210200f6e1bd13f699c42a74cd2bb7364c11248e2348f677c7f6dfb` (x86_64) |
| Transformer Engine | `42b840051647eef89761a16dfdff87e82bb253ab` (`2.15.0+42b8400`) |
| Base image | `nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04@sha256:731f274d76b8585d9d2e0a78b1d0b8773d47a415b670c8eed88f6eb11543cb41` |
| ntrace (optional NVIDIA-internal instrumentation) | `f5ad3e960709435b4837ca7947a1ecefd25fc439` |

## Upstreaming status

| Fix (present in the pinned source or optimized image) | Target | Status |
|---|---|---|
| vllm_worker_async self.cfg AttributeError | NVIDIA-NeMo/RL | not opened |
| replay-buffer ready check `==`→`>=` | NVIDIA-NeMo/RL | not opened |
| configurable NCCL timeout (NRL_NCCL_TIMEOUT_MINUTES) | NVIDIA-NeMo/RL | not opened |
| configurable check_for_nan_in_grad | NVIDIA-NeMo/RL | not opened |
| attention_backend selection + forbid TE unfused fallback | NVIDIA-NeMo/RL | not opened |
| greedy/deterministic validation | NVIDIA-NeMo/RL | not opened |
| _stable_group_ids advantage-grouping convergence fix | NVIDIA-NeMo/RL | not opened |
| async-grpo collection-after-refit + refit diagnostics | NVIDIA-NeMo/RL | not opened |
| MoE per-expert HF export in refit | NVIDIA-NeMo/RL | not opened |
| Qwen3.5 multi-turn prefix repair | NVIDIA-NeMo/RL | retained; no current-main equivalent; covered by suffix-repair/strict-failure unit tests |
| grouped validation and pass@k accuracy | NVIDIA-NeMo/RL | narrow downstream patch; absent from current main |
| delayed periodic validation in all three GRPO trainers | NVIDIA-NeMo/RL | narrow downstream patch; absent from current main |
| obsolete Gym tool-call reasoning bypass | NVIDIA-NeMo/Gym | removed; pinned Gym normalization and exact Qwen template are equivalent |
| atomic recoverable trajectory metrics | NVIDIA-NeMo/Gym | narrow downstream patch retained |
| bounded transient Gym disconnect retries | NVIDIA-NeMo/Gym | narrow downstream patch retained; 120-second default bound |
| sanitized R2E testbed environment | nv-R2E-Gym | narrow downstream patch; not upstream |
| TE flash-attn hdim-256 sm10.3 gate | TransformerEngine | guarded one-line build-time change in the policy actor venv; not upstream |
| FLA Blackwell GDN backward autotune restriction | fla | narrow PR #1000 backport retained |
| FLA global-autotuner diagnostic monkey patch | fla | removed; absent from the qualified image |

# 2. Hardware Requirements

- GB300 NVL; reference scale is 64 nodes x 4 GPUs, split into 16 policy and
  48 generation nodes.
- `/dev/fuse` available on compute nodes (apptainer SIF sandboxes run inside
  the training container).

# 3. Set up

## Build the container

```bash
source_commit="$(git rev-parse HEAD)"
docker buildx build --platform linux/arm64 \
  --build-arg GIT_COMMIT_ID="$source_commit" \
  --build-arg NEMO_RL_REVISION="$source_commit" \
  -t <registry>/qwen35_397b_grpo:pytorch \
  -f docker/mlperf/Dockerfile docker/mlperf
```

Useful build args: `NEMO_RL_REPO`, `NEMO_RL_REVISION`,
`HYBRID_EP_REPO`, `HYBRID_EP_REVISION`, and `NEMO_GYM_REVISION` (`SKIP` keeps
the recorded submodule pin). `NEMO_RL_REVISION` and `GIT_COMMIT_ID` are
mandatory immutable commits; do not pass branch names. `GIT_COMMIT_ID` is
written to the OCI label and
`/NEMO_RL_PROVENANCE.txt`.

The default build uses the public PyPI index and builds NeMo-RL from the public
GitHub source; it does not inherit from a gated nightly image or require
private services.

An internal package mirror can still be selected explicitly with the
`PIP_INDEX_URL` and `UV_INDEX_URL` build arguments; it is not the default.

The HybridEP build fetches the official DeepEP repository at the exact
qualified commit, verifies that the detached checkout is clean, and installs
it without any downstream source patch. It replaces DeepEP only in the
Megatron policy-worker venv; vLLM retains the dependency selected by the 413
lock. Do not set `NRL_FORCE_REBUILD_VENVS=true` for the authoritative profile:
that emergency mode recreates the policy venv from the lock and would replace
the qualified `e0a5b1d` build with `a484936`.

## Prepare inputs

See `data_scripts/README.md` (HF snapshot, mcore checkpoint cache, R2E-Gym
SIF images, train/val jsonl). Keep site paths in an external, untracked data
config and pass it with `--data-config`. The authoritative profile selects the
`easy-curriculum-subset` training JSONL relative to that cluster's R2E-Gym
artifact root; set `QWEN35_CURRICULUM_DATA_PATH` only when the same file is
staged elsewhere. Its SHA-256 is
`c07bcd64ed1c558e28d091239104e38295a5e696c1d21bb0b61f0346c7eaa0f7`.

# 4. Launch training

The production profile uses only code and configuration baked into the image.
Set the image and result directory, then use the audited submission wrapper:

```bash
export CONT=<image or .sqsh path>
export LOGDIR=<shared-filesystem>/results

./submit_rcp.sh \
    --data-config <external-site-config.sh> \
    --gbs 256 \
    --val-start 18 \
    --max-steps 30 \
    --replicas 6
```

The wrapper requires an external site data config, either through
`--data-config PATH` or `RCP_DATA_CONFIG`. It submits each replica as an
independent Slurm job and prints the shared seed base. `--lr`, `--clip`,
`--val-start`, and `--max-steps` are allowlisted study overrides. Omitting one
uses the value from the selected GBS recipe.

The default MLPerf pass@4 target is `0.7`. Pass an empty target explicitly to
disable target-based early stopping:

```bash
./submit_rcp.sh --data-config <external-site-config.sh> --gbs 512 --target "" --replicas 1
```

With no target, validation is still measured and the job runs until its
configured maximum steps or wall time. It does not report an artificial
successful convergence. `EXTRA_ARGS` remains available as an explicit testing
escape hatch and is printed by the wrapper when nonempty; qualified production
runs should use the named options.

`run.sub` brings up the Ray cluster (head on the first sorted node), waits
for all `DGXNNODES * DGXNGPU` worker units, then runs `run_and_time.sh` ONCE
on the head node per experiment (NEXP). `config_common.sh` exports the
`OccupiedIdleGPUsJobReaper` SBATCH_COMMENT exemption — required for async
GRPO whose training GPUs legitimately idle during rollout buffer-fill.

For dev iteration without an image rebuild, the narrowest temporary option is
to apply a patch to each node's writable container at launch:

```bash
export NRL_RUNTIME_PATCH=/path/to/02-grouped-validation-pass-at-k.patch
```

The launcher mounts that file read-only and applies it before Ray starts; the
host checkout and image remain unchanged. For broader source iteration,
`NRL_SOURCE_OVERLAY=1` plus `REPO_LOCATION=<host nemo-rl checkout>` still
overlay-mounts `nemo_rl/`, `examples/`, and `qwen_35/` (CI always runs baked
sources); use that mode separately from `NRL_RUNTIME_PATCH`.

The patch and source-overlay controls above are development facilities only.
The authoritative launch leaves them unset and does not mount `/workspace/llm`.

## Authoritative multi-GBS profiles

One full common YAML owns the qualified algorithm and system behavior. Four
small child YAMLs own only batch-dependent study hyperparameters:

| GBS | Prompts x generations | Learning rate | Gradient clip | Validation starts | Maximum steps |
|---:|---:|---:|---:|---:|---:|
| 256 | 16 x 16 | `1.0e-6` | `0.125` | 20 | 30 |
| 512 | 32 x 16 | `1.4142135624e-6` | `0.08838834765` | 10 | 20 |
| 768 | 48 x 16 | `1.7320508076e-6` | `0.0721687836` | 7 | 14 |
| 1024 | 64 x 16 | `2.0e-6` | `0.0625` | 5 | 10 |

`config_GB300_64x4_t16g48_tp4pp2ep32gtp8.sh` supplies the common 16-policy /
48-generation-node topology. The GBS256 profile is the qualified baseline
unless an allowlisted option is supplied.

| Setting | Value |
|---|---:|
| Policy + generation nodes | 16 + 48 |
| Prompts x generations / GBS | selected by `_gbs*.yaml` |
| Training sampling | temperature 1.0, top-p 1.0 |
| Validation sampling | temperature 0.1, top-p 0.95 |
| Loss filtering | `seq-mask-tis`, `[0.999, 1.002]` |
| Learning rate / gradient clip | selected by `_gbs*.yaml` |
| Gym train/validation concurrency | 256 / 256 |
| Train/validation agent timeout | 1800 s / 1800 s |
| MoE dispatcher | `flex` / `hybridep`, 32 SMs |
| CUDA graphs | `VLLM_COMPILE`, `FULL_AND_PIECEWISE` |
| Prefix cache | enabled, reset after every refit |
| vLLM batching | 16,384 tokens, throughput mode |
| Validation | four generations, every step from the profile's start step |
| MLPerf metric / target | `pass@4` / `0.7` |
| Maximum steps | selected by `_gbs*.yaml` |
| Checkpoints / optimizer state | disabled / disabled |
| Segment size | 8 |

Global `CUDA_DEVICE_MAX_CONNECTIONS=1` and
`NCCL_LAUNCH_ORDER_IMPLICIT=1` are exported before Ray starts so policy and
generation actors inherit the same communicator settings. The common recipe
fixes Gym concurrency to 256 symmetrically. GBS is derived from prompts times
generations, so selecting a profile cannot leave a conflicting independent
batch-size field.

The qualification run reached grouped `pass@4=0.73828125` at step 20 and
emitted a successful MLPerf stop against the `0.7` target. Checkpointing and
optimizer-state saves are disabled in this authoritative workload.

## Cross-system qualification risk

Qualification on one system does not establish portability to another
scheduler, network, or memory configuration. A cross-system attempt OOMed
during an early refit in grouped expert export after large `torch.stack` and
packed-broadcast `torch.cat` allocations. Refit memory and buffer sizing remain
explicit production-qualification requirements.

# 5. Quality

- Metric: grouped observed `pass@4` (a prompt passes when any of its four
  validation trajectories has positive reward), emitted as MLLOG
  `eval_accuracy` by `mlperf_grpo_logging.py` to the mllog file
  (`/results/<datestamp>_<n>_mllog.log`), checked by
  `mlperf_logging.compliance_checker`.
- Default target: `MLPERF_TARGET_ACCURACY=0.7`; `--target ""` disables target
  stopping without forcing a successful status.

# 6. Additional notes

- Config naming: `config_<SYSTEM>_<TOTAL>x<GPUS>_t<TRAIN>g<GEN>_<parallelism>.sh`
  where tp/pp/ep = Megatron policy parallelism and gtp = vLLM generation TP
  (both configured in the recipe yaml; the name documents them).
- CI: registered in `ci/scripts/generate_pipeline.py` as
  `qwen35_397b_grpo_pytorch`; the exposed production system configuration is
  the 64-node t16g48 profile above, with batch shape selected by recipe.
