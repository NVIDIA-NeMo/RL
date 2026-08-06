# HybridEP Upstream #5008 Validation

These launchers validate 20 packed-sequence Qwen3-30B-A3B GRPO steps with the
upstream HybridEP uneven-dispatch fix on CW H100 and OCI-HSG GB200. They use the
canonical performance recipes unchanged and apply all validation settings as
runtime overrides.

## Contracts

| Cluster | Login | Allocation | Launcher |
|---|---|---|---|
| CW H100 | `cw-dfw-cs-001-login-01.nvidia.com` | `batch`, 4×8 H100, 1 hour | `submit-cw-qwen30.sh` |
| OCI-HSG GB200 | `oci-hsg-cs-001-vscode-02` | `batch`, 4×4 GB200, 4 hours | `submit-oci-qwen30.sh` |

The checked-in cluster manifests record the canonical recipe, runtime
overrides, topology variables, and output root for each platform. The derived
uneven-dispatch setting is intentionally not a YAML or CLI override; each job
probes `_apply_moe_config` and fails unless the effective value is `True`.

## Required runtime inputs

- `ACCOUNT`: select only after rechecking cluster FairShare. The launcher runs
  `sshare` again immediately before `sbatch` and records its output.
- `CONTAINER`: either a local squashfs plus `CONTAINER_SHA256`, or a container
  reference pinned with `@sha256:<64-hex-digest>`.
- `DEEPEP_WHEEL`: platform wheel built from
  `17cfb817bccec3a9c247013360cc550c2bac441e`.
- `DEEPEP_METADATA`: JSON metadata emitted by the wheel build pipeline. It must
  bind the exact `commit`, `platform`, `architecture`, `wheel`, and `sha256`.
- `SOURCE_PATH` (optional): defaults to the current repository root.
- `HF_CACHE` and `OUTPUT_ROOT` (optional): default beneath the platform's
  experiment-specific Lustre output root.

The source and every recursive submodule must be clean, including untracked
files, and `HEAD` must exactly match
`fork:sna/hybridep-always-pad-uneven-20260805`. The launchers fail closed on
the NeMo-RL commit, submodule origins and SHAs, MCore #5008 ancestry, DeepEP
artifact and import, Python 3.13.14, local GPU model/count, and effective model
configuration. A deterministic node-local overlay makes the verified wheel
precede actor-venv packages through `PYTHONPATH` and `LD_LIBRARY_PATH`. Before
training, a Ray probe resolves the registered `MegatronPolicyWorker` executable,
materializes the same MCore actor venv used by `worker_groups.py`, and confirms
that `deep_ep`, `deep_ep_cpp`, and `hybrid_ep_cpp` all import from the overlay on
every Ray node. Actor venvs are stored under `OUTPUT_ROOT`, never in the source
tree. The launchers write only selected provenance fields and never print
`WANDB_API_KEY`.

## Preflight and submission

Run on the corresponding cluster login node from the fresh recursive clone:

```bash
export ACCOUNT=<fairshare-selected-account>
export CONTAINER=/lustre/path/to/nemo-rl-nightly.sqsh
export CONTAINER_SHA256=<sha256>
export DEEPEP_WHEEL=/lustre/path/to/deep_ep-17cfb817-<platform>.whl
export DEEPEP_METADATA=/lustre/path/to/build-generated-deep_ep-metadata.json

TEST_ONLY=1 bash experiments/hybridep-upstream5008-validation/submit-cw-qwen30.sh
bash experiments/hybridep-upstream5008-validation/submit-cw-qwen30.sh
```

Use `submit-oci-qwen30.sh` for OCI-HSG. `TEST_ONLY=1` adds
`sbatch --test-only`; it performs all login-node provenance and FairShare
checks but does not launch training. Complete Slurm, Ray driver, training, node
hardware, and provenance logs remain beneath the configured `OUTPUT_ROOT`.
