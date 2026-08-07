# Qwen3-30B-A3B HybridEP Padding A/B

This experiment isolates dispatcher padding and DeepEP revision effects on the
canonical four-node, eight-GPU-per-node H100 performance recipe. All arms run
20 packed-sequence GRPO steps from the frozen NeMo-RL `ba473d4752` lineage and
use the same container, model cache, topology, batch settings, and node class.

| Arm | Dispatcher | DeepEP | Padding path |
|---|---|---|---|
| `official-alltoall` | all-to-all | unused | none |
| `official-pr5008-17cf` | HybridEP | `17cfb817` | MCore PR 5008 per-dispatch padding |
| `official-pr5008-f725` | HybridEP | `f725d296` | MCore PR 5008 per-dispatch padding |
| `legacy-prepad-17cf` | HybridEP | `17cfb817` | one NeMo pre-pad per microbatch; PR 5008 padding disabled |

`arm_matrix.py` is the authoritative machine-readable contract. The launcher
fails closed on source and recursive submodule state, pushed branch identity,
container SHA, DeepEP wheel metadata and SHA, H100 count, effective MCore
configuration, and the legacy padding contract.

## Render and validate

Rendering is side-effect free and does not require cluster tools or artifacts.

```bash
python3 experiments/hybridep-padding-ab-q30/arm_matrix.py --list --format json
ARM=official-pr5008-17cf RENDER_ONLY=1 \
  bash experiments/hybridep-padding-ab-q30/submit-cw-qwen30-matrix.sh
```

Before every real submission, select `ACCOUNT` from a fresh FairShare check and
set the immutable container and source inputs. HybridEP arms also require the
wheel and build-generated metadata for their selected DeepEP commit.

```bash
export ACCOUNT=<fairshare-selected-account>
export SOURCE_PATH=/lustre/path/to/pushed-recursive-checkout
export FORK_BRANCH=sna/hybridep-padding-ab-q30-20260807
export CONTAINER=/lustre/path/to/nemo-rl-nightly.sqsh
export CONTAINER_SHA256=<sha256>
export DEEPEP_17CF_WHEEL=/lustre/path/to/deep_ep-17cf-x86_64.whl
export DEEPEP_17CF_METADATA=/lustre/path/to/17cf-build-metadata.json
export DEEPEP_F725_WHEEL=/lustre/path/to/deep_ep-f725-x86_64.whl
export DEEPEP_F725_METADATA=/lustre/path/to/f725-build-metadata.json
export HF_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home

ARM=official-pr5008-17cf TEST_ONLY=1 \
  bash experiments/hybridep-padding-ab-q30/submit-cw-qwen30-matrix.sh
ARM=official-pr5008-17cf \
  bash experiments/hybridep-padding-ab-q30/submit-cw-qwen30-matrix.sh
```

The launcher requests all eight GPUs on every node with `--gpus-per-node=8`,
does not request exclusive access, and does not set explicit CPU or memory
requests. Each arm has an independent output directory and runtime overlay.

## Analysis contract

The primary comparison uses matched Steps 2–20. Steps 5–20 are reported as a
steady-state sensitivity window. Report arithmetic mean phase times and
ratio-of-sums tokens/s/GPU for Policy, LogProb, generation, and end-to-end.
Also report reward, KL, validation accuracy, failures, and bounded padding
telemetry separately from the performance runs.
