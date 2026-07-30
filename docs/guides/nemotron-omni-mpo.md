# Nemotron Omni MPO

This path ports offline image MPO to the canonical `NemotronOmniModel`
from NeMo-RL PR #3290 (which supersedes #3227) and Megatron-Bridge PR #4885.
Development is based on the rewritten
`aroshanghias/nemotron-omni-main-migration` head. It deliberately does not
restore the legacy `LLaVAModel` collapse/expand flow.

## Container

PR #3290 does not name a published, immutable production image for this
workflow. Its migration document uses an Ali-owned placeholder image for an
intermediate launch example and lists the old `nvcr.io/nvidia/nemo:25.04`
environment only for the legacy vLLM 0.7 baseline. Neither is the target MPO
container.

Build the release image from the checked-out #3290-based source so the
NeMo-RL, Megatron-Bridge, Megatron-LM, TransformerEngine, and vLLM pins stay
consistent:

```bash
docker buildx build --build-context nemo-rl=. \
  --target release -f docker/Dockerfile \
  --tag nemorl-omni-mpo:pr3290 --load .
```

At this revision, `docker/Dockerfile` defaults to
`nvcr.io/nvidia/cuda-dl-base:26.03-cuda13.2-devel-ubuntu24.04`; use the
repository build rather than treating that base image as a complete NeMo-RL
runtime.

For Slurm/Pyxis, publish that image to the registry available to the cluster
and use it as `--container-image`. The same source can be used directly for
development with `uv run --extra mcore`, but a release-image build is the
reproducible qualification path.

MPO is offline preference training. It computes policy and reference
log-probabilities with Megatron and does not start a vLLM generation worker.
The vLLM compatibility work in #3290 therefore does not block the first
MPO run.

## Data and launch

PR #3290 adds maintained Nano image-GRPO references for Clevr
(`vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-megatron-tp8ep8.v1.yaml`) and
MMPR
(`vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-megatron-tp8ep16.v1.yaml`).
They are architecture references, not MPO recipes. This migration adds the
offline MPO recipe below.

The recipe accepts the existing MMPR meta-recipe JSON. Override its placeholder
path at launch:

```bash
uv run --extra mcore python examples/run_vlm_mpo.py \
  --config examples/configs/recipes/vlm/vlm_mpo-nemotron-omni-30ba3b-mmpr-1n8g-megatron-tp8.v1.yaml \
  data.train.data_path=/absolute/path/to/mmpr/meta.json
```

For first light, append `data.train.max_samples=1024 mpo.max_num_steps=2`.
Prepared MMPR rows are cached under `$HF_DATASETS_CACHE` (or
`$HF_HOME/datasets`) because scanning the full legacy meta-recipe on Lustre is
expensive.

For a faster integration-only check, also set
`policy.train_global_batch_size=8` and reduce `data.train.max_samples` to `64`.
Restore the recipe's batch size of 256 for parity and throughput qualification.

The Slurm helper exposes the same smoke-test controls:

```bash
CONTAINER=<registry-image> SBATCH_ACCOUNT=<account> \
MPO_MODEL_NAME=/path/to/canonical/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
MPO_DATA_PATH=/absolute/path/to/mmpr/meta.json \
MPO_MAX_SAMPLES=64 MPO_MAX_NUM_STEPS=2 \
MPO_TRAIN_GLOBAL_BATCH_SIZE=8 scripts/vlm_mpo.sh
```

The launcher explicitly enables online W&B logging. It defaults to entity
`joc`, project `nemotron-omni-main-migration`, and a unique run name; override
these with `WANDB_ENTITY`, `WANDB_PROJECT`, and `WANDB_NAME`. It forwards
`WANDB_API_KEY` or read-only mounts the submit host's `$HOME/.netrc`. The
container driver defaults to `WANDB_PIN_VERSION=0.28.1`, which supports current
`wandb_v1_...` tokens. Legacy 36-character keys can explicitly select
`WANDB_PIN_VERSION=0.21.1`. Set it to another semantic version, or to an empty
string to use the container SDK unchanged.

The data processor emits normal processor-expanded image tensors and tokens.
NeMo-RL's Megatron data pipeline keeps each chosen/rejected pair in one
microbatch and packs its expanded token rows into a full THD sequence.
`NemotronOmniModel` inserts media embeddings into that full sequence and then
selects the context-parallel slice.

## Qualification order

1. Run a short Nano single-image MMPR smoke test with CP=1 and MTP disabled.
2. Compare loss components, pair accuracy, chosen/rejected rewards, and the
   checkpointed BCO `reward_shift` against the legacy implementation.
3. Resume from a checkpoint and verify the first resumed shift update.
4. Qualify CP>1 separately with a valid parallel topology and
   `make_sequence_length_divisible_by` divisible by `2 * CP * TP` when
   sequence parallelism is enabled.
5. Defer multi-image, video/audio, Super, and MTP qualification until the Nano
   image baseline is stable.
