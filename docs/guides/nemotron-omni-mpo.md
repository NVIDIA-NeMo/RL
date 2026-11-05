# Nemotron Omni MPO

Mixed Preference Optimization (MPO) combines preference, supervised
fine-tuning, and binary-classification losses for offline preference data.
Nemotron Omni MPO uses `NemotronOmniModel` with NeMo-RL's Megatron data
pipeline and supports single-image MMPR chosen/rejected pairs.

## Container

Build the release image from the checked-out source so the
NeMo-RL, Megatron-Bridge, Megatron-LM, TransformerEngine, and vLLM pins stay
consistent:

```bash
docker buildx build --build-context nemo-rl=. \
  --target release -f docker/Dockerfile \
  --tag nemorl-omni-mpo:latest --load .
```

For Slurm/Pyxis, publish that image to the registry available to the cluster
and use it as `--container-image`. The same source can be used directly for
development with `uv run --extra mcore`, but a release-image build is the
reproducible qualification path.

MPO is offline preference training. It computes policy and reference
log-probabilities with Megatron and does not start a vLLM generation worker.

## Data and launch

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

The launcher enables online W&B logging and defaults to project `vlm-mpo`.
Override the destination with `WANDB_ENTITY`, `WANDB_PROJECT`, and
`WANDB_NAME`. It forwards `WANDB_API_KEY` or read-only mounts the submit host's
`$HOME/.netrc`. The container driver defaults to `WANDB_PIN_VERSION=0.28.1`;
set it to another semantic version, or to an empty string to use the container
SDK unchanged.

The data processor emits normal processor-expanded image tensors and tokens.
NeMo-RL's Megatron data pipeline keeps each chosen/rejected pair in one
microbatch and packs its expanded token rows into a full THD sequence.
`NemotronOmniModel` inserts media embeddings into that full sequence and then
selects the context-parallel slice.

The pinned Megatron-Bridge distinguishes NeMo-RL's dense 4D causal mask from
the 2D media-token validity mask, so unpacked image batches are supported by
the integration. Packed training remains the recommended path for useful Super
context lengths because unpacked batches have substantially higher memory use.

## MPO configuration

The algorithm-specific settings live under `mpo`:

- `reference_policy_kl_penalty`: scales policy/reference log-ratios.
- `preference_loss_weight`: weights the pairwise preference term.
- `sft_loss_weight`: weights chosen-response negative log-likelihood.
- `bco_loss_weight`: weights the binary-classification objective.
- `preference_average_log_probs`, `sft_average_log_probs`, and
  `quality_average_log_probs`: select sum or token-average normalization for
  the three terms.
- `reward_shift`: initial BCO reward centering value.
- `reward_shift_momentum`: exponential-moving-average momentum for driver-side
  reward-shift updates. The updated value is checkpointed.
- `max_num_steps`: final training and scheduler horizon.
- `stop_after_step`: optional segment boundary for time-limited jobs. Keep
  `max_num_steps` unchanged across resumed segments.

Packed MPO additionally requires:

- `policy.sequence_packing.fuse_loss=true`
- `policy.sequence_packing.pair_grouping_key=pair_index`
- a Megatron policy backend

`max_sequences_per_bin=1` bounds each bin to one chosen/rejected pair. When it
is enabled, `policy.train_global_batch_size` must be divisible by the
data-parallel degree.

## Chained jobs

Use `scripts/vlm_mpo_chain.sh` when the scheduler wall-clock limit is shorter
than the intended run. `CHAIN_STEP_TARGETS` contains cumulative, strictly
increasing boundaries, while `MPO_MAX_NUM_STEPS` remains the final horizon:

```bash
CHAIN_SEGMENTS=2 CHAIN_STEP_TARGETS=50,100 MPO_MAX_NUM_STEPS=100 \
CONTAINER=<registry-image> SBATCH_ACCOUNT=<account> \
MPO_MODEL_NAME=/path/to/model MPO_DATA_PATH=/path/to/mmpr/meta.json \
scripts/vlm_mpo_chain.sh
```

All segments use one checkpoint directory and one W&B run ID. Each dependent
segment restores model, optimizer, scheduler, dataloader, and MPO reward-shift
state before continuing.

## Current scope

The provided recipe targets single-image MMPR data with Nemotron Omni Nano.
Packed and unpacked image batches are supported. Multi-image, video/audio
preference data, and MTP training require separate qualification.
