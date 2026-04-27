# Nemotron 3 Nano Omni

This recipe explains how to run the Nemotron 3 Nano Omni workflow end to end. The first step is to create a container, then you can run three different algorithms, depending on your use case: MPO, text-only GRPO, and vision GRPO. Text-only GRPO is provided, in case you want to improve text-related capabilities. Otherwise, MPO and vision GRPO are the two algorithms you need for vision RL.

## 0) Clone the nano-v3-omni branch
We have built the code to fine-tune the Omni model into a separate branch:

```bash
git clone --branch nano-v3-omni --single-branch --recurse-submodules https://github.com/NVIDIA-NeMo/RL.git /path/to/nemo-rl-omni
```
Make sure all the submodules are downloaded properly.

## 1) Build the container

Start by building a container image and converting it to an Enroot `.sqsh`.

1. SSH to your cluster login node and enter the repo:

```bash
ssh <cluster-login-host>
cd /path/to/nemo-rl-omni
```

2. Use the `nano-v3-omni` branch and refresh submodules:

```bash
git checkout nano-v3-omni
git submodule update --init --recursive
```

3. Build the Docker image locally on that compute node:

```bash
TAG="$(git rev-parse --short HEAD)"

docker buildx build \
  --target release \
  --build-arg MAX_JOBS=4 \
  --build-arg UV_VERSION=0.9.30 \
  --build-arg VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/27ca95b3c9e6e32ac02508e729c01865800fd036/vllm-0.14.0rc2.dev196%2Bg27ca95b3c-cp38-abi3-manylinux_2_31_x86_64.whl" \
  --build-arg SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM="0.14.0rc2.dev196+g27ca95b3c" \
  -f docker/Dockerfile \
  --tag "nemo-rl:nano-v3-vl-${TAG}" \
  --load .
```

4. Convert the local Docker image into an Enroot `.sqsh` on the same compute node:

```bash
mkdir -p /tmp/$USER/containers /tmp/$USER/enroot-tmp /tmp/$USER/enroot-cache

export ENROOT_TEMP_PATH=/tmp/$USER/enroot-tmp
export ENROOT_CACHE_PATH=/tmp/$USER/enroot-cache
DATE_TAG="$(date +%Y%m%d)"
enroot import \
  -o "/tmp/$USER/containers/nemo-rl-nano-v3-vl-${DATE_TAG}-${TAG}.sqsh" \
  "dockerd://nemo-rl:nano-v3-vl-${TAG}"
```

5. The final container path will be:

```bash
/tmp/$USER/containers/
```

## 2) Configure runtime settings

The Nano v3 Omni launchers source `${NEMORL}/.env`. Put your cluster, container, checkpoint, and dataset paths there once instead of exporting them before every run.

Create `/path/to/nemo-rl-omni/.env` and fill in the values for the stages you plan to run.

```bash
# Shared launcher settings
SBATCH_ACCOUNT=your_slurm_account
CONTAINER=/path/to/containers/nemo-rl-nano-v3-vl-tag.sqsh
MOUNTS=/lustre:/lustre # (for example)
NRL_FORCE_REBUILD_VENVS=false

# Optional shared settings
# SBATCH_PARTITION=batch
# SBATCH_TIME=4:00:00
# GPUS_PER_NODE=8
# CACHE_ROOT=/path/to/users/$USER/.cache/nemo-rl
# RESULTS_ROOT=/path/to/results

# MPO
MPO_MODEL_NAME=/path/to/initial_policy_checkpoint
MPO_DATA_PATH=/path/to/data/mmpr_public/processed/MMPR-v1.2/meta_public.json

# Text GRPO
TEXT_GRPO_MODEL_NAME=/path/to/mpo_or_sft_checkpoint
TEXT_GRPO_TRAIN_DATA_PATH=/path/to/data/text_only_rl/processed/train.jsonl

# Image GRPO
IMAGE_GRPO_MODEL_NAME=/path/to/mpo_or_sft_checkpoint
IMAGE_GRPO_CACHE_DIR=/path/to/data/mmpr_tiny/processed
```

The first launch may spend a few minutes creating worker-env caches under `CACHE_ROOT/ray_venvs`; later reruns with the same container and dependency fingerprint reuse the ready caches automatically. Set `NRL_FORCE_REBUILD_VENVS=true` only when you intentionally want to rebuild those worker environments after a dependency or container change.

## 3) Run Public MMPR MPO

Prepare the public `OpenGVLab/MMPR` dataset and submit Nano Nemotron 3 Omni MPO with `scripts/nanov3_mpo.sh`.

The launcher:

- reads `data/mmpr_public/processed/MMPR-v1.2/meta_public.json`
- runs `examples/run_vlm_mpo.py` with `examples/omni/nanov3_mpo.yaml`
- submits the MPO job through `ray.sub`

The MPO data loader expects one combined `meta_public.json`. Validation is still split internally by `mmpr.py`, so this flow does not create separate train/val files.

1. Download the public MMPR dataset:

```bash
mkdir -p data/mmpr_public

uvx --from huggingface-hub hf download \
  OpenGVLab/MMPR \
  --repo-type dataset \
  --local-dir data/mmpr_public/raw
```

2. Convert the downloaded dataset into the layout expected by the MPO launcher. This step can take 30-45 minutes.

```bash
uvx --with tqdm python scripts/prepare_public_mmpr_for_mpo.py \
  --input-dir data/mmpr_public/raw \
  --output-dir data/mmpr_public/processed/MMPR-v1.2 \
  --meta-name meta_public.json
```

3. Confirm that the processed MPO input looks like:

```text
data/mmpr_public/processed/MMPR-v1.2/
  annotations/
  images/
  meta_public.json
  prepare_public_mmpr_for_mpo_summary.json
```

4. Set these `.env` values to point at the processed dataset and starting checkpoint:

```bash
MPO_MODEL_NAME=/path/to/initial_policy_checkpoint
MPO_DATA_PATH=/path/to/data/mmpr_public/processed/MMPR-v1.2/meta_public.json
```

5. Submit the MPO job from the repo root:

```bash
bash scripts/nanov3_mpo.sh
```

That wrapper points `data.data_path` at the generated `meta_public.json`, sets `policy.model_name` to the configured checkpoint, and submits `examples/run_vlm_mpo.py` through `ray.sub`.

## 4) Run Text-only RL

Prepare the text-only RL dataset and submit text-only GRPO training with `scripts/nanov3_text_rl.sh`.

The launcher:

- reads the processed JSONL file produced below
- runs `examples/omni/nanov3_text_rl.yaml`
- submits a SLURM job through `ray.sub`

1. Download the dataset:

```bash
mkdir -p data/text_only_rl

uvx --from huggingface-hub hf download \
  nvidia/Nemotron-3-Nano-RL-Training-Blend \
  --repo-type dataset \
  --local-dir data/text_only_rl/raw
```

2. Process the raw dataset:

```bash
mkdir -p data/text_only_rl/processed

uv run --script data/text_only_rl/raw/create_nanov3_jsonl.py \
  --input data/text_only_rl/raw/train.jsonl \
  --output data/text_only_rl/processed/train.jsonl
```

Use `data/text_only_rl/processed/train.jsonl` for RL training.

3. Set these `.env` values to point at the processed JSONL and starting checkpoint:

```bash
TEXT_GRPO_MODEL_NAME=/path/to/mpo_or_sft_checkpoint
TEXT_GRPO_TRAIN_DATA_PATH=/path/to/data/text_only_rl/processed/train.jsonl
```

4. Submit the job from the repo root:

```bash
bash scripts/nanov3_text_rl.sh
```

On success, `sbatch` prints the submitted job ID.

## 5) Run Vision RL

Prepare the public `OpenGVLab/MMPR-Tiny` dataset for the next Nano Nemotron 3 Omni vision RL stage.

The prep script:

- reads the raw Hugging Face snapshot under `data/mmpr_tiny/raw`
- extracts `images.zip` into the cache layout expected by `nemo_rl.data.datasets.response_datasets.mmpr_tiny`
- copies `mmpr_tiny.parquet`, writes an inspection JSONL, and records a preparation summary under `data/mmpr_tiny/processed`

This step prepares the cache directory that the vision RL launcher passes to `data.cache_dir`.

Run the commands below from the repo root.

1. Download the public MMPR-Tiny dataset:

```bash
mkdir -p data/mmpr_tiny/raw

uvx --from huggingface-hub hf download \
  OpenGVLab/MMPR-Tiny \
  images.zip \
  mmpr_tiny.parquet \
  --repo-type dataset \
  --local-dir data/mmpr_tiny/raw
```

2. Convert the raw snapshot into the local cache layout expected by the existing `mmpr_tiny` loader:

```bash
mkdir -p data/mmpr_tiny/processed

uv run --no-project --with pandas --with pyarrow --with tqdm \
  python scripts/prepare_mmpr_tiny_for_vision_rl.py \
  --input-dir data/mmpr_tiny/raw \
  --output-dir data/mmpr_tiny/processed
```

3. Confirm that the processed vision RL input looks like:

```text
data/mmpr_tiny/processed/
  .mmpr_ready
  MMPR-Tiny/
    images/
  mmpr_tiny.parquet
  mmpr_tiny_preview.jsonl
  prepare_mmpr_tiny_for_vision_rl_summary.json
```

4. Set these `.env` values to point at the processed cache and starting checkpoint:

```bash
IMAGE_GRPO_MODEL_NAME=/path/to/mpo_or_sft_checkpoint
IMAGE_GRPO_CACHE_DIR=/path/to/data/mmpr_tiny/processed
```

The existing `mmpr_tiny` response dataset already reads:

- `<cache_dir>/mmpr_tiny.parquet`
- `<cache_dir>/MMPR-Tiny/images/...`

The generated `mmpr_tiny_preview.jsonl` is only for inspection and debugging. The current GRPO loader still reads the parquet file and extracted image cache directly.

5. Before launching vision RL, keep these dataset knobs in mind:

- `data.dataset_name=mmpr_tiny`
- `data.cache_dir=<processed dir>`
- `data.val_size=<number of validation samples>`

6. Run vision RL

```bash
bash scripts/nanov3_vision_rl.sh
```