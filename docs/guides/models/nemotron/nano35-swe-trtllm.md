# Nano 3.5 SWE training with TRT-LLM

This side branch runs asynchronous GRPO in NeMo-RL: SWE agents collect
trajectories, Megatron updates the policy, TRT-LLM receives updated weights,
and checkpoints preserve the optimizer and asynchronous data progress.

The recipe has completed a 48-step training run with checkpoint save and
cross-job resume. See [validation scope](#validation-scope) for the tested
behavior and known limitation.

Choose one of the following ways to prepare the training environment and SWE
assets. After completing either option, prepare an [initial model](#initial-model)
and follow the same [preflight and launch](#preflight-and-launch) steps.

| Setup option | Training image | SWE data and task environments |
| --- | --- | --- |
| [1. Download from Hugging Face](#option-1-download-from-hugging-face) (recommended) | Download the prebuilt SQSH, including patched TRT-LLM and actor environments | Download the prepared 403-row JSONL and all 403 ARM64 SIFs |
| [2. Build the assets yourself](#option-2-build-the-assets-yourself) | Build the Docker image and export a SQSH; this includes compiling TRT-LLM | Generate the JSONL from public SWE-bench data and build the 403 SIFs |

Option 1 requires no image builds or TRT-LLM compilation. Option 2 follows the
source-build delivery pattern of the [Ultra guide](nemotron-3-ultra.md) and is
useful when changing the environment. Both options run on your own cluster and
require no access to the author's cluster, S3, or W&B account. Initial model
selection and the validation limits below apply to both options.

The side branch contains source, dependency pins, build scripts, the recipe,
launchers, and asset manifests. Large images, datasets, model weights, logs,
and checkpoints remain outside Git. The validated training platform is ARM64
GB200/SM100, with four GPUs per node; other platforms are unvalidated.

## Before you start

Use a preparation host with [uv](https://docs.astral.sh/uv/getting-started/installation/)
and storage shared with the compute nodes. Slurm/Pyxis/Enroot and host support
for `/dev/fuse` must be available on the target cluster. The full training recipe
uses 32 four-GPU GB200 nodes; GPU/SWE preflight uses one four-GPU node.

Clone the code and choose a shared asset directory once, then follow either
Option 1 or Option 2. Replace `/lustre/PROJECT/nano35-assets` with your writable
shared directory. The supplied launchers mount `/lustre`; adapt their container
mounts if your cluster uses another storage root.

```bash
git clone --branch nano35-trtllm-rc28-public https://github.com/NVIDIA-NeMo/RL.git
cd RL
git checkout --detach 8de12437669895c83bd969ad6aaf9b03ec502916

export NANO35_ASSET_DIR=/lustre/PROJECT/nano35-assets
mkdir -p "$NANO35_ASSET_DIR"
```

Both options provide the same paths for the launchers:

```text
$NANO35_ASSET_DIR/
  images/nano35.sqsh
  data/swe_verified_403.jsonl
  sifs/swe-bench.eval.arm64.{instance_id}.sif   # one file per task, 403 total
```

Initial model weights are prepared separately after either option.

## Option 1: Download from Hugging Face

The public [asset repository](https://huggingface.co/datasets/shikicloud/nano35-swe-assets/tree/9dc883a50812cf3e5d0d3cd17151df178175681e)
contains the cleaned training SQSH (25.9 GB), the 403-row dataset, all 403 ARM64
SWE SIFs (371.4 GB), and checksum/validation records. Allow about 397.3 GB for
these assets, plus space for the separately downloaded initial model, runtime
caches, training outputs, and checkpoints. Model weights and trained
checkpoints are not included in this asset repository.

```bash
export NANO35_HF_ASSET_REPO=shikicloud/nano35-swe-assets
export NANO35_HF_ASSET_REVISION=9dc883a50812cf3e5d0d3cd17151df178175681e

HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
uvx --from huggingface-hub==0.34.4 hf download "$NANO35_HF_ASSET_REPO" \
  --repo-type dataset --revision "$NANO35_HF_ASSET_REVISION" \
  --local-dir "$NANO35_ASSET_DIR" --max-workers 1
```

The conservative download settings avoid Xet buffering and parallel file
downloads on login nodes with small per-user memory limits. Use the same
revision and local directory when restarting an interrupted download; preserve
the download cache under the asset directory.

Verify the release image, dataset, and all 403 SIFs before running jobs:

```bash
printf '%s  %s\n' \
  197e2009bcad1b1fb2c617a2995ba77391ce726594b0c719c566dd7a1565e36c "$NANO35_ASSET_DIR/images/nano35.sqsh" \
  72e8e2b4a41751ce347a662b99739019a60608d9614f74cd673357d834d52212 "$NANO35_ASSET_DIR/data/swe_verified_403.jsonl" \
  | sha256sum -c -
(cd "$NANO35_ASSET_DIR" && sha256sum -c metadata/sif-sha256sums.txt)
```

This completes Option 1. Continue directly at [Initial model](#initial-model),
then [Preflight and launch](#preflight-and-launch). Skip Option 2 when using
these downloaded assets.

The release image's filename, size, and SHA-256 are recorded in
[`validated-image.json`](../../../../docker/nano35/validated-image.json).
The manifest distinguishes the historical 48-step run from the file-integrity
and runtime checks performed after publication cleanup. The cleaned export
removes builder metadata and uses `nano3.5-e2e` as its neutral W&B project;
training parameters and installed package bytes are preserved. Use the cleaned
export when distributing a prebuilt image.
The older W&B `nano35-rc28-py313-arm64-sqsh:v0` lacks the refit statistics fix
and must not be used. W&B is optional for metrics, not required for asset access.

## Option 2: Build the assets yourself

Complete all three steps below: build the training image, prepare the dataset,
and build the SWE SIFs. Run from the checkout and use the asset directory set in
[Before you start](#before-you-start). This option does not require the prebuilt
HF asset package.

Use a native ARM64 CUDA build host with Docker/buildx, network access to the
pinned public dependencies, and sufficient build scratch. SWE SIF construction
also requires a working Docker daemon and Apptainer 1.5.0. Allow additional
space for build layers and temporary archives.

### Build the training image

```bash
git submodule update --init --recursive

docker buildx build --platform linux/arm64 --progress=plain --load \
  -f docker/nano35/Dockerfile -t nano35-swe:local .

mkdir -p "$NANO35_ASSET_DIR/images"
enroot import -o "$NANO35_ASSET_DIR/images/nano35.sqsh" \
  dockerd://nano35-swe:local
sha256sum "$NANO35_ASSET_DIR/images/nano35.sqsh" \
  > "$NANO35_ASSET_DIR/images/nano35.sqsh.sha256"
```

This source build compiles the pinned, patched TRT-LLM wheel and prebuilds
OpenHands and all required actor environments. Compilation happens during image
preparation; jobs reuse the resulting image. Kernel JIT and graph warmup may
still occur during job startup. Ultra uses vLLM and skips TRT-LLM compilation;
the delivery pattern is shared, but the inference environment differs.

Build in a normal clone, with initialized recursive submodules. If the build
host and cluster differ, push to your own container registry and use
`enroot import ... docker://REGISTRY/IMAGE:TAG` on the cluster, or transfer the
SQSH with its checksum. No particular registry provider is required.

The final build step removes builder accounts, home directories, Git reflogs,
authentication configuration, internal build records, cached HF modules, tool
caches, and CUDA compatibility markers containing build-host names.
Git commits, submodule state, installed environments, and the dependency
fingerprint remain available. Docker mounts the source context read-only, then
copies, builds, and cleans it within one layer so removed metadata cannot be
recovered from an earlier source-copy layer. Inspect the resulting artifact
before sharing it.

A rebuild produces a new image checksum. Run the GPU/SWE and data preflights
below for that exact image; the launcher does not require the historical
release checksum. See [build notes](#build-notes) for the pinned patches.

### Prepare the 403 SWE tasks

The workload remains the recorded SWE-bench Verified subset. It does not switch
to Ultra's SWE-Gym / SWE-rebench blend.
[`prepare_swe_data.py`](../../../../tools/nano35/prepare_swe_data.py) downloads
a fixed revision of the public
[SWE-bench Verified dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified),
selects the checked-in ordered
[403 instance IDs](../../../../tools/nano35/swe_verified_403.json), and restores
the Gym request fields:

```bash
uv run --no-project --no-config --python 3.12 --script tools/nano35/prepare_swe_data.py \
  --output-dir "$NANO35_ASSET_DIR/data"
```

The resulting `data/swe_verified_403.jsonl` must have SHA-256
`72e8e2b4a41751ce347a662b99739019a60608d9614f74cd673357d834d52212`,
matching the completed training run byte for byte. The script also writes the
raw source rows for environment building and a data manifest. It rejects a
different source checksum, missing IDs, or conflicting existing output.
An existing pinned parquet can be supplied with `--parquet` for offline use.

### Build the SWE SIFs

On a native ARM64 host with a working Docker daemon and Apptainer 1.5.0, run:

```bash
uv run --no-project --no-config --python 3.12 --script tools/nano35/build_swe_sifs.py \
  --data "$NANO35_ASSET_DIR/data/swe_verified_403.raw.jsonl" \
  --sif-dir "$NANO35_ASSET_DIR/sifs" \
  --work-dir /LOCAL_SCRATCH/nano35-swe-build --workers 4
```

The builder uses the same pinned public SWE-bench harness as the training
image. It builds native ARM64 Docker images, applies the gold patch in a
temporary test container, checks the required tests, and converts the original
unpatched image to `swe-bench.eval.arm64.{instance_id}.sif`. Docker archives are
temporary; build logs and per-SIF receipts record the results. Completed,
checksum-matching SIFs are reused on the next invocation.

The pinned harness normally falls back to x86 for ten selected instances.
This entrypoint explicitly requests ARM64 for every instance and requires gold
tests to pass before export. Any build or test failure makes the command fail
with `sif-build-report.json`; it does not silently remove questions. Allow space
for Docker build layers and archives as well as the SIF set (the historical
403-image set occupies about 371 GB).

**Validation boundary:** the new public SIF build entrypoint has not been run
through all 403 builds. The completed training run used previously prepared
ARM64 SIFs. Rebuilding from source can expose upstream package/build failures;
build receipts and the GPU/SWE preflight are required before using a rebuilt
set. New SIFs are not claimed to be byte-identical to the historical set.

This completes Option 2. Continue with the common model preparation and
preflight steps below, using the assets you just built.

## Initial model

This section and the remaining launch instructions apply to both setup options.

Supply a compatible Nano 3.5 BF16 HF checkpoint, with all indexed safetensors
shards, through `NANO35_MODEL`. For an external starting point, the public
[Nemotron 3.5 Lightning BF16 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16)
can be downloaded at a fixed revision:

```bash
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
uvx --from huggingface-hub==0.34.4 hf download \
  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 \
  --revision a9904d24bcc1d289a1950fa9d2b978c47cf903b9 \
  --local-dir "$NANO35_ASSET_DIR/models/nano35-initial-hf" --max-workers 1
```

That public checkpoint has not been verified to be identical to the initial
checkpoint of the completed run, nor validated by that run. Treat it as a new
initialization: complete preflight and start fresh. Reproducing the historical
training results exactly also requires the original initial weights and SWE
environments. The launcher checks model shards and every referenced SIF before
submission; changing the dataset requires a new recipe and data preflight.

## Environment

| Component | Pin |
| --- | --- |
| NeMo-RL base | `2a70a4a155165058fea864877128a511fc475cdd` plus this branch |
| TRT-LLM base | `fca831eac3da515461e148b05e0663c29ab6aaa5` plus `tools/trtllm-nano35.patch` |
| TRT-LLM version | `1.3.0rc28+nano35.1` |
| Driver and Ray worker Python | 3.13.14 |
| OpenHands Python | Separate 3.12 environment |
| Torch / CUDA compiler | 2.13.0+cu130 / 13.2 |

The image prebuilds separate driver, training, inference, collector, Gym, and
OpenHands environments. SWE harness pins live in
[`swe_harness_pins.json`](../../../../docker/nano35/swe_harness_pins.json);
installed package manifests and source checksums are under
`/opt/nano35-metadata` inside the image.

The image includes the Gym/TRT-LLM integration, conversation-ID forwarding,
and asynchronous weight-refit fixes required by this recipe. Resume requires
checkpoints with the current asynchronous data state; legacy checkpoints are
not supported by these launchers.

## Workload and topology

| Setting | Validated recipe |
| --- | --- |
| Allocation | 32 nodes, four GPUs each; `--segment=16` |
| Training | 16 nodes / 64 GPUs; TP4, CP16, PP1, expert TP1, EP32 |
| Inference | 16 nodes / 16 replicas; TP4, MoE TP4, MoE EP1 (TEP4), AsyncLLM |
| Dataset / route | 403 SWE-bench Verified rows / `swe_agents_train` |
| Prompts / generations / batch | 32 / 16 / 512 |
| Gym trajectory concurrency / Response API workers | 1024 / 16 |
| Max agent turns | 200 |
| Context / generation limits | 196608 / 196608 tokens |
| Precision / temperature / top-p | BF16 / 1.0 / 1.0 |
| Async GRPO | Maximum rollout age 1; in-flight weight refit enabled |
| TRT scheduling | 32768 batched tokens, batch cap 64, `MAX_UTILIZATION` |
| GPU memory fraction | 0.7 |
| Cache | Per-conversation retention for three turns; prompt-end Mamba snapshot; FP32 SSM; 64-token blocks |
| Graph / communication | MNNVL and breakable prefill graphs |
| Fastokens / router replay | Disabled |
| Training target | Learning rate 3e-6; four epochs, 48 steps |
| Checkpoints | Optimizer and async data state; every five steps and at the predictive job boundary |
| Metrics | TensorBoard, with an optional live host exporter to W&B `nano3.5-e2e` |

The exact recipe is
[`grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml`](../../../../examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml).
The launcher freezes the recipe in `<run>/recipe.yaml` and mounts it read-only
over the image's recipe. Resume reuses those same bytes. Native W&B logging is
disabled; the optional host exporter selects its own destination as shown below.

A 64-token block is an allocation unit for attention KV cache, not a trajectory
count or Mamba state size. Gym concurrency covers whole agent trajectories,
including tools and evaluation; it does not mean 1024 requests decode at once.

Each step uses 32 question groups with 16 attempts per group. With ordered data
and `drop_last=True`, an epoch uses the first 384 of the 403 rows and drops the
last 19. Four epochs therefore give 48 steps. Different steps usually contain
different questions. The first two steps use the initial generation weights
because collection overlaps training with maximum age one.

`train/swe_agents_train/reward/mean` is the mean binary reward over 512 attempts.
For this async collector, `reward/median` is the average of each question's
16-attempt median; it is not the median of all 512 attempts. Equal step means
can occur naturally. This workload does not enable periodic or final fixed-set
validation within 48 steps, so these training rewards do not establish accuracy
improvement over the initial model.

## Preflight and launch

Run from the repository root. Compute nodes must expose `/dev/fuse` to the outer
container for Apptainer. The supplied launchers mount it along with `/lustre`.
They use the validated Lyris Slurm topology; a different site's segment/account
options need checking before use.

Set site-specific absolute paths and scheduler settings:

```bash
export NANO35_IMAGE="$NANO35_ASSET_DIR/images/nano35.sqsh"
export NANO35_MODEL="$NANO35_ASSET_DIR/models/nano35-initial-hf"
export NANO35_DATA="$NANO35_ASSET_DIR/data/swe_verified_403.jsonl"
export NANO35_SIF_TEMPLATE="$NANO35_ASSET_DIR/sifs/swe-bench.eval.arm64.{instance_id}.sif"
export NANO35_ACCOUNT=YOUR_ACCOUNT
export NANO35_PARTITION=YOUR_GB200_PARTITION
export NANO35_VALIDATION_ROOT=/lustre/PROJECT/validation
mkdir -p "$NANO35_VALIDATION_ROOT"
sbatch --account="$NANO35_ACCOUNT" --partition="$NANO35_PARTITION" \
  --job-name="${NANO35_ACCOUNT}-nano35.preflight" \
  --output="$NANO35_VALIDATION_ROOT/slurm-%j.out" tools/nano35/preflight.sbatch
```

Submit only after the preceding Nano job has ended. GPU preflight uses one
four-GPU node to check TP4 generation, effective MNNVL/breakable graphs,
conversation reuse, cache reset, and one real SWE agent/evaluation trajectory.

After successful completion, run the separate data check:

```bash
export NANO35_PREFLIGHT_REPORT="$NANO35_VALIDATION_ROOT/GPU_JOB_ID/result/gpu-preflight.json"
sbatch --account="$NANO35_ACCOUNT" --partition="$NANO35_PARTITION" \
  --job-name="${NANO35_ACCOUNT}-nano35.data-check" \
  --output="$NANO35_VALIDATION_ROOT/data-%j.out" tools/nano35/data_preflight.sbatch
```

The data check reads all 403 rows in each split through the image's tokenizer,
Gym processor, collator, and worker-based `StatefulDataLoader`. It checks payload
and order without generating or training. Once it passes:

```bash
export NANO35_DATA_PREFLIGHT_REPORT="$NANO35_VALIDATION_ROOT/DATA_JOB_ID/data-preflight.json"
export NANO35_RUN_DIR=/lustre/PROJECT/runs/nano35-fresh
bash tools/nano35/launch_e2e.sh fresh "$NANO35_RUN_DIR"
```

Training requires reports matching the exact image, dataset and recipe hashes.
Fresh mode starts from the initial model with a new optimizer and data cursor.
The launcher submits one five-hour Slurm job. The recipe uses a **3h30 internal
budget**: if elapsed time plus the average step time would exceed it, training
saves and exits early. A job can therefore finish in roughly three hours, with
a varying number of steps. The Slurm limit includes startup and save headroom.

## Serial continuation and live metrics

For automatic continuation, keep the launch environment above in a persistent
login session, such as tmux, and run:

```bash
uv run --no-project --no-config tools/nano35/continue_training.py "$NANO35_RUN_DIR" \
  >> "$NANO35_RUN_DIR/monitor.log" 2>&1
```

The monitor checks immediately and then every two hours. It submits exactly one
continuation after Slurm reports `COMPLETED`, the driver reports completion,
and checkpoint metadata passes. It never prequeues dependent jobs. It stops on
a failed attempt or when 48 steps are complete. `touch "$NANO35_RUN_DIR/monitor.stop"`
prevents future submissions without cancelling an already running job. Remove
that file only when intentionally restarting the monitor. To inspect once, add
`--once`; a successful check can submit a continuation.

Manual continuation uses the same guarded path:

```bash
bash tools/nano35/launch_e2e.sh resume "$NANO35_RUN_DIR"
```

Resume checks that the image, data and configuration match the original run,
and verifies the optimizer shards, asynchronous data progress and checkpoint
files before submitting the next job.

In a separate persistent session, authenticate and start live metrics:

```bash
uvx --from wandb==0.30.0 wandb login
uv run --no-project --no-config --with wandb==0.30.0 --with tensorboard==2.20.0 \
  tools/nano35/stream_wandb_results.py "$NANO35_RUN_DIR" \
  --entity YOUR_WANDB_TEAM --project nano3.5-e2e
```

The exporter reads TensorBoard every 60 seconds and creates one W&B run per
Slurm job, grouped by experiment directory. Training tags retain their original
names and `global_step`; `ray/*` uses the separate `ray_step` time axis. It
preserves repeated values and late-arriving metrics. It uploads scalars and run
metadata, not checkpoints, dataset rows or the container image. Local cursors
support ordinary restarts; they are not a transactional delivery guarantee if
the exporter crashes during upload.

Use `--dry-run` to inspect local events without contacting W&B or writing state.
Only one exporter may use a run directory; its saved state is tied to one
entity/project. Do not replace an existing experiment's exporter with this
version mid-run. `touch "$NANO35_RUN_DIR/wandb-sync/stop"` stops the exporter
without stopping training. Training itself continues to log locally if the
exporter is offline.

## Validation scope

The pinned image and recipe completed 48 training steps, including policy
updates, asynchronous weight refit, checkpoint saves and seven cross-job
restores followed by training. The final checkpoint includes optimizer and
asynchronous data state; it was saved and inspected, but was not restored in
a subsequent job. No fixed-set accuracy evaluation was performed.

Known limitation: TRT/Gloo can emit peer-disconnect errors during worker
shutdown. In the validated run, these occurred after the generation-worker
shutdown marker; all eight training jobs exited successfully and saved
checkpoints passed verification. Confirm job completion and checkpoint status
before treating a disconnect as a shutdown-only message.

## Build notes

The runtime contains the TRT-LLM patch
[`tools/trtllm-nano35.patch`](../../../../tools/trtllm-nano35.patch) and the OpenHands
patch [`docker/nano35/openhands.patch`](../../../../docker/nano35/openhands.patch).
The build creates a CPython 3.13 TRT-LLM wheel once. Build-tool dependencies use an
independent directory so they do not alter the locked actor environment or
hold its uv lock. Rebuilding creates a new image and checksum; validate that
image separately; the historical manifest is evidence for the original image,
not an expected checksum for every rebuild.
The build installs uv 0.11.28. Use that version for local lock checks too;
older uv versions may reject the scoped dependency overrides in `pyproject.toml`.

For the local Enroot workflow, `tools/nano35/export_oci.sh` converts a completed
SQSH to a local OCI archive using the image's umoci and Skopeo. It performs no
publication. NGC can be used later; this delivery does not require an NGC upload.
