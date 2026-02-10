# Reproduction Report: NeMo-RL v0.5.0 + vLLM Speculative Decoding with Draft Models

**Date:** 2026-02-10
**Author:** Shaunak J
**Host:** viking-prod-228 (8x NVIDIA H20, computelab-sc-01 cluster)

## Overview

This report documents how to build a custom NeMo-RL Docker image that includes vLLM speculative decoding with draft models (vLLM PR #24322), and how to run GRPO training with it.

## Source Components

| Component | Source | Commit / Version | How Obtained |
|-----------|--------|-------------------|--------------|
| NeMo-RL | `https://github.com/NVIDIA-NeMo/RL.git` | `946862e7` (HEAD of `main` on 2026-02-10) | Already cloned locally |
| vLLM (custom) | `https://github.com/vllm-project/vllm.git` | `4a5299c93ff97c26def537b92562df5ada530fea` | Merge commit of PR #24322 |
| Precompiled vLLM wheel | `https://wheels.vllm.ai` | Same commit as above | Auto-downloaded during build |
| Base Docker image | `nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04` | Pulled by Dockerfile | From NGC registry |

## Commit Hash Provenance

### `946862e7` — NeMo-RL

The latest commit on the NeMo-RL `main` branch at the time of this build:

```
946862e7 docs: add release runs to front page readme for 0.5 (#1879)
```

This is the same codebase as the `nvcr.io/nvidia/nemo-rl:v0.5.0` release.

### `4a5299c93ff97c26def537b92562df5ada530fea` — vLLM

The merge commit of [vllm-project/vllm#24322](https://github.com/vllm-project/vllm/pull/24322) ("feat: spec decode with draft models"), merged on 2026-01-19. Retrieved via:

```bash
gh pr view 24322 --repo vllm-project/vllm --json mergeCommit
# => 4a5299c93ff97c26def537b92562df5ada530fea
```

The precompiled wheel for this commit exists at:
```
https://wheels.vllm.ai/4a5299c93ff97c26def537b92562df5ada530fea/vllm-0.14.0rc2.dev156%2Bg4a5299c93-cp38-abi3-manylinux_2_31_x86_64.whl
```

This was verified by HTTP HEAD request returning 200.

### Replaced defaults in `tools/build-custom-vllm.sh`

The build script had two hardcoded defaults that were written by the NeMo-RL team for their v0.10-era vLLM pin:

- `GIT_REF` default: `cc99baf14dacc2497d0c5ed84e076ef2c37f6a4d` — the previous vLLM commit pin
- `VLLM_WHEEL_COMMIT` default: `862f2ef893d9751db0a92bd2d4ae0e3d9677872f` — chosen as `git merge-base --fork-point origin/main tags/v0.10.0`

Both were replaced with the PR #24322 merge commit. The wheel URL template was also updated because vLLM changed their naming convention from `vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl` to `vllm-<version>-cp38-abi3-manylinux_2_31_x86_64.whl`.

## Local Modifications

### 1. `tools/build-custom-vllm.sh`

**Line 24** — Changed default `GIT_REF`:
```diff
-GIT_REF=${2:-cc99baf14dacc2497d0c5ed84e076ef2c37f6a4d}
+GIT_REF=${2:-4a5299c93ff97c26def537b92562df5ada530fea}
```

**Lines 27-28** — Changed default wheel commit and URL pattern, made env var overridable:
```diff
-VLLM_WHEEL_COMMIT=${3:-862f2ef893d9751db0a92bd2d4ae0e3d9677872f}  # use full commit hash from the main branch
-export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${VLLM_WHEEL_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"
+VLLM_WHEEL_COMMIT=${3:-4a5299c93ff97c26def537b92562df5ada530fea}  # merge commit of vllm PR #24322 (spec decode with draft models)
+export VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://wheels.vllm.ai/${VLLM_WHEEL_COMMIT}/vllm-0.14.0rc2.dev156%2Bg4a5299c93-cp38-abi3-manylinux_2_31_x86_64.whl}"
```

### 2. `docker/Dockerfile`

**Line 117** — Commented out sglang extra (broken upstream dependency):
```diff
-uv sync --link-mode symlink --locked --extra sglang --no-install-project
+# Skipped: sgl-kernel build broken (DeepGEMM ref 54f99a8 missing upstream)
+# uv sync --link-mode symlink --locked --extra sglang --no-install-project
```

**Reason:** The `sgl-kernel` build fails because it tries to fetch DeepGEMM at commit `54f99a8af537b3c6eb4819b69907ccbe2b600792` which no longer exists in the upstream repo. This only affects the sglang inference backend, not vLLM.

## Step-by-Step Reproduction

### Prerequisites

- Docker with buildx support
- GPU node with NVIDIA drivers
- ~100 GB disk space for the build
- Internet access (to pull base image, clone vLLM, download wheel, download HF models)

### 1. Clone and prepare the source

```bash
git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL
git checkout 946862e7

# Initialize submodules (required for Megatron-LM, Automodel, etc.)
git submodule update --init --depth 1
```

### 2. Apply patches to build-custom-vllm.sh

```bash
# Update GIT_REF default to PR #24322 merge commit
sed -i 's|GIT_REF=${2:-cc99baf14dacc2497d0c5ed84e076ef2c37f6a4d}|GIT_REF=${2:-4a5299c93ff97c26def537b92562df5ada530fea}|' \
  tools/build-custom-vllm.sh

# Update VLLM_WHEEL_COMMIT default
sed -i 's|VLLM_WHEEL_COMMIT=${3:-862f2ef893d9751db0a92bd2d4ae0e3d9677872f}.*|VLLM_WHEEL_COMMIT=${3:-4a5299c93ff97c26def537b92562df5ada530fea}  # merge commit of vllm PR #24322|' \
  tools/build-custom-vllm.sh

# Update wheel URL pattern and make it overridable
sed -i 's|^export VLLM_PRECOMPILED_WHEEL_LOCATION="https://wheels.vllm.ai/${VLLM_WHEEL_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"|export VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://wheels.vllm.ai/${VLLM_WHEEL_COMMIT}/vllm-0.14.0rc2.dev156%2Bg4a5299c93-cp38-abi3-manylinux_2_31_x86_64.whl}"|' \
  tools/build-custom-vllm.sh
```

### 3. Apply patch to Dockerfile

```bash
sed -i 's|^uv sync --link-mode symlink --locked --extra sglang --no-install-project$|# Skipped: sgl-kernel build broken (DeepGEMM ref missing upstream)\n# uv sync --link-mode symlink --locked --extra sglang --no-install-project|' \
  docker/Dockerfile
```

### 4. Build the Docker image

```bash
docker buildx build \
  --build-arg BUILD_CUSTOM_VLLM=1 \
  --target release \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --tag nemo-rl:v0.5.0-spec-decode \
  .
```

This takes approximately 20-30 minutes. The build will:
1. Pull the CUDA base image
2. Install system dependencies, uv, Python 3.12
3. Clone vLLM at commit `4a5299c` and build it with the precompiled wheel
4. Sync all dependency extras (vllm, mcore, automodel) except sglang
5. Prefetch Ray worker virtual environments
6. Generate container fingerprint

### 5. Save the image (optional)

```bash
docker save nemo-rl:v0.5.0-spec-decode | gzip > nemo-rl-v0.5.0-spec-decode.tar.gz
```

The resulting image is approximately 21 GB compressed.

To load on another machine:
```bash
docker load < nemo-rl-v0.5.0-spec-decode.tar.gz
```

### 6. Run the container

```bash
docker run --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /path/to/your/scratch:/path/to/your/scratch \
  -it nemo-rl:v0.5.0-spec-decode bash
```

### 7. Create the training config

Save as `grpo-qwen3-4b-spec-decode-1n8g.yaml`:

```yaml
defaults: ../../grpo_math_1B.yaml
grpo:
  max_num_steps: 200
  num_prompts_per_step: 32
  num_generations_per_prompt: 16
checkpointing:
  checkpoint_dir: results/grpo-qwen3-4b-spec-decode-1n8g
policy:
  model_name: Qwen/Qwen3-4B
  tokenizer:
    name: Qwen/Qwen3-4B
  train_global_batch_size: 512
  train_micro_batch_size: 1
  logprob_batch_size: 1
  max_total_sequence_length: 2048
  dynamic_batching:
    enabled: true
  sequence_packing:
    enabled: false
  make_sequence_length_divisible_by: 1
  generation:
    max_new_tokens: 1024
    vllm_cfg:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.85
      max_model_len: 2048
    vllm_kwargs:
      speculative_config:
        method: draft_model
        model: Qwen/Qwen3-0.6B
        num_speculative_tokens: 5
        draft_tensor_parallel_size: 1
data:
  max_input_seq_length: 1024
logger:
  log_dir: logs/grpo-qwen3-4b-spec-decode-1n8g
  wandb_enabled: false
  tensorboard_enabled: true
cluster:
  gpus_per_node: 8
```

**Model pairing note:** The target and draft models must have the same vocabulary size. Qwen3 models all use vocab_size=151936 across all sizes, so any Qwen3 pair works. Qwen2.5 models have inconsistent vocab sizes (7B+ use 152064, smaller models use 151936), which prevents cross-size draft pairing.

### 8. Run GRPO training

```bash
python examples/run_grpo.py \
  --config /path/to/grpo-qwen3-4b-spec-decode-1n8g.yaml \
  ++logger.log_dir=/path/to/your/scratch/logs/grpo-qwen3-4b-spec-decode \
  ++checkpointing.checkpoint_dir=/path/to/your/scratch/results/grpo-qwen3-4b-spec-decode
```

**Important:** Use `python` (not `uv run`) inside the container, as custom vLLM containers use frozen environments. See `docs/guides/use-custom-vllm.md` for details.

## Verified Results

Training was verified running on viking-prod-228 (8x H20 GPUs). Key metrics from the first 3 steps:

| Metric | Step 1 | Step 2 | Step 3 |
|--------|--------|--------|--------|
| Loss | -0.035 | -0.014 | -0.016 |
| Avg Reward | 0.340 | 0.272 | 0.229 |
| Mean Gen Length | 1008 | 1015 | 1001 |
| E2E Tokens/sec (total) | 2488 | 2564 | 2555 |
| Step Time | 228s | 224s | 224s |
| Generation % of step | 51.5% | 46.2% | 46.2% |

vLLM confirmed speculative decoding active:
```
Initializing a V1 LLM engine (v0.14.0rc2.dev156+g4a5299c93.d20260210)
  speculative_config=SpeculativeConfig(method='draft_model', model='Qwen/Qwen3-0.6B', num_spec_tokens=5)
Loading drafter model...
Starting to load draft model Qwen/Qwen3-0.6B. TP=1, rank=0
Model loading took 8.67 GiB memory
```

## Known Issues

1. **sglang extra is disabled** — `sgl-kernel` cannot build due to a missing DeepGEMM commit upstream. This does not affect vLLM-based inference or training.
2. **Async scheduling disabled with draft model spec decode** — vLLM logs: "Async scheduling not supported with draft_model-based speculative decoding and will be disabled." This is expected behavior from the vLLM implementation.
3. **`min_p`, `logit_bias`, `min_tokens` unsupported with spec decode** — vLLM warning, not an issue for standard GRPO training.

## File Locations (on viking-prod-228)

- Source: `/home/scratch.shaunakj_other/Development/RL/`
- Saved image: `/home/scratch.shaunakj_other/Development/nemo-rl-v0.5.0-spec-decode.tar.gz`
- Training logs: `/tmp/logs/grpo-qwen3-4b-spec-decode/exp_001/`
- Training config: `/home/scratch.shaunakj_other/Development/RL/examples/configs/recipes/llm/grpo-qwen2.5-7b-spec-decode-1n8g.yaml`
