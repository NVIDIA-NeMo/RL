# MLPerf TRT-LLM Bring-up

Qwen3.5-35B-A3B is used only as a smaller proxy for the Rubin bring up. It shares the same architecture as the MLPerf target model - Qwen3.5 397B.

## Pre-requisite
- All training data/SIFs have been moved to Rubin cluster.


## Branches

- **NeMo-RL:** [`NVIDIA-NeMo/RL:opt/dev-backup`](https://github.com/NVIDIA-NeMo/RL/tree/opt/dev-backup).
  — It layers TRTLLM SWE MR [#3130](https://github.com/NVIDIA-NeMo/RL/pull/3130) on Michal's MLPerf fork. Note that we'll formalize this branch into Nemo-RL repo side branch once branch base is aligned, which shouldn't be a blocker for Rubin bring up at this stage.
- **TensorRT-LLM:** `1.3.0rc21` plus [PR #16642](https://github.com/NVIDIA/TensorRT-LLM/pull/16642) for Qwen3.5 refit.
- **[Optimized repo](https://gitlab-master.nvidia.com/dl/mlperf/optimized):** `main` plus these tentative 35B scripts in [fork](https://gitlab-master.nvidia.com/erinh/optimized/-/tree/erinh/mlperf-trtllm-bringup-scripts?ref_type=heads):
  - [optimized/qwen35_397b_grpo/pytorch/config_GB200_4x4_t2g2_tp2pp1ep4gtp4_trtllm.sh](https://gitlab-master.nvidia.com/erinh/optimized/-/blob/erinh/mlperf-trtllm-bringup-scripts/qwen35_397b_grpo/pytorch/config_GB200_4x4_t2g2_tp2pp1ep4gtp4_trtllm.sh?ref_type=heads)
  - [optimized/qwen35_397b_grpo/pytorch/conf/grpo_qwen35_35b_a3b_swe_openhands_async_trtllm.yaml](https://gitlab-master.nvidia.com/erinh/optimized/-/blob/erinh/mlperf-trtllm-bringup-scripts/qwen35_397b_grpo/pytorch/conf/grpo_qwen35_35b_a3b_swe_openhands_async_trtllm.yaml?ref_type=heads)

Update the W&B project/name and all user-specific model, data, log, and
checkpoint paths before running.

## Environment

Please refer to this [Dockerfile](docker/mlperf/Dockerfile) to build an image.
In [tools/build-custom-trtllm.sh L105](tools/build-custom-trtllm.sh#L105) the Dockerfile used, change it to include Rubin. Since we've never built Nemo-RL + TRTLLM on Rubin, please review these build scripts and make HW-specific changes as needed.

To build the image, clone the [optimized repo](https://gitlab-master.nvidia.com/dl/mlperf/optimized)
locally and check out commit `95891c03b`. The build uses `docker/mlperf/Dockerfile`
from this repo, but the build context must be the optimized recipe dir
`<optimized_repo>/qwen35_397b_grpo/pytorch` — that is where the Dockerfile's
`COPY`s pull `patches/`, `conf/`, `run_and_time.sh`, `requirements-mlperf.txt`,
etc. from.

Run from this repo's root (`<optimized_repo>` is the path to your local
`optimized` checkout):

```bash
docker buildx build \
    --pull \
    -f docker/mlperf/Dockerfile \
    --target mlperf \
    --build-arg GITLAB_CLONE_ACCESS_TOKEN="${GITLAB_CLONE_ACCESS_TOKEN}" \
    --tag "${GITLAB_REGISTRY}:${IMAGE_NAME}" \
    --push \
    <optimized_repo>/qwen35_397b_grpo/pytorch
```

The baked-in TRT-LLM is built from the source pinned by these two build args
(defaults set in `docker/mlperf/Dockerfile`):

```dockerfile
ARG BUILD_CUSTOM_TRTLLM_URL=https://github.com/hchings/TensorRT-LLM.git
ARG BUILD_CUSTOM_TRTLLM_REF=983e7ff57dd26ce662e1b34cce94b76c9181be05
```

- `BUILD_CUSTOM_TRTLLM_URL` — the TRT-LLM git remote to build from.
- `BUILD_CUSTOM_TRTLLM_REF` — the commit / branch / tag to check out.

Override either with `--build-arg` (e.g.
`--build-arg BUILD_CUSTOM_TRTLLM_REF=<sha>`) to bake a different TRT-LLM into
the image.

The image already bakes in a prebuilt TRT-LLM. If you instead want to run with
an **editable** TRT-LLM (e.g. to iterate on local TRT-LLM changes) rather than
the baked-in one, first build its C++ components and package them in a separate
job — the build mutates the driver environment, so it must not run in the
training job:

```bash
TRTLLM_SRC=/path/to/TensorRT-LLM bash tools/build-editable-trtllm.sh
```

Then set:

```bash
export NRL_TRTLLM_EDITABLE=/path/to/TensorRT-LLM
```

This requires the NeMo-RL editable-TRT-LLM support change and rebuilds the
TRT-LLM virtual environment from the prebuilt editable package.


Image for Blackwell on Lyris cluster:
- `master.nvidia.com:5005/shuyix/docker-images:nemo-rl-trtllm-20260722-aarch64-mlperf-cudnnfix`

The `-cudnnfix` suffix above is a thin overlay
([docker/mlperf/Dockerfile.cudnn-fix](docker/mlperf/Dockerfile.cudnn-fix)) that
bakes in cuDNN sublibrary symlinks. You only need it if you hit a cuDNN failure
at runtime (e.g. `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` / `_VERSION_MISMATCH`);
overlay your image with:

```bash
docker buildx build \
    --pull \
    --build-arg BASE_IMAGE="${GITLAB_REGISTRY}:${IMAGE_NAME}" \
    --tag "${GITLAB_REGISTRY}:${IMAGE_NAME}-cudnnfix" \
    --push \
    - < docker/mlperf/Dockerfile.cudnn-fix
```


## Run scripts

Please ensure all file paths used in [run_qwen35_35b_trtllm_4n.sh](run_qwen35_35b_trtllm_4n.sh) match your cluster's paths. 

This script is only tested on 4 nodes X GB200/GB300. For Rubin, it might only need 2 nodes or less. 
Please adjust the parallel config in it correspondingly.

```
bash run_qwen35_35b_trtllm_4n.sh
```

## Repro validation & reference Wandbs
- Look out for any file not found / apptainer / Gym / r2egym errors in your output log. If you see any of this it means your image or uv has issues.
- You should see `reward/mean` at step 0 to be around `0.1` and trending upward as the training goes.
- Example runs for TRT-LLM TP4/TP8 and vLLM on Blackwell are below. See runs with `tp4-gb200-0722-erinh` in [wandb](https://wandb.ai/nvidia/grpo-dev-erinh/workspace?nw=nwusererinh).

---

Contact @Erin Ho @Shuyi Xiong @Chunwei Yan for questions/issues.

