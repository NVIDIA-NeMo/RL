# Build Docker Images

This guide explains how to build the NeMo RL Docker image.

The **release** image is our recommended option as it provides the most complete environment. It includes the base image with pre-fetched NeMo RL python packages in the `uv` cache, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers. This is the ideal choice for production deployments.

## Building the Release Image

```sh
# Self-contained build (default: builds from main):
docker buildx build -f docker/Dockerfile \
    --tag <registry>/nemo-rl:latest \
    --push .

# Self-contained build (specific git ref):
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=r0.6.0 \
    --tag <registry>/nemo-rl:r0.6.0 \
    --push .

# Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL):
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=r0.6.0 \
    --tag <registry>/nemo-rl:r0.6.0 \
    --push https://github.com/NVIDIA-NeMo/RL.git

# Local NeMo RL source override:
docker buildx build --build-context nemo-rl=. -f docker/Dockerfile \
    --tag <registry>/nemo-rl:latest \
    --push .
```

> [!NOTE]
> The `--tag <registry>/nemo-rl:latest --push` flags are not necessary if you just want to build locally.

## uv Cache Pruning

The release image ships `uv`'s download cache (`/opt/nemo_rl_cache/uv`) so an unprivileged runtime user can later run `uv run --extra <backend>` fully offline. Because the shared build cache this is seeded from persists across many builds, routine dependency-version bumps (torch, vLLM, SGLang, ...) leave each predecessor version's wheel behind -- `uv` only ever adds to its cache, it never removes a version a prior `uv.lock` pinned and the current one doesn't.

Right after the seed is copied in, before any `uv sync`, `docker/prune_uv_cache.py` removes exactly those superseded entries. Running first means it only judges what the seed carried in, so the build-isolation dependencies the syncs resolve outside `uv.lock` (setuptools, a build-time torch, ...) are kept. It runs in the same layer the syncs fill, because deleting files in a later layer only hides them and the image still ships them. It removes index wheels and source builds whose `(package, version)` the current `uv.lock` no longer pins, and git builds/checkouts for commits it no longer pins, across every extra (installed in this build or not). It never removes a wheel the offline-install promise depends on, since that promise is itself "every wheel `uv.lock` could need." This runs automatically on every build; there is no flag to disable it.

## Skipping vLLM or SGLang Dependencies

If you don't need vLLM, SGLang, or TRT-LLM support, you can skip building those dependencies to reduce build time and image size. Use the `SKIP_VLLM_BUILD`, `SKIP_SGLANG_BUILD`, and/or `SKIP_TRTLLM_BUILD` build arguments:

```sh
# Skip vLLM dependencies:
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_VLLM_BUILD=1 \
    --tag <registry>/nemo-rl:latest \
    .

# Skip SGLang dependencies:
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_SGLANG_BUILD=1 \
    --tag <registry>/nemo-rl:latest \
    .

# Skip TRT-LLM dependencies:
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_TRTLLM_BUILD=1 \
    --tag <registry>/nemo-rl:latest \
    .

# Skip all three:
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_VLLM_BUILD=1 \
    --build-arg SKIP_SGLANG_BUILD=1 \
    --build-arg SKIP_TRTLLM_BUILD=1 \
    --tag <registry>/nemo-rl:latest \
    .
```

When these build arguments are set, the corresponding `uv sync --extra` commands are skipped, and the virtual environment prefetching will exclude actors that depend on those packages.

> [!NOTE]
> If you skip vLLM, SGLang, or TRT-LLM during the build but later try to use those backends at runtime, the dependencies will be fetched and built on-demand. This may add significant setup time on first use.

## Custom Setup Commands

By default, the Docker image installs [apptainer](https://apptainer.org/) (with a `singularity` symlink) via a pluggable `custom-setup` build stage. The default script is `docker/install_apptainer.sh`. You can override or skip this step at build time.

### Override with a custom script

Create a directory containing your setup script(s), then pass it as a build context along with the script filename:

```sh
# my-setup-dir/my_script.sh
#!/bin/bash
set -euo pipefail
apt-get update && apt-get install -y my-custom-package
apt-get clean && rm -rf /var/lib/apt/lists/*
```

```sh
docker buildx build \
  --build-context custom-setup=my-setup-dir/ \
  --build-arg CUSTOM_SETUP_FNAME=my_script.sh \
  -f docker/Dockerfile --tag <registry>/nemo-rl:latest .
```

### Skip custom setup entirely

To build without any custom setup commands, set `CUSTOM_SETUP_FNAME` to empty:

```sh
docker buildx build --build-arg CUSTOM_SETUP_FNAME= -f docker/Dockerfile --tag <registry>/nemo-rl:latest .
```
