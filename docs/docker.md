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
    --build-arg NRL_GIT_REF=r0.3.0 \
    --tag <registry>/nemo-rl:r0.3.0 \
    --push .

# Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL):
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=r0.3.0 \
    --tag <registry>/nemo-rl:r0.3.0 \
    --push https://github.com/NVIDIA-NeMo/RL.git

# Local NeMo RL source override:
docker buildx build --build-context nemo-rl=. -f docker/Dockerfile \
    --tag <registry>/nemo-rl:latest \
    --push .
```

> [!NOTE]
> The `--tag <registry>/nemo-rl:latest --push` flags are not necessary if you just want to build locally.

## Skipping vLLM or SGLang Dependencies

If you don't need vLLM or SGLang support, you can skip building those dependencies to reduce build time and image size. Use the `SKIP_VLLM_BUILD` and/or `SKIP_SGLANG_BUILD` build arguments:

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

# Skip both vLLM and SGLang dependencies:
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_VLLM_BUILD=1 \
    --build-arg SKIP_SGLANG_BUILD=1 \
    --tag <registry>/nemo-rl:latest \
    .
```

When these build arguments are set, the corresponding `uv sync --extra` commands are skipped, and the virtual environment prefetching will exclude actors that depend on those packages.

> [!NOTE]
> If you skip vLLM or SGLang during the build but later try to use those backends at runtime, the dependencies will be fetched and built on-demand. This may add significant setup time on first use.
