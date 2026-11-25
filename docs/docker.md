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
    --build-arg NRL_GIT_REF=r0.8.0 \
    --tag <registry>/nemo-rl:r0.8.0 \
    --push .

# Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL):
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=r0.8.0 \
    --tag <registry>/nemo-rl:r0.8.0 \
    --push https://github.com/NVIDIA-NeMo/RL.git

# Local NeMo RL source override:
docker buildx build --build-context nemo-rl=. -f docker/Dockerfile \
    --tag <registry>/nemo-rl:latest \
    --push .
```

> [!NOTE]
> The `--tag <registry>/nemo-rl:latest --push` flags are not necessary if you just want to build locally.

## uv Cache Pruning

The release image ships `uv`'s download cache (`/opt/nemo_rl_cache/uv`) so an unprivileged runtime user can later run `uv run --extra <backend>` fully offline. Each build starts that cache from a BuildKit cache mount (seeded from the `uv-cache` registry image), and `uv` only ever adds to it: it never removes a version a prior `uv.lock` pinned and the current one doesn't. The seed export's `uv cache prune --ci` drops prebuilt wheels but keeps every version ever built from source, and a long-lived builder's cache mount keeps everything.

Right after that cache is restored, before any `uv sync`, `docker/prune_uv_cache.py` removes exactly those superseded entries: index wheels and source builds whose `(package, version)` the current `uv.lock` no longer pins, and git builds/checkouts for commits it no longer pins, across every extra (installed in this build or not). Running first means it only judges restored content, so the build-isolation dependencies the syncs resolve outside `uv.lock` (setuptools, a build-time torch, ...) are kept. It runs in the same layer the syncs fill, because deleting files in a later layer only hides them and the image still ships them, and the result is synced back into the cache mount. It never removes a wheel the offline-install promise depends on, since that promise is itself "every wheel `uv.lock` could need." This runs automatically on every build; there is no flag to disable it.

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

## Overriding Packages in the vLLM Worker Venvs

`VLLM_VENV_PACKAGE_OVERRIDES` installs the listed packages over the locked ones in every worker venv that carries the `vllm` extra. By default it installs torch 2.15.0 built for CUDA 13.4 (with torchvision, torchaudio, Triton and cuBLAS to match), so the vLLM workers also run on Rubin (sm_107). `uv.lock` cannot express a torch for a single extra. The default is equivalent to:

```sh
docker buildx build -f docker/Dockerfile \
    --build-arg VLLM_VENV_PACKAGE_OVERRIDES=torch==2.15.0+cu134,torchvision==0.29.2+cu134,torchaudio==2.11.1+cu134,triton==3.9.0,nvidia-cublas==13.7.0.27 \
    --tag <registry>/nemo-rl:latest \
    .
```

The value is a comma-separated list of requirements, normally exact pins. Override torch together with its companion packages (torchvision, torchaudio): torchaudio refuses to import next to a torch built for a different CUDA version. Packages come from PyPI and from `VLLM_VENV_PACKAGE_OVERRIDES_INDEX`, a comma-separated list of extra indexes (default: the PyTorch cu134 index).

The list may include `vllm` itself and nightly wheels. The listed requirements also override what the installed packages ask for, so a vLLM wheel that pins an older torch still installs next to the torch you list. A requirement that names a pre-release allows one for that package:

```sh
--build-arg VLLM_VENV_PACKAGE_OVERRIDES_INDEX=https://download.pytorch.org/whl/nightly/cu135,https://download.pytorch.org/whl/nightly/cu134,https://wheels.vllm.ai/4f0ecfff6be6f4d28ba91ad4d44b43b4100d4f2d \
--build-arg VLLM_VENV_PACKAGE_OVERRIDES=vllm==0.34.0rc1.dev1193,torch==2.16.0.dev20261116+cu135,torchvision==0.30.0.dev20261116+cu134,torchaudio==2.11.0.dev20261116+cu134
```

When the overridden venv has a Triton wheel with its own `ptxas-blackwell`, the build also sets `TRITON_PTXAS_BLACKWELL_PATH` to it for that venv (through a `.pth` file, unless the variable is already set). Otherwise flashinfer points Triton at the base image's `ptxas`, which can be too old for the GPU the overridden torch targets.

`MCORE_VENV_PACKAGE_OVERRIDES` does the same for the worker venvs that carry the `mcore` extra, with the same index list. By default it installs Triton 3.9.0, whose compiler knows Rubin (sm_107). The compiled training extensions are built against the locked torch, so use it for packages that do not link to torch. The default is equivalent to:

```sh
--build-arg MCORE_VENV_PACKAGE_OVERRIDES=triton==3.9.0
```

The overridden venvs no longer match `uv.lock`. Anything that rebuilds them from the lock restores the locked packages: `uv sync`, `uv run --extra vllm`, `NRL_FORCE_REBUILD_VENVS=true`, or a `NEMO_RL_VENV_DIR` that points away from the prebuilt venvs.

## Stripping the uv Cache

By default, the release image keeps `uv`'s download cache (`/opt/nemo_rl_cache/uv`) on disk. This lets the image's unprivileged runtime user re-run `uv run --extra <backend>` fully offline, since every wheel it could need is already cached in the image. The tradeoff is a larger image.

For size-sensitive builds that don't need that offline capability, drop the cache from the final image with `STRIP_UV_CACHE`:

```sh
docker buildx build -f docker/Dockerfile \
    --build-arg STRIP_UV_CACHE=1 \
    --tag <registry>/nemo-rl:latest \
    .
```

> [!NOTE]
> On a long-lived local builder, BuildKit's cache mount (keyed by base image, uv version, and the HybridEP setting) persists across builds regardless of `STRIP_UV_CACHE` -- it just controls whether a copy also ships in the image layers. To reset that mount, run `docker buildx prune --filter type=exec.cachemount`.

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
