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

## Building Only Some Backends

Each backend is a uv extra (`vllm`, `sglang`, `mcore`, `automodel`, `modelopt`, `nemo_gym`, `trtllm`). By default the image builds all of them. To build only the ones you use, list them in `NRL_EXTRAS`:

```sh
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_EXTRAS="mcore vllm nemo_gym" \
    --tag <registry>/nemo-rl:latest \
    .
```

| Build argument | Default | Effect |
|---|---|---|
| `NRL_EXTRAS` | all extras | Only these extras are installed, and only the worker virtual environments that need no other extra are prefetched. An unknown name fails the build. |
| `NRL_GROUPS` | all dependency groups | Only these dependency groups (for example `build`) are installed in the main virtual environment. |

The Rust toolchain is only installed when `sglang` is built.

> [!NOTE]
> A backend left out of the build is fetched and built on first use, which can add a long setup time to that run.

`NRL_EXTRAS` replaces the `SKIP_VLLM_BUILD`, `SKIP_SGLANG_BUILD` and `SKIP_TRTLLM_BUILD` arguments, and passing one of them now fails the build. To leave a backend out, list the others. For example, `SKIP_SGLANG_BUILD=1` with `SKIP_TRTLLM_BUILD=1` becomes:

```sh
--build-arg NRL_EXTRAS="vllm mcore automodel modelopt nemo_gym"
```

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
