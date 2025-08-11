# Build Docker Images

This guide provides two methods for building Docker images:

* **release**: Contains everything from the hermetic image, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers.
* **hermetic**: Includes the base image plus pre-fetched NeMo RL python packages in the `uv` cache.

Use the:
* **release** (recommended): if you want to pre-fetch the NeMo RL [worker virtual environments](./design-docs/uv.md#worker-configuration) and copy in the project source code.
* **hermetic**: if you want to pre-fetch NeMo RL python packages into the `uv` cache to eliminate the initial overhead of program start.

## Release Image

The release image is our recommended option as it provides the most complete environment. It includes everything from the hermetic image, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers. This is the ideal choice for production deployments.

```sh
# Self-contained build (default: builds from main):
docker buildx build --target release -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .

# Self-contained build (specific git ref):
docker buildx build --target release -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.3.0 --tag <registry>/nemo-rl:r0.3.0 --push .

# Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL):
docker buildx build --target release -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.3.0 --tag <registry>/nemo-rl:r0.3.0 --push https://github.com/NVIDIA-NeMo/RL.git

# Local NeMo RL source override:
docker buildx build --target release --build-context nemo-rl=. -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
```

**Note:** The `--tag <registry>/nemo-rl:latest --push` flags are not necessary if you just want to build locally.

## Hermetic Image

The hermetic image includes all Python dependencies pre-downloaded in the `uv` cache, eliminating the initial overhead of downloading packages at runtime. This is useful when you need a more predictable environment or have limited network connectivity.

```sh
# Self-contained build (default: builds from main):
docker buildx build --target hermetic -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .

# Self-contained build (specific git ref):
docker buildx build --target hermetic -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.3.0 --tag <registry>/nemo-rl:r0.3.0 --push .

# Self-contained build (remote NeMo RL source; no need for a local clone of NeMo RL):
docker buildx build --target hermetic -f docker/Dockerfile --build-arg NRL_GIT_REF=r0.3.0 --tag <registry>/nemo-rl:r0.3.0 --push https://github.com/NVIDIA-NeMo/RL.git

# Local NeMo RL source override:
docker buildx build --target hermetic --build-context nemo-rl=. -f docker/Dockerfile --tag <registry>/nemo-rl:latest --push .
```

**Note:** The `--tag <registry>/nemo-rl:latest --push` flags are not necessary if you just want to build locally.
