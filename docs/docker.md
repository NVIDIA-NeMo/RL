# Build Docker Images

This guide provides three methods for building Docker images:

* **release**: Contains everything from the hermetic image, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers.
* **hermetic**: Includes the base image plus pre-fetched NeMo RL python packages in the `uv` cache.
* **base**: A minimal image with CUDA, `ray`, and `uv` installed, ideal for specifying Python dependencies at runtime.

Use the:
* **release** (recommended): if you want to pre-fetch the NeMo RL [worker virtual environments](./design-docs/uv.md#worker-configuration) and copy in the project source code.
* **hermetic**: if you want to pre-fetch NeMo RL python packages into the `uv` cache to eliminate the initial overhead of program start.
* **base**: if you just need a minimal image with CUDA, `ray`, and `uv` installed and are okay with dynamically downloading your requirements at runtime. This option trades off fast container download/startup with slower initial overhead to download python packages.

## Release Image

The release image is our recommended option as it provides the most complete environment. It includes everything from the hermetic image, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers. This is the ideal choice for production deployments.

```sh
cd docker/
docker buildx build --target release -t nemo_rl -f Dockerfile ..
```

## Hermetic Image

The hermetic image includes all Python dependencies pre-downloaded in the `uv` cache, eliminating the initial overhead of downloading packages at runtime. This is useful when you need a more predictable environment or have limited network connectivity.

```sh
cd docker/
docker buildx build --target hermetic -t nemo_rl -f Dockerfile ..
```

## Base Image

The base image provides a minimal environment with CUDA, `ray`, and `uv` installed. While it's the smallest image, it requires downloading Python dependencies at runtime, which may not be ideal for all use cases.

```sh
cd docker/
docker buildx build --target base -t nemo_rl -f Dockerfile ..
```
