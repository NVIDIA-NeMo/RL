# Building the Docker Container
NOTE: *We use `docker buildx` instead of `docker build` for these containers*

This directory contains the `Dockerfile` for NeMo-RL Docker images.
You can build two types of images:
- A **release image** (recommended): Contains everything from the hermetic image, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers.
- A **hermetic image**: Includes the base image plus pre-fetched NeMo RL python packages in the `uv` cache.


For detailed instructions on building these images, please see [docs/docker.md](../docs/docker.md).