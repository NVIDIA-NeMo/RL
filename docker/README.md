# Building the Docker Container
NOTE: *We use `docker buildx` instead of `docker build` for these containers*

This directory contains the `Dockerfile` for NeMo-RL Docker images.
You can build two types of images:
- A **base image**: A minimal image where Python dependencies can be specified at runtime.
- A **hermetic image**: An image that includes default dependencies for offline use.


For detailed instructions on building these images, please see [docs/docker.md](../docs/docker.md).