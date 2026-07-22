# Building the Docker Container
NOTE: *We use `docker buildx` instead of `docker build` for these containers*

This directory contains the `Dockerfile` for NeMo-RL Docker images.
You can build two types of images:
- A **release image** (recommended): Contains everything from the hermetic image, plus the nemo-rl source code and pre-fetched virtual environments for isolated workers.
- A **hermetic image**: Includes the base image plus pre-fetched NeMo RL python packages in the `uv` cache.


For detailed instructions on building these images, please see [docs/docker.md](../docs/docker.md).

## Qwen 3.5 MLPerf image

The audited Qwen 3.5 GRPO image and launcher context lives in
[`docker/mlperf`](mlperf). Build it from the repository root with:

```bash
source_commit="$(git rev-parse HEAD)"
docker buildx build --platform linux/arm64 \
  --build-arg GIT_COMMIT_ID="$source_commit" \
  --build-arg NEMO_RL_REVISION="$source_commit" \
  -f docker/mlperf/Dockerfile docker/mlperf
```

See [`docker/mlperf/README.md`](mlperf/README.md) for immutable dependency pins,
downstream patches, provenance, and the authoritative runtime profile.
That recipe builds the explicitly pinned revision from the public Qwen branch
and does not inherit from a gated NeMo-RL nightly image.
