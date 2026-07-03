#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:?Set IMAGE to a registry tag, for example nvcr.io/<org>/nemo-rl-dynamo-slurm:<tag>}
PLATFORM=${PLATFORM:-linux/amd64}
PUSH=${PUSH:-0}
NEMO_RL_IMAGE=${NEMO_RL_IMAGE:-${IMAGE}-nemo-base}
NEMO_RL_COMMIT=${NEMO_RL_COMMIT:-$(git rev-parse HEAD)}

OUTPUT=(--load)
if [[ "${PUSH}" = "1" ]]; then
  OUTPUT=(--push)
fi

docker buildx build \
  --platform "${PLATFORM}" \
  --target release \
  --build-arg NEMO_RL_COMMIT="${NEMO_RL_COMMIT}" \
  --tag "${NEMO_RL_IMAGE}" \
  "${OUTPUT[@]}" \
  --file docker/Dockerfile \
  .

docker buildx build \
  --platform "${PLATFORM}" \
  --build-arg NEMO_RL_BASE_IMAGE="${NEMO_RL_IMAGE}" \
  --tag "${IMAGE}" \
  "${OUTPUT[@]}" \
  --file docker/Dockerfile.dynamo-slurm \
  .

echo "Built ${IMAGE}"
if [[ "${PUSH}" = "1" ]]; then
  echo "On the Slurm login node, import it with:"
  echo "  enroot import -o nemo-rl-dynamo-slurm.sqsh docker://${IMAGE}"
fi
