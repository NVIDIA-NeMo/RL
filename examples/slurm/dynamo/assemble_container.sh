#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-new}
REPO=${REPO:-${ROOT}/RL}
DYNAMO_COMMIT=${DYNAMO_COMMIT:-59358c26d0aeed19300706462b63ada25a0a6d7c}
DYNAMO_PYTHON_VERSION=${DYNAMO_PYTHON_VERSION:-3.12.11}
ETCD_VERSION=${ETCD_VERSION:-v3.5.21}
NATS_VERSION=${NATS_VERSION:-v2.11.6}

test "$(uname -m)" = x86_64
UV=$(find /usr/local/bin /usr/bin /root/.local/bin -maxdepth 1 -type f -name uv -print -quit)
test -n "${UV}"

# Replace the release image source with this branch, then resolve its locked
# all-groups environment in-place for both the driver and Ray actors.
rsync -a --delete --exclude=.git --exclude=.venv "${REPO}/" /opt/nemo-rl/
cd /opt/nemo-rl
export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
export UV_CACHE_DIR=${ROOT}/cache/uv
export NRL_CONTAINER=1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
"${UV}" sync --locked --all-groups --no-install-project --reinstall --link-mode copy
test -x /opt/nemo_rl_venv/bin/python
(
  cd "${REPO}"
  PYTHONPATH="${REPO}" /opt/nemo_rl_venv/bin/python \
    tools/generate_fingerprint.py > /opt/nemo_rl_container_fingerprint
)

# Dynamo/vLLM has an independent Python solve. The pinned revision's matching
# native runtime wheel is not yet published, so build both packages from the
# exact checkout instead of mixing source with an older runtime wheel.
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
DYNAMO_COMMIT="${DYNAMO_COMMIT}" \
DYNAMO_PYTHON_VERSION="${DYNAMO_PYTHON_VERSION}" \
UV_BIN="${UV}" \
  /opt/nemo-rl/docker/install-dynamo-source.sh

curl --fail --location --retry 3 \
  "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-amd64.tar.gz" \
  --output /tmp/etcd.tgz
tar -xzf /tmp/etcd.tgz -C /tmp
install -m 0755 "/tmp/etcd-${ETCD_VERSION}-linux-amd64/etcd" /usr/local/bin/etcd

curl --fail --location --retry 3 \
  "https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-linux-amd64.tar.gz" \
  --output /tmp/nats.tgz
tar -xzf /tmp/nats.tgz -C /tmp
install -m 0755 "/tmp/nats-server-${NATS_VERSION}-linux-amd64/nats-server" /usr/local/bin/nats-server

rm -rf /tmp/etcd* /tmp/nats*

/opt/dynamo_venv/bin/python -c \
  'import dynamo.vllm.handlers, vllm; print("Dynamo/vLLM ready", vllm.__version__)'
/opt/dynamo_venv/bin/python \
  /opt/nemo-rl/nemo_rl/models/generation/dynamo/validate_dynamo_vllm_args.py \
  '["--model","Qwen/Qwen2.5-1.5B","--served-model-name","Qwen/Qwen2.5-1.5B","--namespace","nemo-rl-build","--discovery-backend","etcd","--request-plane","tcp","--enable-rl","--weight-transfer-config","{\"backend\":\"nccl\"}","--tensor-parallel-size","1","--pipeline-parallel-size","1","--dtype","bfloat16","--endpoint-types","chat,completions"]'
/opt/dynamo_venv/bin/python \
  /opt/nemo-rl/nemo_rl/models/generation/dynamo/validate_dynamo_vllm_args.py \
  '["--model","nvidia/NVIDIA-Nemotron-Nano-9B-v2","--served-model-name","nvidia/NVIDIA-Nemotron-Nano-9B-v2","--namespace","nemo-rl-build","--discovery-backend","etcd","--request-plane","tcp","--enable-rl","--weight-transfer-config","{\"backend\":\"nccl\"}","--tensor-parallel-size","1","--pipeline-parallel-size","1","--dyn-tool-call-parser","nemotron_deci","--dyn-reasoning-parser","nemotron_nano","--compilation-config","{\"backend\":\"eager\"}","--endpoint-types","chat,completions"]'
if /opt/dynamo_venv/bin/python \
  /opt/nemo-rl/nemo_rl/models/generation/dynamo/validate_dynamo_vllm_args.py \
  '["--model","Qwen/Qwen2.5-1.5B","--served-model-name","Qwen/Qwen2.5-1.5B","--namespace","nemo-rl-build","--discovery-backend","etcd","--request-plane","tcp","--enable-rl","--weight-transfer-config","{\"backend\":\"nccl\"}","--dyn-tool-call-parser","not-a-real-parser"]'; then
  echo "Dynamo parser unexpectedly accepted an invalid tool parser" >&2
  exit 1
fi
etcd --version
nats-server --version
