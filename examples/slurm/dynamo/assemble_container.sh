#!/usr/bin/env bash
set -euo pipefail
set -x

ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-new}
REPO=${REPO:-${ROOT}/RL}
DYNAMO_COMMIT=${DYNAMO_COMMIT:-59358c26d0aeed19300706462b63ada25a0a6d7c}
DYNAMO_PYTHON_VERSION=${DYNAMO_PYTHON_VERSION:-3.12.11}
ETCD_VERSION=${ETCD_VERSION:-v3.5.21}
NATS_VERSION=${NATS_VERSION:-v2.11.6}
MATERIALIZE_NEMO_ENV=${MATERIALIZE_NEMO_ENV:-1}
EXPECTED_GYM_COMMIT=${EXPECTED_GYM_COMMIT:-eddd5e98a541cc90e0ee41f1b5e9bd146b5be665}

case "$(uname -m)" in
  x86_64)
    RELEASE_ARCH=amd64
    DEFAULT_CUDA_ARCH=9.0
    ;;
  aarch64)
    RELEASE_ARCH=arm64
    DEFAULT_CUDA_ARCH=10.0
    ;;
  *)
    echo "Unsupported architecture: $(uname -m)" >&2
    exit 1
    ;;
esac
UV=$(find /usr/local/bin /usr/bin /root/.local/bin -maxdepth 1 -type f -name uv -print -quit)
test -n "${UV}"

# Replace the release image source with this branch. Ruit's HSG image already
# contains the tested NeMo-RL/Gym environment, so its aarch64 build can retain
# that environment while the source tree and Gym editable checkout are updated.
rsync -a --delete --exclude=.git --exclude=.venv "${REPO}/" /opt/nemo-rl/
cd /opt/nemo-rl
if [[ "${MATERIALIZE_NEMO_ENV}" = "1" ]]; then
  export UV_PROJECT_ENVIRONMENT=/opt/nemo_rl_venv
  export UV_CACHE_DIR=${ROOT}/cache/uv
  export NRL_CONTAINER=1
  export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-${DEFAULT_CUDA_ARCH}}
  "${UV}" sync --locked --all-groups --no-install-project --reinstall --link-mode copy
elif [[ "${MATERIALIZE_NEMO_ENV}" != "0" ]]; then
  echo "MATERIALIZE_NEMO_ENV must be 0 or 1, got ${MATERIALIZE_NEMO_ENV}" >&2
  exit 1
fi
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
  "https://github.com/etcd-io/etcd/releases/download/${ETCD_VERSION}/etcd-${ETCD_VERSION}-linux-${RELEASE_ARCH}.tar.gz" \
  --output /tmp/etcd.tgz
tar -xzf /tmp/etcd.tgz -C /tmp
install -m 0755 "/tmp/etcd-${ETCD_VERSION}-linux-${RELEASE_ARCH}/etcd" /usr/local/bin/etcd

curl --fail --location --retry 3 \
  "https://github.com/nats-io/nats-server/releases/download/${NATS_VERSION}/nats-server-${NATS_VERSION}-linux-${RELEASE_ARCH}.tar.gz" \
  --output /tmp/nats.tgz
tar -xzf /tmp/nats.tgz -C /tmp
install -m 0755 "/tmp/nats-server-${NATS_VERSION}-linux-${RELEASE_ARCH}/nats-server" /usr/local/bin/nats-server

rm -rf /tmp/etcd* /tmp/nats*

/opt/dynamo_venv/bin/python -c \
  'import importlib.metadata as m; import dynamo.vllm.handlers, vllm; assert m.version("ai-dynamo") == "1.3.0"; assert m.version("ai-dynamo-runtime") == "1.3.0"; assert vllm.__version__ == "0.23.0"; print("Dynamo", m.version("ai-dynamo"), "runtime", m.version("ai-dynamo-runtime"), "vLLM", vllm.__version__)'
/opt/dynamo_venv/bin/python \
  /opt/nemo-rl/nemo_rl/models/generation/dynamo/validate_dynamo_vllm_args.py \
  '["--model","Qwen/Qwen2.5-1.5B","--served-model-name","Qwen/Qwen2.5-1.5B","--namespace","nemo-rl-build","--discovery-backend","etcd","--request-plane","tcp","--enable-rl","--weight-transfer-config","{\"backend\":\"nccl\"}","--tensor-parallel-size","1","--pipeline-parallel-size","1","--dtype","bfloat16","--endpoint-types","chat,completions"]'
/opt/dynamo_venv/bin/python \
  /opt/nemo-rl/nemo_rl/models/generation/dynamo/validate_dynamo_vllm_args.py \
  '["--model","/models/nemotron-nano-v3.5","--served-model-name","/models/nemotron-nano-v3.5","--namespace","nemo-rl-build","--discovery-backend","etcd","--request-plane","tcp","--event-plane","nats","--enable-rl","--weight-transfer-config","{\"backend\":\"nccl\"}","--trust-remote-code","--seed","0","--tensor-parallel-size","4","--pipeline-parallel-size","1","--dtype","bfloat16","--kv-cache-dtype","auto","--gpu-memory-utilization","0.85","--max-model-len","196608","--no-enforce-eager","--load-format","auto","--attention-backend","FLASH_ATTN","--moe-backend","triton","--mamba-ssm-cache-dtype","float32","--compilation-config","{\"cudagraph_capture_sizes\":[1,2,4,8,16,32,64],\"pass_config\":{\"fuse_allreduce_rms\":false}}","--dyn-tool-call-parser","qwen3_coder","--dyn-reasoning-parser","nemotron_nano","--exclude-tools-when-tool-choice-none","--no-dyn-enable-structural-tag","--dyn-structural-tag-scope","auto","--dyn-structural-tag-schema","auto","--custom-jinja-template","/models/nemotron-nano-v3.5/chat_template.jinja","--endpoint-types","chat,completions"]'
if /opt/dynamo_venv/bin/python \
  /opt/nemo-rl/nemo_rl/models/generation/dynamo/validate_dynamo_vllm_args.py \
  '["--model","Qwen/Qwen2.5-1.5B","--served-model-name","Qwen/Qwen2.5-1.5B","--namespace","nemo-rl-build","--discovery-backend","etcd","--request-plane","tcp","--enable-rl","--weight-transfer-config","{\"backend\":\"nccl\"}","--dyn-tool-call-parser","not-a-real-parser"]'; then
  echo "Dynamo parser unexpectedly accepted an invalid tool parser" >&2
  exit 1
fi

test "$(git -C "${REPO}/3rdparty/Gym-workspace/Gym" rev-parse HEAD)" = "${EXPECTED_GYM_COMMIT}"
printf '%s\n' "${EXPECTED_GYM_COMMIT}" > /opt/nemo_gym_commit
GYM_PYTHON=$(find /opt/gym_venvs -path '*/.venv/bin/python' -type f -print -quit)
test -x "${GYM_PYTHON}"
"${GYM_PYTHON}" -c \
  'import pathlib, nemo_gym; expected = pathlib.Path("/opt/nemo-rl/3rdparty/Gym-workspace/Gym").resolve(); actual = pathlib.Path(nemo_gym.__file__).resolve(); assert actual.is_relative_to(expected), (actual, expected); print("Gym source", actual)'
grep -R -q '_attach_native_token_information' \
  /opt/nemo-rl/3rdparty/Gym-workspace/Gym

etcd --version
nats-server --version
