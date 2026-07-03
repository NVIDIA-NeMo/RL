#!/usr/bin/env bash
set -euo pipefail
set -x

DYNAMO_COMMIT=${DYNAMO_COMMIT:-59358c26d0aeed19300706462b63ada25a0a6d7c}
DYNAMO_PYTHON_VERSION=${DYNAMO_PYTHON_VERSION:-3.12.11}
DYNAMO_VENV=${DYNAMO_VENV:-/opt/dynamo_venv}
DYNAMO_SOURCE_DIR=${DYNAMO_SOURCE_DIR:-/opt/dynamo-src}
RUST_VERSION=${RUST_VERSION:-1.93.1}
UV_BIN=${UV_BIN:-$(command -v uv)}

export RUSTUP_HOME=${RUSTUP_HOME:-/opt/rustup}
export CARGO_HOME=${CARGO_HOME:-/opt/cargo}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/opt/dynamo-target}
export PATH="${CARGO_HOME}/bin:${DYNAMO_VENV}/bin:${PATH}"

# The pinned revision is ahead of the latest published release: ai-dynamo
# 1.3.0 requires an ai-dynamo-runtime 1.3.0 wheel that is not yet on PyPI.
# Build both packages from the same checkout so their native/Python contracts
# cannot drift.
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  build-essential \
  cmake \
  curl \
  git \
  libclang-dev \
  libhwloc-dev \
  libudev-dev \
  pkg-config \
  protobuf-compiler \
  python3-dev
rm -rf /var/lib/apt/lists/*

if [[ ! -x "${CARGO_HOME}/bin/cargo" ]]; then
  curl --proto '=https' --tlsv1.2 --fail --location --retry 3 \
    https://sh.rustup.rs --output /tmp/rustup-init.sh
  sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal \
    --default-toolchain "${RUST_VERSION}"
  rm -f /tmp/rustup-init.sh
fi

"${UV_BIN}" python install "${DYNAMO_PYTHON_VERSION}"
"${UV_BIN}" venv --clear --python "${DYNAMO_PYTHON_VERSION}" "${DYNAMO_VENV}"
"${UV_BIN}" pip install --python "${DYNAMO_VENV}/bin/python" \
  pip 'maturin[patchelf]'

git clone --filter=blob:none https://github.com/ai-dynamo/dynamo.git \
  "${DYNAMO_SOURCE_DIR}"
git -C "${DYNAMO_SOURCE_DIR}" checkout --detach "${DYNAMO_COMMIT}"
test "$(git -C "${DYNAMO_SOURCE_DIR}" rev-parse HEAD)" = "${DYNAMO_COMMIT}"

(
  cd "${DYNAMO_SOURCE_DIR}/lib/bindings/python"
  VIRTUAL_ENV="${DYNAMO_VENV}" \
    "${DYNAMO_VENV}/bin/maturin" develop --uv --release
)
"${UV_BIN}" pip install --python "${DYNAMO_VENV}/bin/python" \
  "${DYNAMO_SOURCE_DIR}/lib/gpu_memory_service"
(
  cd "${DYNAMO_SOURCE_DIR}"
  "${UV_BIN}" pip install --python "${DYNAMO_VENV}/bin/python" '.[vllm]'
)

printf '%s\n' "${DYNAMO_COMMIT}" > /opt/dynamo_commit
"${DYNAMO_VENV}/bin/python" -c \
  'import importlib.metadata as m, pathlib; assert m.version("ai-dynamo") == "1.3.0"; assert m.version("ai-dynamo-runtime") == "1.3.0"; assert pathlib.Path("/opt/dynamo_commit").read_text().strip() == "59358c26d0aeed19300706462b63ada25a0a6d7c"; import dynamo.vllm.handlers, vllm; print("Dynamo", m.version("ai-dynamo"), "runtime", m.version("ai-dynamo-runtime"), "vLLM", vllm.__version__)'
