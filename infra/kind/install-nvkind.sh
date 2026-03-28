#!/bin/bash
# Installs kind (dependency) and nvkind.
# Requires: go, curl

set -eou pipefail

KIND_VERSION=${KIND_VERSION:-v0.31.0}

# --- Install kind (required by nvkind) ---
NAMED_KIND=~/bin/kind-$KIND_VERSION
mkdir -p ~/bin

if [[ ! -f $NAMED_KIND ]]; then
  ARCH=$(uname -m)
  case $ARCH in
    x86_64)  ARCH=amd64 ;;
    aarch64) ARCH=arm64 ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
  esac
  curl -Lo "$NAMED_KIND" "https://kind.sigs.k8s.io/dl/$KIND_VERSION/kind-linux-$ARCH"
  chmod +x "$NAMED_KIND"
fi
ln -sf "$NAMED_KIND" ~/bin/kind
echo "Installed kind $KIND_VERSION at $NAMED_KIND"

# --- Install nvkind ---
if ! command -v nvkind &>/dev/null; then
  GOBIN=~/bin go install github.com/NVIDIA/nvkind/cmd/nvkind@latest
fi
echo "Installed nvkind at $(which nvkind || echo ~/bin/nvkind)"
