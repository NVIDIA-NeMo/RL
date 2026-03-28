#!/bin/bash

set -eou pipefail
HELM_VERSION=${HELM_VERSION:-v3.17.3}

mkdir -p ~/bin/
NAMED_HELM=~/bin/helm-$HELM_VERSION

if [[ ! -f $NAMED_HELM ]]; then
  ARCH=$(uname -m)
  case $ARCH in
    x86_64)  ARCH=amd64 ;;
    aarch64) ARCH=arm64 ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
  esac
  tmp_helm_dir=$(mktemp -d)
  curl -sSL "https://get.helm.sh/helm-${HELM_VERSION}-linux-${ARCH}.tar.gz" | tar -xz -C "$tmp_helm_dir" --strip-components=1
  cp "$tmp_helm_dir/helm" "$NAMED_HELM"
  rm -rf "$tmp_helm_dir"
fi

echo "Installed helm at $NAMED_HELM"
echo "To use, you may set 'alias helm=$NAMED_HELM'"
