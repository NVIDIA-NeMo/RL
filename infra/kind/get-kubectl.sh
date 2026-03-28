#!/bin/bash

set -eou pipefail

KUBECTL_VERSION=${KUBECTL_VERSION:-latest}

echo ==============================================
echo "Currently installed versions of kubectl:"
ls ~/bin/kubectl* 2>/dev/null || true
echo ==============================================

mkdir -p ~/bin
NAMED_KUBECTL=~/bin/kubectl-$KUBECTL_VERSION

if [[ ! -f $NAMED_KUBECTL ]]; then
  ARCH=$(uname -m)
  case $ARCH in
    x86_64)  ARCH=amd64 ;;
    aarch64) ARCH=arm64 ;;
    *) echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
  esac
  if [[ $KUBECTL_VERSION == latest ]]; then
    curl -L "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/$ARCH/kubectl" -o "$NAMED_KUBECTL"
  else
    curl -L "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/$ARCH/kubectl" -o "$NAMED_KUBECTL"
  fi
fi
chmod +x "$NAMED_KUBECTL"

echo "Installed kubectl at $NAMED_KUBECTL"
echo "To use, you may set 'alias kubectl=$NAMED_KUBECTL'"

if [[ ! -f ~/.krew/bin/kubectl-krew ]]; then
  (
    set -x; cd "$(mktemp -d)" &&
    OS="$(uname | tr '[:upper:]' '[:lower:]')" &&
    ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/\(arm\)\(64\)\?.*/\1\2/' -e 's/aarch64$/arm64/')" &&
    KREW="krew-${OS}_${ARCH}" &&
    curl -fsSLO "https://github.com/kubernetes-sigs/krew/releases/latest/download/${KREW}.tar.gz" &&
    tar zxvf "${KREW}.tar.gz" &&
    ./"${KREW}" install krew
  )
else
  echo "krew already installed"
fi

$NAMED_KUBECTL krew install ctx
$NAMED_KUBECTL krew install ns
$NAMED_KUBECTL krew install stern
$NAMED_KUBECTL krew install view-allocations
$NAMED_KUBECTL krew install whoami

cat <<EOF
Installed krew at $HOME/.krew
Add the following to .bashrc/.bash_aliases

export PATH="\${KREW_ROOT:-\$HOME/.krew}/bin:\$PATH"
EOF
