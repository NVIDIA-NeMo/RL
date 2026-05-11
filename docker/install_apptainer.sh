#!/bin/bash
set -euo pipefail
apt-get update
apt-get install -y --no-install-recommends software-properties-common
add-apt-repository -y ppa:apptainer/ppa
apt-get update
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
apt-get install -y --no-install-recommends apptainer=1.5.0-1~${CODENAME}
ln -sf /usr/bin/apptainer /usr/bin/singularity
apt-get clean && rm -rf /var/lib/apt/lists/*
