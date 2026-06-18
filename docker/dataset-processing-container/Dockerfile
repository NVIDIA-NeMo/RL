ARG UBUNTU_VERSION=24.04
FROM ubuntu:${UBUNTU_VERSION} AS base

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=xterm-256color \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace

RUN <<'EOF'
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  gnupg \
  software-properties-common \
  tmux \
  vim \
  wget
add-apt-repository -y ppa:apptainer/ppa
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
DOCKER_CLI_VERSION="$(apt-cache madison docker-ce-cli \
  | awk '$3 ~ /^5:27\./ { print $3; exit }')"
test -n "${DOCKER_CLI_VERSION}"
apt-get install -y --no-install-recommends \
  apptainer \
  docker-buildx-plugin \
  docker-ce-cli="${DOCKER_CLI_VERSION}"
docker --version | grep -E '^Docker version 27\.'
docker buildx version
rm -rf /var/lib/apt/lists/*
EOF

# Install Python
RUN <<'EOF'
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install --default python3.12
EOF

RUN <<'EOF'
git clone --branch mlperf-grpo https://github.com/filaretov/r2e-gym-arm-build /workspace/repos/r2e-gym-arm-build
EOF

COPY --chmod=0755 run-r2e-gym-build-images.sh .
COPY --chmod=0755 run-build-sif-images.sh .
COPY --chmod=0755 entrypoint.sh .
COPY --chmod=0666 r2e-gym-instances-to-build.txt .
COPY --chmod=0666 build_swe_sif_images.py .

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["/bin/bash"]

