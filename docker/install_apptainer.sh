#!/bin/bash
set -euo pipefail

APPTAINER_VERSION=1.4.5
APPTAINER_SOURCE_SHA256=d323a8b9a0a9e5e131b396d0049fdaa99beceb83a3d7ffb80dd91d15331e3b9a
GO_VERSION=1.26.5
GO_AMD64_SHA256=5c2c3b16caefa1d968a94c1daca04a7ca301a496d9b086e17ad77bb81393f053
GO_ARM64_SHA256=fe4789e92b1f33358680864bbe8704289e7bb5fc207d80623c308935bd696d49
# Bumping APPTAINER_VERSION itself is not an option here (functionality
# regressions on a newer release), so instead this script builds the SAME
# 1.4.5 release from source on both architectures and patches only the
# vendored dependency versions via `go get` + `go mod tidy` + `go mod vendor`
# before compiling -- same technique Megatron-Bridge uses to patch
# wandb-core's vendored deps
# (see 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/docker/Dockerfile.fw_final).
GO_CRYPTO_VERSION=0.55.0
GO_TEXT_VERSION=0.39.0
GRPC_VERSION=1.83.1
FULCIO_VERSION=1.8.5
# `go mod tidy` re-resolves the WHOLE dependency graph, not just the four
# modules above -- left unconstrained, it cascade-upgrades
# github.com/docker/docker and github.com/containers/image/v5 to versions
# whose APIs (archive.Compression, client.ImageLoadOption) apptainer 1.4.5's
# vendored source doesn't compile against. Pin both to the exact versions
# already in apptainer 1.4.5's own go.mod so tidy has no reason to move them.
DOCKER_VERSION=27.5.1
CONTAINERS_IMAGE_VERSION=5.33.1
DEB_ARCH="$(dpkg --print-architecture)"
export DEBIAN_FRONTEND=noninteractive

install_go() {
    local arch="$1"
    local sha256="$2"
    local go_tarball="/tmp/go${GO_VERSION}.linux-${arch}.tar.gz"

    wget --progress=dot:giga -O "${go_tarball}" "https://go.dev/dl/go${GO_VERSION}.linux-${arch}.tar.gz"
    echo "${sha256}  ${go_tarball}" | sha256sum -c -
    rm -rf /usr/local/go
    tar -C /usr/local -xzf "${go_tarball}"
    rm -f "${go_tarball}"
    export PATH="/usr/local/go/bin:${PATH}"
}

verify_install() {
    apptainer --version
    singularity --version
}

install_from_source() {
    local arch="$1"
    local go_sha256="$2"
    local build_dir="/tmp/apptainer-build"
    local source_tarball="/tmp/apptainer-${APPTAINER_VERSION}.tar.gz"
    local source_url="https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer-${APPTAINER_VERSION}.tar.gz"
    # curl, git, and wget are installed by docker/Dockerfile and must remain in the final image.
    local build_packages=(
        autoconf
        automake
        dh-apparmor
        libfuse3-dev
        liblzo2-dev
        liblz4-dev
        liblzma-dev
        libseccomp-dev
        libsubid-dev
        libtool
        libzstd-dev
        pkg-config
        zlib1g-dev
    )
    local runtime_packages=(
        build-essential
        ca-certificates
        cryptsetup
        fakeroot
        fuse3
        libfuse3-3
        liblzo2-2
        liblz4-1
        liblzma5
        libseccomp2
        libsubid4
        libzstd1
        squashfs-tools
        tzdata
        uidmap
        zlib1g
    )
    local new_build_packages=()
    local dependency_download_max_attempts=5
    local dependency_download_retry_delay=5
    local dependency_download_status
    local attempt

    for package in "${build_packages[@]}"; do
        if ! dpkg-query -W -f='${db:Status-Abbrev}' "${package}" 2>/dev/null | grep -q '^ii '; then
            new_build_packages+=("${package}")
        fi
    done

    apt-get update
    apt-get install -y --no-install-recommends "${runtime_packages[@]}" "${build_packages[@]}"

    install_go "${arch}" "${go_sha256}"

    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    wget --progress=dot:giga -O "${source_tarball}" "${source_url}"
    echo "${APPTAINER_SOURCE_SHA256}  ${source_tarball}" | sha256sum -c -
    tar -C "${build_dir}" --strip-components=1 -xzf "${source_tarball}"
    rm -f "${source_tarball}"

    # Anonymous GitHub patch downloads can return transient HTTP 429 responses.
    # The downloader cleans partial dependency files before every attempt.
    cd "${build_dir}"
    for ((attempt = 1; attempt <= dependency_download_max_attempts; attempt++)); do
        if ./scripts/download-dependencies; then
            break
        else
            dependency_download_status=$?
        fi

        if ((attempt == dependency_download_max_attempts)); then
            echo "Apptainer dependency download failed after ${attempt} attempts" >&2
            return "${dependency_download_status}"
        fi

        echo "Apptainer dependency download attempt ${attempt} failed; retrying in ${dependency_download_retry_delay}s" >&2
        sleep "${dependency_download_retry_delay}"
        dependency_download_retry_delay=$((dependency_download_retry_delay * 2))
    done
    ./scripts/compile-dependencies

    # Patch vendored dependency versions in place, keeping APPTAINER_VERSION
    # unchanged. `go mod tidy` resolves any transitive deps (e.g. x/net, which
    # grpc pulls in) to versions compatible with the bumped direct deps.
    go get "golang.org/x/crypto@v${GO_CRYPTO_VERSION}"
    go get "golang.org/x/text@v${GO_TEXT_VERSION}"
    go get "google.golang.org/grpc@v${GRPC_VERSION}"
    go get "github.com/sigstore/fulcio@v${FULCIO_VERSION}"
    go get "github.com/docker/docker@v${DOCKER_VERSION}"
    go get "github.com/containers/image/v5@v${CONTAINERS_IMAGE_VERSION}"
    go mod tidy
    go mod vendor

    ./mconfig --without-suid
    make -C builddir
    make -C builddir install
    ./scripts/install-dependencies
    rm -rf "${build_dir}"

    apt-get install -y --no-install-recommends "${runtime_packages[@]}"
    if ((${#new_build_packages[@]} > 0)); then
        apt-get purge -y --auto-remove "${new_build_packages[@]}"
    fi
    rm -rf /usr/local/go /root/go /root/.cache/go-build
}

case "${DEB_ARCH}" in
    amd64)
        install_from_source amd64 "${GO_AMD64_SHA256}"
        ;;
    arm64)
        install_from_source arm64 "${GO_ARM64_SHA256}"
        ;;
    *)
        echo "Unsupported architecture for Apptainer ${APPTAINER_VERSION}: ${DEB_ARCH}" >&2
        exit 1
        ;;
esac

ln -sf /usr/local/bin/apptainer /usr/bin/apptainer
ln -sf /usr/local/bin/apptainer /usr/bin/singularity
verify_install

apt-get clean
rm -rf /var/lib/apt/lists/*
