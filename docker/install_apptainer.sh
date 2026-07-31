#!/bin/bash
set -euo pipefail

APPTAINER_VERSION=1.4.5
APPTAINER_DEB_SHA256=70f19af846501acfbc2e42e7cfeee9ee11ddbbfa1c3502d0d99cde34e8e0af05
APPTAINER_SOURCE_SHA256=d323a8b9a0a9e5e131b396d0049fdaa99beceb83a3d7ffb80dd91d15331e3b9a
GO_VERSION=1.23.6
GO_ARM64_SHA256=561c780e8f4a8955d32bf72e46af0b5ee5e0debe1e4633df9a03781878219202
DEB_ARCH="$(dpkg --print-architecture)"
export DEBIAN_FRONTEND=noninteractive

install_amd64_deb() {
    local deb_path="/tmp/apptainer_${APPTAINER_VERSION}_amd64.deb"
    local deb_url="https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}_amd64.deb"

    apt-get update
    apt-get install -y --no-install-recommends ca-certificates wget
    wget --progress=dot:giga -O "${deb_path}" "${deb_url}"
    echo "${APPTAINER_DEB_SHA256}  ${deb_path}" | sha256sum -c -
    apt-get install -y --no-install-recommends "${deb_path}"
    rm -f "${deb_path}"
}

install_go_arm64() {
    local go_tarball="/tmp/go${GO_VERSION}.linux-arm64.tar.gz"

    wget --progress=dot:giga -O "${go_tarball}" "https://go.dev/dl/go${GO_VERSION}.linux-arm64.tar.gz"
    echo "${GO_ARM64_SHA256}  ${go_tarball}" | sha256sum -c -
    rm -rf /usr/local/go
    tar -C /usr/local -xzf "${go_tarball}"
    rm -f "${go_tarball}"
    export PATH="/usr/local/go/bin:${PATH}"
}

verify_install() {
    apptainer --version
    singularity --version
}

install_arm64_from_source() {
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

    install_go_arm64

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
        install_amd64_deb
        ln -sf /usr/bin/apptainer /usr/bin/singularity
        verify_install
        ;;
    arm64)
        install_arm64_from_source
        ln -sf /usr/local/bin/apptainer /usr/bin/apptainer
        ln -sf /usr/local/bin/apptainer /usr/bin/singularity
        verify_install
        ;;
    *)
        echo "Unsupported architecture for Apptainer ${APPTAINER_VERSION}: ${DEB_ARCH}" >&2
        exit 1
        ;;
esac

apt-get clean
rm -rf /var/lib/apt/lists/*
