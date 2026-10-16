#!/bin/bash
set -euo pipefail

APPTAINER_VERSION=1.4.5
APPTAINER_SOURCE_SHA256=d323a8b9a0a9e5e131b396d0049fdaa99beceb83a3d7ffb80dd91d15331e3b9a
GO_VERSION=1.26.6
GO_AMD64_SHA256=708effb774be8237570d0add163225abbdfaf4fca28b2611df167beba4feef89
GO_ARM64_SHA256=d0507e9e9d7fe012aae570108cbd76c15de879e17130ab8cb90d4d7445cb1f2e
# Bumping APPTAINER_VERSION itself is not an option here (functionality
# regressions on a newer release), so instead this script builds the SAME
# 1.4.5 release from source on both architectures and patches only the
# vendored dependency versions via `go get` + `go mod tidy` + `go mod vendor`
# before compiling -- same technique Megatron-Bridge uses to patch
# wandb-core's vendored deps
# (see 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/docker/Dockerfile.fw_final).
GO_CRYPTO_VERSION=0.55.0
# x/crypto 0.55.0 requires x/text >= 0.41.0; pinning x/text lower makes the
# later `go get` silently downgrade x/crypto (and x/net) to match.
GO_TEXT_VERSION=0.41.0
# Only used for the bundled gocryptfs helper (see patch_gocryptfs).
GO_SYS_VERSION=0.49.0
GO_TERM_VERSION=0.45.0
GRPC_VERSION=1.83.2
FULCIO_VERSION=1.8.5
REKOR_VERSION=1.5.2
DOCKER_CLI_VERSION=29.3.0
# `go mod tidy` re-resolves the WHOLE dependency graph, not just the modules
# above -- left unconstrained, it cascade-upgrades
# github.com/docker/docker, github.com/containers/image/v5 and
# github.com/google/go-containerregistry to versions whose APIs
# (archive.Compression, client.ImageLoadOption, docker/docker/api) apptainer
# 1.4.5's vendored source doesn't compile against. Pin docker/docker and
# go-containerregistry with `replace` directives rather than `go get`:
# fulcio >= 1.8.x requires docker/docker 28.x, so a later `go get
# docker/docker@27.5.1` silently DOWNGRADES fulcio (and sigstore) instead of
# holding docker back, whereas `replace` pins only the one module.
# docker/docker 27.5.1 / go-containerregistry 0.20.3 are the versions known to
# compile; github.com/moby/moby GHSA-x744-4wpc-v9h2 (CVE-2026-34040) is a
# dockerd AuthZ-plugin bypass that apptainer, only a client of this module,
# does not contain, and no fixed release exists under the docker/docker path.
DOCKER_VERSION=27.5.1
GO_CONTAINERREGISTRY_VERSION=0.20.3
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
    # Never let `go get` / a go.mod `toolchain` line download a different Go.
    export GOTOOLCHAIN=local
}

# apptainer bundles gocryptfs as a separate upstream project. It is downloaded
# (and vendored) by scripts/download-dependencies and built by
# scripts/compile-dependencies from its OWN go.mod, so the `go get` patching of
# apptainer's go.mod below never reaches it. Patch its vendored deps here, i.e.
# after download and before compile (compile runs with GOPROXY=off -mod=vendor).
patch_gocryptfs() {
    local tarball
    local dir

    tarball="$(echo gocryptfs-*.tar.gz)"
    if [[ ! -f "${tarball}" ]]; then
        echo "gocryptfs-*.tar.gz not found in $(pwd)" >&2
        return 1
    fi
    dir="${tarball%.tar.gz}"

    tar -xzf "${tarball}"
    (
        cd "${dir}"
        go get "golang.org/x/crypto@v${GO_CRYPTO_VERSION}"
        go get "golang.org/x/sys@v${GO_SYS_VERSION}"
        go get "golang.org/x/term@v${GO_TERM_VERSION}"
        go mod tidy
        go mod vendor
    )
    tar -czf "${tarball}" "${dir}"
    rm -rf "${dir}"
}

# Fail the image build, rather than the downstream scanner, if a shipped Go
# binary embeds a toolchain or module version other than the pinned one.
assert_go_binary() {
    local binary="$1"
    shift
    local build_info
    local expected

    build_info="$(go version -m "${binary}")"
    for expected in "go${GO_VERSION}" "$@"; do
        if ! grep -qF -- "${expected}" <<<"${build_info}"; then
            echo "${binary}: expected '${expected}' in Go build info, got:" >&2
            echo "${build_info}" >&2
            return 1
        fi
    done
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
    patch_gocryptfs
    ./scripts/compile-dependencies

    # Patch vendored dependency versions in place, keeping APPTAINER_VERSION
    # unchanged. `go mod tidy` resolves any transitive deps (e.g. x/net, which
    # grpc pulls in) to versions compatible with the bumped direct deps.
    go mod edit \
        -replace "github.com/docker/docker=github.com/docker/docker@v${DOCKER_VERSION}+incompatible" \
        -replace "github.com/google/go-containerregistry=github.com/google/go-containerregistry@v${GO_CONTAINERREGISTRY_VERSION}"
    go get "golang.org/x/crypto@v${GO_CRYPTO_VERSION}"
    go get "golang.org/x/text@v${GO_TEXT_VERSION}"
    go get "google.golang.org/grpc@v${GRPC_VERSION}"
    go get "github.com/sigstore/fulcio@v${FULCIO_VERSION}"
    go get "github.com/sigstore/rekor@v${REKOR_VERSION}"
    go get "github.com/containers/image/v5@v${CONTAINERS_IMAGE_VERSION}"
    go get "github.com/docker/cli@v${DOCKER_CLI_VERSION}+incompatible"
    go mod tidy
    go mod vendor

    ./mconfig --without-suid
    make -C builddir
    make -C builddir install
    ./scripts/install-dependencies

    assert_go_binary /usr/local/bin/apptainer \
        "golang.org/x/crypto	v${GO_CRYPTO_VERSION}" \
        "golang.org/x/text	v${GO_TEXT_VERSION}" \
        "google.golang.org/grpc	v${GRPC_VERSION}" \
        "github.com/sigstore/fulcio	v${FULCIO_VERSION}" \
        "github.com/sigstore/rekor	v${REKOR_VERSION}" \
        "github.com/docker/cli	v${DOCKER_CLI_VERSION}+incompatible"
    assert_go_binary /usr/local/libexec/apptainer/bin/starter \
        "golang.org/x/crypto	v${GO_CRYPTO_VERSION}"
    assert_go_binary /usr/local/libexec/apptainer/bin/gocryptfs \
        "golang.org/x/crypto	v${GO_CRYPTO_VERSION}" \
        "golang.org/x/sys	v${GO_SYS_VERSION}"
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
