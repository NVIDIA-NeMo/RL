#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Rebuild the Go c-shared library bundled in mooncake-transfer-engine wheels
# (libetcd_wrapper.so) with a patched Go toolchain and patched dependencies.
#
# Why: the upstream wheels ship libetcd_wrapper.so compiled with an older Go
# toolchain and etcd/grpc/x-net/x-text versions that carry known CVEs. The library
# is a tiny standalone Go module (mooncake-common/etcd/ in the Mooncake repo), so
# rather than rebuild the whole wheel this script rebuilds only that module from the
# matching upstream source tag and overwrites every installed copy in place.
#
# Usage: install_mooncake.sh <dir> [<dir> ...]
#   Every libetcd_wrapper*.so found below the given directories (venvs and the uv
#   cache) is rebuilt and replaced. Run it in the SAME Docker layer that creates
#   the venvs: files are overwritten in place, so venvs that hardlink the uv cache
#   all pick up the patched file through the shared inode, whereas patching in a
#   later layer would make overlayfs copy every path up separately.
#
# TODO: delete this script once upstream Mooncake wheels ship with these (or newer)
# versions.
set -euo pipefail
# Without this a failure inside a $(...) command substitution would not abort the script.
shopt -s inherit_errexit

GO_VERSION=1.26.6
GO_AMD64_SHA256=708effb774be8237570d0add163225abbdfaf4fca28b2611df167beba4feef89
GO_ARM64_SHA256=d0507e9e9d7fe012aae570108cbd76c15de879e17130ab8cb90d4d7445cb1f2e
ETCD_VERSION=3.5.33
GO_TEXT_VERSION=0.41.0
# gRPC 1.83.2 requires x/net >= 0.58.0.
GO_NET_VERSION=0.58.0
GRPC_VERSION=1.83.2

# Mooncake wheel versions this script knows how to patch, with the sha256 of the
# matching GitHub source tarball (v<version>). The build fails for any other
# installed version so that bumping the mooncake pin forces a conscious decision.
declare -A MOONCAKE_SOURCE_SHA256=(
    [0.3.11.post1]=13bf6529362b3587e150e6b25cc0085d52a6d8b300b2c1912f2b280fed2baa57
    [0.3.13.post1]=5c6ca3413c054cb3486e68775528ee7ffc305069c2656cf790c7637adee15308
)

if (($# == 0)); then
    echo "Usage: $0 <dir> [<dir> ...]" >&2
    exit 2
fi

case "$(dpkg --print-architecture)" in
    amd64) GO_ARCH=amd64; GO_SHA256="${GO_AMD64_SHA256}" ;;
    arm64) GO_ARCH=arm64; GO_SHA256="${GO_ARM64_SHA256}" ;;
    *) echo "Unsupported architecture: $(dpkg --print-architecture)" >&2; exit 1 ;;
esac

for tool in gcc curl tar sha256sum readelf nm jq; do
    command -v "${tool}" >/dev/null || { echo "${tool} is required to build libetcd_wrapper.so" >&2; exit 1; }
done

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
export GOTOOLCHAIN=local
export GOPATH="${WORK_DIR}/gopath"
export GOCACHE="${WORK_DIR}/gocache"
export GOFLAGS=-modcacherw

install_go() {
    local tarball="${WORK_DIR}/go.tar.gz"

    curl -fsSL --retry 3 -o "${tarball}" "https://go.dev/dl/go${GO_VERSION}.linux-${GO_ARCH}.tar.gz"
    echo "${GO_SHA256}  ${tarball}" | sha256sum -c -
    mkdir "${WORK_DIR}/goroot"
    tar -C "${WORK_DIR}/goroot" -xzf "${tarball}"
    export PATH="${WORK_DIR}/goroot/go/bin:${PATH}"
    go version
}

# Version of the mooncake wheel that owns a libetcd_wrapper*.so, read from the
# dist-info directory next to the wheel's mooncake/ and *.libs/ directories.
wheel_version() {
    local lib="$1"
    local site="${lib%/*}"
    local dist_info

    site="${site%/*}"
    dist_info="$(echo "${site}"/mooncake_transfer_engine*.dist-info)"
    if [[ ! -d "${dist_info}" ]]; then
        echo "no mooncake_transfer_engine dist-info next to ${lib}" >&2
        return 1
    fi
    dist_info="${dist_info##*/}"
    dist_info="${dist_info#mooncake_transfer_engine*-}"
    echo "${dist_info%.dist-info}"
}

# Exported C API of the library, excluding cgo's per-build hash-suffixed symbols.
exported_api() {
    nm -D --defined-only "$1" | awk '{print $3}' | grep -E '^(Etcd|Snapshot|New)' | sort
}

is_patched() {
    local build_info

    # The original auditwheel-processed copies carry Go build info that `go version -m`
    # cannot read; treat that as "not patched" rather than printing an error.
    build_info="$(go version -m "$1" 2>/dev/null)" || return 1
    grep -qF "go${GO_VERSION}" <<<"${build_info}" \
        && grep -qF "go.etcd.io/etcd/client/pkg/v3	v${ETCD_VERSION}" <<<"${build_info}" \
        && grep -qF "golang.org/x/text	v${GO_TEXT_VERSION}" <<<"${build_info}" \
        && grep -qF "golang.org/x/net	v${GO_NET_VERSION}" <<<"${build_info}" \
        && grep -qF "google.golang.org/grpc	v${GRPC_VERSION}" <<<"${build_info}"
}

download_source() {
    local version="$1"
    local src_dir="${WORK_DIR}/src-${version}"
    if [[ ! -d "${src_dir}" ]]; then
        if [[ -z "${MOONCAKE_SOURCE_SHA256[${version}]:-}" ]]; then
            echo "Mooncake ${version}: no verified source checksum available" >&2
            return 1
        fi
        local tarball="${WORK_DIR}/mooncake-${version}.tar.gz"
        curl -fsSL --retry 3 -o "${tarball}" \
            "https://github.com/kvcache-ai/Mooncake/archive/refs/tags/v${version}.tar.gz"
        echo "${MOONCAKE_SOURCE_SHA256[${version}]}  ${tarball}" | sha256sum -c - >&2
        mkdir "${src_dir}"
        tar -C "${src_dir}" --strip-components=1 -xzf "${tarball}"
    fi
}

prepare_source() {
    local version="$1"
    local src_dir="${WORK_DIR}/src-${version}"
    local resolved

    if [[ ! -d "${src_dir}" ]]; then
        download_source "${version}"
        (
            cd "${src_dir}/mooncake-common/etcd"
            go mod edit -toolchain="go${GO_VERSION}"
            # One `go get` so the pins are resolved jointly: a conflict fails here
            # instead of one pin silently downgrading another.
            go get \
                "go.etcd.io/etcd/api/v3@v${ETCD_VERSION}" \
                "go.etcd.io/etcd/client/pkg/v3@v${ETCD_VERSION}" \
                "go.etcd.io/etcd/client/v3@v${ETCD_VERSION}" \
                "golang.org/x/text@v${GO_TEXT_VERSION}" \
                "golang.org/x/net@v${GO_NET_VERSION}" \
                "google.golang.org/grpc@v${GRPC_VERSION}" >&2
            go mod tidy >&2
            resolved="$(go list -m \
                go.etcd.io/etcd/api/v3 go.etcd.io/etcd/client/pkg/v3 go.etcd.io/etcd/client/v3 \
                golang.org/x/text golang.org/x/net google.golang.org/grpc | tr ' ' '@')"
            for pin in \
                "go.etcd.io/etcd/api/v3@v${ETCD_VERSION}" \
                "go.etcd.io/etcd/client/pkg/v3@v${ETCD_VERSION}" \
                "go.etcd.io/etcd/client/v3@v${ETCD_VERSION}" \
                "golang.org/x/text@v${GO_TEXT_VERSION}" \
                "golang.org/x/net@v${GO_NET_VERSION}" \
                "google.golang.org/grpc@v${GRPC_VERSION}"; do
                if ! grep -qxF "${pin}" <<<"${resolved}"; then
                    echo "dependency resolution moved a pin; wanted ${pin}, got:" >&2
                    echo "${resolved}" >&2
                    exit 1
                fi
            done
        )
    fi
}

preserve_mooncake_license() {
    local version="$1" notice_dir="$2"
    local src_dir="${WORK_DIR}/src-${version}"
    local license_file
    local -a notices

    install -d "${notice_dir}"
    license_file="$(find "${src_dir}" -maxdepth 1 -type f \( -iname 'license*' -o -iname 'copying*' \) -print -quit)"
    if [[ -z "${license_file}" ]]; then
        echo "No Mooncake license file found for ${version}" >&2
        return 1
    fi
    mapfile -d '' -t notices < <(find "${src_dir}" -maxdepth 1 -type f \
        \( -iname 'license*' -o -iname 'notice*' -o -iname 'copying*' -o -iname 'copyright*' \) -print0)
    install -m 644 "${notices[@]}" "${notice_dir}/"
}

preserve_notices() {
    local version="$1" lib="$2" notice_dir="$3"
    local src_dir="${WORK_DIR}/src-${version}"
    local module module_version module_dir notice relative license_file
    local -a notices

    preserve_mooncake_license "${version}" "${notice_dir}"
    install -D -m 644 "$(go env GOROOT)/LICENSE" "${notice_dir}/go/LICENSE"

    # Enumerate the modules used by the wrapper, retaining their original
    # notice filenames and module paths to avoid attribution collisions.
    (
        cd "${src_dir}/mooncake-common/etcd"
        go list -deps -json etcd_wrapper.go \
            | jq -r 'select(.Module and (.Module.Main != true)) | [.Module.Path, .Module.Version, .Module.Dir] | @tsv' \
            | sort -u
    ) > "${WORK_DIR}/notice-modules-${version}.tsv"
    test -s "${WORK_DIR}/notice-modules-${version}.tsv"
    printf 'Module\tVersion\n' > "${notice_dir}/DEPENDENCIES.tsv"
    while IFS=$'\t' read -r module module_version module_dir; do
        test -d "${module_dir}"
        license_file="$(find "${module_dir}" -maxdepth 1 -type f \( -iname 'license*' -o -iname 'copying*' \) -print -quit)"
        if [[ -z "${license_file}" ]]; then
            echo "No license file found for ${module}@${module_version}" >&2
            return 1
        fi
        mapfile -d '' -t notices < <(find "${module_dir}" -maxdepth 1 -type f \
            \( -iname 'license*' -o -iname 'notice*' -o -iname 'copying*' -o -iname 'copyright*' -o -iname 'authors*' \) -print0)
        for notice in "${notices[@]}"; do
            relative="${notice##*/}"
            install -D -m 644 "${notice}" "${notice_dir}/dependencies/${module}/${module_version}/${relative}"
        done
        printf '%s\t%s\n' "${module}" "${module_version}" >> "${notice_dir}/DEPENDENCIES.tsv"
    done < "${WORK_DIR}/notice-modules-${version}.tsv"
    go version -m "${lib}" > "${notice_dir}/BUILD-INFO.txt"
    cat > "${notice_dir}/MODIFICATIONS.txt" <<EOF
NVIDIA NeMo-RL rebuild, $(date -u +%Y-%m-%d).
Mooncake ${version}: replaced libetcd_wrapper*.so with a rebuild using
Go ${GO_VERSION}, etcd ${ETCD_VERSION}, grpc ${GRPC_VERSION},
golang.org/x/net ${GO_NET_VERSION}, and golang.org/x/text ${GO_TEXT_VERSION}.
The wrapper's go.mod and go.sum were updated; etcd_wrapper.go is unchanged.
DEPENDENCIES.tsv records the resolved modules and BUILD-INFO.txt records the
library build metadata. The original wheel SBOM describes the upstream build.
Corresponding source is provided through the distribution's NOTICES.txt URL
(/opt/nemo-rl/NOTICES.txt in the release image); no local source archive is added.
EOF
    : > "${notice_dir}/REBUILT-LIBRARIES.txt"
}

# Build libetcd_wrapper.so for one mooncake version, optionally with a SONAME.
# Prints the path of the built library.
build_wrapper() {
    local version="$1"
    local soname="$2"
    local src_dir="${WORK_DIR}/src-${version}"
    local out="${WORK_DIR}/out-${version}-${soname:-nosoname}.so"
    local -a ldflags=()

    prepare_source "${version}"

    if [[ -n "${soname}" ]]; then
        ldflags=(-ldflags="-extldflags=-Wl,-soname,${soname}")
    fi
    (
        cd "${src_dir}/mooncake-common/etcd"
        # Same invocation as upstream's mooncake-common/etcd/CMakeLists.txt.
        go build -buildmode=c-shared "${ldflags[@]}" -o "${out}" etcd_wrapper.go >&2
    )
    echo "${out}"
}

mapfile -t libs < <(find "$@" -xdev -type f -name 'libetcd_wrapper*.so' | sort)
if ((${#libs[@]} == 0)); then
    # The upstream aarch64 wheels do not bundle libetcd_wrapper.so (and none of
    # their binaries embed Go), so there is nothing to patch there.
    if [[ "${GO_ARCH}" == arm64 ]]; then
        # Preserve upstream attribution even when this wheel has no Go library.
        declare -A wheel_notices_written
        mapfile -t dist_infos < <(find "$@" -xdev -type d -name 'mooncake_transfer_engine*.dist-info' | sort)
        for dist_info in "${dist_infos[@]}"; do
            version="${dist_info##*/}"
            version="${version#mooncake_transfer_engine*-}"
            version="${version%.dist-info}"
            if [[ -z "${wheel_notices_written[${version}]:-}" ]]; then
                download_source "${version}"
                preserve_mooncake_license "${version}" "/usr/local/share/licenses/mooncake/${version}"
                wheel_notices_written["${version}"]=1
            fi
        done
        echo "no libetcd_wrapper*.so found under: $*; nothing to patch on arm64"
        exit 0
    fi
    echo "no libetcd_wrapper*.so found under: $*" >&2
    exit 1
fi
echo "Found ${#libs[@]} libetcd_wrapper copies to patch"

# Validate everything before building anything.
declare -A lib_version
for lib in "${libs[@]}"; do
    lib_version["${lib}"]="$(wheel_version "${lib}")"
    if [[ -z "${MOONCAKE_SOURCE_SHA256[${lib_version[${lib}]}]:-}" ]]; then
        echo "${lib}: mooncake ${lib_version[${lib}]} is not supported by $0;" >&2
        echo "add its source tarball sha256 to MOONCAKE_SOURCE_SHA256 after verifying the patch" >&2
        exit 1
    fi
done

install_go

patched=0
for lib in "${libs[@]}"; do
    version="${lib_version[${lib}]}"
    soname="$(readelf -d "${lib}" | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p')"

    if is_patched "${lib}"; then
        continue
    fi

    new_lib="$(build_wrapper "${version}" "${soname}")"

    # The C++ side (engine.so, store.so, mooncake_master) is linked against this
    # library's C API, so refuse to replace it with anything that exports a different one.
    if ! diff <(exported_api "${lib}") <(exported_api "${new_lib}") >&2; then
        echo "${lib}: rebuilt library exports a different C API" >&2
        exit 1
    fi
    if ! is_patched "${new_lib}"; then
        echo "${new_lib}: rebuilt library does not embed the expected versions" >&2
        go version -m "${new_lib}" >&2
        exit 1
    fi

    # Overwrite in place (same inode) so hardlinked copies stay in sync.
    cat "${new_lib}" > "${lib}"
    patched=$((patched + 1))
    echo "patched ${lib} (mooncake ${version})"
done

declare -A notices_written
for lib in "${libs[@]}"; do
    is_patched "${lib}" || { echo "${lib} is not patched after rebuild" >&2; exit 1; }
    version="${lib_version[${lib}]}"
    notice_dir="/usr/local/share/licenses/mooncake/${version}"
    if [[ -z "${notices_written[${version}]:-}" ]]; then
        # Cache seeds may already contain rebuilt libraries. Still fetch their
        # matching source/module metadata so missing notices are restored.
        prepare_source "${version}"
        preserve_notices "${version}" "${lib}" "${notice_dir}"
        notices_written["${version}"]=1
    fi
    printf '%s\n' "${lib}" >> "${notice_dir}/REBUILT-LIBRARIES.txt"
done
echo "libetcd_wrapper: ${patched} patched, ${#libs[@]} verified (go${GO_VERSION}, etcd ${ETCD_VERSION}, x/net ${GO_NET_VERSION}, x/text ${GO_TEXT_VERSION}, grpc ${GRPC_VERSION})"
