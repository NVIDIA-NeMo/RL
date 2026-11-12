#!/usr/bin/env bash
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

# Build NeMo Gym server venvs once into an archive, and unpack that archive on every node.
#
# Keeps the venvs on each node's local disk instead of a shared filesystem.
# See "Python Environments on Shared Filesystems" in docs/cluster.md.
#
#   gym_venv_archive.sh build <archive.tar> <recipe config>...
#       Run inside the training image, from the NeMo RL checkout.
#       Builds the Gym server venvs that the configs use into /opt/gym_venvs (NEMO_GYM_VENV_DIR),
#       then writes <archive.tar>, <archive.tar>.servers and <archive.tar>.size.
#       Uses the same CUDA torch backend and vLLM wheel override as the image's own Gym prefetch step.
#   gym_venv_archive.sh unpack <archive.tar> [<file to touch on failure>]
#       Run as ray.sub's SETUP_COMMAND on every node, before Ray starts.
#       Replaces /opt/gym_venvs with the archive's venvs, or does nothing if it already holds them.
#       ray.sub ignores SETUP_COMMAND's exit status, so pass ray.sub's ENDED file to end the job on failure.

set -euo pipefail

VENV_DIR="${NEMO_GYM_VENV_DIR:-/opt/gym_venvs}"

usage() {
    sed -n '/^#   gym_venv_archive.sh/,/^$/p' "$0" >&2
    exit 2
}

build() {
    local archive="$1"
    shift
    [[ $# -gt 0 ]] || usage
    [[ "${archive}" == /* ]] || { echo "Error: the archive path must be absolute: ${archive}" >&2; exit 1; }
    local part=".partial.${SLURM_JOB_ID:-$$}"

    rm -rf "${VENV_DIR:?}"
    mkdir -p "${VENV_DIR}"

    # Same environment as docker/Dockerfile's Gym prefetch step: CUDA torch and vLLM's prebuilt CUDA wheel, so a
    # build on a node without GPUs does not pick CPU torch for vLLM-based Gym servers.
    local cuda="${NEMO_GYM_CUDA-cu130}"
    local vllm_version="${NEMO_GYM_VLLM_VERSION-0.29.0}"
    local override_file
    override_file="$(mktemp)"
    if [[ -n "${vllm_version}" ]]; then
        local wheel_arch
        case "$(uname -m)" in aarch64) wheel_arch=aarch64 ;; *) wheel_arch=x86_64 ;; esac
        printf 'vllm @ https://github.com/vllm-project/vllm/releases/download/v%s/vllm-%s-cp38-abi3-manylinux_2_28_%s.whl\n' \
            "${vllm_version}" "${vllm_version}" "${wheel_arch}" > "${override_file}"
    fi
    UV_TORCH_BACKEND="${cuda}" UV_OVERRIDE="${override_file}" \
        uv run --no-sync python examples/nemo_gym/prefetch_venvs.py "$@"
    rm -f "${override_file}"

    find "${VENV_DIR}" -mindepth 3 -maxdepth 3 -name .venv -printf '%P\n' | sed 's#/.venv$##' | sort \
        > "${archive}.servers${part}"
    [[ -s "${archive}.servers${part}" ]] || {
        echo "Error: no Gym server venvs in ${VENV_DIR}; does a config set env.nemo_gym.uv_venv_dir elsewhere?" >&2
        exit 1
    }
    tar -C "$(dirname "${VENV_DIR}")" -cf "${archive}${part}" "$(basename "${VENV_DIR}")"
    stat -c %s "${archive}${part}" > "${archive}.size${part}"
    mv "${archive}${part}" "${archive}"
    mv "${archive}.size${part}" "${archive}.size"
    mv "${archive}.servers${part}" "${archive}.servers"
    echo "Gym venvs: ${archive} ($(du -h "${archive}" | cut -f1)), servers:"
    sed 's/^/  /' "${archive}.servers"
}

unpack() {
    local archive="$1"
    local ended_file="${2:-}"
    local marker="${VENV_DIR}/.from-archive"
    [[ "$(cat "${marker}" 2>/dev/null)" == "${archive}" ]] && return 0

    local staging
    if staging="$(mktemp -d "$(dirname "${VENV_DIR}")/.gym_venvs.XXXXXX")" &&
        [[ "$(stat -c %s "${archive}" 2>/dev/null)" == "$(cat "${archive}.size" 2>/dev/null)" ]] &&
        tar -C "${staging}" -xf "${archive}" &&
        ! while read -r server; do
            [[ -x "${staging}/$(basename "${VENV_DIR}")/${server}/.venv/bin/python" ]] || echo "missing: ${server}"
        done < "${archive}.servers" | grep . >&2 &&
        rm -rf "${VENV_DIR}" &&
        mv "${staging}/$(basename "${VENV_DIR}")" "${VENV_DIR}" &&
        printf '%s\n' "${archive}" > "${marker}"; then
        rm -rf "${staging}"
        echo "Gym venvs: unpacked ${archive} into ${VENV_DIR}"
        return 0
    fi
    echo "Error: unpacking ${archive} into ${VENV_DIR} failed on $(hostname)." >&2
    echo "  Check that the archive is mounted at this path and that its .size and .servers files sit next to it." >&2
    [[ -n "${staging:-}" ]] && rm -rf "${staging}"
    [[ -n "${ended_file}" ]] && touch "${ended_file}"
    exit 1
}

case "${1:-}" in
    build) [[ $# -ge 3 ]] || usage; shift; build "$@" ;;
    unpack) [[ $# -ge 2 && $# -le 3 ]] || usage; shift; unpack "$@" ;;
    *) usage ;;
esac
