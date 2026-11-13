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

# The CUDA torch backend and vLLM wheel that Gym server venvs are built with.
#
# Sourced by docker/Dockerfile's Gym prefetch step and by gym_venv_archive.sh, so the image and an archive
# build the same environments from one set of defaults.
# A caller's NEMO_GYM_CUDA or NEMO_GYM_VLLM_VERSION takes precedence.
# An empty NEMO_GYM_VLLM_VERSION writes no vLLM override.

NEMO_GYM_CUDA="${NEMO_GYM_CUDA-cu130}"
NEMO_GYM_VLLM_VERSION="${NEMO_GYM_VLLM_VERSION-0.29.0}"

# gym_write_vllm_override <file> [arch]
# Writes a uv override that pins vllm to vLLM's prebuilt CUDA wheel, or an empty file.
# arch is a Docker TARGETARCH or uname -m value; it defaults to this machine's.
gym_write_vllm_override() {
    local out="$1"
    local arch="${2:-$(uname -m)}"
    local wheel_arch
    : > "${out}"
    [[ -n "${NEMO_GYM_VLLM_VERSION}" ]] || return 0
    case "${arch}" in
        arm64 | aarch64) wheel_arch=aarch64 ;;
        *) wheel_arch=x86_64 ;;
    esac
    printf 'vllm @ https://github.com/vllm-project/vllm/releases/download/v%s/vllm-%s-cp38-abi3-manylinux_2_28_%s.whl\n' \
        "${NEMO_GYM_VLLM_VERSION}" "${NEMO_GYM_VLLM_VERSION}" "${wheel_arch}" > "${out}"
}
