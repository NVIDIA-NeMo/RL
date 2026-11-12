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

# Which uv extras (backends) an image builds. Sourced by docker/Dockerfile so
# the base stage, the dependency layer and the TRT-LLM layer all agree.

# Every extra the image can build, in the order the main venv warms them.
NRL_ALL_EXTRAS="vllm sglang mcore automodel modelopt nemo_gym trtllm"

# Print the extras to build, space-separated: NRL_EXTRAS (default: all of them)
# minus any that a SKIP_<NAME>_BUILD arg turns off.
nrl_wanted_extras() {
    local wanted="${NRL_EXTRAS:-$NRL_ALL_EXTRAS}" extra out=""
    for extra in $wanted; do
        case " $NRL_ALL_EXTRAS " in
            *" $extra "*) ;;
            *)
                echo "NRL_EXTRAS names an unknown extra: '$extra' (known: $NRL_ALL_EXTRAS)" >&2
                return 1
                ;;
        esac
    done
    for extra in $NRL_ALL_EXTRAS; do
        case " $wanted " in
            *" $extra "*) ;;
            *) continue ;;
        esac
        if [ "$extra" = vllm ] && [ -n "${SKIP_VLLM_BUILD:-}" ]; then continue; fi
        if [ "$extra" = sglang ] && [ -n "${SKIP_SGLANG_BUILD:-}" ]; then continue; fi
        if [ "$extra" = trtllm ] && [ -n "${SKIP_TRTLLM_BUILD:-}" ]; then continue; fi
        out="$out $extra"
    done
    echo "${out# }"
}

# Print the extras NOT to build, space-separated: the list
# nemo_rl/distributed/actor_environments.py takes to drop actors that need them.
nrl_skipped_extras() {
    local wanted extra out=""
    wanted="$(nrl_wanted_extras)" || return 1
    for extra in $NRL_ALL_EXTRAS; do
        case " $wanted " in
            *" $extra "*) ;;
            *) out="$out $extra" ;;
        esac
    done
    echo "${out# }"
}
