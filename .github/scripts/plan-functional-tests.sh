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

#!/bin/bash

set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
    echo "Usage: $0 <test-level> <script-pattern> <platform> [scripts-directory]" >&2
    exit 2
fi

test_level=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
script_pattern=$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')
platform=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')
scripts_directory=${4:-tests/functional}

case "$platform" in
    all | h100 | gb200) ;;
    *)
        echo "functional_test_platform must be one of: all, h100, gb200." >&2
        exit 2
        ;;
esac

if [[ "$test_level" == "lfast" && "$platform" == "gb200" ]]; then
    echo "functional_test_platform=gb200 is not supported for Lfast tests." >&2
    exit 2
fi

h100_scripts='[]'
gb200_scripts='[]'

while IFS= read -r script_path; do
    script=$(basename "$script_path" .sh)
    script_lower=$(printf '%s' "$script" | tr '[:upper:]' '[:lower:]')

    if [[ "$test_level" == "megatron" && "$script_lower" != *megatron* ]]; then
        continue
    fi
    if [[ -n "$script_pattern" && "$script_lower" != *"$script_pattern"* ]]; then
        continue
    fi

    if [[ "$platform" != "gb200" && "$script_lower" != *gb200* ]]; then
        h100_scripts=$(jq -cn --argjson scripts "$h100_scripts" --arg script "$script" '$scripts + [$script]')
    fi
    if [[ "$platform" != "h100" ]]; then
        gb200_scripts=$(jq -cn --argjson scripts "$gb200_scripts" --arg script "$script" '$scripts + [$script]')
    fi
done < <(find "$scripts_directory" -maxdepth 1 -type f -name 'L1_Functional*.sh' | sort)

h100_count=$(jq 'length' <<< "$h100_scripts")
gb200_count=$(jq 'length' <<< "$gb200_scripts")

if ((h100_count + gb200_count == 0)); then
    echo "No functional test scripts match the requested selection." >&2
    exit 1
fi
if [[ "$test_level" == "lfast" && "$h100_count" == "0" ]]; then
    echo "No H100 functional test scripts match the requested Lfast selection." >&2
    exit 1
fi

jq -cn \
    --argjson h100 "$h100_scripts" \
    --argjson gb200 "$gb200_scripts" \
    --argjson h100_count "$h100_count" \
    --argjson gb200_count "$gb200_count" \
    '{h100: $h100, gb200: $gb200, h100_count: $h100_count, gb200_count: $gb200_count}'
