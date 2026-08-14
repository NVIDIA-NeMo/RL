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

if [[ "$#" -lt 3 || "$#" -gt 5 ]]; then
    echo "Usage: $0 <test-level> <script-pattern> <platform> [unit-directory] [functional-directory]" >&2
    exit 2
fi

test_level=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
script_pattern=$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')
platform=$(printf '%s' "$3" | tr '[:upper:]' '[:lower:]')
unit_directory=${4:-tests/unit}
functional_directory=${5:-tests/functional}

case "$platform" in
    all | h100 | gb200) ;;
    *)
        echo "test_platform must be one of: all, h100, gb200." >&2
        exit 2
        ;;
esac

unit_scripts='[]'
h100_functional_scripts='[]'
gb200_functional_scripts='[]'
run_unit_tests=false
run_functional_tests=false

case "$test_level" in
    lfast | l0 | l1 | l2) run_unit_tests=true ;;
esac
case "$test_level" in
    lfast | l1 | l2 | megatron) run_functional_tests=true ;;
esac

if [[ "$platform" != "gb200" && "$run_unit_tests" == "true" ]]; then
    while IFS= read -r script_path; do
        script=$(basename "$script_path" .sh)
        script_lower=$(printf '%s' "$script" | tr '[:upper:]' '[:lower:]')

        if [[ -n "$script_pattern" && "$script_lower" != *"$script_pattern"* ]]; then
            continue
        fi

        unit_scripts=$(jq -cn --argjson scripts "$unit_scripts" --arg script "$script" '$scripts + [$script]')
    done < <(find "$unit_directory" -maxdepth 1 -type f -name 'L0_Unit*.sh' | sort)
fi

if [[ "$run_functional_tests" == "true" ]]; then
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
            h100_functional_scripts=$(jq -cn --argjson scripts "$h100_functional_scripts" --arg script "$script" '$scripts + [$script]')
        fi
        if [[ "$platform" != "h100" && "$test_level" != "lfast" ]]; then
            gb200_functional_scripts=$(jq -cn --argjson scripts "$gb200_functional_scripts" --arg script "$script" '$scripts + [$script]')
        fi
    done < <(find "$functional_directory" -maxdepth 1 -type f -name 'L1_Functional*.sh' | sort)
fi

unit_count=$(jq 'length' <<< "$unit_scripts")
h100_functional_count=$(jq 'length' <<< "$h100_functional_scripts")
gb200_functional_count=$(jq 'length' <<< "$gb200_functional_scripts")

if [[ "$run_unit_tests" == "true" || "$run_functional_tests" == "true" ]] &&
    ((unit_count + h100_functional_count + gb200_functional_count == 0)); then
    echo "No test scripts match the requested selection." >&2
    exit 1
fi

jq -cn \
    --argjson unit "$unit_scripts" \
    --argjson h100_functional "$h100_functional_scripts" \
    --argjson gb200_functional "$gb200_functional_scripts" \
    --argjson unit_count "$unit_count" \
    --argjson h100_functional_count "$h100_functional_count" \
    --argjson gb200_functional_count "$gb200_functional_count" \
    '{
        unit: $unit,
        h100_functional: $h100_functional,
        gb200_functional: $gb200_functional,
        unit_count: $unit_count,
        h100_functional_count: $h100_functional_count,
        gb200_functional_count: $gb200_functional_count
    }'
