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

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 3 ]]; then
    echo "Usage: $0 <script-pattern> [unit-directory] [functional-directory]" >&2
    exit 2
fi

script_pattern=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
unit_directory=${2:-tests/unit}
functional_directory=${3:-tests/functional}

unit_scripts='[]'
functional_scripts='[]'

while IFS= read -r script_path; do
    script=$(basename "$script_path" .sh)
    script_lower=$(printf '%s' "$script" | tr '[:upper:]' '[:lower:]')

    if [[ -n "$script_pattern" && "$script_lower" != *"$script_pattern"* ]]; then
        continue
    fi

    unit_scripts=$(jq -cn --argjson scripts "$unit_scripts" --arg script "$script" '$scripts + [$script]')
done < <(find "$unit_directory" -maxdepth 1 -type f -name 'L0_Unit*.sh' | sort)

while IFS= read -r script_path; do
    script=$(basename "$script_path" .sh)
    script_lower=$(printf '%s' "$script" | tr '[:upper:]' '[:lower:]')

    if [[ -n "$script_pattern" && "$script_lower" != *"$script_pattern"* ]]; then
        continue
    fi

    functional_scripts=$(jq -cn --argjson scripts "$functional_scripts" --arg script "$script" '$scripts + [$script]')
done < <(find "$functional_directory" -maxdepth 1 -type f -name 'L1_Functional*.sh' | sort)

unit_count=$(jq 'length' <<< "$unit_scripts")
functional_count=$(jq 'length' <<< "$functional_scripts")

if ((unit_count + functional_count == 0)); then
    echo "No test scripts match the requested selection." >&2
    exit 1
fi

jq -cn \
    --argjson unit "$unit_scripts" \
    --argjson functional "$functional_scripts" \
    --argjson unit_count "$unit_count" \
    --argjson functional_count "$functional_count" \
    '{
        unit: $unit,
        functional: $functional,
        unit_count: $unit_count,
        functional_count: $functional_count
    }'
