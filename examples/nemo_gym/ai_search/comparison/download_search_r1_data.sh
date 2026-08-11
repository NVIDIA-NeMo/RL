#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

output_dir=$1
mkdir -p "${output_dir}"

download_and_verify() {
  local filename=$1
  local expected_sha256=$2
  local url=$3
  local destination="${output_dir}/${filename}"

  if [[ -f "${destination}" ]] && \
    echo "${expected_sha256}  ${destination}" | sha256sum --check --status; then
    echo "verified ${destination}"
    return
  fi

  curl --fail --location --retry 5 --continue-at - \
    --output "${destination}" "${url}"
  echo "${expected_sha256}  ${destination}" | sha256sum --check --status
  echo "verified ${destination}"
}

download_and_verify \
  train.parquet \
  c3cc21e862a8469105de666101578cbff23cdc77e91a803cef102622c89cc4f6 \
  'https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/resolve/main/train.parquet?download=true'

download_and_verify \
  test.parquet \
  30aa887b6d47e06e8c0f6f5307c88fe4e13461ac25a20ec0a5433ad7a4fe25dc \
  'https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/resolve/main/test.parquet?download=true'
