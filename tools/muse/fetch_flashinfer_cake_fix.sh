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

# Fetch the two FlashInfer Cake kernel packages that the flashinfer-python 0.7.3 wheel ships
# without their build.json manifest and CUDA sources (cake_muse_spark_norm_backend and
# cake_tp_moe_finalize_rmsnorm_backend). vLLM's Muse Spark model needs both; their Python files
# in the v0.7.3 tag are identical to the wheel's. Mount each <out_dir>/<package> over
# <vLLM worker venv>/lib/python3.*/site-packages/flashinfer/<package>; FlashInfer JIT-compiles them
# on first use.
#
#   bash tools/muse/fetch_flashinfer_cake_fix.sh <out_dir> [tag]   # default tag v0.7.3
set -euo pipefail
OUT=${1:?usage: fetch_flashinfer_cake_fix.sh <out_dir> [tag]}
TAG=${2:-v0.7.3}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git -C "$TMP" init -q
git -C "$TMP" remote add origin https://github.com/flashinfer-ai/flashinfer.git
git -C "$TMP" sparse-checkout set --no-cone \
  flashinfer/cake_muse_spark_norm_backend flashinfer/cake_tp_moe_finalize_rmsnorm_backend
git -C "$TMP" fetch -q --depth 1 --filter=blob:none origin "refs/tags/$TAG"
git -C "$TMP" checkout -q FETCH_HEAD
mkdir -p "$OUT"
for b in cake_muse_spark_norm_backend cake_tp_moe_finalize_rmsnorm_backend; do
  [[ -f $TMP/flashinfer/$b/build.json ]] || { echo "$TAG has no flashinfer/$b/build.json" >&2; exit 1; }
  rm -rf "${OUT:?}/$b"
  cp -r "$TMP/flashinfer/$b" "$OUT/$b"
done
echo "wrote $OUT/{cake_muse_spark_norm_backend,cake_tp_moe_finalize_rmsnorm_backend}"
