#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Pull a maintainer-published immutable NGC image; do not install dependencies.
set -euo pipefail
IMAGE_REF=${1:?Usage: fetch_image.sh nvcr.io/org/image@sha256:digest /absolute/image.sqsh}
OUTPUT=${2:?Pass the destination SQSH path}
[[ "$IMAGE_REF" =~ ^nvcr.io/[A-Za-z0-9_./-]+@sha256:[a-f0-9]{64}$ ]] || {
    echo 'Use the exact nvcr.io digest from the validated release manifest' >&2; exit 2;
}
[[ "$OUTPUT" == /* && "$OUTPUT" == *.sqsh ]] || exit 2
[[ ! -e "$OUTPUT" && ! -e "$OUTPUT.partial" ]] || {
    echo 'Destination exists; choose a new filename' >&2; exit 1;
}
command -v enroot >/dev/null
mkdir -p "$(dirname "$OUTPUT")"
REGISTRY=${IMAGE_REF%%/*}
REPOSITORY=${IMAGE_REF#*/}
enroot import --output "$OUTPUT.partial" "docker://${REGISTRY}#${REPOSITORY}"
mv "$OUTPUT.partial" "$OUTPUT"
sha256sum "$OUTPUT" > "$OUTPUT.sha256"
printf '%s\n' "$IMAGE_REF" > "$OUTPUT.source.txt"
echo "Image ready: $OUTPUT"
