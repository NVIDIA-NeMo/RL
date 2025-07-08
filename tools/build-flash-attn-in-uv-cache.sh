#!/bin/bash

set -eou pipefail

if ! command -v uv &> /dev/null; then
    echo "uv could not be found, please install it with 'pip install uv'"
    exit 1
fi

# setuptools, torch, psutil (required by flash-attn), ninja (enables parallel flash-attn build)
uv sync --no-install-project
uv pip install ninja
uv sync --extra automodel --no-install-project
uv sync --no-install-project
echo "✅ flash-attn successfully added to uv cache"
