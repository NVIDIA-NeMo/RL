#!/bin/bash

set -eou pipefail

if ! command -v uv &> /dev/null; then
    echo "uv could not be found, please install it with 'pip install uv'"
    exit 1
fi

# setuptools, torch, psutil (required by flash-attn), ninja (enables parallel flash-attn build)
uv pip install setuptools torch==2.7.0 psutil ninja --torch-backend=cu128