#!/bin/bash

set -eou pipefail

if ! command -v uv &> /dev/null; then
    echo "uv could not be found, please install it with 'pip install uv'"
    exit 1
fi

# setuptools, torch, psutil (required by flash-attn), ninja (enables parallel flash-attn build)
uv sync --link-mode symlink --locked --no-install-project
if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  VIRTUAL_ENV=$UV_PROJECT_ENVIRONMENT uv pip install ninja
else
  uv pip install ninja
fi
uv sync --link-mode symlink --locked --extra automodel --no-install-project
uv sync --link-mode symlink --locked --no-install-project
echo "✅ flash-attn successfully added to uv cache"
