#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Create the separate Harbor runtime the gym-native agent drives (upstream
# harbor==0.6.6 plus python-dotenv for the judge credentials file). The managed
# interpreter is placed on the shared filesystem so the venv also runs inside
# the training container. link_shared_assets.sh reuses an existing one instead.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
code_dir=$(cd "$script_dir/../.." && pwd)
data_root=${AVB_DATA_ROOT:-$code_dir/data/agentic_vbench}
venv=${AVB_HARBOR_VENV:-$data_root/harbor-venv}
command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
mkdir -p "$data_root/python"
export UV_PYTHON_INSTALL_DIR=$data_root/python
uv python install 3.13
uv venv --python 3.13 "$venv"
uv pip install --python "$venv/bin/python" "harbor==0.6.6" "python-dotenv>=1.0"
"$venv/bin/python" -c 'import importlib.metadata as m, dotenv; assert m.version("harbor") == "0.6.6"; print("harbor 0.6.6 ready")'
