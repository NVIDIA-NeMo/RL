#!/bin/bash
set -eoux pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd $SCRIPT_DIR

uv sync
# Just the first call with --extra mcore is invoked with --reinstall in case submodules were recently updated/downloaded
uv run --reinstall --extra mcore --no-build-isolation python <<"EOF"
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048

# Initialize model and inputs.
model = te.Linear(in_features, out_features, bias=True)
inp = torch.randn(hidden_size, in_features, device="cuda")

# TODO: Disabling FP8 testing since CI machines may not support FP8
## Create an FP8 recipe. Note: All input args are optional.
#fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)
#
## Enable autocasting for the forward pass
#with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
#    out = model(inp)

out = model(inp)

loss = out.sum()
loss.backward()
print("[TE hello world succeessful]")
EOF

uv run --extra mcore --no-build-isolation python <<"EOF"
import is_megatron_installed
import is_megatron_bridge_installed
assert is_megatron_installed.INSTALLED, "Megatron is not installed. Please check if the submodule has been initialized. May need to run `git submodule update --init --recursive`"
assert is_megatron_bridge_installed.INSTALLED, "Megatron Bridge is not installed. Please check if the submodule has been initialized. May need to run `git submodule update --init --recursive`"

# This must be the first import to get all of the megatron non-core packages added to the path
import nemo_rl
import megatron.core
from megatron.training.utils import get_ltor_masks_and_position_ids
from megatron.bridge import AutoBridge
print("[Megatron-Core/Megatron-Bridge imports successful]")
EOF

# Sync just to return the environment to the original base state
uv sync --link-mode symlink --locked --no-install-project
uv sync --link-mode symlink --locked --extra vllm --no-install-project
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project
echo Success
