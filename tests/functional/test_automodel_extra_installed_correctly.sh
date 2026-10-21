#!/bin/bash
set -eoux pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd $SCRIPT_DIR

uv sync
# Just the first call with --extra automodel is invoked with --reinstall in case submodules were recently updated/downloaded
uv run --reinstall --extra automodel --no-build-isolation python <<"EOF"
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Test basic transformers functionality that automodel extends
config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")
print(f"Loaded config: {config.model_type}")

# Test nemo_automodel import
import nemo_automodel
from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
print("[NeMo Automodel import successful]")

# Test flash-attn import (part of automodel extra)
import flash_attn
print(f"[Flash Attention available: {flash_attn.__version__}]")

print("[Automodel extra dependencies test successful]")
EOF

# Test that automodel components can be accessed
uv run --extra automodel --no-build-isolation python <<"EOF"
# This must be the first import to get all of the automodel packages added to the path
import nemo_rl

# Test automodel utilities
from nemo_rl.models.automodel.checkpoint import AutomodelCheckpointManager
print("[Automodel checkpoint utilities import successful]")

# Test automodel factory
from nemo_rl.models.automodel.utils import AUTOMODEL_FACTORY
print(f"[Automodel factory available: {len(AUTOMODEL_FACTORY)} entries]")

print("[Automodel integration test successful]")
EOF

# Sync just to return the environment to the original base state
uv sync --link-mode symlink --locked --no-install-project
uv sync --link-mode symlink --locked --extra vllm --no-install-project
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --extra automodel --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project
echo Success
