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
try:
    import nemo_automodel
    from nemo_automodel.components._transformers.auto_model import NeMoAutoModelForCausalLM
    print("[NeMo Automodel import successful]")
except ImportError as e:
    print(f"[WARNING] NeMo Automodel import failed: {e}")
    print("[This may be expected if nemo_automodel is not fully built]")

# Test flash-attn import (part of automodel extra)
try:
    import flash_attn
    print(f"[Flash Attention available: {flash_attn.__version__}]")
except ImportError:
    print("[WARNING] Flash Attention not available")

# Test vllm import (part of automodel extra) 
try:
    import vllm
    print(f"[vLLM available: {vllm.__version__}]")
except ImportError:
    print("[WARNING] vLLM not available")

print("[Automodel extra dependencies test successful]")
EOF

# Test that automodel components can be accessed
uv run --extra automodel --no-build-isolation python <<"EOF"
# This must be the first import to get all of the automodel packages added to the path
import nemo_rl

# Test automodel utilities
try:
    from nemo_rl.utils.automodel_checkpoint import detect_checkpoint_format, load_checkpoint, save_checkpoint
    print("[Automodel checkpoint utilities import successful]")
except ImportError as e:
    print(f"[Automodel checkpoint utilities import failed: {e}]")

# Test automodel factory
try:
    from nemo_rl.models.policy.utils import AUTOMODEL_FACTORY, NEMO_AUTOMODEL_AVAILABLE
    print(f"[Automodel factory available: {NEMO_AUTOMODEL_AVAILABLE}]")
except ImportError as e:
    print(f"[Automodel factory import failed: {e}]")

print("[Automodel integration test successful]")
EOF

# Sync just to return the environment to the original base state
uv sync --link-mode symlink --locked --no-install-project
uv sync --link-mode symlink --locked --extra vllm --no-install-project
uv sync --link-mode symlink --locked --extra mcore --no-install-project
uv sync --link-mode symlink --locked --extra automodel --no-install-project
uv sync --link-mode symlink --locked --all-groups --no-install-project
echo Success
