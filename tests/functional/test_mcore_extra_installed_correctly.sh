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
import is_nemo_installed
assert is_megatron_installed.INSTALLED, "Megatron is not installed. Please check if the submodule has been initialized. May need to run `git submodule update --init --recursive`"
assert is_nemo_installed.INSTALLED, "NeMo is not installed. Please check if the submodule has been initialized. May need to run `git submodule update --init --recursive`"

# This must be the first import to get all of the megatron non-core packages added to the path
import nemo_rl
import megatron.core
from megatron.training.utils import get_ltor_masks_and_position_ids
from nemo.tron.init import initialize_megatron
from nemo.tron.config import (
    ConfigContainer,
    TrainingConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    CheckpointConfig,
    DistributedDataParallelConfig,
)
from nemo.tron.utils.common_utils import get_rank_safe
from nemo.tron.config import TokenizerConfig
from nemo.tron.model import get_model_from_config
from nemo.tron.checkpointing import checkpoint_exists, load_checkpoint
from nemo.tron.init import initialize_megatron, set_jit_fusion_options
from nemo.tron.setup import _init_checkpointing_context, _update_model_config_funcs
from nemo.tron.state import GlobalState
from nemo.tron.optim import setup_optimizer
from nemo.tron import fault_tolerance
from nemo.tron.tokenizers.tokenizer import build_tokenizer
from nemo.tron.utils.train_utils import (
    calc_params_l2_norm,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from nemo.tron.train import train_step
from nemo.tron.setup import HAVE_FSDP2
print("[Nemo/Mcore imports successful]")
EOF

# Sync just to return the environment to the original base state
uv sync
echo Success
