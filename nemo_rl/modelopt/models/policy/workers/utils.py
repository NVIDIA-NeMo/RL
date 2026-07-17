# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


import os
from functools import partial
from pathlib import Path

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from megatron.bridge.models.gpt_provider import transformer_engine_layer_spec
from megatron.bridge.models.mamba.mamba_provider import (
    modelopt_mamba_stack_spec,
    transformer_engine_mamba_stack_spec,
)
from megatron.core import parallel_state
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from modelopt.torch.utils.plugins import (
    get_megatron_calibration_forward_loop,
    megatron_prefill,
)
from torch.utils.data import DataLoader, Dataset

from nemo_rl.algorithms.utils import get_tokenizer as _base_get_tokenizer
from nemo_rl.modelopt.utils import resolve_quant_cfg

MAX_SEQ_LEN = 2048
MAX_OUTPUT_LEN = 512
# Calibration dataloader defaults used when the optional
# policy.quant_batch_size / policy.quant_sequence_length config keys are unset.
DEFAULT_CALIB_BATCH_SIZE = 32
DEFAULT_CALIB_SAMPLE_LENGTH = 1024


def symlink_pre_quantized_model(src: str, pretrained_path: str) -> None:
    """Symlink an external pre-quantized checkpoint as ``<pretrained_path>/iter_0000000``."""
    iter0_path = Path(pretrained_path) / "iter_0000000"
    absolute_src = Path(src).resolve()
    iter0_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(absolute_src.as_posix(), iter0_path.as_posix(), target_is_directory=True)
    assert iter0_path.exists(), f"Symlink target does not exist: {absolute_src}"
    print(f"Using pre-quantized model at: {absolute_src}")


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN):
    """Returns a tokenizer configured for ModelOpt calibration.

    Wraps :func:`nemo_rl.algorithms.utils.get_tokenizer` and applies the
    extra configuration needed for batched calibration forward passes:
    ``padding_side="left"`` and ``model_max_length`` truncation.
    """
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = _base_get_tokenizer({"name": ckpt_path})
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = max_seq_len
    return tokenizer


class _DictDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    def __len__(self):
        return len(next(iter(self.data.values())))


def get_forward_loop_func(
    *,
    is_megatron: bool,
    tokenizer,
    dataset_name: str,
    batch_size: int,
    num_samples: int,
    sample_length: int,
    device: torch.device,
):
    """Build the calibration forward loop for the requested backend and data."""
    if is_megatron and dataset_name == "random":
        # The upstream helper owns CP/DP sharding for named datasets. The local
        # synthetic loop has no tokenizer/dataset for that helper to partition.
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1:
            raise RuntimeError(
                "Random ModelOpt Megatron calibration requires "
                f"context_parallel_size=1, got {cp_size}; use a named dataset"
            )

    if dataset_name != "random" and is_megatron:
        return get_megatron_calibration_forward_loop(
            tokenizer,
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_samples=num_samples,
            seq_length=sample_length,
            device=device,
            apply_chat_template=False,
            pack=True,
        )

    if dataset_name == "random":
        calib_dataloader = DataLoader(
            _DictDataset(
                {
                    "input_ids": torch.randint(
                        0,
                        100,
                        (num_samples, sample_length),
                        device=device,
                    )
                }
            ),
            batch_size=batch_size,
        )
    else:
        calib_dataloader = get_dataset_dataloader(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_samples=num_samples,
            device=device,
            include_labels=False,
            max_sample_length=sample_length,
        )

    if not is_megatron:
        return create_forward_loop(dataloader=calib_dataloader)

    def _forward_loop(model):
        for batch in calib_dataloader:
            megatron_prefill(model, batch["input_ids"], skip_return_logits=True)

    return _forward_loop


def quantize_model(
    model: nn.Module,
    quant_cfg: str,
    tokenizer,
    calib_size,
    is_megatron: bool = False,
    batch_size=None,
    data=None,
    max_sample_length=None,
):
    """Quantizes the model with the provided calibration dataset.

    Args:
        model: the model to be quantized.
        quant_cfg: the quantization algorithm config name if simple quantization is used.
                   the list of quantization algorithm config names if auto quantization is used.
        tokenizer: the tokenizer.
        batch_size: the calibration batch size for each calibration inference run.
        calib_size: the total calibration dataset size.
        auto_quantize_bits: The effective bits constraint for auto_quantize.
        data: the name of the calibration dataset.
    """
    mtq_cfg = resolve_quant_cfg(quant_cfg)
    use_calibration = need_calibration(mtq_cfg)
    if not use_calibration:
        print("Dynamic quantization. Calibration skipped.")
        forward_loop = None
    else:
        if data is None:
            raise ValueError(
                "policy.quant_calib_data is required by this quantization config."
            )
        if calib_size is None:
            raise ValueError(
                "policy.quant_calib_size is required by this quantization config."
            )
        device = (
            model.device
            if hasattr(model, "device")
            else next(model.parameters()).device
        )
        calib_batch_size = (
            batch_size if batch_size is not None else DEFAULT_CALIB_BATCH_SIZE
        )
        calib_sample_length = (
            max_sample_length
            if max_sample_length is not None
            else DEFAULT_CALIB_SAMPLE_LENGTH
        )
        forward_loop = get_forward_loop_func(
            is_megatron=is_megatron,
            tokenizer=tokenizer,
            dataset_name=data,
            batch_size=calib_batch_size,
            num_samples=calib_size,
            sample_length=calib_sample_length,
            device=device,
        )

    model = mtq.quantize(model, mtq_cfg, forward_loop)
    mtq.print_quant_summary(model)


def get_modelopt_checkpoint_dir() -> str:
    """Gets the default modelopt checkpoint directory.

    1. Use NRL_MODELOPT_CHECKPOINT_DIR environment variable if set.
    2. Use HF_HOME/nemo_rl if HF_HOME is set.
    3. Use ~/.cache/huggingface/nemo_rl if neither are set.
    """
    nrl_modelopt_checkpoint_dir = os.environ.get("NRL_MODELOPT_CHECKPOINT_DIR")
    if nrl_modelopt_checkpoint_dir is not None and nrl_modelopt_checkpoint_dir.strip():
        modelopt_checkpoint_dir = nrl_modelopt_checkpoint_dir
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home is not None and hf_home.strip():
            modelopt_checkpoint_dir = os.path.join(hf_home, "nemo_rl")
        else:
            modelopt_checkpoint_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "nemo_rl"
            )
    print(f"Using default modelopt checkpoint dir: {modelopt_checkpoint_dir}")
    return modelopt_checkpoint_dir


def get_quantization_layer_spec(
    disable_modelopt_layer_spec: bool = False,
):
    """Return a checkpoint-serializable GPT layer-spec callback.

    Megatron-Bridge serializes ``transformer_layer_spec`` as the callback's
    importable target. Return Megatron functions/partials directly so saved
    configs stay within the built-in target allowlist. The partial keeps the
    sequence-packing requirement of disabling arbitrary attention masks.
    """
    if disable_modelopt_layer_spec:
        return transformer_engine_layer_spec
    return partial(
        get_gpt_modelopt_spec,
        local_core_attention=False,
        remap_te_layernorm=True,
        real_quant_cfg="None",
        use_arbitrary_attention_mask=False,
    )


def get_quantization_mamba_stack_spec(
    disable_modelopt_layer_spec: bool = False,
):
    """Return a checkpoint-serializable Mamba stack-spec callback."""
    if disable_modelopt_layer_spec:
        return transformer_engine_mamba_stack_spec
    return modelopt_mamba_stack_spec
