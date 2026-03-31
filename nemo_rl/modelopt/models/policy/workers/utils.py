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

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from megatron.bridge.models.gpt_provider import transformer_engine_layer_spec
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from modelopt.torch.quantization.config import need_calibration
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from modelopt.torch.utils.plugins import megatron_prefill
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from nemo_rl.modelopt.utils import resolve_quant_cfg

MAX_SEQ_LEN = 2048
MAX_OUTPUT_LEN = 512


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN, trust_remote_code=False):
    """Returns the tokenizer from the model ckpt_path."""
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    if type(tokenizer).__name__ == "QWenTokenizer":
        # qwen use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class _DictDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

    def __len__(self):
        return len(next(iter(self.data.values())))


def get_forward_loop_func(
    is_megatron: bool,
    calib_dataloader: DataLoader,
):
    """Gets the forward loop function for the model."""
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
    batch_size=32,
    data="cnn_dailymail",
    max_sample_length=1024,
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
    device = (
        model.device if hasattr(model, "device") else next(model.parameters()).device
    )
    if data == "random":
        calib_size = 1
        calib_dataloader = DataLoader(
            _DictDataset({"input_ids": torch.randint(0, 100, (1, 5), device=device)}),
            batch_size=1,
        )
    else:
        calib_dataloader = get_dataset_dataloader(
            dataset_name=data,
            tokenizer=tokenizer,
            batch_size=batch_size,
            num_samples=calib_size,
            device=device,
            include_labels=False,
            max_sample_length=max_sample_length,
        )

    mtq_cfg = resolve_quant_cfg(quant_cfg)

    forward_loop = None
    use_calibration = need_calibration(mtq_cfg)
    if not use_calibration:
        print("Dynamic quantization. Calibration skipped.")
    else:
        forward_loop = get_forward_loop_func(is_megatron, calib_dataloader)

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


def quantization_layer_spec(config):
    """Layer specification for quantization with ModelOpt.

    We need to disable arbitrary attention mask for sequence packing.
    """
    disable_modelopt_layer_spec = int(
        os.environ.get("DISABLE_MODELOPT_LAYER_SPEC", "0")
    )
    if disable_modelopt_layer_spec:
        return transformer_engine_layer_spec(config)
    print("Using quantization_layer_spec without arbitrary attention mask")
    return get_gpt_modelopt_spec(
        config=config,
        local_core_attention=False,
        remap_te_layernorm=True,
        real_quant_cfg="None",
        use_arbitrary_attention_mask=False,
    )
