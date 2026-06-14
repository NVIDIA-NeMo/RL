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


import functools
import os
from pathlib import Path
from typing import Any

import modelopt.torch.quantization as mtq
import torch
import torch.nn as nn
from megatron.bridge.models.gpt_provider import transformer_engine_layer_spec
from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec
from megatron.training.config import target_allowlist
from modelopt.torch.quantization.config import normalize_quant_cfg_list
from modelopt.torch.utils.dataset_utils import (
    create_forward_loop,
    get_dataset_dataloader,
)
from modelopt.torch.utils.plugins import megatron_prefill
from torch.utils.data import DataLoader, Dataset

from nemo_rl.algorithms.utils import get_tokenizer as _base_get_tokenizer
from nemo_rl.modelopt.utils import resolve_quant_cfg

MAX_SEQ_LEN = 2048
MAX_OUTPUT_LEN = 512


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


def _pad_or_truncate_input_ids(
    input_ids: torch.Tensor,
    *,
    max_length: int | None = None,
    multiple: int = 1,
    pad_token_id: int = 0,
) -> torch.Tensor:
    if max_length is not None and input_ids.shape[-1] > max_length:
        input_ids = input_ids[..., :max_length]

    if multiple <= 1:
        return input_ids.contiguous()

    seq_len = input_ids.shape[-1]
    padded_len = ((seq_len + multiple - 1) // multiple) * multiple
    pad_len = padded_len - seq_len
    if pad_len == 0:
        return input_ids.contiguous()

    padding = torch.full(
        (*input_ids.shape[:-1], pad_len),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    return torch.cat((input_ids, padding), dim=-1).contiguous()


def get_forward_loop_func(
    is_megatron: bool,
    calib_dataloader: DataLoader,
):
    """Gets the forward loop function for the model."""
    if not is_megatron:
        return create_forward_loop(dataloader=calib_dataloader)

    from megatron.core import parallel_state

    cp_size = parallel_state.get_context_parallel_world_size()
    pad_multiple = max(1, 2 * cp_size)

    def _forward_loop(model):
        for batch in calib_dataloader:
            input_ids = _pad_or_truncate_input_ids(
                batch["input_ids"],
                multiple=pad_multiple,
            )
            megatron_prefill(model, input_ids, skip_return_logits=True)

    return _forward_loop


def _requires_forward_calibration(config: dict[str, Any]) -> bool:
    """Return whether this config needs data-driven activation calibration."""
    algorithm = config.get("algorithm")
    if algorithm is not None and algorithm != "max":
        return True

    def _needs_stats(cfg: dict[str, Any]) -> bool:
        if not cfg.get("enable", True):
            return False
        if cfg.get("type") == "dynamic":
            return False
        return not cfg.get("use_constant_amax", False)

    for entry in normalize_quant_cfg_list(config.get("quant_cfg") or []):
        name = entry["quantizer_name"]
        if "weight_quantizer" in name:
            continue

        raw_cfg = entry.get("cfg")
        if isinstance(raw_cfg, list):
            if any(_needs_stats(dict(cfg)) for cfg in raw_cfg):
                return True
            continue

        cfg = dict(raw_cfg or {})
        if "enable" in entry:
            cfg["enable"] = entry["enable"]
        if _needs_stats(cfg):
            return True

    return False


def quantize_model(
    model: nn.Module,
    quant_cfg: str,
    tokenizer,
    calib_size,
    is_megatron: bool = False,
    batch_size=32,
    data=None,
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
    mtq_cfg = resolve_quant_cfg(quant_cfg)
    use_calibration = _requires_forward_calibration(mtq_cfg)
    if not use_calibration:
        forward_loop = None
    else:
        if data is None:
            raise ValueError(
                "policy.quant_calib_data is required by this quantization config."
            )
        device = (
            model.device
            if hasattr(model, "device")
            else next(model.parameters()).device
        )
        if data == "random":
            calib_size = 1
            calib_dataloader = DataLoader(
                _DictDataset(
                    {"input_ids": torch.randint(0, 100, (1, 5), device=device)}
                ),
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
        forward_loop = get_forward_loop_func(is_megatron, calib_dataloader)

    mtq.quantize(model, mtq_cfg, forward_loop)


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


def get_quantization_layer_spec():
    """Build the ModelOpt quantization transformer-layer spec as a portable target.

    Returns `partial` function to allow Megatron-Bridge's allowlist to accept the layer spec.
    """
    if int(os.environ.get("DISABLE_MODELOPT_LAYER_SPEC", "0")):
        return transformer_engine_layer_spec
    return functools.partial(
        get_gpt_modelopt_spec,
        local_core_attention=False,
        remap_te_layernorm=True,
        real_quant_cfg="None",
        use_arbitrary_attention_mask=False,
    )


target_allowlist.add_exact(
    "nemo_rl.modelopt.models.policy.workers.utils.quantization_layer_spec"
)
