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
"""Offline conversion of Automodel DCP checkpoints to Hugging Face format."""

import os
from typing import Any, Optional

from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoTokenizer


def convert_dcp_to_hf(
    dcp_ckpt_path: str,
    hf_ckpt_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    overwrite: bool = False,
    hf_overrides: Optional[dict[str, Any]] = {},
) -> str:
    """Convert a Torch DCP checkpoint to a Hugging Face checkpoint.

    This is not an optimized utility. If checkpoint is too large, consider saving DCP during training
    and using this utility to convert to HF format.

    Args:
        dcp_ckpt_path: Checkpoint dir that contains model/, for example
            step_N/policy/weights. The DCP shards are read from <dcp_ckpt_path>/model.
        hf_ckpt_path: Path to save HF checkpoint.
        model_name_or_path: Model name or path for config.
        tokenizer_name_or_path: Tokenizer name or path.
        overwrite: Whether to overwrite existing checkpoint. Defaults to False.
        hf_overrides: Extra keyword arguments forwarded to AutoConfig.from_pretrained.

    Returns:
        Path to the saved HF checkpoint.

    Raises:
        FileExistsError: If HF checkpoint already exists and overwrite is False.
        FileNotFoundError: If <dcp_ckpt_path>/model/.metadata does not exist.
    """
    # Checkpoints are written at <ckpt_dir>/model. Check the input before creating
    # the output dir, so a bad path does not leave an empty dir that blocks a rerun.
    model_dir = os.path.join(dcp_ckpt_path, "model")
    if not os.path.exists(os.path.join(model_dir, ".metadata")):
        if os.path.exists(os.path.join(dcp_ckpt_path, ".metadata")):
            raise FileNotFoundError(
                f"{dcp_ckpt_path} holds DCP shards directly instead of under 'model/'. "
                "This is the DTensor v1 layout (NeMo RL v0.7 and older, with _v2 unset "
                "or false), which this converter no longer reads. Convert it with "
                "examples/converters/convert_dcp_to_hf.py from NeMo RL v0.7.x."
            )
        raise FileNotFoundError(
            f"No DCP .metadata file found in {model_dir}. Only checkpoints saved with "
            "model_save_format='torch_save' can be converted. 'safetensors' checkpoints "
            "are already Hugging Face format: use <dcp_ckpt_path>/model/consolidated "
            "(written when save_consolidated is 'final' or 'every')."
        )
    dcp_ckpt_path = model_dir

    if os.path.exists(hf_ckpt_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {hf_ckpt_path}. Delete it to run or set overwrite=True."
        )
    os.makedirs(hf_ckpt_path, exist_ok=True)

    weights_path = os.path.join(hf_ckpt_path, "pytorch_model.bin")
    dcp_to_torch_save(dcp_ckpt_path, weights_path)

    config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True, **hf_overrides
    )
    config.save_pretrained(hf_ckpt_path)

    # TODO: After the following PR gets merged:
    # https://github.com/NVIDIA-NeMo/RL/pull/148/files
    # tokenizer should be copied from policy/tokenizer/* instead of relying on the model name
    # We can expose a arg at the top level --tokenizer_path to plumb that through.
    # This is more stable than relying on the current NeMo-RL get_tokenizer() which can
    # change release to release.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=True
    )
    tokenizer.save_pretrained(hf_ckpt_path)

    return hf_ckpt_path
