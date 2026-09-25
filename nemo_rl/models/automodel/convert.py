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
        dcp_ckpt_path (str): Path to DCP checkpoint
        hf_ckpt_path (str): Path to save HF checkpoint
        model_name_or_path (str): Model name or path for config
        tokenizer_name_or_path (str, optional): Tokenizer name or path.
                                               Defaults to model_name_or_path if None.
        overwrite (bool, optional): Whether to overwrite existing checkpoint. Defaults to False.

    Returns:
        str: Path to the saved HF checkpoint

    Raises:
        FileExistsError: If HF checkpoint already exists and overwrite is False
    """
    if os.path.exists(hf_ckpt_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {hf_ckpt_path}. Delete it to run or set overwrite=True."
        )
    os.makedirs(hf_ckpt_path, exist_ok=True)

    # Checkpoints are written at <ckpt_dir>/model.
    dcp_ckpt_path = os.path.join(dcp_ckpt_path, "model")
    if not os.path.exists(os.path.join(dcp_ckpt_path, ".metadata")):
        raise FileNotFoundError(f"No metadata file found in {dcp_ckpt_path}.")

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
