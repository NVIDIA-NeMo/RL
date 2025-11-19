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

"""Type definitions for automodel training framework."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from transformers import AutoConfig

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@dataclass(frozen=True)
class RuntimeConfig:
    """Unified runtime configuration for model training and inference.

    This replaces ValidatedState with a cleaner, more intuitive structure
    that groups all runtime settings in one place.
    """

    # Model architecture properties
    is_reward_model: bool
    is_vlm: bool
    is_hf_model: bool
    is_moe_model: bool
    model_class: type
    model_config: AutoConfig
    hf_config_overrides: dict[str, Any]

    # Attention configuration
    allow_flash_attn_args: bool
    attn_impl: Optional[str]

    # Training/inference settings
    dtype: torch.dtype
    enable_seq_packing: bool
    max_grad_norm: float

    # Memory management
    cpu_offload: bool = False
    offload_optimizer_for_logprob: bool = False

    # Generation configuration
    is_generation_colocated: Optional[bool] = None


@dataclass
class ProcessedInputs:
    """Processed microbatch inputs ready for model forward pass.

    This structure contains all necessary tensors and metadata for a forward pass,
    including context parallel buffers and flash attention configuration.
    """

    # Core inputs (always present)
    input_ids: torch.Tensor
    seq_len: int

    # Optional tensors (None when not applicable)
    attention_mask: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None

    # Flash attention configuration
    flash_attn_kwargs: dict[str, Any] = field(default_factory=dict)

    # Multimodal (VLM) inputs
    vlm_kwargs: dict[str, Any] = field(default_factory=dict)

    # Context parallel support (cp_size > 1)
    cp_buffers: list[torch.Tensor] = field(default_factory=list)
    seq_index: Optional[torch.Tensor] = None

    @property
    def has_context_parallel(self) -> bool:
        """Check if context parallel is enabled."""
        return len(self.cp_buffers) > 0

    @property
    def has_flash_attention(self) -> bool:
        """Check if flash attention is configured."""
        return len(self.flash_attn_kwargs) > 0

    @property
    def is_multimodal(self) -> bool:
        """Check if this is a multimodal input."""
        return len(self.vlm_kwargs) > 0


@dataclass
class LossInputs:
    """Everything needed to compute loss.

    Groups together microbatch data, loss function, normalization factors,
    and temperature processing function.
    """

    microbatch: BatchedDataDict[Any]
    loss_fn: LossFunction
    global_valid_seqs: torch.Tensor
    global_valid_toks: torch.Tensor
    apply_temperature_fn: Callable[[torch.Tensor], torch.Tensor]
