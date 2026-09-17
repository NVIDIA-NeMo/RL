# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_rl.models.generation.vllm.config import VllmConfig
    from nemo_rl.models.policy import PolicyConfig


def resolve_worker_cls(
    default_cls: str,
    config: PolicyConfig | VllmConfig,
    *,
    quantized_cls: str | None,
) -> str:
    """Resolve the worker from config, reconciling a quantization requirement.

    Args:
        default_cls: Backend worker used without an extension or quantization.
        config: Worker config, updated in place for legacy quantization recipes.
        quantized_cls: Required worker for quantization, or None when disabled.

    Returns:
        The configured worker FQN, or the backend default when unset.

    Raises:
        ValueError: The configured worker does not match the quantization worker.
    """
    configured_cls = config.get("worker_extension_cls_fqn")
    if quantized_cls is not None:
        if configured_cls is None:
            warnings.warn(
                "quant_cfg requires "
                f"worker_extension_cls_fqn={quantized_cls!r}; setting it automatically. "
                "Set this field explicitly to silence this warning.",
                stacklevel=3,
            )
            config["worker_extension_cls_fqn"] = quantized_cls
        elif configured_cls != quantized_cls:
            raise ValueError(
                f"quant_cfg requires worker_extension_cls_fqn={quantized_cls!r}, "
                f"got {configured_cls!r}."
            )
    configured_cls = config.get("worker_extension_cls_fqn")
    return configured_cls if configured_cls is not None else default_cls
