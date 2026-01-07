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

"""Megatron setup utilities for NeMo RL."""

from nemo_rl.models.megatron.setup import (
    ModelAndOptimizerState,
    RuntimeConfig,
    handle_model_import,
    setup_distributed,
    setup_model_and_optimizer,
    setup_reference_model_state,
    validate_and_set_config,
    validate_model_paths,
    finalize_megatron_setup,
)

__all__ = [
    "ModelAndOptimizerState",
    "RuntimeConfig",
    "handle_model_import",
    "validate_and_set_config",
    "setup_distributed",
    "setup_model_and_optimizer",
    "setup_reference_model_state",
    "validate_model_paths",
    "finalize_megatron_setup",
]