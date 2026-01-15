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
"""NeMo RL Infrastructure Module.

This module provides infrastructure abstractions including:
- Resource management for GPU allocation
- Unified logging interface
- Checkpoint management

These abstractions simplify distributed training setup and provide
consistent interfaces across different backends.
"""

from nemo_rl.infra.checkpointing import (
    CheckpointBackend,
    CheckpointError,
    CheckpointFormat,
    CheckpointManager,
    CheckpointMetadata,
)
from nemo_rl.infra.logging import (
    LoggerFacade,
    LogLevel,
    configure_logging,
    create_logger_from_config,
    get_logger,
    log_debug,
    log_error,
    log_info,
    log_warning,
)
from nemo_rl.infra.resources import (
    AllocationError,
    Resource,
    ResourceAllocation,
    ResourceManager,
    ResourceType,
)

__all__ = [
    # Resources
    "ResourceManager",
    "ResourceAllocation",
    "Resource",
    "ResourceType",
    "AllocationError",
    # Logging
    "LoggerFacade",
    "LogLevel",
    "configure_logging",
    "create_logger_from_config",
    "get_logger",
    "log_info",
    "log_debug",
    "log_warning",
    "log_error",
    # Checkpointing
    "CheckpointManager",
    "CheckpointFormat",
    "CheckpointError",
    "CheckpointMetadata",
    "CheckpointBackend",
]
