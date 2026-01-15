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
"""NeMo RL utilities.

This module provides utility functions and classes for NeMo RL,
including error handling, logging, checkpointing, and more.
"""

from nemo_rl.utils.errors import (
    BackendError,
    CheckpointError,
    ConfigError,
    DataError,
    EnvironmentError,
    NeMoRLError,
    TrainingError,
    format_options,
    fuzzy_match,
    get_common_suggestion,
    get_doc_link,
)

__all__ = [
    # Error classes
    "NeMoRLError",
    "ConfigError",
    "BackendError",
    "TrainingError",
    "DataError",
    "EnvironmentError",
    "CheckpointError",
    # Error utilities
    "fuzzy_match",
    "format_options",
    "get_doc_link",
    "get_common_suggestion",
]
