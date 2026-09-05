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

import os

LAYER_QUANTIZATION_LOG_ENV = "NRL_LOG_LAYER_QUANTIZATION"
FP8_QUANTIZATION_IGNORE_DUMP_ENV = "NRL_DUMP_FP8_QUANTIZATION_IGNORE"
FP8_QUANTIZATION_IGNORE_DUMP_PATH_ENV = "NRL_DUMP_FP8_QUANTIZATION_IGNORE_PATH"

TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def is_truthy_env_var(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUTHY_ENV_VALUES
