# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""run.sub-side mllogger shim (training-side logging is mlperf_grpo_logging.py)."""

import os

from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper
from mlperf_logging import mllog

if log_file := os.environ.get("MLPERF_MLLOG_FILE"):
    mllog.config(filename=log_file)

mllogger = MLLoggerWrapper(PyTCommunicationHandler(), value=None)
