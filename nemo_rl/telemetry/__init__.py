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

"""NeMo-RL telemetry: optional OpenTelemetry instrumentation via nemo-lens.

Public surface:

* :class:`~nemo_rl.telemetry.config.TelemetryConfig` — the ``telemetry:`` config
  block.
* :class:`~nemo_rl.telemetry.span_groups.RLSpanGroup` — RL span-group presets.
* :func:`~nemo_rl.telemetry.setup.init_telemetry_driver` /
  :func:`~nemo_rl.telemetry.setup.init_telemetry_worker` /
  :func:`~nemo_rl.telemetry.setup.get_telemetry` /
  :func:`~nemo_rl.telemetry.setup.shutdown_telemetry` — lifecycle helpers.

The instrumentation primitives (``managed_span`` / ``trace_fn`` / ``span_cm`` /
``is_span_group_enabled`` / ``safe_set_span_attributes``) come from
:mod:`nemo_rl.telemetry._fallbacks`, which re-exports the real nemo-lens
implementations when it is installed and no-op stubs when it is not. Importing
this package never requires nemo-lens.
"""

from nemo_rl.telemetry.config import TelemetryConfig
from nemo_rl.telemetry.setup import (
    get_telemetry,
    init_telemetry_driver,
    init_telemetry_worker,
    shutdown_telemetry,
)
from nemo_rl.telemetry.span_groups import RLSpanGroup

__all__ = [
    "TelemetryConfig",
    "RLSpanGroup",
    "get_telemetry",
    "init_telemetry_driver",
    "init_telemetry_worker",
    "shutdown_telemetry",
]
