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

"""What prefix recovery needs from the worker that hosts it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nemo_rl.models.generation.prefix_recovery.progress import ProgressReader
from nemo_rl.models.generation.prefix_recovery.state import CaptureRegistry

if TYPE_CHECKING:
    from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource


@dataclass(frozen=True)
class PrefixRecoveryParts:
    """Inputs the host worker builds once in ``setup_token_capture``."""

    # Gym's capture core (an optional extra, so untyped); builds prefix and
    # chunk records and fails calls.
    capture: Any
    # Stages cut chunks and clears retired ones.
    sink: TQTokenSink
    # Fetches a restored call's staged chunks.
    staging_source: TQTokenSource
    # The worker's live calls; recovery reads them, the worker adds and removes.
    registry: CaptureRegistry
    # Checks that a cut ends on a whole character.
    tokenizer: Any
    # The rollout weight version new calls are stamped with.
    weight_version: Callable[[], int]
    # Bearer token the cut route requires.
    control_token: str | None
    # Bound on one cut, measured from its arrival; None waits indefinitely.
    control_timeout_s: float | None
    progress_reader: ProgressReader
