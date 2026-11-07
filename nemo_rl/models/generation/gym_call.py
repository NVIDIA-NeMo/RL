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

"""The extra fields NeMo RL and NeMo-Gym exchange on a chat call, defined once.

The native vLLM server reads :class:`GymCallExtras` as request fields and
answers with :class:`GymCallResult`'s fields on the message. Dynamo's frontend
carries only OpenAI fields plus ``nvext``, so through it NeMo-Gym's Dynamo
dialect packs the extras into one ``nemo_rl:gym:<json>`` annotation and NeMo
RL's chat engine returns the result as ``nvext["nemo_rl"]``. A field added here
reaches both paths.
"""

import json
from collections.abc import Iterable
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

# The annotation NeMo-Gym's Dynamo dialect carries the extras in.
GYM_EXTRAS_ANNOTATION_PREFIX = "nemo_rl:gym:"
# The response nvext key the call result rides in through Dynamo.
GYM_RESULT_NVEXT_KEY = "nemo_rl"


class GymCallExtras(BaseModel):
    """What a NeMo-Gym chat call adds to an OpenAI request."""

    # A field Gym sends that NeMo RL does not know is refused, not ignored.
    model_config = ConfigDict(extra="forbid")

    # The token-capture admission (capture_key, call ids, prev_len, mode, ...).
    ng_capture: Optional[dict[str, Any]] = None
    # Without capture: the last echoed assistant turn's prompt + generation ids.
    required_prefix_token_ids: Optional[list[int]] = None
    # Return only the prompt's length (Gym's /context/measure); no generation.
    ng_measure: bool = False

    @classmethod
    def from_annotations(cls, annotations: Iterable[str]) -> "GymCallExtras | None":
        """The extras in ``nvext.annotations``; None when the call carries none."""
        for annotation in annotations or ():
            if annotation.startswith(GYM_EXTRAS_ANNOTATION_PREFIX):
                return cls.model_validate(
                    json.loads(annotation[len(GYM_EXTRAS_ANNOTATION_PREFIX) :])
                )
        return None


class GymCallResult(BaseModel):
    """What a NeMo-Gym chat call returns beyond the OpenAI response.

    The native server puts these on ``choices[0].message`` (token fields,
    ``routed_experts``) or the response (``ng_commit_coords``); field names are
    the native ones, so Gym lifts them without renaming.
    """

    # A generation call without token capture.
    prompt_token_ids: Optional[list[int]] = None
    generation_token_ids: Optional[list[int]] = None
    generation_log_probs: Optional[list[float]] = None
    # Router replay: the wire-encoded routes, as the native message carries them.
    routed_experts: Optional[Any] = None
    # A captured call: where its record was staged.
    ng_commit_coords: Optional[dict[str, Any]] = None
    # A measure-only call.
    prompt_token_count: Optional[int] = None
    ng_context_measured: Optional[bool] = None
