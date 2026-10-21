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

"""CUA media alignment at NeMo RL's generation-worker extension point."""

from typing import Any

import ray

from logra.osworld import align_image_placeholders
from nemo_rl.data.captured_media import CapturedMedia
from nemo_rl.models.generation.openai_server_utils import PrefixSplice
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


class CuaGenerationWorkerImpl(VllmAsyncGenerationWorkerImpl):
    def _capture_request_media(
        self,
        engine_prompt: dict[str, Any],
        *,
        admission: Any | None,
        splice: PrefixSplice | None = None,
    ) -> CapturedMedia | None:
        if admission is None and splice is not None:
            align_image_placeholders(engine_prompt, splice.token_ids)
        return super()._capture_request_media(
            engine_prompt, admission=admission, splice=splice
        )


@ray.remote
class CuaGenerationWorker(CuaGenerationWorkerImpl):  # pragma: no cover
    pass
