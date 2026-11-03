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

"""Generation-worker interface and media-alignment regression tests."""

import inspect
from unittest.mock import Mock

import pytest
from logra import generation
from logra.generation import CuaGenerationWorkerImpl
from nemo_rl.data.captured_media import CapturedMedia
from nemo_rl.models.generation.openai_server_utils import PrefixSplice
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


def test_capture_override_accepts_every_base_keyword() -> None:
    base = inspect.signature(VllmAsyncGenerationWorkerImpl._capture_request_media)
    keywords = {
        name: None
        for name, param in base.parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }
    inspect.signature(CuaGenerationWorkerImpl._capture_request_media).bind(
        None, {}, **keywords
    )


@pytest.mark.parametrize("has_admission", [False, True])
@pytest.mark.parametrize("has_splice", [False, True])
def test_capture_forwards_core_keywords_and_aligns_only_uncaptured_splices(
    monkeypatch: pytest.MonkeyPatch, has_admission: bool, has_splice: bool
) -> None:
    worker = object.__new__(CuaGenerationWorkerImpl)
    engine_prompt = {"prompt_token_ids": [1, 2]}
    admission = object() if has_admission else None
    splice = (
        PrefixSplice(token_ids=[3, 4], model_cut_end=1, template_cut_start=1)
        if has_splice
        else None
    )
    captured = Mock(spec=CapturedMedia)
    capture = Mock(return_value=captured)
    align = Mock()
    monkeypatch.setattr(
        VllmAsyncGenerationWorkerImpl, "_capture_request_media", capture
    )
    monkeypatch.setattr(generation, "align_image_placeholders", align)

    result = worker._capture_request_media(
        engine_prompt,
        admission=admission,
        splice=splice,
        verify_retained_media=True,
    )

    capture.assert_called_once_with(
        engine_prompt,
        admission=admission,
        splice=splice,
        verify_retained_media=True,
    )
    assert result is captured
    if admission is None and splice is not None:
        align.assert_called_once_with(engine_prompt, splice.token_ids)
    else:
        align.assert_not_called()
