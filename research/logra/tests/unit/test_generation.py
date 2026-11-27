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

"""Generation-worker interface and exact-token media regression tests."""

from dataclasses import dataclass

import pytest
from logra.generation import CuaGenerationWorker, CuaGenerationWorkerImpl

from nemo_rl.models.generation.openai_server_utils import PrefixSplice
from nemo_rl.models.generation.vllm.utils import remap_multimodal_placeholders
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorker,
    VllmAsyncGenerationWorkerImpl,
)


def test_cua_uses_native_worker_without_a_second_media_remap() -> None:
    assert CuaGenerationWorker is VllmAsyncGenerationWorker
    assert CuaGenerationWorkerImpl is VllmAsyncGenerationWorkerImpl


@pytest.mark.parametrize("history_length", [0, 1, 3, 5])
def test_uncaptured_continuation_remaps_images_once(history_length: int) -> None:
    @dataclass
    class ImageSpan:
        offset: int
        length: int

    image = [17, 18, 18, 19]
    rendered_prefix = [1, *image, 5]
    exact_prefix = [1, *image, *([5] * history_length)]
    template = [*rendered_prefix, *image, 9]
    final = [*exact_prefix, *image, 9]
    spans = [ImageSpan(1, len(image)), ImageSpan(len(rendered_prefix), len(image))]
    placeholders = {"image": spans}
    prompt = {"prompt_token_ids": template, "mm_placeholders": placeholders}
    splice = PrefixSplice(
        token_ids=final,
        model_cut_end=len(exact_prefix),
        template_cut_start=len(rendered_prefix),
    )
    worker = object.__new__(CuaGenerationWorkerImpl)

    media = worker._capture_request_media(
        prompt, admission=None, splice=splice, verify_retained_media=True
    )

    assert media is None
    # The caller remaps uncaptured requests after this hook returns. The old
    # override shifted spans here too, causing out-of-bounds ranges on growth.
    assert prompt["prompt_token_ids"] == template
    assert prompt["mm_placeholders"] is placeholders
    assert [span.offset for span in spans] == [1, len(rendered_prefix)]
    remapped = remap_multimodal_placeholders(
        template_token_ids=template,
        final_token_ids=final,
        mm_placeholders=placeholders,
        splice=splice,
    )
    assert [span.offset for span in remapped["image"]] == [1, len(exact_prefix)]
    for span in remapped["image"]:
        assert final[span.offset : span.offset + span.length] == image
