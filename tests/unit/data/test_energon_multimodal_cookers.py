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

from dataclasses import FrozenInstanceError
from io import BytesIO

import pytest
from PIL import Image

# cookers.generic imports megatron.energon at module scope, and it ships only in
# the `mcore` extra. importorskip must run before that import: the mcore mark is
# applied in pytest_collection_modifyitems, too late to prevent a collection
# error.
pytest.importorskip("megatron.energon")

pytestmark = pytest.mark.mcore

from nemo_rl.data.energon.multimodal.cookers.generic import (  # noqa: E402
    cook_conversation,
)


def _sample(payload, **members):
    return {
        "__key__": "sample-0",
        "__restore_key__": ("sample-0",),
        "__subflavors__": {},
        "__sources__": (),
        "json": payload,
        **members,
    }


def test_generic_cooker_freezes_explicit_media_metadata_without_opening_media():
    media_value = object()
    cooked = cook_conversation(
        _sample(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "media_index": 0}],
                    }
                ],
                "media": [
                    {
                        "type": "image",
                        "value": media_value,
                        "metadata": {"width": 32, "height": 16},
                    }
                ],
            }
        )
    )

    assert cooked.media[0].value is media_value
    assert cooked.media[0].metadata == (("height", 16), ("width", 32))
    with pytest.raises(FrozenInstanceError):
        cooked.media[0].metadata = ()


@pytest.mark.parametrize("encoded", [False, True])
def test_generic_cooker_derives_missing_image_dimensions(encoded):
    image = Image.new("RGB", (32, 16))
    if encoded:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        value = buffer.getvalue()
    else:
        value = image
    cooked = cook_conversation(
        _sample(
            {
                "messages": [
                    {"role": "user", "content": [{"type": "image", "media_index": 0}]}
                ],
                "media": [
                    {"type": "image", "value": value, "metadata": {"source": "test"}}
                ],
            }
        )
    )
    assert cooked.media[0].value is value
    assert dict(cooked.media[0].metadata) == {
        "width": 32,
        "height": 16,
        "source": "test",
    }


def test_generic_cooker_preserves_partial_explicit_dimensions():
    image = Image.new("RGB", (32, 16))
    metadata = {"width": 64}
    cooked = cook_conversation(
        _sample(
            {
                "messages": [{"role": "user", "content": "image"}],
                "media": [{"type": "image", "value": image, "metadata": metadata}],
            }
        )
    )
    assert dict(cooked.media[0].metadata) == {"width": 64, "height": 16}
    assert metadata == {"width": 64}


def test_generic_cooker_resolves_lazy_image_without_replacing_reference():
    class LazyImage:
        def get(self, sample):
            return (Image.new("RGB", (32, 16)),)

    value = LazyImage()
    cooked = cook_conversation(
        _sample(
            {
                "messages": [{"role": "user", "content": "image"}],
                "media": [{"type": "image", "value": value}],
            }
        )
    )
    assert cooked.media[0].value is value
    assert dict(cooked.media[0].metadata) == {"width": 32, "height": 16}
