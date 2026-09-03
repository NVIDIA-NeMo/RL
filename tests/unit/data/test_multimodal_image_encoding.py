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

from collections import Counter

import pytest
from PIL import Image

import nemo_rl.environments.nemo_gym_multimodal as nemo_gym_multimodal
from nemo_rl.data.multimodal_utils import (
    image_to_data_url,
    resolve_to_image,
)
from nemo_rl.environments.nemo_gym_multimodal import (
    normalize_media_in_examples,
)


def _example(*content_parts: dict) -> dict:
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": list(content_parts)}]
        }
    }


def _write_png(tmp_path, name: str, size: tuple[int, int]) -> str:
    path = tmp_path / name
    Image.new("RGB", size, color=(10, 20, 30)).save(path, format="PNG")
    return str(path)


class _CloseTrackingImage:
    def __init__(self) -> None:
        self.closed: bool = False

    def close(self) -> None:
        self.closed = True


def test_image_to_data_url_round_trips_through_resolve_to_image():
    url = image_to_data_url(Image.new("RGB", (4, 3)))
    assert url.startswith("data:image/png;base64,")
    assert resolve_to_image(url).size == (4, 3)


def test_resolve_to_image_accepts_file_scheme(tmp_path):
    path = _write_png(tmp_path, "img.png", (5, 6))
    assert resolve_to_image(f"file://{path}").size == (5, 6)
    assert resolve_to_image(path).size == (5, 6)


def test_normalize_media_encodes_local_image_paths_and_file_urls(tmp_path):
    plain = _write_png(tmp_path, "plain.png", (2, 2))
    file_url = "file://" + _write_png(tmp_path, "scheme.png", (3, 3))

    examples = [
        _example(
            {"type": "input_image", "image_url": plain},
            {"type": "input_image", "image_url": {"url": file_url}},
            {"type": "input_text", "text": "describe"},
        )
    ]
    normalize_media_in_examples(examples)

    parts = examples[0]["responses_create_params"]["input"][0]["content"]
    assert parts[0]["image_url"].startswith("data:image/png;base64,")
    assert parts[1]["image_url"].startswith("data:image/png;base64,")
    assert resolve_to_image(parts[0]["image_url"]).size == (2, 2)
    assert resolve_to_image(parts[1]["image_url"]).size == (3, 3)
    # Non-image parts are untouched.
    assert parts[2] == {"type": "input_text", "text": "describe"}


def test_normalize_media_passes_through_http_and_data_urls():
    data_url = image_to_data_url(Image.new("RGB", (2, 2)))
    examples = [
        _example(
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
            {"type": "input_image", "image_url": "http://example.com/dog.png"},
            {"type": "input_image", "image_url": data_url},
        )
    ]
    normalize_media_in_examples(examples)

    parts = examples[0]["responses_create_params"]["input"][0]["content"]
    assert parts[0]["image_url"] == "https://example.com/cat.png"
    assert parts[1]["image_url"] == "http://example.com/dog.png"
    assert parts[2]["image_url"] == data_url


def test_encode_images_deduplicates_sources_and_uses_a_bounded_thread_pool(
    tmp_path, monkeypatch
):
    first = _write_png(tmp_path, "first.png", (2, 3))
    second = _write_png(tmp_path, "second.png", (4, 5))
    expected = {
        source: nemo_gym_multimodal._encode_single_image_source(source)
        for source in (first, second)
    }
    examples = [
        _example(
            {"type": "input_image", "image_url": first},
            {"type": "image", "image": first},
            {"type": "image_url", "url": second},
        )
        for _ in range(16)
    ]

    resolve_calls = []
    original_resolve = nemo_gym_multimodal.resolve_to_image

    def tracking_resolve(source):
        resolve_calls.append(source)
        return original_resolve(source)

    observed_max_workers = []
    original_executor = nemo_gym_multimodal.ThreadPoolExecutor

    def tracking_executor(*args, **kwargs):
        observed_max_workers.append(kwargs.get("max_workers"))
        return original_executor(*args, **kwargs)

    monkeypatch.setattr(nemo_gym_multimodal, "resolve_to_image", tracking_resolve)
    monkeypatch.setattr(nemo_gym_multimodal, "ThreadPoolExecutor", tracking_executor)

    normalize_media_in_examples(examples)

    assert Counter(resolve_calls) == Counter({first: 1, second: 1})
    assert observed_max_workers == [
        nemo_gym_multimodal._NEMO_GYM_IMAGE_ENCODE_MAX_WORKERS
    ]
    for example in examples:
        parts = example["responses_create_params"]["input"][0]["content"]
        assert parts[0] == {
            "type": "input_image",
            "image_url": expected[first],
        }
        assert parts[1] == {
            "type": "input_image",
            "image_url": expected[first],
        }
        assert parts[2] == {
            "type": "input_image",
            "image_url": expected[second],
        }


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image", "image": "a.png", "url": "b.png"},
        {"type": "image"},
    ],
)
def test_normalize_media_requires_exactly_one_source_key(part):
    with pytest.raises(ValueError, match="requires exactly one"):
        normalize_media_in_examples([_example(part)])


@pytest.mark.parametrize(
    "source",
    ["", {"url": ""}],
)
def test_normalize_media_rejects_empty_urls(source):
    with pytest.raises(ValueError, match="requires a non-empty media URL"):
        normalize_media_in_examples(
            [_example({"type": "input_image", "image_url": source})]
        )


def test_normalize_media_preserves_input_image_file_id():
    part = {"type": "input_image", "file_id": "file-123", "detail": "high"}
    examples = [_example(part)]

    normalize_media_in_examples(examples)

    assert examples[0]["responses_create_params"]["input"][0]["content"][0] == {
        "type": "input_image",
        "file_id": "file-123",
        "detail": "high",
    }


def test_normalize_media_promotes_nested_image_detail():
    data_url = image_to_data_url(Image.new("RGB", (2, 2)))
    examples = [
        _example(
            {
                "type": "image",
                "image": {"url": data_url, "detail": "high"},
            }
        )
    ]

    normalize_media_in_examples(examples)

    assert examples[0]["responses_create_params"]["input"][0]["content"][0] == {
        "type": "input_image",
        "image_url": data_url,
        "detail": "high",
    }


def test_normalize_media_canonicalizes_local_video(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    encoded_url = "data:video/mp4;base64,dmlkZW8="
    monkeypatch.setattr(
        nemo_gym_multimodal,
        "video_path_to_data_url",
        lambda source: encoded_url,
    )
    examples = [_example({"type": "video", "url": str(video_path)})]

    normalize_media_in_examples(examples)

    assert examples[0]["responses_create_params"]["input"][0]["content"][0] == {
        "type": "input_video",
        "video_url": encoded_url,
    }


def test_encode_images_does_not_partially_mutate_on_error(tmp_path):
    existing = _write_png(tmp_path, "existing.png", (2, 3))
    missing = str(tmp_path / "missing.png")
    examples = [
        _example(
            {"type": "input_image", "image_url": existing},
            {"type": "input_image", "image_url": missing},
        )
    ]

    with pytest.raises(FileNotFoundError):
        normalize_media_in_examples(examples)

    parts = examples[0]["responses_create_params"]["input"][0]["content"]
    assert parts[0]["image_url"] == existing
    assert parts[1]["image_url"] == missing


def test_encode_single_image_source_closes_image_on_success(monkeypatch):
    image = _CloseTrackingImage()
    monkeypatch.setattr(nemo_gym_multimodal, "resolve_to_image", lambda _: image)
    monkeypatch.setattr(
        nemo_gym_multimodal,
        "image_to_data_url",
        lambda _: "data:image/png;base64,AA",
    )

    assert (
        nemo_gym_multimodal._encode_single_image_source("image.png")
        == "data:image/png;base64,AA"
    )
    assert image.closed


def test_encode_single_image_source_closes_image_on_error(monkeypatch):
    image = _CloseTrackingImage()
    monkeypatch.setattr(nemo_gym_multimodal, "resolve_to_image", lambda _: image)

    def fail_to_encode(_):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(nemo_gym_multimodal, "image_to_data_url", fail_to_encode)

    with pytest.raises(RuntimeError, match="encoding failed"):
        nemo_gym_multimodal._encode_single_image_source("image.png")
    assert image.closed


def test_normalize_media_is_a_noop_for_text_only_examples():
    examples = [_example({"type": "input_text", "text": "no images here"})]
    before = [
        dict(part)
        for part in examples[0]["responses_create_params"]["input"][0]["content"]
    ]
    assert normalize_media_in_examples(examples) is examples
    assert examples[0]["responses_create_params"]["input"][0]["content"] == before

    # Missing/oddly-shaped payloads must not raise.
    assert (
        normalize_media_in_examples([{}, {"responses_create_params": {}}]) is not None
    )
    assert normalize_media_in_examples([{"responses_create_params": {"input": "nope"}}])
