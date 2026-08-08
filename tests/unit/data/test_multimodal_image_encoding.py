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

from PIL import Image

from nemo_rl.data.multimodal_utils import (
    encode_images_in_examples,
    image_to_data_url,
    resolve_to_image,
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


def test_image_to_data_url_round_trips_through_resolve_to_image():
    url = image_to_data_url(Image.new("RGB", (4, 3)))
    assert url.startswith("data:image/png;base64,")
    assert resolve_to_image(url).size == (4, 3)


def test_resolve_to_image_accepts_file_scheme(tmp_path):
    path = _write_png(tmp_path, "img.png", (5, 6))
    assert resolve_to_image(f"file://{path}").size == (5, 6)
    assert resolve_to_image(path).size == (5, 6)


def test_encode_images_encodes_local_paths_and_file_urls(tmp_path):
    plain = _write_png(tmp_path, "plain.png", (2, 2))
    file_url = "file://" + _write_png(tmp_path, "scheme.png", (3, 3))

    examples = [
        _example(
            {"type": "input_image", "image_url": plain},
            {"type": "input_image", "image_url": {"url": file_url}},
            {"type": "input_text", "text": "describe"},
        )
    ]
    encode_images_in_examples(examples)

    parts = examples[0]["responses_create_params"]["input"][0]["content"]
    assert parts[0]["image_url"].startswith("data:image/png;base64,")
    assert parts[1]["image_url"].startswith("data:image/png;base64,")
    assert resolve_to_image(parts[0]["image_url"]).size == (2, 2)
    assert resolve_to_image(parts[1]["image_url"]).size == (3, 3)
    # Non-image parts are untouched.
    assert parts[2] == {"type": "input_text", "text": "describe"}


def test_encode_images_passes_through_http_and_data_urls():
    data_url = image_to_data_url(Image.new("RGB", (2, 2)))
    examples = [
        _example(
            {"type": "input_image", "image_url": "https://example.com/cat.png"},
            {"type": "input_image", "image_url": "http://example.com/dog.png"},
            {"type": "input_image", "image_url": data_url},
        )
    ]
    encode_images_in_examples(examples)

    parts = examples[0]["responses_create_params"]["input"][0]["content"]
    assert parts[0]["image_url"] == "https://example.com/cat.png"
    assert parts[1]["image_url"] == "http://example.com/dog.png"
    assert parts[2]["image_url"] == data_url


def test_encode_images_is_a_noop_for_text_only_examples():
    examples = [_example({"type": "input_text", "text": "no images here"})]
    before = [
        dict(part)
        for part in examples[0]["responses_create_params"]["input"][0]["content"]
    ]
    assert encode_images_in_examples(examples) is examples
    assert examples[0]["responses_create_params"]["input"][0]["content"] == before

    # Missing/oddly-shaped payloads must not raise.
    assert encode_images_in_examples([{}, {"responses_create_params": {}}]) is not None
    assert encode_images_in_examples([{"responses_create_params": {"input": "nope"}}])
