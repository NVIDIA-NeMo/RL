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

import base64
from io import BytesIO

import pytest
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


def _split(data_url: str) -> tuple[str, bytes]:
    """Split a data URL into its MIME subtype and decoded payload."""
    header, payload = data_url.split(",", 1)
    assert header.startswith("data:image/") and header.endswith(";base64")
    return header[len("data:image/") : -len(";base64")], base64.b64decode(payload)


def test_image_to_data_url_round_trips_through_resolve_to_image():
    url = image_to_data_url(Image.new("RGB", (4, 3)))
    assert url.startswith("data:image/png;base64,")
    assert resolve_to_image(url).size == (4, 3)


@pytest.mark.parametrize(
    "name,pil_format,expected_subtype",
    [
        ("img.png", "PNG", "png"),
        ("img.jpg", "JPEG", "jpeg"),
        ("img.jpeg", "JPEG", "jpeg"),
        ("img.gif", "GIF", "gif"),
        ("img.webp", "WEBP", "webp"),
    ],
)
def test_image_to_data_url_embeds_local_files_verbatim(
    tmp_path, name, pil_format, expected_subtype
):
    """Web-safe files keep their MIME subtype and their exact bytes."""
    path = tmp_path / name
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(path, format=pil_format)

    subtype, payload = _split(image_to_data_url(str(path)))
    assert subtype == expected_subtype
    assert payload == path.read_bytes()
    with Image.open(BytesIO(payload)) as decoded:
        assert decoded.format == pil_format
        assert decoded.size == (8, 6)


def test_image_to_data_url_reencodes_unembeddable_file_as_png(tmp_path):
    path = tmp_path / "img.bmp"
    Image.new("RGB", (4, 5), color=(1, 2, 3)).save(path, format="BMP")

    subtype, payload = _split(image_to_data_url(str(path)))
    assert subtype == "png"
    with Image.open(BytesIO(payload)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.size == (4, 5)


def test_image_to_data_url_keeps_jpeg_format_for_pil_images(tmp_path):
    """A PIL image opened from a JPEG must not come back out as a PNG."""
    path = tmp_path / "img.jpg"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(path, format="JPEG")

    with Image.open(path) as opened:
        subtype, payload = _split(image_to_data_url(opened))
    assert subtype == "jpeg"
    with Image.open(BytesIO(payload)) as decoded:
        assert decoded.format == "JPEG"


def test_image_to_data_url_falls_back_to_png_for_in_memory_images():
    """``Image.new`` has no ``format``, so PNG stays the default."""
    image = Image.new("RGB", (4, 3))
    assert image.format is None
    assert _split(image_to_data_url(image))[0] == "png"


@pytest.mark.parametrize("fmt", ["JPEG", "jpg", "JPG"])
def test_image_to_data_url_maps_jpg_aliases_to_the_jpeg_mime_type(fmt):
    """``image/jpg`` is not a real MIME type -- every alias must emit jpeg."""
    subtype, payload = _split(image_to_data_url(Image.new("RGB", (8, 6)), fmt=fmt))
    assert subtype == "jpeg"
    with Image.open(BytesIO(payload)) as decoded:
        assert decoded.format == "JPEG"


def test_image_to_data_url_explicit_fmt_overrides_source_format(tmp_path):
    path = tmp_path / "img.jpg"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(path, format="JPEG")

    with Image.open(path) as opened:
        assert _split(image_to_data_url(opened, fmt="PNG"))[0] == "png"


def test_image_to_data_url_embeds_multi_picture_jpeg_as_jpeg(tmp_path):
    """PIL reports phone-camera JPEGs as MPO; they must not be rewritten to PNG."""
    path = tmp_path / "img.mpo"
    frame = Image.new("RGB", (8, 6), color=(10, 20, 30))
    try:
        frame.save(path, format="MPO", save_all=True, append_images=[frame])
    except (OSError, ValueError, KeyError) as e:
        pytest.skip(f"Pillow cannot write MPO files: {e}")
    with Image.open(path) as opened:
        if opened.format != "MPO":
            pytest.skip(f"Pillow read the file back as {opened.format}, not MPO")

    subtype, payload = _split(image_to_data_url(str(path)))
    assert subtype == "jpeg"
    assert payload == path.read_bytes()


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


def test_encode_images_preserves_source_encoding(tmp_path):
    """JPEG sources stay JPEG and are embedded byte-for-byte (no PIL re-encode)."""
    path = tmp_path / "img.jpg"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path, format="JPEG")
    raw = path.read_bytes()

    examples = [_example({"type": "input_image", "image_url": str(path)})]
    encode_images_in_examples(examples)

    url = examples[0]["responses_create_params"]["input"][0]["content"][0]["image_url"]
    assert _split(url) == ("jpeg", raw)


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
