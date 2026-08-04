from PIL import Image

from nemo_rl.data.multimodal_utils import image_to_data_url
from nemo_rl.environments.nemo_gym import (
    _extract_input_images_from_message,
    _index_per_turn_images,
)


def _image(size: tuple[int, int]) -> str:
    """Return a data URL for a solid RGB image of the given size."""
    return image_to_data_url(Image.new("RGB", size))


def _user(*data_urls: str) -> dict:
    return {
        "role": "user",
        "content": [{"type": "input_image", "image_url": url} for url in data_urls],
    }


def _assistant(token_ids: list[int]) -> dict:
    return {"role": "assistant", "generation_token_ids": token_ids}


def test_extract_input_images_handles_flat_and_dict_image_url():
    item = {
        "role": "user",
        "content": [
            {"type": "input_image", "image_url": _image((2, 2))},
            {"type": "input_image", "image_url": {"url": _image((3, 3))}},
            {"type": "input_text", "text": "ignore me"},
        ],
    }
    images = _extract_input_images_from_message(item)
    assert [img.size for img in images] == [(2, 2), (3, 3)]


def test_extract_input_images_returns_empty_for_string_content():
    assert _extract_input_images_from_message({"role": "user", "content": "hi"}) == []
    assert _extract_input_images_from_message({"role": "user"}) == []


def test_index_per_turn_images_bins_seed_and_intermediate_images():
    seed_obs = [_user(_image((2, 2)))]
    output = [
        _assistant([1, 2]),
        _user(_image((3, 3)), _image((4, 4))),
        _assistant([3, 4]),
    ]
    per_turn = _index_per_turn_images(seed_obs, output)

    assert len(per_turn) == 2
    assert [img.size for img in per_turn[0]] == [(2, 2)]
    assert [img.size for img in per_turn[1]] == [(3, 3), (4, 4)]


def test_index_per_turn_images_text_only_rollout_yields_empty_buckets():
    output = [
        {"role": "user", "content": "solve this"},
        _assistant([1, 2]),
        {"role": "user", "content": "and this"},
        _assistant([3, 4]),
    ]
    assert _index_per_turn_images([], output) == [[], []]


def test_index_per_turn_images_assigns_tool_result_image_to_next_turn():
    """A tool-result image contributes to the following assistant turn."""
    output = [
        _user(_image((2, 2))),
        _assistant([1, 2]),
        {"type": "function_call_output", "output": _image((5, 5))},
        _assistant([3, 4]),
    ]
    per_turn = _index_per_turn_images([], output)

    assert len(per_turn) == 2
    assert [img.size for img in per_turn[0]] == [(2, 2)]
    assert [img.size for img in per_turn[1]] == [(5, 5)]


def test_index_per_turn_images_aligns_with_postprocess_skip_of_empty_generations():
    """Turns skipped by the postprocess loop must not consume an image bucket.

    ``_postprocess_nemo_gym_to_nemo_rl_result`` skips output items whose
    ``generation_token_ids`` is present but empty, so the bucket list must skip
    them too or every later turn is attached to the wrong images.
    """
    output = [
        _user(_image((2, 2))),
        _assistant([]),  # all-EOS generation, skipped by the postprocess loop
        _user(_image((6, 6))),
        _assistant([7, 8]),
    ]
    per_turn = _index_per_turn_images([], output)

    assert len(per_turn) == 1
    assert [img.size for img in per_turn[0]] == [(2, 2), (6, 6)]
