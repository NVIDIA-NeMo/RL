"""Actual worker/capture/TQ boundary; only vLLM processing/storage are doubled."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from nemo_gym.token_id_capture.staging.digest import compute_extras_digest

from nemo_rl.data_plane.schema import ROUTED_EXTRAS_METADATA_FIELD
from nemo_rl.models.generation.vllm.utils import capture_image_geometry
from tests.unit.models.generation.test_vllm_context_measurement import (
    captured_server,  # noqa: F401
    render,
    server,  # noqa: F401
    wiring,
)

pytestmark = pytest.mark.nemo_gym


def image_prompt(sizes=((2, 4), (4, 2))):
    tokens, spans, items = [10], [], []
    for height, width in sizes:
        spans.append(SimpleNamespace(offset=len(tokens), length=2, is_embed=None))
        tokens.extend([18, 18, 11])
        data = dict(
            pixel_values_flat=torch.zeros(3, height, width),
            imgs_sizes=(height, width),
            num_tokens_per_image=2,
        )
        items.append(SimpleNamespace(get_data=lambda data=data: data))
    return dict(
        prompt_token_ids=tokens,
        mm_placeholders={"image": spans},
        mm_kwargs={"image": items},
    )


def test_geometry_records_order_and_true_hw_not_equal_token_counts():
    prompt = image_prompt()
    root = capture_image_geometry(prompt, prev_len=0)
    assert root["images"] == [
        dict(offset=1, length=2, height=2, width=4, token_id=18),
        dict(offset=4, length=2, height=4, width=2, token_id=18),
    ]
    child = capture_image_geometry(prompt, prev_len=4)
    assert child["images"] == root["images"][1:]
    assert child["prefix_digest"] == compute_extras_digest(
        {"images": root["images"][:1]}
    )
    # Same token count and area, different H/W: retained geometry must differ.
    changed = capture_image_geometry(image_prompt(((4, 2), (4, 2))), prev_len=4)
    assert changed["prefix_digest"] != child["prefix_digest"]
    # A text-only continuation still commits that it retained all earlier images.
    terminal = capture_image_geometry(prompt, prev_len=len(prompt["prompt_token_ids"]))
    assert terminal["images"] == []
    assert terminal["prefix_digest"] == compute_extras_digest(
        {"images": root["images"]}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "cached",
        "hw",
        "pixels",
        "count",
        "tokens",
        "overlap",
        "crossing",
        "modality",
        "mask",
    ],
)
def test_bad_engine_geometry_is_rejected(mutation):
    prompt = image_prompt()
    prev_len = 0
    if mutation == "missing":
        prompt["mm_kwargs"]["image"].pop()
    elif mutation == "cached":
        prompt["mm_kwargs"]["image"][0] = None
    elif mutation == "hw":
        prompt["mm_kwargs"]["image"][0].get_data()["imgs_sizes"] = (4, 2)
    elif mutation == "pixels":
        prompt["mm_kwargs"]["image"][0].get_data()["pixel_values_flat"] = torch.zeros(
            1, 3, 2, 4
        )
    elif mutation == "count":
        prompt["mm_kwargs"]["image"][0].get_data()["num_tokens_per_image"] = 3
    elif mutation == "tokens":
        prompt["prompt_token_ids"][2] = 19
    elif mutation == "overlap":
        prompt["mm_placeholders"]["image"][1].offset = 1
    elif mutation == "crossing":
        prev_len = 2
    elif mutation == "modality":
        prompt["mm_kwargs"]["audio"] = []
    else:
        prompt["mm_placeholders"]["image"][0].is_embed = torch.tensor([False, False])
    with pytest.raises(ValueError):
        capture_image_geometry(prompt, prev_len=prev_len)


def test_image_embedding_mask_selects_only_contiguous_embedding_positions():
    prompt = image_prompt(((2, 4),))
    span = prompt["mm_placeholders"]["image"][0]
    span.offset, span.length = 0, 4
    span.is_embed = torch.tensor([False, True, True, False])
    assert capture_image_geometry(prompt, prev_len=0)["images"][0]["offset"] == 1


@pytest.mark.parametrize("cc_enabled", [False, True])
def test_real_renderer_stages_geometry_without_returning_it_to_gym(
    request, monkeypatch, cc_enabled
):
    stack = request.getfixturevalue("captured_server")
    stack.worker._cc_capture_enabled = cc_enabled
    observed_cache_policy = []
    prompt = image_prompt(((2, 4),))

    async def process(self, *, messages, skip_mm_cache, **kwargs):
        observed_cache_policy.append(skip_mm_cache)
        return messages, [deepcopy(prompt)]

    monkeypatch.setattr(wiring._OnlineRenderer, "preprocess_chat", process)
    request = stack.chat_type(
        messages=[{"role": "user", "content": "image"}],
        ng_capture={
            "rollout_id": "images",
            "model_call_id": "root",
            "mode": "text",
        },
    )
    asyncio.run(render(stack.renderer, request))
    content = {
        "choices": [
            {"message": {"generation_token_ids": [31], "generation_log_probs": [-0.5]}}
        ]
    }
    response = stack.worker._finish_request_capture(request, content)
    assert response["ng_commit_coords"]["disposition"] == "staged"
    assert "cc_image_geometry" not in response
    assert observed_cache_policy == [cc_enabled]
    staged = stack.data_plane.get_samples(
        ["images/root"], "staged", [ROUTED_EXTRAS_METADATA_FIELD]
    )
    metadata = json.loads(bytes(staged[ROUTED_EXTRAS_METADATA_FIELD][0].tolist()))
    assert metadata == (
        {"cc_image_geometry": capture_image_geometry(prompt, prev_len=0)}
        if cc_enabled
        else None
    )
    assert not stack.worker._capture_calls
