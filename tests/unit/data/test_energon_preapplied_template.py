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

from copy import deepcopy
from typing import Any

import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

pytest.importorskip("megatron.energon")
pytestmark = pytest.mark.mcore

from nemo_rl.data.energon.multimodal.cookers.generic import cook_conversation  # noqa: E402
from nemo_rl.data.energon.multimodal.task_encoders.generic_sft import (  # noqa: E402
    HFMultimodalSFTProcessorAdapter,
)
from nemo_rl.data.llm_message_utils import add_loss_mask_to_message_log  # noqa: E402


def _sample(subflavors: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "__key__": "sample-0",
        "__restore_key__": ("sample-0",),
        "__subflavors__": subflavors,
        "__sources__": (),
        "json": {
            "messages": [
                {"role": "user", "content": "A: "},
                {"role": "assistant", "content": "Paris"},
            ],
            # Payload fields must not override the dataset's subflavors.
            "skip_chat_template": True,
        },
    }


@pytest.mark.parametrize(
    "subflavors, expected",
    [
        (None, False),
        ({}, False),
        ({"skip_chat_template": False}, False),
        ({"skip_chat_template": True}, True),
    ],
)
def test_cooker_maps_skip_chat_template_to_preapplied(
    subflavors: dict[str, Any] | None, expected: bool
) -> None:
    crude = _sample(subflavors)
    original = deepcopy(crude)
    cooked = cook_conversation(crude)
    assert cooked.chat_template_preapplied is expected
    assert crude == original
    assert cooked.__subflavors__ == subflavors


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_cooker_rejects_non_boolean_skip_chat_template(value: Any) -> None:
    with pytest.raises(ValueError, match="skip_chat_template.*boolean"):
        cook_conversation(_sample({"skip_chat_template": value}))


class _TextProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.template_calls = 0

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.template_calls += 1
        # A multimodal processor accepts structured text parts; Qwen's text
        # tokenizer needs their string content before rendering its template.
        messages = [
            {**message, "content": "".join(part["text"] for part in message["content"])}
            if isinstance(message["content"], list)
            else message
            for message in messages
        ]
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, **kwargs: Any) -> Any:
        return self.tokenizer(**kwargs)


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def test_adapter_mixed_sources_only_templates_normal_samples(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    processor = _TextProcessor(tokenizer)
    adapter = HFMultimodalSFTProcessorAdapter(
        processor=processor,
        max_sequence_length=1024,
        add_bos=False,
        add_eos=False,
        add_generation_prompt=False,
    )
    # The same adapter serves both source types; the flag is per sample.
    for preapplied in [True, False, True]:
        sample = cook_conversation(_sample({"skip_chat_template": preapplied}))
        template_calls = processor.template_calls
        encoded = adapter.encode(sample)
        assert encoded.__key__ == sample.__key__
        assert encoded.length == sum(
            m["token_ids"].numel() for m in encoded.message_log
        )
        assert sample.messages[0]["content"] == "A: "
        if preapplied:
            assert processor.template_calls == template_calls
            assert [m["content"] for m in encoded.message_log] == ["A: ", "Paris"]
            for message, text in zip(
                encoded.message_log, ["A: ", "Paris"], strict=True
            ):
                assert message["token_ids"].tolist() == tokenizer.encode(
                    text, add_special_tokens=False
                )
        else:
            assert processor.template_calls == template_calls + 2
            assert "<|im_start|>" in encoded.message_log[0]["content"][0]["text"]
        add_loss_mask_to_message_log(
            [encoded.message_log], roles_to_train_on=["assistant"]
        )
        assert torch.all(encoded.message_log[0]["token_loss_mask"] == 0)
        assert encoded.message_log[1]["token_loss_mask"].sum() > 0


@pytest.mark.parametrize("option", ["tools", "generation_prompt"])
def test_adapter_preapplied_rejects_template_options(
    tokenizer: PreTrainedTokenizerBase, option: str
) -> None:
    sample = cook_conversation(_sample({"skip_chat_template": True}))
    if option == "tools":
        sample.tools = []
    adapter = HFMultimodalSFTProcessorAdapter(
        processor=_TextProcessor(tokenizer),
        max_sequence_length=1024,
        add_bos=False,
        add_eos=False,
        add_generation_prompt=option == "generation_prompt",
    )
    with pytest.raises(ValueError, match="chat_template_preapplied"):
        adapter.encode(sample)
