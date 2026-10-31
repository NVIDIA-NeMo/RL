# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import copy
from collections.abc import Mapping
from typing import Any

from nemo_rl.data.deepseek_v4_encoding import encode_messages

DEEPSEEK_V4_CHAT_TEMPLATE = "deepseek_v4"


def should_use_deepseek_v4_chat_template(tokenizer_config: Mapping[str, Any]) -> bool:
    """Return whether the config explicitly requests the DeepSeek V4 renderer."""
    return tokenizer_config.get("chat_template") == DEEPSEEK_V4_CHAT_TEMPLATE


def get_deepseek_v4_tokenizer(
    tokenizer: Any, chat_template_kwargs: dict[str, Any] | None = None
) -> Any:
    """Wrap a fast tokenizer with vLLM 0.25.1's DeepSeek V4 encoder."""
    dsv4_tokenizer = copy.copy(tokenizer)
    default_chat_template_kwargs = dict(chat_template_kwargs or {})
    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):  # type: ignore
        def apply_chat_template(
            self,
            conversation: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str | list[int]:
            kwargs = {**default_chat_template_kwargs, **kwargs}
            thinking = kwargs.get("thinking", False) or kwargs.get(
                "enable_thinking", False
            )
            thinking_mode = "thinking" if thinking else "chat"

            deepseek_messages = list(conversation)
            if tools:
                deepseek_messages.insert(0, {"role": "system", "tools": tools})

            reasoning_effort = kwargs.get("reasoning_effort")
            if reasoning_effort == "none":
                thinking_mode = "chat"
                reasoning_effort = None
            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            elif isinstance(reasoning_effort, str):
                reasoning_effort = "high"
            else:
                reasoning_effort = None

            prompt = encode_messages(
                deepseek_messages,
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            if not kwargs.get("tokenize", True):
                return prompt

            tokenizer_kwargs = {
                key: kwargs[key]
                for key in ("truncation", "max_length")
                if key in kwargs
            }
            return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)

        def num_special_tokens_to_add(self, *args: Any, **kwargs: Any) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self) -> tuple[Any, tuple[Any, dict[str, Any]]]:
            return get_deepseek_v4_tokenizer, (
                tokenizer,
                default_chat_template_kwargs,
            )

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"
    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer

    return dsv4_tokenizer
