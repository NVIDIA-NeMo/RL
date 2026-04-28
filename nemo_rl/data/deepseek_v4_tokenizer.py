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

import copy
import json
from pathlib import Path
from typing import Any

from nemo_rl.data.deepseek_v4_encoding import encode_messages

DEEPSEEK_V4_CHAT_TEMPLATE = "deepseek_v4"


def _is_deepseek_v4_model_path(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    return config.get("model_type") == "deepseek_v4"


def should_use_deepseek_v4_chat_template(tokenizer_config: dict[str, Any]) -> bool:
    chat_template = tokenizer_config.get("chat_template")
    if (
        isinstance(chat_template, str)
        and chat_template.lower() == DEEPSEEK_V4_CHAT_TEMPLATE
    ):
        return True

    tokenizer_name = tokenizer_config.get("name")
    return isinstance(tokenizer_name, str) and _is_deepseek_v4_model_path(
        tokenizer_name
    )


def get_deepseek_v4_tokenizer(tokenizer: Any) -> Any:
    """Wrap a tokenizer with DeepSeek V4's Python chat encoder.

    This mirrors vLLM's DeepseekV4Tokenizer wrapper: the underlying tokenizer is
    still the standard fast tokenizer, but apply_chat_template delegates to the
    DeepSeek V4 reference encoder instead of relying on tokenizer_config.json.
    """
    if getattr(tokenizer, "_nemo_rl_deepseek_v4_chat_template", False):
        return tokenizer

    dsv4_tokenizer = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):  # type: ignore
        _nemo_rl_deepseek_v4_chat_template = True

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str | list[int]:
            thinking = kwargs.get("thinking", False)
            enable_thinking = kwargs.get("enable_thinking", False)
            thinking_mode = "thinking" if thinking or enable_thinking else "chat"

            conversation = kwargs.get("conversation", messages)
            deepseek_messages = list(conversation)
            if tools is not None and len(tools) > 0:
                deepseek_messages.insert(0, {"role": "system", "tools": tools})

            reasoning_effort = kwargs.get("reasoning_effort")
            if reasoning_effort not in ("max", "high"):
                reasoning_effort = None

            prompt_str = encode_messages(
                deepseek_messages,
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            if kwargs.get("tokenize", True):
                tokenizer_kwargs = {
                    key: kwargs[key]
                    for key in ("truncation", "max_length")
                    if key in kwargs
                }
                return self.encode(
                    prompt_str,
                    add_special_tokens=False,
                    **tokenizer_kwargs,
                )

            return prompt_str

        def num_special_tokens_to_add(self, *args: Any, **kwargs: Any) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self) -> tuple[Any, tuple[Any]]:
            return get_deepseek_v4_tokenizer, (tokenizer,)

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"

    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    dsv4_tokenizer.chat_template = DEEPSEEK_V4_CHAT_TEMPLATE
    return dsv4_tokenizer
