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

from types import SimpleNamespace

from nemo_rl.utils.prefix_reuse import (
    derive_required_prefix_token_ids,
    messages_to_last_assistant,
    replace_prefix_tokens,
)


class _Tokenizer:
    eos_token_id = 2

    def decode(self, token_ids):
        return repr(token_ids)


def test_replace_prefix_tokens_preserves_prior_model_tokens() -> None:
    result = replace_prefix_tokens(
        _Tokenizer(),
        model_prefix_token_ids=[11, 12, 220, 17, 2],
        template_prefix_token_ids=[11, 12, 1001, 2],
        template_token_ids=[11, 12, 1001, 2, 21, 22],
    )

    assert result == [11, 12, 220, 17, 2, 21, 22]


def test_derive_required_prefix_token_ids_uses_latest_message() -> None:
    messages = [
        {"role": "assistant", "prompt_token_ids": [1], "generation_token_ids": [2]},
        {"role": "user", "content": "next"},
        SimpleNamespace(
            role="assistant",
            prompt_token_ids=[3, 4],
            generation_token_ids=[5],
        ),
    ]

    assert derive_required_prefix_token_ids(messages) == [3, 4, 5]


def test_messages_to_last_assistant_includes_latest_assistant_turn() -> None:
    messages = [
        {"role": "system"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "after"},
    ]

    assert messages_to_last_assistant(messages) == messages[:4]
