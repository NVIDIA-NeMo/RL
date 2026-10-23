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

"""Exercise the supported AutoTokenizer remote-code fallback without downloads."""

import json
from textwrap import dedent

from transformers import AutoTokenizer


def test_deepseek_remote_tokenizer_without_tokenizer_json(tmp_path):
    # The DeepSeek blocklist first chooses TokenizersBackend, but this local
    # checkpoint intentionally supplies only a declared custom tokenizer.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoTokenizer": ["tokenization_fixture.FixtureTokenizer", None]
                },
            }
        )
    )
    (tmp_path / "tokenization_fixture.py").write_text(
        dedent("""
        from transformers import PreTrainedTokenizer

        class FixtureTokenizer(PreTrainedTokenizer):
            def __init__(self, **kwargs):
                super().__init__(unk_token="<unk>", **kwargs)

            @property
            def vocab_size(self):
                return 2

            def get_vocab(self):
                return {"<unk>": 0, "hello": 1}

            def _tokenize(self, text):
                return text.split()

            def _convert_token_to_id(self, token):
                return self.get_vocab().get(token, 0)

            def _convert_id_to_token(self, index):
                return "hello" if index == 1 else "<unk>"
    """)
    )
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=True)
    assert tokenizer.__class__.__name__ == "FixtureTokenizer"
    assert tokenizer.encode("hello", add_special_tokens=False) == [1]
