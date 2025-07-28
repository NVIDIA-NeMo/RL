# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import pytest
from transformers import AutoTokenizer

from nemo_rl.data.hf_datasets.squad import SquadDataset


@pytest.mark.hf_gated
@pytest.mark.skip(reason="dataset download is flaky")
def test_squad_dataset():
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    squad_dataset = SquadDataset()

    # check that the dataset is formatted correctly
    for example in squad_dataset.formatted_ds["train"].take(5):
        assert "messages" in example
        assert len(example["messages"]) == 3

        assert example["messages"][0]["role"] == "system"
        assert example["messages"][1]["role"] == "user"
        assert example["messages"][2]["role"] == "assistant"

        template = "{% for message in messages %}{%- if message['role'] == 'system'  %}{{'Context: ' + message['content'].strip()}}{%- elif message['role'] == 'user'  %}{{' Question: ' + message['content'].strip() + ' Answer:'}}{%- elif message['role'] == 'assistant'  %}{{' ' + message['content'].strip()}}{%- endif %}{% endfor %}"

        ## check that applying chat template works as expected
        default_templated = tokenizer.apply_chat_template(
            example["messages"],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )

        assert default_templated == (
            "Context: "
            + example["messages"][0]["content"]
            + " Question: "
            + example["messages"][1]["content"]
            + " Answer: "
            + example["messages"][2]["content"]
        )
