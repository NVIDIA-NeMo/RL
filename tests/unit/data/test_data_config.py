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

from pydantic import TypeAdapter

from nemo_rl.data import DataConfig


def test_data_config_preserves_native_openai_chat_adapter_fields() -> None:
    train = {
        "dataset_name": "openai_format",
        "data_path": "<ASSET_DIR>/openmath-chat.jsonl",
        "data_files": None,
        "text_key": None,
        "characters_per_sample": None,
        "processor": "kd_data_processor",
        "split": "train",
        "chat_key": "messages",
        "use_preserving_dataset": False,
        "system_key": None,
        "system_prompt": None,
        "tool_key": None,
        "prompt_file": None,
    }

    validated = TypeAdapter(DataConfig).validate_python(
        {
            "max_input_seq_length": 4096,
            "shuffle": False,
            "collator_mode": "chat",
            "train": train,
        }
    )

    assert validated["train"] == train
