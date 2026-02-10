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

import json
import tempfile

import pytest

from nemo_rl.data.datasets import load_response_dataset


def create_sample_general_conversation_jsonl():
    """Create a temporary jsonl file with one sample: video + user/assistant conversation."""
    sample = [
        {
            "video": "path_to_video.mp4",
            "conversations": [
                {"from": "user", "value": "<video>\nPlease describe this video."},
                {"from": "assistant", "value": "Two kids are playing ping pong in this video."},
            ],
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")
        return f.name


def test_general_conversation_jsonl_preprocessor_converts_to_openai_format():
    """Test that a mock local jsonl sample is converted to OpenAI-compatible message form by the preprocessor."""
    data_path = create_sample_general_conversation_jsonl()
    try:
        data_config = {
            "dataset_name": "general-conversation-jsonl",
            "data_path": data_path,
        }
        dataset = load_response_dataset(data_config)

        # Raw first example from the jsonl
        first_raw = dataset.dataset[0]
        # Run the preprocessor (same as used in the pipeline)
        formatted = dataset.preprocessor(first_raw)

        # Expected OpenAI-compatible structure
        assert "messages" in formatted
        assert "task_name" in formatted
        assert formatted["task_name"] == "general-conversation-jsonl"

        assert len(formatted["messages"]) == 2

        # User message: content is list of video block + text block
        user_msg = formatted["messages"][0]
        assert user_msg["role"] == "user"
        user_content = user_msg["content"]
        assert isinstance(user_content, list)
        assert user_content[0] == {"type": "video", "video": "path_to_video.mp4"}
        assert user_content[1]["type"] == "text"
        assert "Please describe this video." in user_content[1]["text"]

        # Assistant message: content is list of text block(s)
        assistant_msg = formatted["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assistant_content = assistant_msg["content"]
        assert isinstance(assistant_content, list)
        assert assistant_content == [
            {"type": "text", "text": "Two kids are playing ping pong in this video."}
        ]
    finally:
        import os

        try:
            os.unlink(data_path)
        except OSError:
            pass
