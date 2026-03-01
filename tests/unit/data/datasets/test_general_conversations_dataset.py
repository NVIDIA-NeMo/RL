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

from nemo_rl.data.datasets import load_response_dataset


def create_sample_general_conversation_jsonl_multimodal_interleaved_multiturn():
    """Create a temporary jsonl file with one sample: audio + video + image + user/assistant conversations."""
    sample = [
        {
            "sound": ["sample_000001.2345ew.flac", "sample_000001.gd1dtg.wav"],
            "video-audio": "sample_000001.35tags.mp4",
            "image": ["sample_000001.as23ds.jpg", "sample_000001.gds233.jpg"],
            "conversations": [
                {"from": "user", "value": "<sound>"},
                {
                    "from": "assistant",
                    "value": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition.",
                },
                {
                    "from": "user",
                    "value": "Describe what is NeMo based on the tutorial video: <video-audio> and the information in the two images: <image> <image>. Combine that information with sound <sound>. Answer: ",
                },
                {
                    "from": "assistant",
                    "value": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability.",
                },
            ],
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")
        return f.name


def test_general_conversation_jsonl_multimodal_interleaved_multiturn():
    """Test that a mock local jsonl sample is converted to OpenAI-compatible message form by the preprocessor."""
    data_path = (
        create_sample_general_conversation_jsonl_multimodal_interleaved_multiturn()
    )
    try:
        data_config = {
            "dataset_name": "general-conversation-jsonl",
            "data_path": data_path,
        }
        dataset = load_response_dataset(data_config)

        assert len(dataset.dataset) == 1

        # Raw first example from the jsonl
        first_raw = dataset.dataset[0]
        # Run the preprocessor (same as used in the pipeline)
        formatted = dataset.preprocessor(first_raw)

        # Expected OpenAI-compatible structure
        assert "messages" in formatted
        assert "task_name" in formatted
        assert formatted["task_name"] == "general-conversation-jsonl"

        assert len(formatted["messages"]) == 4

        # User message: content is list of audio/video/image block + text block
        user_msg0 = formatted["messages"][0]
        assert user_msg0["role"] == "user"
        user_content0 = user_msg0["content"]
        assert isinstance(user_content0, list)
        assert len(user_content0) == 1
        # the "sound" tag will be converted to the "audio" tag
        assert user_content0[0] == {
            "type": "audio",
            "audio": "sample_000001.2345ew.flac",
        }

        # Assistant message: content is list of text block(s).
        # Multimodal tokens are also supported in a similar fashion as for the user message.
        assistant_msg0 = formatted["messages"][1]
        assert assistant_msg0["role"] == "assistant"
        assistant_content0 = assistant_msg0["content"]
        assert isinstance(assistant_content0, list)
        assert len(assistant_content0) == 1
        assert assistant_content0[0] == {
            "type": "text",
            "text": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition.",
        }

        user_msg1 = formatted["messages"][2]
        assert user_msg1["role"] == "user"
        user_content1 = user_msg1["content"]
        assert isinstance(user_content1, list)
        assert len(user_content1) == 9
        assert user_content1[0] == {
            "type": "text",
            "text": "Describe what is NeMo based on the tutorial video: ",
        }
        # video-audio tag will be splitted into one video tag followed by one audio tag
        # TODO: more advanced video-audio interleaving technique? Should be handled on the model level.
        assert user_content1[1] == {
            "type": "video",
            "video": "sample_000001.35tags.mp4",
        }
        assert user_content1[2] == {
            "type": "audio",
            "audio": "sample_000001.35tags.mp4",
        }
        assert user_content1[3] == {
            "type": "text",
            "text": " and the information in the two images: ",
        }
        assert user_content1[4] == {
            "type": "image",
            "image": "sample_000001.as23ds.jpg",
        }
        assert user_content1[5] == {
            "type": "image",
            "image": "sample_000001.gds233.jpg",
        }
        assert user_content1[6] == {
            "type": "text",
            "text": ". Combine that information with sound ",
        }
        assert user_content1[7] == {
            "type": "audio",
            "audio": "sample_000001.gd1dtg.wav",
        }
        assert user_content1[8] == {"type": "text", "text": ". Answer: "}

        assistant_msg1 = formatted["messages"][3]
        assert assistant_msg1["role"] == "assistant"
        assistant_content1 = assistant_msg1["content"]
        assert isinstance(assistant_content1, list)
        assert len(assistant_content1) == 1
        assert assistant_content1[0] == {
            "type": "text",
            "text": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability.",
        }

    finally:
        import os

        try:
            os.unlink(data_path)
        except OSError:
            pass


def create_sample_general_conversation_jsonl_multimodal_singleturn():
    """Create a temporary jsonl file with multiple samples with each sample contains one modality."""
    sample = [
        {
            "image": "sample_000001.as23ds.jpg",
            "conversations": [
                {"from": "user", "value": "<image>\nPlease describe this image."},
                {
                    "from": "assistant",
                    "value": "Two kids are playing ping pong in this image.",
                },
            ],
        },
        {
            "audio": ["sample_000001.2345ew.flac"],
            "conversations": [
                {"from": "user", "value": "<audio>"},
                {
                    "from": "assistant",
                    "value": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition.",
                },
            ],
        },
        {
            "video-audio": "sample_000001.35tags.mp4",
            "conversations": [
                {
                    "from": "user",
                    "value": "<video-audio>\nDescribe what is NeMo based on the tutorial video. Answer: ",
                },
                {
                    "from": "assistant",
                    "value": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability.",
                },
            ],
        },
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")
        return f.name


def test_general_conversation_jsonl_multimodal_singleturn():
    """Test that a mock local jsonl sample is converted to OpenAI-compatible message form by the preprocessor."""
    data_path = create_sample_general_conversation_jsonl_multimodal_singleturn()
    try:
        data_config = {
            "dataset_name": "general-conversation-jsonl",
            "data_path": data_path,
        }
        dataset = load_response_dataset(data_config)

        assert len(dataset.dataset) == 3

        # Raw first example from the jsonl
        first_raw = dataset.dataset[0]
        # Run the preprocessor (same as used in the pipeline)
        formatted = dataset.preprocessor(first_raw)

        # Expected OpenAI-compatible structure
        assert "messages" in formatted
        assert "task_name" in formatted
        assert formatted["task_name"] == "general-conversation-jsonl"

        assert len(formatted["messages"]) == 2

        # User message: content is list of audio/video/image block + text block
        user_msg0 = formatted["messages"][0]
        assert user_msg0["role"] == "user"
        user_content0 = user_msg0["content"]
        assert isinstance(user_content0, list)
        assert len(user_content0) == 2
        # the "sound" tag will be converted to the "audio" tag
        assert user_content0[0] == {
            "type": "image",
            "image": "sample_000001.as23ds.jpg",
        }
        assert user_content0[1] == {
            "type": "text",
            "text": "\nPlease describe this image.",
        }

        # Assistant message: content is list of text block(s).
        # Multimodal tokens are also supported in a similar fashion as for the user message.
        assistant_msg0 = formatted["messages"][1]
        assert assistant_msg0["role"] == "assistant"
        assistant_content0 = assistant_msg0["content"]
        assert isinstance(assistant_content0, list)
        assert len(assistant_content0) == 1
        assert assistant_content0[0] == {
            "type": "text",
            "text": "Two kids are playing ping pong in this image.",
        }

        # Raw Second example from the jsonl
        second_raw = dataset.dataset[1]
        # Run the preprocessor (same as used in the pipeline)
        formatted = dataset.preprocessor(second_raw)

        # Expected OpenAI-compatible structure
        assert "messages" in formatted
        assert "task_name" in formatted
        assert formatted["task_name"] == "general-conversation-jsonl"

        assert len(formatted["messages"]) == 2

        # User message: content is list of audio/video/image block + text block
        user_msg0 = formatted["messages"][0]
        assert user_msg0["role"] == "user"
        user_content0 = user_msg0["content"]
        assert isinstance(user_content0, list)
        assert len(user_content0) == 1
        # the "sound" tag will be converted to the "audio" tag
        assert user_content0[0] == {
            "type": "audio",
            "audio": "sample_000001.2345ew.flac",
        }

        # Assistant message: content is list of text block(s).
        # Multimodal tokens are also supported in a similar fashion as for the user message.
        assistant_msg0 = formatted["messages"][1]
        assert assistant_msg0["role"] == "assistant"
        assistant_content0 = assistant_msg0["content"]
        assert isinstance(assistant_content0, list)
        assert len(assistant_content0) == 1
        assert assistant_content0[0] == {
            "type": "text",
            "text": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition.",
        }

        # Raw Third example from the jsonl
        third_raw = dataset.dataset[2]
        # Run the preprocessor (same as used in the pipeline)
        formatted = dataset.preprocessor(third_raw)

        # Expected OpenAI-compatible structure
        assert "messages" in formatted
        assert "task_name" in formatted
        assert formatted["task_name"] == "general-conversation-jsonl"

        assert len(formatted["messages"]) == 2

        # User message: content is list of audio/video/image block + text block
        user_msg0 = formatted["messages"][0]
        assert user_msg0["role"] == "user"
        user_content0 = user_msg0["content"]
        assert isinstance(user_content0, list)
        assert len(user_content0) == 3
        # the "sound" tag will be converted to the "audio" tag
        assert user_content0[0] == {
            "type": "video",
            "video": "sample_000001.35tags.mp4",
        }
        assert user_content0[1] == {
            "type": "audio",
            "audio": "sample_000001.35tags.mp4",
        }
        assert user_content0[2] == {
            "type": "text",
            "text": "\nDescribe what is NeMo based on the tutorial video. Answer: ",
        }

        # Assistant message: content is list of text block(s).
        # Multimodal tokens are also supported in a similar fashion as for the user message.
        assistant_msg0 = formatted["messages"][1]
        assert assistant_msg0["role"] == "assistant"
        assistant_content0 = assistant_msg0["content"]
        assert isinstance(assistant_content0, list)
        assert len(assistant_content0) == 1
        assert assistant_content0[0] == {
            "type": "text",
            "text": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability.",
        }

    finally:
        import os

        try:
            os.unlink(data_path)
        except OSError:
            pass
