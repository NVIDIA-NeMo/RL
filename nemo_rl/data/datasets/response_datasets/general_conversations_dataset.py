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

import os
from functools import partial
from typing import Any, Optional, Literal
from collections import defaultdict

from nemo_rl.data import multimodal_utils
from nemo_rl.data.datasets.utils import load_dataset_from_path
from nemo_rl.data.datasets.response_datasets import conversation_base
from nemo_rl.data.interfaces import TaskDataSpec


_DEBUG=True


class GeneralConversationsJsonlDataset:
    """Loads general conversation datasets that have the json (manifest) files and media files in separate files (jsonl datasets).
    Each sample can be single/multi-turn converstaions with multiple modalities.
    Each modality can have one or more number of media objects.
    There is no requiement of where the media tag (e.g. '<sound>') should appear in the conversations.

    The structure of the jsonl files could be like this:

    ```media files
    sample_000001.2345ew.flac
    sample_000001.35tags.mp4
    sample_000001.as23ds.jpg
    sample_000001.gd1dtg.wav
    sample_000001.gds233.jpg
    sample_000002.asf234.wav
    ...
    ```

    ```json structure
    {
      "sound": ["sample_000001.2345ew.flac", "sample_000001.gd1dtg.wav"],
      "video": "sample_000001.35tags.mp4",
      "image": ["sample_000001.as23ds.jpg", "sample_000001.gds233.jpg"],
      "conversations": [
        {
          "from": "user",
          "value": "<sound>"
        },
        {
          "from": "assistant",
          "value": "Automatic speech recognition is a technology that allows computers to recognize and transcribe spoken language. In the NeMo Framework, ASR is used for tasks such as speech-to-text and voice recognition."
        },
        {
          "from": "user",
          "value": "Describe what is NeMo based on the tutorial video: <video> and the information in the two images: <image> <image>. Combine that information with sound <sound>. Answer: "
        },
        {
          "from": "assistant",
          "value": "The NeMo Framework provides a range of tools and features for training and deploying ASR models, including model parallelism, data parallelism, and distributed checkpointing. This allows for faster training and inference times, as well as improved model accuracy and reliability."
        }
      ]
    }
    ```
    """

    task_name = "general-conversation-jsonl"

    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
        train_media_data_dir: Optional[str] = None,
        val_media_data_dir: Optional[str] = None,
    ):
        self.train_media_data_dir = train_media_data_dir
        self.val_media_data_dir = val_media_data_dir
        train_ds = load_dataset_from_path(train_data_path, train_split)
        if val_data_path:
            val_ds = load_dataset_from_path(val_data_path, val_split)
        else:
            val_ds = None

        # store the formatted dataset
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }

        self.datum_preprocessor = {
            "train": partial(self._datum_preprocessor, media_directory=train_media_data_dir),
            "val": partial(self._datum_preprocessor, media_directory=val_media_data_dir)
        }
        
        self.task_spec = TaskDataSpec(task_name="GeneralConversationsJsonlDataset")

    @classmethod
    def process_message_fragment(cls, tag: str, fragment: Any, media_directory: Optional[str] = None) -> dict[str, Any]:
        if media_directory is not None and \
            tag in multimodal_utils.media_tags and \
            isinstance(fragment, str) and \
            not os.path.isfile(fragment):
            media_path = os.path.join(media_directory, fragment)
            if os.path.isfile(media_path):
                fragment = media_path
        ret = []
        for t in tag.split('-'):
            ret.append({"type": t, t: fragment})
        return ret

    @classmethod
    def _datum_preprocessor(
        cls, example: dict[str, Any],
        media_directory: Optional[str] = None
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Convert the json structure into an OpenAI-API-like message log.
        """
        processed_example = {
            "messages": [],
            "task_name": cls.task_name,
        }

        if "conversations" in example:
            media_index = defaultdict(int)
            tried_default_extensions = set()
            data = conversation_base.convert_metadata(example)

            for message in data["conversations"]:
                role = message["from"]
                if role not in {"user", "assistant"}:
                    role = conversation_base.conversation_sender_mapping_sample_to_allowed[role]
                content = conversation_base.conversation_process_message(
                    data,
                    message,
                    media_index,
                    allow_empty_text=True,
                    check_if_media_file_exist=False,
                    tried_default_extensions=tried_default_extensions,
                    process_message_fragment=partial(cls.process_message_fragment, media_directory=media_directory),
                )

                processed_example["messages"].append({"role": role, "content": content})

        return processed_example
