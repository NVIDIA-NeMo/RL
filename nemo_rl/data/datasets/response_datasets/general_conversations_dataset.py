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
import re
import warnings
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Optional

from nemo_rl.data import multimodal_utils
from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import load_dataset_from_path

# map the senders from the sample to the allowed ones
conversation_sender_mapping_sample_to_allowed = {
    "human": "user",
    "gpt": "assistant",
    "agent": "assistant",
}


# convert
def convert_metadata(metadata: Dict[str, Any]):
    data = metadata.copy()

    for tag in multimodal_utils.MEDIA_TAGS_TO_ALLOWED:
        if tag in data:
            tag_mapped = multimodal_utils.MEDIA_TAGS_TO_ALLOWED[tag]
            if tag_mapped not in data:
                data[tag_mapped] = data[tag]
                del data[tag]
            else:
                warnings.warn(
                    f"Trying to map {tag} to {tag_mapped}, but {tag_mapped} already exists in the raw data. Mapping is not carried out."
                )

    for idx, message in enumerate(data["conversations"]):
        msg_str = message["value"]
        for tag in multimodal_utils.MEDIA_TAGS_TO_ALLOWED:
            tag_str = "<" + tag + ">"
            if tag_str in msg_str:
                tag_str_mapped = multimodal_utils.MEDIA_TAGS[
                    multimodal_utils.MEDIA_TAGS_TO_ALLOWED[tag]
                ]
                msg_str = msg_str.replace(tag_str, tag_str_mapped)
        message["value"] = msg_str
        data["conversations"][idx] = message

    return data


def conversation_process_message(
    metadata: Dict[str, Any],
    message: Dict[str, str],
    media_index: dict,
    raw: Optional[Dict[str, Any]] = None,
    allow_empty_text: bool = False,
    check_if_media_file_exist: bool = True,
    tried_default_extensions: Optional[set] = None,
    process_message_fragment: Callable = lambda tag, fragment: [{tag: fragment}],
) -> list[Dict[str, Any]]:
    """Convert one conversation message from a string to a list of dictionaries representing media or text.

    Args:
        raw: dictionary with all webdataset compliant keys of a sample.
            Emtpy for jsonl dataset, non-empty otherwise.
        metadata:
    """
    if raw is None:
        raw = {}
    if tried_default_extensions is None:
        tried_default_extensions = set()
    fragments = []
    parts = re.split(multimodal_utils.MEDIA_TAG_PATTERN, message["value"])

    # Convert the parts to message fragments
    empty_text = True
    for i, part in enumerate(parts):
        if part in multimodal_utils.MEDIA_TAGS.values():
            # process multimodal tags
            tag = multimodal_utils.MEDIA_TAGS_REVERSED[part]
            if tag not in metadata:
                raise ValueError(
                    f"{part} is found in the message, but no corresponding {tag} key can be found in {metadata}"
                )
            if not isinstance(metadata[tag], list):
                metadata[tag] = [metadata[tag]]
            # try to extract the media object from the shard
            basename = os.path.basename(metadata[tag][media_index[tag]])
            ext = basename.split(".", 1)[1] if "." in basename else ""
            if (
                raw
                and ext not in raw
                and ext not in tried_default_extensions
                and tag in multimodal_utils.DEFAULT_MEDIA_EXTENSIONS
            ):
                # try the default extension
                for ext in multimodal_utils.DEFAULT_MEDIA_EXTENSIONS[tag]:
                    if ext in raw:
                        tried_default_extensions.add(ext)
                        break
            media_file = None
            if ext in raw:
                media_file = ext
            elif isinstance(metadata[tag][media_index[tag]], str) and os.path.isfile(
                metadata[tag][media_index[tag]]
            ):
                # if cannot get it from the shard files, try to find the local file
                media_file = metadata[tag][media_index[tag]]
            elif check_if_media_file_exist:
                sample_to_print = raw if raw else metadata
                raise ValueError(
                    f"Cannot find the media file {metadata[tag][media_index[tag]]} from {sample_to_print} or locally."
                )
            else:
                media_file = metadata[tag][media_index[tag]]
            media_index[tag] += 1
            fragments += process_message_fragment(tag, media_file)
        else:
            # process text
            if part.strip():
                fragments += process_message_fragment("text", part)
                empty_text = False

    if not allow_empty_text and empty_text:
        fragments += process_message_fragment("text", " ")

    return fragments


class GeneralConversationsJsonlDataset(RawDataset):
    """Loads general conversation datasets that have the json (manifest) files and media files in separate files (jsonl datasets).

    Each sample can be single/multi-turn conversations with multiple modalities.
    Each modality can have one or more number of media objects.
    There is no requirement of where the media tag (e.g. '<sound>') should appear in the conversations.

    The structure of the jsonl files could be like this.

    Example media filenames::

        sample_000001.2345ew.flac
        sample_000001.35tags.mp4
        sample_000001.as23ds.jpg
        sample_000001.gd1dtg.wav
        sample_000001.gds233.jpg
        sample_000002.asf234.wav
        ...

    Example JSON structure::

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
    """

    task_name = "general-conversation-jsonl"

    def __init__(
        self,
        data_path: str,
        media_data_dir: Optional[str] = None,
        split_validation_size: float = 0,
        seed: int = 42,
        **kwargs,
    ):
        self.media_data_dir = media_data_dir
        self.dataset = load_dataset_from_path(data_path)
        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )

        self.preprocessor = partial(
            self._datum_preprocessor, media_directory=media_data_dir
        )

        # `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)

    @classmethod
    def process_message_fragment(
        cls, tag: str, fragment: Any, media_directory: Optional[str] = None
    ) -> list[dict[str, Any]]:
        if (
            media_directory is not None
            and tag in multimodal_utils.MEDIA_TAGS
            and isinstance(fragment, str)
            and not os.path.isfile(fragment)
        ):
            media_path = os.path.join(media_directory, fragment)
            if os.path.isfile(media_path):
                fragment = media_path
        ret = []
        for t in tag.split("-"):
            ret.append({"type": t, t: fragment})
        return ret

    @classmethod
    def _datum_preprocessor(
        cls, example: dict[str, Any], media_directory: Optional[str] = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Convert the json structure into an OpenAI-API-like message log."""
        processed_example = {
            "messages": [],
            "task_name": cls.task_name,
        }

        if "conversations" in example:
            media_index = defaultdict(int)
            tried_default_extensions = set()
            data = convert_metadata(example)

            for message in data["conversations"]:
                role = message["from"]
                if role not in {"user", "assistant"}:
                    role = conversation_sender_mapping_sample_to_allowed.get(role)
                    if role is None:
                        raise ValueError(
                            f"Unknown conversation role: {message['from']}"
                        )
                content = conversation_process_message(
                    data,
                    message,
                    media_index,
                    allow_empty_text=True,
                    check_if_media_file_exist=False,
                    tried_default_extensions=tried_default_extensions,
                    process_message_fragment=partial(
                        cls.process_message_fragment, media_directory=media_directory
                    ),
                )

                processed_example["messages"].append({"role": role, "content": content})

        return processed_example
