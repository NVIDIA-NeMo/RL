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
import io
import copy
import warnings
import dataclasses
from PIL import Image
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Callable, Optional

from nemo_rl.data import multimodal_utils


# map the senders from the sample to the allowed ones
conversation_sender_mapping_sample_to_allowed = {
    'human': 'user',
    'gpt': 'assistant',
    'agent': 'assistant',
}


# convert 
def convert_metadata(metadata: Dict[str, Any], return_inplace=False):
    data = metadata
    if not return_inplace:
        data = metadata.copy()

    for tag in multimodal_utils.media_tags_to_allowed:
        if tag in data:
            tag_mapped = multimodal_utils.media_tags_to_allowed[tag]
            if tag_mapped not in data:
                data[tag_mapped] = data[tag]
                del  data[tag]
            else:
                warnings.warn(
                    f"Trying to map {tag} to {tag_mapped}, but {tag_mapped} already exists in the raw data. Mapping is not carried out."
                )

    for idx, message in enumerate(data["conversations"]):
        msg_str = message["value"]
        for tag in multimodal_utils.media_tags_to_allowed:
            tag_str = '<' + tag + '>'
            if tag_str in msg_str:
                tag_str_mapped = multimodal_utils.media_tags[
                    multimodal_utils.media_tags_to_allowed[tag]
                ]
                msg_str = msg_str.replace(tag_str, tag_str_mapped)
        message["value"] = msg_str
        data["conversations"][idx] = message

    if not return_inplace:
        return data


def conversation_process_message(
    metadata: Dict[str, Any],
    message: Dict[str, str],
    media_index: dict,
    raw: Dict[str, Any] = {},
    allow_empty_text: bool = False,
    check_if_media_file_exist: bool = True,
    tried_default_extensions: set = set(),
    tags_mapping_sample_to_allowed: Dict[str, str] = multimodal_utils.media_tags_to_allowed,
    process_message_fragment: Callable = lambda tag, fragment: [{tag: fragment}],
) -> list[Dict[str, Any]]:
    """
    Args:
        raw: dictionary with all webdataset compliant keys of a sample. 
            Emtpy for jsonl dataset, non-empty otherwise
        metadata: 
    """
    fragments = []    
    parts = re.split(multimodal_utils.media_tag_pattern, message["value"])
    
    # Convert the parts to message fragments
    empty_text = True
    for i, part in enumerate(parts):
        if part in multimodal_utils.media_tags.values():
            # process multimodal tags
            tag = multimodal_utils.media_tags_reversed[part]
            if not isinstance(metadata[tag], list):
                metadata[tag] = [metadata[tag]]
            # try to extract the media object from the shard
            ext = os.path.basename(metadata[tag][media_index[tag]]).split('.', 1)[1]
            if raw and ext not in raw and \
                tag not in tried_default_extensions and \
                tag in multimodal_utils.default_media_extensions:
                # try the default extension
                for ext in multimodal_utils.default_media_extensions[tag]:
                    if ext in raw:
                        tried_default_extensions.add(ext)
                        break
            media_file = None
            if ext in raw:
                media_file = ext
            elif isinstance(metadata[tag][media_index[tag]], str) and \
                os.path.isfile(metadata[tag][media_index[tag]]):
                # if cannot get it from the shard files, try to find the local file
                media_file = metadata[tag][media_index[tag]]
            elif check_if_media_file_exist:
                sample_to_print = raw if raw else metadata
                raise ValueError(f"Cannot find the media file {metadata[tag][media_index[tag]]} from {sample_to_print} or locally.")
            else:
                media_file = metadata[tag][media_index[tag]]
            media_index[tag] += 1
            fragments += process_message_fragment(tag, media_file)
        else:
            # process text
            if part.strip():
                fragments += process_message_fragment('text', part)
                empty_text = False
                
    if not allow_empty_text and empty_text:
        fragments += process_message_fragment('text', ' ')

    return fragments
