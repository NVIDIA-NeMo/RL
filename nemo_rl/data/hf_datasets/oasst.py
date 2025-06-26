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

import copy
import gzip
import json
import os
import random

import requests

from nemo_rl.data.interfaces import TaskDataSpec

SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"


def parse_conversations(tree_obj, first: bool = False):
    """Recusive function that returns all the sub converstaions in a list starting from node tree_obj.

    Args:
        tree_obj (obj): current conversation node

    Returns:
        a list of sub conversation threads including the current conversation node
    """
    turns = []
    if first:
        turn = {"content": SYSTEM_PROMPT, "role": "system"}
        turns.append(turn)

    if "prompt" in tree_obj:
        prompt_obj = tree_obj["prompt"]
    elif "text" in tree_obj and "role" in tree_obj:
        prompt_obj = tree_obj
    else:
        return [[]]
    if prompt_obj["role"] == "prompter":
        role = "user"
    elif prompt_obj["role"] == "assistant":
        role = "assistant"
    else:
        raise ValueError(f"unknown role {prompt_obj['role']}")
    turn = {"content": prompt_obj["text"], "role": role}
    turns.append(turn)

    all_conversations = []
    multiple_sub_threads = []
    for next_obj in prompt_obj["replies"]:
        multiple_threads = parse_conversations(next_obj)
        multiple_sub_threads.extend(multiple_threads)
    if len(multiple_sub_threads) != 0:
        for sub_thread in multiple_sub_threads:
            all_conversations.append(copy.deepcopy(turns) + sub_thread)
    else:
        all_conversations.append(copy.deepcopy(turns))
    return all_conversations


def get_data_records(objs):
    ## TODO: old format was multi-conversation per example, but ours is single conversation
    ## is this just because of the input data format?
    output = []
    for obj in objs:
        multi_conversations = parse_conversations(obj, first=True)
        for conversations in multi_conversations:
            if len(conversations) <= 2:
                # remove single turn conversations
                ## system prompt is always first turn
                continue

            conversation_obj = {
                "messages": conversations,
            }
            output.append(conversation_obj)
    return output


def download_and_process_oasst(
    output_directory: str = ".",
    seed: int = 42,
    split_ratio: float = 0.95,
) -> dict[str, list]:
    os.makedirs(output_directory, exist_ok=True)
    filename = f"{output_directory}/2023-04-12_oasst_all.trees.jsonl.gz"

    # only download if doesn't exist
    if not os.path.isfile(filename):
        url = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_all.trees.jsonl.gz"
        response = requests.get(url)
        with open(filename, mode="wb") as fw:
            fw.write(response.content)

    with gzip.open(filename) as f:
        file_content = f.readlines()

    all_objs = [json.loads(dp.decode("utf-8")) for dp in file_content]

    random.seed(seed)
    random.shuffle(all_objs)
    train_num = int(len(all_objs) * split_ratio)
    train_objs = all_objs[:train_num]
    val_objs = all_objs[train_num:]
    train_records = get_data_records(train_objs)
    val_records = get_data_records(val_objs)

    formatted_ds = {
        "train": train_records,
        "validation": val_records,
    }

    return formatted_ds


class OasstDataset:
    def __init__(self, output_dir: str = ".") -> None:
        self.formatted_ds = download_and_process_oasst(output_dir)
        self.task_spec = TaskDataSpec(
            task_name="OASST",
        )
