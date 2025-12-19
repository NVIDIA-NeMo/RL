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

from datasets import Dataset
from huggingface_hub import hf_hub_download

from nemo_rl.data.datasets.raw_dataset import RawDataset

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


def get_data_records(objs, task_name: str = "oasst"):
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
                "task_name": task_name,
            }
            output.append(conversation_obj)
    return output


class OasstDataset(RawDataset):
    def __init__(
        self,
        split_validation_size: float = 0.05,
        seed: int = 42,
        **kwargs,
    ):
        self.task_name = "oasst"

        # load from huggingface
        filename = hf_hub_download(
            repo_id="OpenAssistant/oasst1",
            filename="2023-04-12_oasst_all.trees.jsonl.gz",
            repo_type="dataset",
        )
        with gzip.open(filename) as f:
            file_content = f.readlines()

        # format the dataset
        all_objs = [json.loads(dp.decode("utf-8")) for dp in file_content]
        self.dataset = get_data_records(all_objs, task_name=self.task_name)
        self.dataset = Dataset.from_list(self.dataset)

        # `self.val_dataset` is used (not None) only when current dataset is used for both training and validation
        self.val_dataset = None
        self.split_train_validation(split_validation_size, seed)
