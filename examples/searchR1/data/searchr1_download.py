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

# Adapted from https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-train/examples/search/searchr1_download.py

import argparse

from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(
    description="Download files from a Hugging Face dataset repository."
)
parser.add_argument(
    "--repo_id",
    type=str,
    default="PeterJinGo/wiki-18-e5-index",
    help="Hugging Face repository ID",
)
parser.add_argument(
    "--local_dir", type=str, required=True, help="Local directory to save files"
)

args = parser.parse_args()

repo_id = "PeterJinGo/wiki-18-e5-index"
for file in ["part_aa", "part_ab"]:
    hf_hub_download(
        repo_id=repo_id,
        filename=file,  # e.g., "e5_Flat.index"
        repo_type="dataset",
        local_dir=args.local_dir,
    )

repo_id = "PeterJinGo/wiki-18-corpus"
hf_hub_download(
    repo_id=repo_id,
    filename="wiki-18.jsonl.gz",
    repo_type="dataset",
    local_dir=args.local_dir,
)
