## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from huggingface_hub import snapshot_download

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import (
    load_dataset_from_path,
    get_huggingface_cache_path,
)


class DailyOmniDataset(RawDataset):
    """Simple wrapper around the CLEVR-CoGenT dataset.

    Args:
        split: Split name for the dataset, default is "train"
    """

    task_name = "daily-omni"

    def __init__(self, split: str = "train", **kwargs):
        # train, valA, and valB are supported splits.
        SPLIT_TO_HF_NAME = {
            "train": "liarliar/Daily-Omni",
        }
        if split not in SPLIT_TO_HF_NAME:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train'."
            )

        self.hf_cache_dir = get_huggingface_cache_path(SPLIT_TO_HF_NAME[split])
        if not self.hf_cache_dir:
            # download the dataset
            self.hf_cache_dir = snapshot_download(repo_id=SPLIT_TO_HF_NAME[split], repo_type='dataset')
        if not self.hf_cache_dir:
            raise ValueError(f"Cannot download DailyOmniDataset.")

        json_file = os.path.join(self.hf_cache_dir, "qa.json")
        
        if not os.path.isfile(json_file):
            raise ValueError(f"{json_file} cannot be found.")

        files_folder = os.path.join(self.hf_cache_dir, 'Videos')
        if not os.path.isdir(files_folder):
            # prepare the dataset
            # TODO: move untar, unzip func to utils?
            import tarfile
            archive_filename = os.path.join(self.hf_cache_dir, "Videos.tar")
            if not os.path.isfile(archive_filename):
                raise ValueError(f"{archive_filename} cannot be found.")
            try:
                with tarfile.open(archive_filename, "r:*") as tar:
                    # Extract all contents to the specified path
                    tar.extractall()
                print(f"Successfully extracted '{archive_filename}' to '{files_folder}'")
            except tarfile.ReadError:
                raise tarfile.ReadErro(f"Error: Could not read the tar file. It might be corrupted or not a tar file.")
            except Exception as e:
                raise Exception(f"An unexpected error occurred: {e}")

        self.dataset = load_dataset_from_path(json_file)

        # format the dataset
        self.dataset = self.dataset.map(
            self.format_data,
            remove_columns=self.dataset.column_names,
        )

    @classmethod
    def get_prompt(cls, data: dict[str, Any]) -> str:
        # WARNING: model could have preference of a different prompt
        prompt = data["Question"] + '\n' + '\n'.join(data["Choice"])
        candidate_answers = [chr(ord("A")+idx) for idx in range(len(data["Choice"]))]
        candidate_answers_all_but_last = ",".join(candidate_answers[:-1])
        prompt += '\n' + f"Your replies must contain only a single letter " + \
            f"(either {candidate_answers_all_but_last} or {candidate_answers[-1]})."
        return prompt

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        user_content = [
            {
                "type": "video",
                "video": os.path.join(
                    self.hf_cache_dir, 
                    "Videos", 
                    data["video_id"], 
                    data["video_id"]+"_video.mp4"
                ),
            },
            {
                "type": "text",
                "text": self.get_prompt(data),
            },
        ]
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": data["Answer"]},
            ],
            "task_name": self.task_name,
        }
