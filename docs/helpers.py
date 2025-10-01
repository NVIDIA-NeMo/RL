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


def make_dpo_dataset():
    train_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    val_file = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)

    # Write train data
    train_data = [
        {"context": "What is 2+2?", "chosen": "4", "rejected": "5"},
        {"context": "What is 3*3?", "chosen": "9", "rejected": "6"},
    ]
    for item in train_data:
        lines = train_file.write(json.dumps(item) + "\n")
    train_file.flush()

    # Write validation data
    val_data = [
        {"context": "What is 4+4?", "chosen": "8", "rejected": "7"},
        {"context": "What is 5*5?", "chosen": "25", "rejected": "20"},
    ]
    for item in val_data:
        lines = val_file.write(json.dumps(item) + "\n")
    val_file.flush()

    return train_file, val_file
