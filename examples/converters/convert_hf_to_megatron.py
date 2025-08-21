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

"""Import a Hugging Face model into Megatron checkpoint format compatiable with the NeMo-RL Megatron policy.

NOTE: this script requires mcore. Make sure to launch with the mcore extra:
uv run --extra mcore python examples/convert_hf_to_megatron.py \
  --hf-model-name <hf_model_or_path> \
  --output-path </path/to/output_dir>
"""

import argparse
import os

from nemo_rl.models.megatron.community_import import import_model_from_hf_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import a Hugging Face model into Megatron checkpoint format",
    )
    parser.add_argument(
        "--hf-model-name",
        type=str,
        required=True,
        help=(
            "Hugging Face model ID or local path (e.g., 'meta-llama/Llama-3.1-8B-Instruct')."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help=(
            "Directory to write the Megatron checkpoint (e.g., /tmp/megatron_ckpt)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    hf_model_name = args.hf_model_name
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    import_model_from_hf_name(hf_model_name, output_path)


if __name__ == "__main__":
    main()


