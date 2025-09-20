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

import argparse

import yaml

from nemo_rl.models.megatron.community_import import export_model_from_megatron

""" NOTE: this script requires mcore. Make sure to launch with the mcore extra:

Option 1 - Using config file:
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
  --config <path_to_ckpt>/config.yaml \
  --megatron-ckpt-path <path_to_ckpt>/policy/weights/iter_xxxxx \
  --hf-ckpt-path <path_to_save_hf_ckpt>

Option 2 - Using HF model name directly:
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
  --hf-model <hf_model_name> \
  --megatron-ckpt-path <path_to_ckpt>/policy/weights/iter_xxxxx \
  --hf-ckpt-path <path_to_save_hf_ckpt>
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Torch DCP checkpoint to HF checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file in the checkpoint directory",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=None,
        help="HuggingFace model name to use directly (alternative to --config)",
    )
    parser.add_argument(
        "--megatron-ckpt-path",
        type=str,
        default=None,
        help="Path to Megatron checkpoint",
    )
    parser.add_argument(
        "--hf-ckpt-path", type=str, default=None, help="Path to save HF checkpoint"
    )
    # Parse known args for the script
    args = parser.parse_args()

    # Validate that either config or hf_model is provided, but not both (XOR)
    if not (bool(args.config) ^ bool(args.hf_model)):
        parser.error("Exactly one of --config or --hf-model must be provided.")

    return args


def main():
    """Main entry point."""
    args = parse_args()

    if args.config:
        # Load model and tokenizer names from config file and populate args
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        args.hf_model = config["policy"]["model_name"]
        args.hf_tokenizer = config["policy"]["tokenizer"]["name"]
    else:
        # Use the provided hf_model name for both model and tokenizer
        args.hf_tokenizer = args.hf_model

    export_model_from_megatron(
        hf_model_name=args.hf_model,
        input_path=args.megatron_ckpt_path,
        output_path=args.hf_ckpt_path,
        hf_tokenizer_path=args.hf_tokenizer,
    )


if __name__ == "__main__":
    main()
