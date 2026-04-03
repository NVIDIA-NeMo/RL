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
import os
import yaml

from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf


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
        "--dcp-ckpt-path", type=str, default=None, help="Path to DCP checkpoint"
    )
    parser.add_argument(
        "--hf-ckpt-path", type=str, default=None, help="Path to save HF checkpoint"
    )
    # Parse known args for the script
    args = parser.parse_args()

    return args


def main():
    """Main entry point."""
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name_or_path = config["policy"]["model_name"]    
    # Since we have now merged https://github.com/NVIDIA-NeMo/RL/pull/148/files
    # we want to copy the tokenizer from policy/tokenizer/* instead of relying on the model name
    # from the config file
    # The reason is that some algos may change the tokenizer.chat_template property at runtime
    # and we want this to persist correctly from DCP ckpts down to HF converted ckpts
    
    # this (somewhat brittle) logic assumes that dcp_ckpt_path will always be policy/weights/ and the
    # tokenizer files are always located at policy/tokenizer/*
    if os.path.exists(tokenizer_path := os.path.join(args.dcp_ckpt_path, "..", "tokenizer")):
        print(f"Using local tokenizer path at {tokenizer_path} for HF conversion")
        tokenizer_name_or_path = tokenizer_path
    else:
        print(f"WARNING: No local tokenizer path found at {tokenizer_path}. Falling back to loading the vanilla tokenizer based on the config file. Please ensure this is what you want.")
        tokenizer_name_or_path = config["policy"]["model_name"]
    hf_overrides = config["policy"].get("hf_overrides", {}) or {}

    hf_ckpt = convert_dcp_to_hf(
        dcp_ckpt_path=args.dcp_ckpt_path,
        hf_ckpt_path=args.hf_ckpt_path,
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        hf_overrides=hf_overrides,
    )
    print(f"Saved HF checkpoint to: {hf_ckpt}")


if __name__ == "__main__":
    main()
