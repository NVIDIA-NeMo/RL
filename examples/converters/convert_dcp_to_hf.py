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
import glob
import os
import yaml

from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf

from transformers import AutoModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Torch DCP checkpoint to HF checkpoint"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to the checkpoint directory with config.yaml in it",
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

    config = f"{args.checkpoint_path}/config.json"

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    model_name_or_path = config["policy"]["model_name"]
    tokenizer_name_or_path = f"{args.checkpoint_path}/policy/tokenizer"

    hf_ckpt = convert_dcp_to_hf(
        dcp_ckpt_path=f"{args.checkpoint_path}/policy/weights",
        hf_ckpt_path=args.hf_ckpt_path,
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        overwrite=True,
    )

    model = AutoModel.from_pretrained(args.hf_ckpt_path)
    model.save_pretrained(
        args.hf_ckpt_path,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    for f in glob.glob(os.path.join(args.hf_ckpt_path, "*.bin")) + glob.glob(
        os.path.join(args.hf_ckpt_path, "*.bin.index.json")
    ):
        os.remove(f)

    print(f"Saved HF checkpoint to: {hf_ckpt}")


if __name__ == "__main__":
    main()
