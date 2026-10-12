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
from typing import Any, Optional

import torch
import yaml
from megatron.bridge import AutoBridge

""" NOTE: this script requires mcore. Make sure to launch with the mcore extra:
uv run --extra mcore torchrun --nproc_per_node=8 --nnodes=1 examples/converters/convert_megatron_to_hf_super.py \
  --config <path_to_ckpt>/config.yaml \
  --megatron-ckpt-path <path_to_ckpt>/policy/weights/iter_xxxxx \
  --hf-ckpt-path <path_to_save_hf_ckpt> \
  --tp <tensor_model_parallel_size> \
  --pp <pipeline_model_parallel_size> \
  --ep <expert_model_parallel_size> \
  --etp <expert_tensor_parallel_size>
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
        "--megatron-ckpt-path",
        type=str,
        default=None,
        help="Path to Megatron checkpoint",
    )
    parser.add_argument(
        "--hf-ckpt-path", type=str, default=None, help="Path to save HF checkpoint"
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="Tensor model parallel size"
    )
    parser.add_argument(
        "--pp", type=int, default=1, help="Pipeline model parallel size"
    )
    parser.add_argument(
        "--ep", type=int, default=1, help="Expert model parallel size"
    )
    parser.add_argument(
        "--etp", type=int, default=1, help="Expert tensor parallel size"
    )
    # Parse known args for the script
    args = parser.parse_args()

    return args


def export_model_from_megatron_gpu(
    hf_model_name: str,
    input_path: str,
    output_path: str,
    hf_tokenizer_path: str,
    overwrite: bool = False,
    hf_overrides: Optional[dict[str, Any]] = {},
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {output_path}. Delete it to run or set overwrite=True."
        )

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **hf_overrides
    )

    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.sequence_parallel=False

    # FIXME: This is a hack to enable cuda graph for the model.
    model_provider.enable_cuda_graph=True
    model_provider.use_te_rng_tracker=True

    # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0, seed_kwargs={"te_rng_tracker": model_provider.use_te_rng_tracker})

    # Load the Megatron model directly
    megatron_model = bridge.load_megatron_model(input_path, wrap_with_ddp=False)

    bridge.save_hf_pretrained(megatron_model, output_path, show_progress=True)


def main():
    """Main entry point."""
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["policy"]["model_name"]
    tokenizer_name = config["policy"]["tokenizer"]["name"]
    hf_overrides = config["policy"].get("hf_overrides", {}) or {}

    export_model_from_megatron_gpu(
        hf_model_name=model_name,
        input_path=args.megatron_ckpt_path,
        output_path=args.hf_ckpt_path,
        hf_tokenizer_path=tokenizer_name,
        hf_overrides=hf_overrides,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        etp=args.etp,
    )


if __name__ == "__main__":
    main()
