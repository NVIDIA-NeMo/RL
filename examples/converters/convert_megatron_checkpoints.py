#!/usr/bin/env python3
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

"""
Megatron-HuggingFace Checkpoint Conversion Example

This script demonstrates how to convert models between HuggingFace and Megatron formats
using the AutoBridge import_ckpt and export_ckpt methods.

Features:
- Import HuggingFace models to Megatron checkpoint format
- Export Megatron checkpoints to HuggingFace format
- Support for various model architectures (GPT, Llama, etc.)
- Configurable model and conversion settings

When used via the shell launcher:
  # HF -> Megatron (import)
  HF_PATH=meta-llama/Llama-3.2-1B MEGATRON_PATH=/path/to/output \
    ./examples/converters/launch_convert_megatron.sh import

  # Megatron -> HF (export)
  MEGATRON_PATH=/path/to/megatron/ckpt HF_PATH=/path/to/hf/output \
    ./examples/converters/launch_convert_megatron.sh export

Direct Python usage (single-node):
  # Import a HuggingFace model to Megatron format
  python examples/converters/convert_megatron_checkpoints.py import \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b \
    --tp 1 --pp 1 --ep 1 --etp 1

  # Export a Megatron checkpoint to HuggingFace format
  python examples/converters/convert_megatron_checkpoints.py export \
    --hf-model meta-llama/Llama-3.2-1B \
    --megatron-path ./checkpoints/llama3_2_1b \
    --hf-path ./exports/llama3_2_1b_hf \
    --tp 1 --pp 1 --ep 1 --etp 1
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import print_rank_0


def validate_path(path: str, must_exist: bool = False) -> Path:
    """Validate and convert string path to Path object."""
    path_obj = Path(path)
    if must_exist and not path_obj.exists():
        raise ValueError(f"Path does not exist: {path}")
    return path_obj


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}. Supported: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def import_hf_to_megatron(
    hf_model: str,
    megatron_path: str,
    torch_dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
) -> None:
    """
    Import a HuggingFace model and save it as a Megatron checkpoint.

    Args:
        hf_model: HuggingFace model ID or path to model directory
        megatron_path: Directory path where the Megatron checkpoint will be saved
        torch_dtype: Model precision ("float32", "float16", "bfloat16")
        device_map: Device placement strategy ("auto", "cuda:0", etc.)
        tp: Tensor parallel size
        pp: Pipeline parallel size
        ep: Expert parallel size
        etp: Expert tensor parallel size
    """
    print(f"Starting import: {hf_model} -> {megatron_path}")

    kwargs = {}
    if torch_dtype:
        kwargs["torch_dtype"] = get_torch_dtype(torch_dtype)
        print(f"   Using torch_dtype: {torch_dtype}")

    if device_map:
        kwargs["device_map"] = device_map
        print(f"   Using device_map: {device_map}")

    # Always allow custom model code execution for compatibility with community models.
    kwargs["trust_remote_code"] = True

    # Import using the convenience method
    print(f"Loading HuggingFace model: {hf_model}")

    bridge = AutoBridge.from_hf_pretrained(hf_model, **kwargs)

    model_provider = bridge.to_megatron_provider(load_weights=True)

    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.sequence_parallel = False

    # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    # Load the Megatron model directly
    print_rank_0("Loading Megatron model...")
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)
    print_rank_0(megatron_model)

    bridge.save_megatron_model(megatron_model, megatron_path, hf_tokenizer_path=hf_model)
    print_rank_0(f"Successfully imported model to: {megatron_path}")


def export_megatron_to_hf(
    hf_model: str,
    megatron_path: str,
    hf_path: str,
    show_progress: bool = True,
    strict: bool = True,
    tp: int = 1,
    pp: int = 1,
    ep: int = 1,
    etp: int = 1,
) -> None:
    """
    Export a Megatron checkpoint to HuggingFace format.

    Args:
        hf_model: HF model ID used as architecture reference (weights are not loaded)
        megatron_path: Directory containing the Megatron checkpoint
        hf_path: Output directory for the HuggingFace model
        show_progress: Display progress bar during weight export
        strict: Require source and target checkpoint keys to match exactly
        tp: Tensor parallel size
        pp: Pipeline parallel size
        ep: Expert parallel size
        etp: Expert tensor parallel size
    """
    print_rank_0(f"Starting export: {megatron_path} -> {hf_path}")

    validate_path(megatron_path, must_exist=True)

    bridge = AutoBridge.from_hf_pretrained(hf_model, trust_remote_code=True)

    print_rank_0("Exporting to HuggingFace format...")
    model_provider = bridge.to_megatron_provider(load_weights=False)
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = pp
    model_provider.expert_model_parallel_size = ep
    model_provider.expert_tensor_parallel_size = etp
    model_provider.pipeline_dtype = torch.bfloat16
    model_provider.sequence_parallel = False

    # Once all overrides are set, finalize the model provider to ensure the post initialization logic is run
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    # Load the Megatron model directly
    megatron_model = bridge.load_megatron_model(megatron_path, wrap_with_ddp=False)

    bridge.save_hf_pretrained(megatron_model, hf_path, show_progress=show_progress)
    print_rank_0(f"Successfully exported model to: {hf_path}")


def main():
    """Main function to handle command line arguments and execute conversions."""
    parser = argparse.ArgumentParser(
        description="Convert models between HuggingFace and Megatron formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Conversion direction")

    # Import subcommand (HF -> Megatron)
    import_parser = subparsers.add_parser("import", help="Import HuggingFace model to Megatron checkpoint format")
    import_parser.add_argument("--hf-model", required=True, help="HuggingFace model ID or path to model directory")
    import_parser.add_argument(
        "--megatron-path", required=True, help="Directory path where the Megatron checkpoint will be saved"
    )
    import_parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    import_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    import_parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    import_parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    import_parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], help="Model precision")
    import_parser.add_argument("--device-map", help='Device placement strategy (e.g., "auto", "cuda:0")')

    # Export subcommand (Megatron -> HF)
    export_parser = subparsers.add_parser("export", help="Export Megatron checkpoint to HuggingFace format")
    export_parser.add_argument(
        "--hf-model", required=True, help="HF model ID for architecture reference (weights are not loaded)"
    )
    export_parser.add_argument(
        "--megatron-path", required=True, help="Directory path where the Megatron checkpoint is stored"
    )
    export_parser.add_argument(
        "--hf-path", required=True, help="Directory path where the HuggingFace model will be saved"
    )
    export_parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    export_parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    export_parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    export_parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallelism size")
    export_parser.add_argument("--no-progress", action="store_true", help="Disable progress bar during export")
    export_parser.add_argument(
        "--not-strict", action="store_true", help="Allow source and target checkpoint to have different keys"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "import":
        import_hf_to_megatron(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            tp=args.tp,
            pp=args.pp,
            ep=args.ep,
            etp=args.etp,
        )

    elif args.command == "export":
        export_megatron_to_hf(
            hf_model=args.hf_model,
            megatron_path=args.megatron_path,
            hf_path=args.hf_path,
            show_progress=not args.no_progress,
            strict=not args.not_strict,
            tp=args.tp,
            pp=args.pp,
            ep=args.ep,
            etp=args.etp,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.command}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
