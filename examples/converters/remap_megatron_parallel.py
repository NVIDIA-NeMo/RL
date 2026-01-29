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

from nemo_rl.models.megatron.community_import import mp_overrides_model_from_megatron

""" NOTE: this script requires mcore. Make sure to launch with the mcore extra:
uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
  --config <path_to_ckpt>/config.yaml \
  --megatron-ckpt-path <path_to_ckpt>/policy/weights/iter_xxxxx \
  --hf-ckpt-path <path_to_save_hf_ckpt>
"""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert Torch DCP checkpoint to HF checkpoint"
    )
    parser.add_argument("--hf-model-name", type=str, default=None, help="HF model name")
    parser.add_argument(
        "--megatron-ckpt-path",
        type=str,
        default=None,
        help="Path to Megatron checkpoint",
    )
    parser.add_argument(
        "--output-path", type=str, default=None, help="Path to save Megatron checkpoint"
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Whether to perform non-strict validation during weight export",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallelism size")
    parser.add_argument(
        "--etp", type=int, default=1, help="Expert tensor parallelism size"
    )
    parser.add_argument("--cp", type=int, default=1, help="Context parallelism size")
    parser.add_argument(
        "--quant-cfg",
        type=str,
        default=None,
        help="Quantization config name (enables post-wrap quantization)",
    )
    parser.add_argument(
        "--quant-calib-data",
        type=str,
        default="random",
        help="Calibration dataset name",
    )
    parser.add_argument(
        "--quant-calib-size",
        type=int,
        default=512,
        help="Total calibration dataset size",
    )
    parser.add_argument(
        "--quant-batch-size",
        type=int,
        default=1,
        help="Calibration batch size",
    )
    parser.add_argument(
        "--quant-sequence-length",
        type=int,
        default=2048,
        help="Max calibration sample length",
    )
    parser.add_argument(
        "--save-hf",
        action="store_true",
        help="Whether to save HF checkpoint",
    )
    parser.add_argument(
        "--restore-modelopt-state",
        action="store_true",
        help="Whether to load modelopt state",
    )
    # Parse known args for the script
    args = parser.parse_args()

    return args


def main():
    """Main entry point."""
    args = parse_args()

    # with open(args.config, "r") as f:
    #     config = yaml.safe_load(f)

    model_name = args.hf_model_name
    tokenizer_name = model_name
    # tokenizer_name = config["policy"]["tokenizer"]["name"]
    # hf_overrides = config["policy"].get("hf_overrides", {}) or {}
    hf_overrides = {}

    mp_overrides_model_from_megatron(
        hf_model_name=model_name,
        input_path=args.megatron_ckpt_path,
        output_path=args.output_path,
        hf_tokenizer_path=tokenizer_name,
        hf_overrides=hf_overrides,
        strict=not args.non_strict,
        quant_cfg=args.quant_cfg,
        quant_calib_data=args.quant_calib_data,
        quant_calib_size=args.quant_calib_size,
        quant_batch_size=args.quant_batch_size,
        quant_sequence_length=args.quant_sequence_length,
        mp_overrides={
            "tensor_model_parallel_size": args.tp,
            "pipeline_model_parallel_size": args.pp,
            "expert_model_parallel_size": args.ep,
            "expert_tensor_parallel_size": args.etp,
            "context_parallel_size": args.cp,
        },
        save_hf=args.save_hf,
        restore_modelopt_state=args.restore_modelopt_state,
    )


if __name__ == "__main__":
    main()
