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

"""Convert a megatron-bridge iter dir to upstream Megatron-LM format.

The two formats share torch_dist sharded weights. They differ only in metadata:

  bridge:  iter_*/run_config.yaml + common.pt (no ``args``)
  MLM:     iter_*/common.pt with ``args`` (no run_config.yaml)

The conversion is lossless for our purposes: copy/symlink everything, drop
``run_config.yaml``, and inject an ``argparse.Namespace`` into ``common.pt``
populated with the TP/PP fields bridge's MLM-load path consults.

This script exists to support functional testing of the
``pretrained_checkpoint.format=megatron_lm`` code path without requiring an
upstream MLM training run to produce a fixture.
"""

import argparse
import os
from typing import Any

import torch
import yaml


def bridge_to_mlm(bridge_iter_dir: str, mlm_iter_dir: str) -> None:
    """Convert a bridge-format iter directory to MLM format in-place under ``mlm_iter_dir``.

    Weight shards are symlinked rather than copied to keep the operation cheap.
    """
    if not os.path.isfile(os.path.join(bridge_iter_dir, "run_config.yaml")):
        raise FileNotFoundError(
            f"{bridge_iter_dir} is not a bridge iter directory "
            "(missing run_config.yaml)."
        )
    if not os.path.isfile(os.path.join(bridge_iter_dir, "metadata.json")):
        raise FileNotFoundError(
            f"{bridge_iter_dir} is not a torch_dist iter directory "
            "(missing metadata.json)."
        )

    os.makedirs(mlm_iter_dir, exist_ok=True)

    with open(os.path.join(bridge_iter_dir, "run_config.yaml")) as f:
        run_config: dict[str, Any] = yaml.safe_load(f) or {}
    model_cfg: dict[str, Any] = run_config.get("model", {}) or {}
    ckpt_cfg: dict[str, Any] = run_config.get("checkpoint", {}) or {}

    common = torch.load(
        os.path.join(bridge_iter_dir, "common.pt"),
        map_location="cpu",
        weights_only=False,
    )
    # Bridge's _extract_megatron_lm_args_from_state_dict reads these via
    # getattr-with-defaults; only TP/PP must be accurate, the rest are flags
    # that don't affect a pretrained_checkpoint load.
    common["args"] = argparse.Namespace(
        tensor_model_parallel_size=model_cfg.get("tensor_model_parallel_size", 1),
        pipeline_model_parallel_size=model_cfg.get("pipeline_model_parallel_size", 1),
        encoder_tensor_model_parallel_size=model_cfg.get(
            "encoder_tensor_model_parallel_size", 0
        ),
        encoder_pipeline_model_parallel_size=model_cfg.get(
            "encoder_pipeline_model_parallel_size", 0
        ),
        no_save_optim=False,
        no_save_rng=False,
        ckpt_fully_parallel_save=ckpt_cfg.get("fully_parallel_save", False),
    )
    torch.save(common, os.path.join(mlm_iter_dir, "common.pt"))

    # Symlink weight shards and the torch_dist metadata; skip the bridge sidecar.
    for name in os.listdir(bridge_iter_dir):
        if name in ("common.pt", "run_config.yaml"):
            continue
        src = os.path.realpath(os.path.join(bridge_iter_dir, name))
        dst = os.path.join(mlm_iter_dir, name)
        if os.path.lexists(dst):
            continue
        os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--bridge-iter-dir",
        required=True,
        help="Path to a bridge iter directory (contains run_config.yaml + metadata.json).",
    )
    parser.add_argument(
        "--mlm-iter-dir",
        required=True,
        help="Output path for the MLM-format iter directory.",
    )
    args = parser.parse_args()
    bridge_to_mlm(args.bridge_iter_dir, args.mlm_iter_dir)
    print(f"Wrote MLM-format iter directory to {args.mlm_iter_dir}")


if __name__ == "__main__":
    main()
