# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Convert the HF weights used by a NeMo-RL SGLang recipe before launching Ray."""

import argparse
import tempfile
from pathlib import Path
from typing import cast

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from nemo_rl.models.generation.sglang.config import (
    SglangQuantizationConfig,
    get_sglang_quantization_scheme,
)
from nemo_rl.models.generation.sglang.mxfp8_setup import convert_mxfp8, mxfp8_cache_path
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="NeMo-RL training recipe")
    args, overrides = parser.parse_known_args()
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(load_config(args.config), overrides)
    generation = config.policy.generation
    if generation.backend != "sglang":
        parser.error("policy.generation.backend must be sglang")
    sglang_cfg = generation.sglang_cfg
    policy = cast(
        SglangQuantizationConfig,
        OmegaConf.to_container(sglang_cfg.quantization, resolve=True),
    )
    if get_sglang_quantization_scheme(policy) != "mxfp8":
        parser.error("policy.generation.sglang_cfg.quantization.scheme must be mxfp8")

    model = sglang_cfg.get("model_path", config.policy.model_name)
    source = Path(model).expanduser()
    explicit = policy.get("converted_model_path")
    if explicit is not None and (not isinstance(explicit, str) or not explicit.strip()):
        parser.error("converted_model_path must name a directory")
    if not source.is_dir():
        if not explicit:
            parser.error("set converted_model_path when using a Hugging Face repo ID")
        source = Path(snapshot_download(repo_id=model))
    source = source.resolve()
    output = (
        Path(
            explicit or mxfp8_cache_path(model_dir=str(source), quantization_cfg=policy)
        )
        .expanduser()
        .resolve()
    )
    if output.is_relative_to(source):
        parser.error("the output directory must be outside the source checkpoint")
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Startup only sees the destination after all shards and metadata are complete.
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        checkpoint = Path(staging) / "checkpoint"
        convert_mxfp8(
            str(source),
            str(checkpoint),
            extra_high_precision_layers_hf=policy.get(
                "extra_high_precision_layers_hf", []
            ),
            modules_to_not_convert=policy.get("modules_to_not_convert", []),
            num_layers_at_start_in_bf16=policy.get("num_layers_at_start_in_bf16", 0),
            num_layers_at_end_in_bf16=policy.get("num_layers_at_end_in_bf16", 0),
        )
        checkpoint.rename(output)
    print(output)


if __name__ == "__main__":
    main()
