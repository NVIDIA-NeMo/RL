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

"""Make an SGLang view of an Inkling2 checkpoint whose routed experts are served in BF16.

The release packs routed experts as NVFP4/MXFP8, and SGLang's Inkling2 model picks
each expert's kernel from hf_quant_config.json. This view symlinks every file of
the release except hf_quant_config.json, which is rewritten so the NVFP4/MXFP8
expert tensors join the plain BF16 group (their scale tensors disappear with them).
SGLang then allocates BF16 expert weights and runs them with grouped_gemm_triton.

The packed safetensors cannot be loaded into BF16 parameters, so serve the view
with load_format=dummy: NeMo RL's first refit writes every weight before any
generation.

    python make_sglang_bf16_view.py SRC DST
"""

import json
import os
import sys


def main(src: str, dst: str) -> None:
    src, dst = os.path.abspath(src), os.path.abspath(dst)
    os.makedirs(dst, exist_ok=False)
    for name in sorted(os.listdir(src)):
        if name != "hf_quant_config.json":
            os.symlink(os.path.join(src, name), os.path.join(dst, name))

    with open(os.path.join(src, "hf_quant_config.json")) as f:
        config = json.load(f)
    groups = config["model"]["parameter_groups"]
    bf16 = [
        g
        for g in groups
        if g["compute_dtype"] == "bfloat16" and "fwd_output_dtype" not in g
    ]
    if len(bf16) != 1:
        raise ValueError(f"Expected one plain BF16 group, found {len(bf16)}")
    kept, moved = [], []
    for group in groups:
        if group["compute_dtype"] in ("nvfp4", "mxfp8"):
            moved += group["tensors"]
        else:
            kept.append(group)
    bf16[0]["tensors"] = bf16[0]["tensors"] + moved
    config["model"]["parameter_groups"] = kept
    with open(os.path.join(dst, "hf_quant_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"{dst}: {len(moved)} expert tensors now BF16")


if __name__ == "__main__":
    main(*sys.argv[1:])
