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
"""Build the tiny Inkling2 checkpoint used by the vLLM MXFP8 GRPO smoke test.

The tiny checkpoint has random BF16 weights and a 256-entry vocabulary with no
tokenizer. This writes a copy that:

* resizes ``embed`` / ``unembed`` to the release tokenizer's 201,024 rows (new
  rows ~ N(0, 0.02)) and sets the release vocab / eos ids, so the real
  tokenizer and math datasets work;
* copies the release tokenizer files;
* stores every layer's routed experts as MXFP8 (e4m3 values, UE8M0 scales per
  32 inputs, like the release's MXFP8 layers) and marks them ``mxfp8`` in
  ``hf_quant_config.json``. Automodel dequantizes them on load; vLLM allocates
  MXFP8 experts, starts from ``load_format: dummy`` and the refit
  re-quantizes the trained BF16 experts.

It writes two views of the same files. ``OUT_DIR`` is vLLM's: the image/text
config as ``model_type: inkling2`` (the only name vLLM registers).
``OUT_DIR-automodel-text`` is Automodel's: the text config as
``inkling2_model`` / ``Inkling2ForCausalLM`` (no vision tower) with the
weights, quant sidecar and tokenizer symlinked. Point ``policy.model_name`` at
the second and ``policy.generation.model_name`` at the first.

Usage: python make_tiny_mxfp8_view.py SRC_TINY RELEASE_DIR OUT_DIR
"""

import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
MXFP8_GROUP_FIELDS = {
    "compute_dtype": "mxfp8",
    "output_dtype": "float8_e4m3fn",
    "required_activation_quantization": {
        "block_size": 32,
        "format": "mxfp8_e4m3",
        "rounding": "nearest",
        "scale_dtype": "ue8m0",
    },
    "weight_block_shape": [32, 32],
    "weight_transform": "mxfp8_nearest",
}
EXPERT_WEIGHT = re.compile(r"model\.llm\.layers\.\d+\.mlp\.experts\.w(13|2)_weight")


def quantize_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[E, N, K]`` -> e4m3 values and UE8M0 scale bytes ``[E, N, K // 32]``.

    Same round-up power-of-two scales as vLLM's ``mxfp8_quantize``.
    """
    blocks = weight.float().reshape(*weight.shape[:-1], -1, 32)
    raw = blocks.abs().amax(-1) / 448.0
    exponent = ((raw.view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= 2.0**-127, 0, exponent.clamp(max=254))
    scale = torch.exp2(exponent.float() - 127).clamp(min=2.0**-127)
    values = (blocks / scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    return values.reshape(weight.shape).to(torch.float8_e4m3fn), exponent.to(
        torch.uint8
    )


def main(src: Path, release: Path, out: Path) -> None:
    release_text = json.loads((release / "config.json").read_text())["text_config"]
    vocab, unpadded = release_text["vocab_size"], release_text["unpadded_vocab_size"]
    eos = json.loads((release / "config.json").read_text())["eos_token_id"]

    out.mkdir(parents=True, exist_ok=True)
    tensors = load_file(src / "model.safetensors")
    generator = torch.Generator().manual_seed(0)
    for name in ("model.llm.embed.weight", "model.llm.unembed.weight"):
        old = tensors[name]
        extra = (
            torch.randn(vocab - old.shape[0], old.shape[1], generator=generator) * 0.02
        )
        tensors[name] = torch.cat([old, extra.to(old.dtype)]).contiguous()
    for name in [n for n in tensors if EXPERT_WEIGHT.fullmatch(n)]:
        tensors[name], tensors[f"{name}_scale"] = quantize_mxfp8(tensors[name])
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})

    config = json.loads((src / "config.json").read_text())
    # vLLM registers only "inkling2"; Automodel accepts it and the legacy
    # "inkling2_mm_model" of the release checkpoint.
    config["model_type"] = "inkling2"
    config["eos_token_id"] = eos
    text = config["text_config"]
    text.update(vocab_size=vocab, unpadded_vocab_size=unpadded)
    # vLLM accepts only the new spelling; Automodel accepts both.
    if "n_layers_per_attnres_block" in text:
        text["n_layers_per_attn_res_block"] = text.pop("n_layers_per_attnres_block")
    (out / "config.json").write_text(json.dumps(config, indent=2))

    quant = json.loads((src / "hf_quant_config.json").read_text())
    groups = quant["model"]["parameter_groups"]
    num_mxfp8 = 0
    for group in groups:
        if all(EXPERT_WEIGHT.fullmatch(t) for t in group["tensors"]):
            group.update(MXFP8_GROUP_FIELDS)
            num_mxfp8 += len(group["tensors"])
    expected = 2 * config["text_config"]["num_hidden_layers"]
    if num_mxfp8 != expected:
        raise ValueError(
            f"marked {num_mxfp8} expert tensors mxfp8, expected {expected}"
        )
    (out / "hf_quant_config.json").write_text(json.dumps(quant, indent=2))

    for name in TOKENIZER_FILES:
        shutil.copy(release / name, out / name)
    shutil.copytree(release / "tiktoken", out / "tiktoken", dirs_exist_ok=True)

    text_view = out.with_name(f"{out.name}-automodel-text")
    text_view.mkdir(exist_ok=True)
    text_config = dict(
        text,
        model_type="inkling2_model",
        architectures=["Inkling2ForCausalLM"],
        eos_token_id=eos,
    )
    (text_view / "config.json").write_text(json.dumps(text_config, indent=2))
    for path in out.iterdir():
        link = text_view / path.name
        if path.name != "config.json" and not link.exists():
            link.symlink_to(path)
    print(
        f"wrote {out} and {text_view}: vocab {vocab}, {num_mxfp8} MXFP8 expert tensors"
    )


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:4]))
