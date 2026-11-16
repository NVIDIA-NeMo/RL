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
"""Build the two views of a released Inkling2 checkpoint for vLLM MXFP8 GRPO.

Both views hold only rewritten metadata plus symlinks to the release files.

``OUT_DIR`` (``policy.generation.model_name``): vLLM's view.
  * ``config.json`` as ``model_type: inkling2`` (the only name vLLM registers)
    with ``n_layers_per_attn_res_block`` (vLLM rejects the old spelling);
  * ``hf_quant_config.json`` lists every layer's routed experts as MXFP8, so
    vLLM allocates MXFP8 experts in all layers. vLLM starts from
    ``load_format: dummy``; each refit quantizes the trained BF16 experts.

``OUT_DIR-automodel-text`` (``policy.model_name``): Automodel's text-only view.
  * ``config.json`` is the text config as ``inkling2_model`` /
    ``Inkling2ForCausalLM`` (no vision tower);
  * the release ``hf_quant_config.json`` is kept, so Automodel decodes the
    stored NVFP4 / MXFP8 experts into BF16 training weights.

Usage: python make_vllm_mxfp8_views.py RELEASE_DIR OUT_DIR
"""

import json
import re
import sys
from pathlib import Path

EXPERT_WEIGHT = re.compile(r"model\.llm\.layers\.\d+\.mlp\.experts\.w(13|2)_weight")
REWRITTEN = ("config.json", "hf_quant_config.json")


def _link_release_files(release: Path, view: Path, skip: tuple[str, ...]) -> None:
    for path in release.iterdir():
        link = view / path.name
        if path.name not in skip and not link.is_symlink():
            link.symlink_to(path.resolve())


def main(release: Path, out: Path) -> None:
    config = json.loads((release / "config.json").read_text())
    text = dict(config["text_config"])
    if "n_layers_per_attnres_block" in text:
        text["n_layers_per_attn_res_block"] = text.pop("n_layers_per_attnres_block")

    quant = json.loads((release / "hf_quant_config.json").read_text())
    groups = quant["model"]["parameter_groups"]
    mxfp8 = [g for g in groups if g["compute_dtype"] == "mxfp8"]
    if len(mxfp8) != 1:
        raise ValueError(f"expected one MXFP8 group in the release, found {len(mxfp8)}")
    experts = [t for g in groups for t in g["tensors"] if EXPERT_WEIGHT.fullmatch(t)]
    if len(experts) != 2 * text["num_hidden_layers"]:
        raise ValueError(f"found {len(experts)} expert tensors, expected 2 per layer")
    kept = []
    for group in groups:
        tensors = [t for t in group["tensors"] if not EXPERT_WEIGHT.fullmatch(t)]
        if tensors:
            kept.append({**group, "tensors": tensors})
    kept.append({**mxfp8[0], "tensors": experts})
    vllm_quant = {**quant, "model": {**quant["model"], "parameter_groups": kept}}

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({**config, "model_type": "inkling2", "text_config": text}, indent=2)
    )
    (out / "hf_quant_config.json").write_text(json.dumps(vllm_quant, indent=2))
    _link_release_files(release, out, REWRITTEN)

    text_view = out.with_name(f"{out.name}-automodel-text")
    text_view.mkdir(exist_ok=True)
    text_config = {
        **text,
        "model_type": "inkling2_model",
        "architectures": ["Inkling2ForCausalLM"],
        "eos_token_id": config["eos_token_id"],
    }
    (text_view / "config.json").write_text(json.dumps(text_config, indent=2))
    _link_release_files(release, text_view, ("config.json",))
    print(f"wrote {out} and {text_view}: {len(experts)} expert tensors served as MXFP8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
