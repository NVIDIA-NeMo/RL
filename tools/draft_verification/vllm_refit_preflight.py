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
"""End-to-end refit preflight for co-trained drafters inside the serving container.

For each drafter (dflash and eagle3 by default) this script starts a real vLLM
engine (target + drafter, ``load_format="dummy"``) with the NeMo-RL worker
extension, mirrors the module-sharing disable the production worker applies,
and checks over ``collective_rpc``:

1. The drafter loads and its expected ``draft.*`` refit key set (derived from
   the live vLLM drafter's parameters) exactly matches the trainer's stream,
   which this script derives from a meta-device build of the vendored trainer
   model for the same checkpoint.
2. ``prepare_refit_info`` accepts the trainer manifest and hard-errors when a
   required draft key is missing.
3. Integer/bool draft buffers (d2t/t2d) keep their dtypes in the trainer
   manifest (token ids are not representable in bf16).
4. No embed_tokens/lm_head storage is aliased between drafter and target after
   the sharing disable — and, as a negative control run first, that WITHOUT
   the disable the eagle-style loaders do alias and the refit alias guard
   rejects the load (the failure mode the disable exists for).

Must run in the vLLM serving venv (vllm importable), e.g.:
  /opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python \
      tools/draft_verification/vllm_refit_preflight.py
"""

import argparse
import gc
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
# The repo checkout takes precedence over any container-installed nemo_rl.
sys.path.insert(0, str(_SCRIPT_DIR.parents[1]))

import torch


def _shim_stale_openai() -> None:
    """Tolerate an older openai package than the pinned vLLM expects.

    ``vllm.tool_parsers.utils`` imports ``NamespaceTool`` from
    ``openai.types.responses`` (only for isinstance checks on Responses-API
    tools, which this preflight never exercises); container venvs built from
    an older lock lack the symbol. Provide a never-matching placeholder.
    """
    import openai.types.responses as responses

    if not hasattr(responses, "NamespaceTool"):
        responses.NamespaceTool = type("NamespaceTool", (), {})


_shim_stale_openai()

# The refit manifest carries torch dtypes through collective_rpc, exactly as
# the production worker streams it (which sets the same flag).
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFTERS = {
    "dflash": {
        "model": "RedHatAI/Qwen3-8B-speculator.dflash",
        "method": "dflash",
        "num_speculative_tokens": 7,
    },
    "eagle3": {
        "model": "RedHatAI/Qwen3-8B-speculator.eagle3",
        "method": "eagle3",
        "num_speculative_tokens": 3,
    },
}
_EXTENSION_CLS = (
    "nemo_rl.models.generation.vllm.vllm_backend.VllmInternalWorkerExtension"
)


def trainer_stream_manifest(algo: str, model_name: str) -> dict[str, tuple]:
    """The trainer's refit manifest: ``draft.`` + vendored state_dict keys.

    Mirrors ``prepare_refit_info`` in the DTensor-v2 worker: floating tensors
    stream as the training dtype, integer/bool buffers keep their own dtype.
    """
    from nemo_rl.models.automodel.draft.integration import load_draft_hf_config

    config = load_draft_hf_config(model_name, algo=algo, target_num_hidden_layers=36)
    with torch.device("meta"):
        if algo == "eagle3":
            from nemo_rl.models.automodel.draft.eagle3_qwen3 import (
                Qwen3Eagle3DraftModel,
            )

            draft = Qwen3Eagle3DraftModel(config)
        else:
            from nemo_rl.models.automodel.draft.draft_qwen3 import Qwen3DSparkModel

            config.num_anchors = 1
            config._attn_implementation = "sdpa"
            draft = Qwen3DSparkModel(config)
    manifest = {}
    for name, tensor in draft.state_dict().items():
        dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
        manifest[f"draft.{name}"] = (tuple(tensor.shape), dtype)
    return manifest


def make_engine(spec: dict):
    from vllm import LLM

    return LLM(
        model=TARGET_MODEL,
        load_format="dummy",
        speculative_config=dict(spec),
        max_model_len=2048,
        gpu_memory_utilization=0.4,
        enforce_eager=True,
        worker_extension_cls=_EXTENSION_CLS,
    )


def destroy_engine(llm) -> None:
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def check_negative_sharing(algo: str, spec: dict) -> None:
    """Without the disable, the loaders alias and the alias guard must fire."""
    llm = make_engine(spec)
    try:
        (report,) = llm.collective_rpc("draft_refit_preflight_report")
        assert report["has_drafter"], f"[{algo}] drafter missing from engine"
        assert report["embed_tokens_aliased"] or report["lm_head_aliased"], (
            f"[{algo}] expected target-shared drafter modules without the "
            f"sharing disable, got {report}"
        )
        fake = [
            ("embed_tokens.weight", torch.zeros(1, 1)),
            ("lm_head.weight", torch.zeros(1, 1)),
        ]
        try:
            llm.collective_rpc("_load_draft_weights", args=(fake,))
        except Exception as err:  # noqa: BLE001
            assert "share storage" in str(err), (
                f"[{algo}] alias guard raised an unexpected error: {err}"
            )
            print(f"[ok] {algo}: alias guard rejects refit while modules are shared")
        else:
            raise AssertionError(
                f"[{algo}] alias guard did not fire on target-shared modules"
            )
    finally:
        destroy_engine(llm)


def check_positive(algo: str, spec: dict) -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        DRAFT_DISABLE_MODULE_SHARING_ENV,
        disable_draft_module_sharing,
    )

    os.environ[DRAFT_DISABLE_MODULE_SHARING_ENV] = "1"
    disable_draft_module_sharing()

    manifest = trainer_stream_manifest(algo, spec["model"])
    d2t_key, t2d_key = "draft.d2t", "draft.t2d"
    assert manifest[d2t_key][1] == torch.int64, manifest[d2t_key]
    assert manifest[t2d_key][1] == torch.bool, manifest[t2d_key]
    print(f"[ok] {algo}: trainer manifest keeps integer/bool buffer dtypes")

    llm = make_engine(spec)
    try:
        (report,) = llm.collective_rpc("draft_refit_preflight_report")
        assert report["has_drafter"] and report["owns_speculator"], report
        assert not report["embed_tokens_aliased"] and not report["lm_head_aliased"], (
            f"[{algo}] drafter modules still aliased after the sharing disable: "
            f"{report}"
        )
        print(f"[ok] {algo}: drafter owns embed_tokens/lm_head after the disable")

        llm.collective_rpc("prepare_refit_info", args=(manifest,))
        print(
            f"[ok] {algo}: trainer manifest matches the vLLM drafter layout "
            f"({len(report['expected_draft_keys'])} expected keys)"
        )

        broken = dict(manifest)
        del broken["draft.fc.weight"]
        try:
            llm.collective_rpc("prepare_refit_info", args=(broken,))
        except Exception as err:  # noqa: BLE001
            assert "missing draft keys" in str(err), err
            print(f"[ok] {algo}: missing draft key is a hard error")
        else:
            raise AssertionError(f"[{algo}] missing draft key was not rejected")
    finally:
        destroy_engine(llm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods", nargs="+", default=["dflash", "eagle3"], choices=list(DRAFTERS)
    )
    parser.add_argument(
        "--skip-negative",
        action="store_true",
        help="Skip the sharing-enabled negative control (saves two engine starts).",
    )
    args = parser.parse_args()

    # The negative control must run before the disable is applied anywhere in
    # this process (the patch is global and irreversible).
    if not args.skip_negative:
        for algo in args.methods:
            check_negative_sharing(algo, DRAFTERS[algo])
    for algo in args.methods:
        check_positive(algo, DRAFTERS[algo])
    print("PASS: vLLM refit preflight green for " + ", ".join(args.methods))


if __name__ == "__main__":
    main()
