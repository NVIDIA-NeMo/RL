# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Populate trusted model modules serially before distributed actor startup."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path


def prewarm(*, model: str, tokenizer: str) -> dict:
    """Exercise the actual runtime's loaders without initializing model weights."""
    # Load heavy dependencies only in the selected container Python environment.
    import transformers
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer

    from nemo_rl.models.huggingface.common import ModelFlag

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    assert type(config).__name__ == "NemotronH_Omni_Reasoning_V3_Config"
    assert ModelFlag.VLLM_LOAD_FORMAT_AUTO.matches(model) is False
    for location in dict.fromkeys((model, tokenizer)):
        AutoProcessor.from_pretrained(
            location, trust_remote_code=True, use_fast=True, fix_mistral_regex=True
        )
        AutoTokenizer.from_pretrained(
            location, trust_remote_code=True, use_fast=True, fix_mistral_regex=True
        )
    module = Path(inspect.getfile(type(config)))
    return {
        "status": "IMAGE_TOOLS_MODEL_MODULES_PREWARMED",
        "transformers_version": transformers.__version__,
        "config_class": type(config).__name__,
        "module": str(module),
        "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "model_initialized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()
    print(json.dumps(prewarm(model=args.model, tokenizer=args.tokenizer)), flush=True)
