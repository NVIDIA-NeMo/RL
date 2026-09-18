# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact source deltas approved for the mixed-run callback-fix rerun."""

from pathlib import Path


BRIDGE_SETUP = "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/training/setup.py"
RL_SETUP = "nemo_rl/models/megatron/setup.py"
PR_HEADS = {
    "NVIDIA-NeMo/Megatron-Bridge#6067": "33c47e44ffeef1fcd1bbd4dc6350e2ab2ee3fde2",
    "NVIDIA-NeMo/RL#4116": "ebfb72e958fbe7aa45fb3a2f13a8a7b613f92299",
}
CALLBACK_EDITS = {
    RL_SETUP: (
        (
            "from megatron.core.transformer.transformer_config import TransformerConfig\n",
            "from megatron.core.transformer.transformer_config import TransformerConfig\n"
            "from megatron.core.utils import get_model_config\n",
        ),
        (
            "        [model],\n        megatron_cfg.model,\n        megatron_cfg.ddp,",
            "        [model],\n"
            "        # Providers such as Nemotron Omni copy their config during model\n"
            "        # construction. Bind callbacks where the MCore scheduler reads them.\n"
            "        get_model_config(model),\n        megatron_cfg.ddp,",
        ),
    ),
    BRIDGE_SETUP: (
        (
            "from megatron.core.transformer.multi_token_prediction import get_mtp_ranks\n",
            "from megatron.core.transformer.multi_token_prediction import get_mtp_ranks\n"
            "from megatron.core.utils import get_model_config\n",
        ),
        (
            "        cfg.model.transformer if isinstance(cfg.model, (GPTModelConfig, HybridModelConfig)) else cfg.model,",
            "        # Providers may copy their configuration while constructing the model.\n"
            "        get_model_config(model[0]),",
        ),
    ),
}


def expected_callback_source(relative: str, baseline: bytes) -> bytes:
    """Apply only the two reviewed PR edits to a qualified baseline file."""
    source = baseline.decode()
    for before, after in CALLBACK_EDITS[relative]:
        if source.count(after) == 1:
            continue
        if source.count(before) != 1:
            raise ValueError(f"Unrecognized callback baseline: {relative}")
        source = source.replace(before, after, 1)
    return source.encode()


def require_callback_sources(root: Path) -> None:
    """Require both independent call-site corrections before training."""
    for relative, edits in CALLBACK_EDITS.items():
        source = (root / relative).read_text()
        if any(source.count(after) != 1 for _, after in edits):
            raise ValueError(f"Missing approved runtime callback fix: {relative}")
