# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks for the cluster-only model qualification lifecycle."""

import ast
from pathlib import Path


def test_learner_qualification_uses_real_mtp_override_and_cached_lifecycle() -> None:
    """Guard the fresh-only probe that missed the first-forward failure."""
    source = Path("tools/check_image_tools_model.py").read_text()
    tree = ast.parse(source)
    finalizations = []
    layer_reads = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name) or node.value.id != "provider":
            continue
        assert not (node.attr == "mtp_num_layers" and isinstance(node.ctx, ast.Store))
        if node.attr == "finalize":
            finalizations.append(node.lineno)
        if node.attr == "num_layers" and isinstance(node.ctx, ast.Load):
            layer_reads.append(node.lineno)
        assert node.attr != "hybrid_override_pattern"
    assert len(finalizations) == 1
    assert layer_reads
    assert finalizations[0] < min(layer_reads)
    assert "get_hybrid_total_layer_count(provider.hybrid_layer_pattern)" in source
    assert "_apply_mtp_config(candidate, policy)" in source
    assert "load_model_config(str(Path(cached_config).parent))" in source
    assert 'providers["cached"] = cached' in source
    assert "parsed.mtp_pattern is None and parsed.mtp_num_depths == 0" in source
