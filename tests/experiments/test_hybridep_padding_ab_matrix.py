# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
MATRIX_PATH = ROOT / "experiments" / "hybridep-padding-ab-q30" / "arm_matrix.py"


def _load_matrix_module():
    spec = importlib.util.spec_from_file_location(
        "hybridep_padding_ab_matrix", MATRIX_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("arm_name", "expected"),
    [
        (
            "official-alltoall",
            {
                "dispatcher": "alltoall",
                "hybridep_backend": False,
                "pad_uneven_dispatch_inputs": False,
                "legacy_prepadding": False,
                "deepep_commit": None,
                "source_profile": "official",
            },
        ),
        (
            "official-pr5008-17cf",
            {
                "dispatcher": "flex",
                "hybridep_backend": True,
                "pad_uneven_dispatch_inputs": True,
                "legacy_prepadding": False,
                "deepep_commit": "17cfb817bccec3a9c247013360cc550c2bac441e",
                "source_profile": "official",
            },
        ),
        (
            "official-pr5008-f725",
            {
                "dispatcher": "flex",
                "hybridep_backend": True,
                "pad_uneven_dispatch_inputs": True,
                "legacy_prepadding": False,
                "deepep_commit": "f725d29699f5bda9ba789456bb9579af69844685",
                "source_profile": "official",
            },
        ),
        (
            "legacy-prepad-17cf",
            {
                "dispatcher": "flex",
                "hybridep_backend": True,
                "pad_uneven_dispatch_inputs": False,
                "legacy_prepadding": True,
                "deepep_commit": "17cfb817bccec3a9c247013360cc550c2bac441e",
                "source_profile": "legacy",
            },
        ),
    ],
)
def test_arm_contract(arm_name: str, expected: dict[str, object]) -> None:
    matrix = _load_matrix_module()

    arm = matrix.get_arm(arm_name)

    assert (
        arm.recipe
        == "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml"
    )
    assert arm.nodes == 4
    assert arm.gpus_per_node == 8
    assert arm.max_steps == 20
    for field_name, expected_value in expected.items():
        assert getattr(arm, field_name) == expected_value
    if arm.source_profile == "official":
        assert arm.nemo_rl_commit == matrix.OFFICIAL_NEMO_RL
        assert arm.bridge_commit == matrix.OFFICIAL_BRIDGE
        assert arm.mcore_commit == matrix.OFFICIAL_MCORE
        assert arm.source_branch == matrix.OFFICIAL_BRANCH
    else:
        assert arm.nemo_rl_commit == matrix.LEGACY_NEMO_RL
        assert arm.bridge_commit == matrix.LEGACY_BRIDGE
        assert arm.mcore_commit == matrix.LEGACY_MCORE
        assert arm.source_branch == matrix.LEGACY_BRANCH


def test_matrix_cli_emits_all_four_arms_as_json() -> None:
    result = subprocess.run(
        [sys.executable, str(MATRIX_PATH), "--list", "--format", "json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)

    assert [arm["name"] for arm in payload] == [
        "official-alltoall",
        "official-pr5008-17cf",
        "official-pr5008-f725",
        "legacy-prepad-17cf",
    ]


def test_unknown_arm_fails_closed() -> None:
    matrix = _load_matrix_module()

    with pytest.raises(ValueError, match="unknown experiment arm"):
        matrix.get_arm("typo")
