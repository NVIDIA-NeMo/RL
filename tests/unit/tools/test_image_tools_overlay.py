# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from tools.build_image_tools_overlay import ROOTS, check_resolution, lock_constraints
from tools.check_image_tools_runtime import environments


def test_overlay_matches_super_branch_and_probes_http_backend():
    assert "openai==2.6.1" in ROOTS
    assert not any(root.startswith("httptools") for root in ROOTS)
    servers = [spec for spec in environments() if "/opt/gym_venvs/" in spec[1]]
    assert len(servers) == 6
    for _, _, modules, _ in servers:
        assert "uvicorn" in modules


@pytest.mark.parametrize(
    "requirement",
    [
        "torch==2.11.0",
        "vllm==0.25.1",
        "nvidia-cublas-cu13==13.1",
        "ray==2.56.1",
        "transformer_engine==2.15.0",
    ],
)
def test_overlay_cannot_replace_framework(requirement):
    with pytest.raises(ValueError, match="must not replace"):
        check_resolution(requirement)


def test_overlay_allows_sdk_and_telemetry():
    check_resolution(
        "# comment\nopenai==2.44.0\nnemo-lens @ git+https://github.com/NVIDIA-NeMo/Lens.git@rev\nopentelemetry-api==1.44.0\n"
    )


def test_constraints_do_not_guess_ambiguous_versions():
    lock = {
        "package": [
            {"name": "a", "version": "1", "source": {"registry": "pypi"}},
            {"name": "b", "version": "1", "source": {"registry": "pypi"}},
            {"name": "b", "version": "2", "source": {"registry": "pypi"}},
            {"name": "local", "source": {"editable": "."}},
        ]
    }
    assert lock_constraints(lock) == "a==1\n"
