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
"""Tests for the xferdtensor transport-selection reporting.

``policy.generation.refit_transport=nccl_reshard`` downgrades to the Python
reshard when ``nccl.m2n.reshard`` is absent.  The downgrade must be announced
loudly but exactly once, and it must stay non-fatal unless a run explicitly
opts into failing fast.
"""

import pytest

from nemo_rl.weight_sync import xferdtensor as mod

_ENV_VARS = (
    "NRL_XFERDTENSOR_PYTHON",
    "NRL_XFERDTENSOR_GOLDEN",
    "NRL_XFERDTENSOR_REQUIRE_M2N",
)


class _ProcessGroup:
    def __init__(self, rank=0):
        self.rank = rank


@pytest.fixture(autouse=True)
def reset_dispatch_state(monkeypatch):
    """Clear the once-per-process log latch and every transport env var."""
    monkeypatch.setattr(mod, "_XFERDTENSOR_PATH_LOGGED", False)
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _banner_records(caplog):
    return [r for r in caplog.records if "REFIT TRANSPORT DOWNGRADE" in r.getMessage()]


def test_downgrade_warns_once_from_rank_zero(monkeypatch, caplog):
    monkeypatch.setattr(mod, "_reshard", None)
    with caplog.at_level("WARNING"):
        mod._check_reshard_path(
            _ProcessGroup(rank=0), use_golden=False, use_python=False
        )

    records = _banner_records(caplog)
    assert len(records) == 1
    message = records[0].getMessage()
    assert "refit_transport=nccl_reshard" in message
    assert "xferdtensor_python" in message
    assert "PERFORMANCE downgrade" in message


def test_downgrade_is_silent_on_non_zero_ranks(monkeypatch, caplog):
    monkeypatch.setattr(mod, "_reshard", None)
    with caplog.at_level("WARNING"):
        mod._check_reshard_path(
            _ProcessGroup(rank=7), use_golden=False, use_python=False
        )

    assert _banner_records(caplog) == []


def test_banner_is_not_repeated_within_a_process(monkeypatch, caplog):
    monkeypatch.setattr(mod, "_reshard", None)
    group = _ProcessGroup(rank=0)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            mod._check_reshard_path(group, use_golden=False, use_python=False)

    assert len(_banner_records(caplog)) == 1


@pytest.mark.parametrize(
    "use_golden,use_python",
    [(False, True), (True, False)],
)
def test_explicitly_forced_path_is_not_reported_as_a_downgrade(
    monkeypatch, caplog, use_golden, use_python
):
    monkeypatch.setattr(mod, "_reshard", None)
    with caplog.at_level("WARNING"):
        mod._check_reshard_path(
            _ProcessGroup(rank=0), use_golden=use_golden, use_python=use_python
        )

    assert _banner_records(caplog) == []


def test_no_warning_when_the_real_op_is_available(monkeypatch, caplog):
    monkeypatch.setattr(mod, "_reshard", lambda *args, **kwargs: None)
    with caplog.at_level("WARNING"):
        mod._check_reshard_path(
            _ProcessGroup(rank=0), use_golden=False, use_python=False
        )

    assert _banner_records(caplog) == []


def test_downgrade_is_non_fatal_by_default(monkeypatch):
    """A missing op must never abort a run that did not opt into failing fast."""
    monkeypatch.setattr(mod, "_reshard", None)
    for rank in (0, 3):
        mod._check_reshard_path(
            _ProcessGroup(rank=rank), use_golden=False, use_python=False
        )


def test_downgrade_raises_when_opted_in(monkeypatch):
    monkeypatch.setattr(mod, "_reshard", None)
    monkeypatch.setenv("NRL_XFERDTENSOR_REQUIRE_M2N", "1")
    with pytest.raises(RuntimeError, match="nccl.m2n.reshard"):
        mod._check_reshard_path(
            _ProcessGroup(rank=0), use_golden=False, use_python=False
        )


def test_opt_in_does_not_raise_when_the_real_op_is_available(monkeypatch):
    monkeypatch.setattr(mod, "_reshard", lambda *args, **kwargs: None)
    monkeypatch.setenv("NRL_XFERDTENSOR_REQUIRE_M2N", "1")
    mod._check_reshard_path(_ProcessGroup(rank=0), use_golden=False, use_python=False)


def test_opt_in_does_not_raise_for_an_explicitly_forced_python_path(monkeypatch):
    monkeypatch.setattr(mod, "_reshard", None)
    monkeypatch.setenv("NRL_XFERDTENSOR_REQUIRE_M2N", "1")
    mod._check_reshard_path(_ProcessGroup(rank=0), use_golden=False, use_python=True)
