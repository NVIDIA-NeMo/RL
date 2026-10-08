# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""``rl.checkpoint.finalize`` against the real ``CheckpointManager``.

Here rather than in ``tests/unit/utils/test_checkpoint.py`` for the autouse
fixture in this package's conftest, which resets the process-global lens and
OTel providers between tests. The subject is the telemetry rather than the
rename, so the span assertions and that fixture belong together.
"""

import threading

import pytest

from nemo_rl.utils.checkpoint import CheckpointManager

try:
    from nemo.lens import NemoLensConfig, setup_telemetry
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from nemo_rl.telemetry.instrumentation import (
        RL_CHECKPOINT_ASYNC_ATTR,
        RL_ITERATION_ATTR,
        iteration_scope,
        managed_span,
    )
    from nemo_rl.telemetry.span_groups import RLSpanGroup

    _HAS_LENS = True
except ImportError:
    _HAS_LENS = False

pytestmark = pytest.mark.skipif(
    not _HAS_LENS, reason="nemo-lens (+ opentelemetry sdk) not installed"
)


@pytest.fixture
def checkpoint_dir(tmp_path):
    return tmp_path.resolve() / "checkpoints"


@pytest.fixture
def checkpointer(checkpoint_dir):
    return CheckpointManager(
        {
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": "loss",
            "higher_is_better": False,
            "save_period": 1,
            "keep_top_k": 3,
            "save_optimizer": True,
        }
    )


@pytest.fixture
def exporter():
    recorder = InMemorySpanExporter()
    handle = setup_telemetry(
        NemoLensConfig(enabled=True, span_groups="all"), span_exporter=recorder
    )
    yield handle, recorder
    handle.shutdown()


def _named(recorder, name):
    (span,) = [s for s in recorder.get_finished_spans() if s.name == name]
    return span


def test_the_span_stays_open_until_the_checkpoint_is_durable(
    checkpointer, checkpoint_dir, exporter
):
    """The gap this closes: ``rl.<algo>.checkpointing`` ends at staging.

    Reproduces the async-save shape. ``wait_fn`` blocks past the step that
    triggered the save, so the only span that lines up with "resumable from
    step 7" is one that is still open while the loop runs step 8.
    """
    handle, recorder = exporter
    writers_done = threading.Event()
    tmp_dir = checkpointer.init_tmp_checkpoint(7, {"loss": 0.5})

    with iteration_scope(7):
        with managed_span(
            RLSpanGroup.CHECKPOINT, "rl.grpo.checkpointing", tracer=handle.tracer
        ):
            checkpointer.begin_finalization(
                tmp_dir, wait_fn=lambda: writers_done.wait(5)
            )
        # Staging is over and the loop is free, but nothing is resumable yet.
        assert not (checkpoint_dir / "step_7").exists()

    with iteration_scope(8):
        writers_done.set()
        checkpointer.finalize_pending()
    handle.shutdown()

    assert (checkpoint_dir / "step_7").exists()
    staging = _named(recorder, "rl.grpo.checkpointing")
    durable = _named(recorder, "rl.checkpoint.finalize")
    assert durable.end_time > staging.end_time
    assert durable.parent.span_id == staging.get_span_context().span_id


def test_the_span_is_filed_under_the_step_that_triggered_the_save(
    checkpointer, exporter
):
    """Not step 8, which is what the loop had moved on to at the rename.

    ``begin_finalization`` runs its work on a bare thread, so the enclosing
    ``iteration_scope`` does not reach it; the step comes from the staging
    directory name instead.
    """
    handle, recorder = exporter
    writers_done = threading.Event()
    tmp_dir = checkpointer.init_tmp_checkpoint(7, {"loss": 0.5})

    with iteration_scope(7):
        checkpointer.begin_finalization(tmp_dir, wait_fn=lambda: writers_done.wait(5))
    with iteration_scope(8):
        writers_done.set()
        checkpointer.finalize_pending()
    handle.shutdown()

    durable = _named(recorder, "rl.checkpoint.finalize")
    assert durable.attributes[RL_ITERATION_ATTR] == 7
    assert durable.attributes[RL_CHECKPOINT_ASYNC_ATTR] is True


def test_a_checkpoint_that_never_landed_is_not_reported_as_a_clean_one(
    checkpointer, exporter
):
    """The error reaches ``finalize_pending`` and the span at the same time."""
    handle, recorder = exporter
    tmp_dir = checkpointer.init_tmp_checkpoint(7, {"loss": 0.5})

    def wait_fn():
        raise RuntimeError("writer rank 3 died")

    checkpointer.begin_finalization(tmp_dir, wait_fn=wait_fn)
    with pytest.raises(RuntimeError, match="Background checkpoint finalization failed"):
        checkpointer.finalize_pending()
    handle.shutdown()

    durable = _named(recorder, "rl.checkpoint.finalize")
    assert not durable.status.is_ok
    assert durable.attributes[RL_ITERATION_ATTR] == 7


def test_a_span_that_cannot_be_opened_does_not_cost_the_checkpoint(
    checkpointer, checkpoint_dir, exporter, monkeypatch
):
    """Instrumentation does not get to decide whether a checkpoint lands.

    The span is opened on the finalization thread, inside the try whose
    exception ``finalize_pending`` re-raises -- so an unguarded failure here
    would both abort the run and skip the rename, reporting a lens
    incompatibility as a failed save. ``NemoLensConfig.from_env`` dropping a
    parameter is the kind of change that reaches this path.
    """

    def _broken_span(*args, **kwargs):
        raise RuntimeError("lens dropped a parameter")

    monkeypatch.setattr(
        "nemo_rl.utils.checkpoint.checkpoint_finalize_span", _broken_span
    )
    tmp_dir = checkpointer.init_tmp_checkpoint(7, {"loss": 0.5})

    # Both calls inside the block: the warning is emitted on the finalization
    # thread, which begin_finalization has already started.
    with pytest.warns(UserWarning, match="Could not open the checkpoint"):
        checkpointer.begin_finalization(tmp_dir)
        checkpointer.finalize_pending()

    assert (checkpoint_dir / "step_7").exists()


def test_a_sync_save_is_distinguishable_from_an_async_one(checkpointer, exporter):
    """No ``wait_fn`` means the bytes already landed and this is just a rename.

    Both shapes emit the same span, so without the attribute a near-zero
    duration could be either one.
    """
    handle, recorder = exporter
    tmp_dir = checkpointer.init_tmp_checkpoint(7, {"loss": 0.5})

    checkpointer.begin_finalization(tmp_dir)
    checkpointer.finalize_pending()
    handle.shutdown()

    assert (
        _named(recorder, "rl.checkpoint.finalize").attributes[RL_CHECKPOINT_ASYNC_ATTR]
        is False
    )
