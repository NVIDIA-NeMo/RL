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

"""P2 unit tests: AReaL two-phase ``_train_pump`` (call-order dry run).

Layer: pure-Python / asyncio unit (no Ray actor, no GPU). Per AREAL.md §9 (P2)
the test is a "dry-run streaming loop with a MOCK trainer recording call order".
``_train_pump`` orchestrates collaborators it does not own — the trainer split
API (``prepare_for_lp_inference`` / ``get_logprobs_from_meta`` /
``prepare_for_training`` / ``begin_train_step`` / ``train_microbatch_from_meta`` /
``finish_train_step`` / ``abort_train_step``), the data plane (``clear_samples``),
``_collect_full_batch`` (P1, tested separately), ``_advantage_pump``, and
``_sync_weights`` (P4, left as-is) — so the unit under test IS the orchestration:
the two-phase ordering and the once-per-step publish. A MOCK trainer is the
correct (and only) tool here because the assertion is about *call order and
counts*, not numerics: numeric equivalence is P3, and real GPU training is the
end-to-end recipe. Nothing about the loop structure is faked; only the GPU/Ray
collaborators are recorded.

SIZE-BASED model (corrected): each RL step consumes the FULL batch
``rl_step_samples`` and splits it into minibatches of ``train_global_batch_size``
sequences each — one ``optimizer.step`` per minibatch. ``num_minibatches`` is
DERIVED (``rl_step_samples // train_global_batch_size``), NOT a config knob, and
``_train_pump`` no longer reads ``self._async_cfg.num_minibatches`` or
``self._dp_world``. It reads ``self._train_minibatch_size`` (== train_gbs),
trims a tail remainder via the real ``_dp_align_batch``, then iterates the real
size-based ``_iter_minibatches``. So we parametrize ``(full_batch_size,
minibatch_size)`` and assert ``finish_train_step`` count ==
``full_batch_size // minibatch_size``.

We exercise the REAL forked ``_train_pump``, ``_iter_minibatches`` and
``_dp_align_batch`` (and the REAL ``aggregate_step_metrics`` /
``reduce_advantage_pump_metrics`` / ``Timer``), feeding REAL ``KVBatchMeta``
batches. The surrounding ``@ray.remote`` actor and its bundle-heavy ``__init__``
are bypassed via ``SingleControllerArealActor.__ray_actor_class__`` (the
undecorated class behind the wrapper), with the methods bound to a tiny stub
holding exactly the state the loop reads.
"""

from __future__ import annotations

import asyncio

import pytest

from nemo_rl.algorithms.single_controller_areal import SingleControllerArealActor
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.utils.timer import Timer

_PARTITION_ID = "rollout_data"

# The original (undecorated) class behind the @ray.remote wrapper.
_ArealClass = SingleControllerArealActor.__ray_actor_class__


def _make_batch(n: int) -> KVBatchMeta:
    """Real full-batch KVBatchMeta with ``n`` rows. ``tags`` carry a
    ``weight_version`` so the loop's lag log path is exercised."""
    return KVBatchMeta(
        partition_id=_PARTITION_ID,
        task_name="train",
        sample_ids=[f"s{i}" for i in range(n)],
        sequence_lengths=[i + 1 for i in range(n)],
        tags=[{"row": i, "weight_version": 0} for i in range(n)],
    )


class _RecordingTrainer:
    """Mock trainer: records the ORDER of every split-API call.

    Each call appends a tuple to the shared ``events`` log. The episode id passed
    to the per-minibatch calls is recorded so we can assert begin/train/finish are
    paired and ordered. ``finish_train_step`` returns an empty dict, which
    ``aggregate_step_metrics`` handles (all ``.get`` defaults) — keeping the test
    free of numerics.
    """

    def __init__(self, events: list) -> None:
        self.events = events

    def prepare_for_lp_inference(self) -> None:
        self.events.append(("prepare_for_lp_inference",))

    def get_logprobs_from_meta(self, meta) -> None:
        self.events.append(("get_logprobs_from_meta",))

    def get_reference_policy_logprobs_from_meta(self, meta) -> None:
        self.events.append(("get_reference_policy_logprobs_from_meta",))

    def prepare_for_training(self) -> None:
        self.events.append(("prepare_for_training",))

    def begin_train_step(self, ep, *, loss_fn=None, gbs=None) -> None:
        self.events.append(("begin_train_step", ep, gbs))

    def train_microbatch_from_meta(self, ep, minibatch_meta) -> None:
        self.events.append(("train_microbatch_from_meta", ep, minibatch_meta.size))

    def finish_train_step(self, ep) -> dict:
        self.events.append(("finish_train_step", ep))
        return {}

    def abort_train_step(self, ep) -> None:
        self.events.append(("abort_train_step", ep))


class _FailingTrainer(_RecordingTrainer):
    """Like ``_RecordingTrainer`` but ``train_microbatch_from_meta`` raises on the
    minibatch index ``fail_on`` — exercises the abort-then-reraise path."""

    def __init__(self, events: list, fail_on: int) -> None:
        super().__init__(events)
        self._fail_on = fail_on
        self._mb_count = 0

    def train_microbatch_from_meta(self, ep, minibatch_meta) -> None:
        super().train_microbatch_from_meta(ep, minibatch_meta)
        if self._mb_count == self._fail_on:
            self._mb_count += 1
            raise RuntimeError("boom in train_microbatch")
        self._mb_count += 1


class _NullLogger:
    def log_metrics(self, *args, **kwargs) -> None:
        pass


class _ArealHelper:
    """Minimal stub carrying exactly the state ``_train_pump`` reads.

    Binds the genuine forked ``_train_pump``, ``_iter_minibatches`` and
    ``_dp_align_batch`` so the REAL two-phase loop + size-based split arithmetic +
    tail-trim are exercised. ``_collect_full_batch``, ``_advantage_pump``,
    ``_sync_weights`` and ``_call_dp`` are recorded shims so the test stays
    pure-asyncio (P1 / P3 / P4 cover those bodies). The Ray actor wrapper and its
    bundle-heavy __init__ are bypassed.

    SIZE-BASED: the loop reads ``self._train_minibatch_size`` (the new attr) — it
    no longer reads ``self._async_cfg.num_minibatches`` or ``self._dp_world``.
    """

    def __init__(
        self,
        *,
        minibatch_size: int,
        max_num_steps: int,
        batches: list,
        trainer,
        events: list,
        reference_logprobs_field=None,
    ) -> None:
        self._master_config = _MasterCfg(max_num_steps)
        self._trainer = trainer
        self._events = events
        self._loss_fn = object()  # opaque; only passed through to begin_train_step
        self._partition_id = _PARTITION_ID
        self._logger = _NullLogger()
        self._timer = Timer()
        # The new size-based knob the loop + _iter_minibatches read.
        self._train_minibatch_size = minibatch_size
        # advantage config knob the loop reads (reference logprobs gating).
        self._advantage_cfg = _AdvCfg(reference_logprobs_field)
        # Phase-1 batch source: pop one per step; None ⇒ exhaustion (clean break).
        self._batches = list(batches)
        # The metrics accumulator the loop reads + resets each step.
        self._step_log_dict = {
            "rewards": [],
            "masked_advantages": [],
            "sequence_lengths": [],
        }
        self._trainer_version = 0
        self._train_steps = 0
        # Observable publish-side counters.
        self.collect_calls = 0
        self.advantage_calls = 0
        self.clear_samples_calls: list[dict] = []
        self.sync_weights_calls = 0
        self.trainer_versions_at_sync: list[int] = []

    # Genuine forked implementations under test.
    _train_pump = _ArealClass._train_pump
    _iter_minibatches = _ArealClass._iter_minibatches
    _dp_align_batch = staticmethod(_ArealClass._dp_align_batch)

    # ── recorded collaborators (bodies are P1/P3/P4 scope) ──────────────────

    async def _collect_full_batch(self):
        self.collect_calls += 1
        self._events.append(("collect_full_batch",))
        if not self._batches:
            return None
        return self._batches.pop(0)

    async def _advantage_pump(self, meta):
        self.advantage_calls += 1
        self._events.append(("advantage_pump",))
        return meta

    async def _call_dp(self, method_name, **kwargs):
        self._events.append(("call_dp", method_name))
        if method_name == "clear_samples":
            self.clear_samples_calls.append(kwargs)
        return None

    async def _sync_weights(self):
        self.sync_weights_calls += 1
        self.trainer_versions_at_sync.append(self._trainer_version)
        self._events.append(("sync_weights",))


class _MasterCfg:
    """Just the two MasterConfig sub-dicts the loop reads."""

    def __init__(self, max_num_steps: int) -> None:
        self.grpo = {"max_num_steps": max_num_steps, "max_num_epochs": None}
        self.cluster = {"num_nodes": 1, "gpus_per_node": 1}


class _AdvCfg:
    """Just the advantage-config field the loop gates reference logprobs on."""

    def __init__(self, reference_logprobs_field) -> None:
        self.reference_logprobs_field = reference_logprobs_field


def _run(helper: _ArealHelper) -> None:
    asyncio.run(helper._train_pump())


def _ep_index(events, name):
    return [i for i, e in enumerate(events) if e[0] == name]


# ── sanity: the bound methods are the real ones ─────────────────────────────


def test_methods_are_the_real_implementations():
    assert _ArealHelper._train_pump is _ArealClass._train_pump
    assert _ArealHelper._iter_minibatches is _ArealClass._iter_minibatches
    # _dp_align_batch is a @staticmethod; compare the underlying function.
    assert (
        _ArealHelper.__dict__["_dp_align_batch"].__func__
        is _ArealClass.__dict__["_dp_align_batch"].__func__
    )


def test_loop_does_not_read_removed_attrs():
    """The SIZE-based loop must drive entirely off ``_train_minibatch_size``; it
    must NOT touch the removed ``_async_cfg.num_minibatches`` or ``_dp_world``.
    The stub deliberately defines neither — a clean full run is proof the real
    ``_train_pump`` never reads them (an AttributeError would surface otherwise)."""
    events: list = []
    helper = _ArealHelper(
        minibatch_size=4,  # 8 / 4 -> 2 minibatches
        max_num_steps=1,
        batches=[_make_batch(8)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    # The stub has the new knob and NOT the removed ones.
    assert hasattr(helper, "_train_minibatch_size")
    assert not hasattr(helper, "_dp_world")
    assert not hasattr(helper, "_async_cfg")
    _run(helper)
    # Ran to completion (2 optimizer steps) without touching removed attrs.
    assert len(_ep_index(events, "finish_train_step")) == 2


# ── get_logprobs_from_meta: exactly once per step, BEFORE any begin ──────────


def test_get_logprobs_once_per_step_before_any_begin():
    events: list = []
    helper = _ArealHelper(
        minibatch_size=3,  # 12 / 3 -> 4 minibatches
        max_num_steps=1,
        batches=[_make_batch(12)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    lp = _ep_index(events, "get_logprobs_from_meta")
    begins = _ep_index(events, "begin_train_step")
    # exactly once.
    assert len(lp) == 1
    # before EVERY begin_train_step (π_prox frozen over the whole batch first).
    assert begins, "expected begin_train_step calls"
    assert lp[0] < begins[0]
    assert all(lp[0] < b for b in begins)


def test_get_logprobs_runs_unconditionally_without_reference_field():
    """π_prox freeze is NOT gated on the advantage config; it runs even when no
    reference logprobs field is configured (decoupled loss always needs it)."""
    events: list = []
    helper = _ArealHelper(
        minibatch_size=4,  # 8 / 4 -> 2 minibatches
        max_num_steps=1,
        batches=[_make_batch(8)],
        trainer=_RecordingTrainer(events),
        events=events,
        reference_logprobs_field=None,
    )
    _run(helper)

    assert len(_ep_index(events, "get_logprobs_from_meta")) == 1
    # reference logprobs stay gated -> NOT called when field is None.
    assert _ep_index(events, "get_reference_policy_logprobs_from_meta") == []


def test_reference_logprobs_called_when_field_configured():
    """When a reference logprobs field IS configured, the gated reference forward
    runs once per step, after get_logprobs_from_meta, before any begin."""
    events: list = []
    helper = _ArealHelper(
        minibatch_size=4,  # 8 / 4 -> 2 minibatches
        max_num_steps=1,
        batches=[_make_batch(8)],
        trainer=_RecordingTrainer(events),
        events=events,
        reference_logprobs_field="reference_policy_logprobs",
    )
    _run(helper)

    ref = _ep_index(events, "get_reference_policy_logprobs_from_meta")
    lp = _ep_index(events, "get_logprobs_from_meta")
    begins = _ep_index(events, "begin_train_step")
    assert len(ref) == 1
    assert lp[0] < ref[0] < begins[0]


# ── finish_train_step count == full_batch // minibatch (SIZE-BASED) ─────────


@pytest.mark.parametrize(
    "full_batch_size,minibatch_size",
    [
        (16, 16),  # 1 minibatch (== whole batch)
        (16, 8),  # 2 minibatches
        (16, 4),  # 4 minibatches
        (16, 2),  # 8 minibatches
        (12, 3),  # 4 minibatches
        (512, 256),  # 2 minibatches (recipe-scale)
    ],
)
def test_finish_count_equals_full_batch_over_minibatch(
    full_batch_size, minibatch_size
):
    """Exactly ``full_batch // minibatch`` optimizer steps per training step:
    that many begin/train/finish triples, each with gbs == minibatch_size, and
    Σ gbs == (full_batch // minibatch) * minibatch."""
    events: list = []
    expected = full_batch_size // minibatch_size
    helper = _ArealHelper(
        minibatch_size=minibatch_size,
        max_num_steps=1,
        batches=[_make_batch(full_batch_size)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    assert len(_ep_index(events, "begin_train_step")) == expected
    assert len(_ep_index(events, "train_microbatch_from_meta")) == expected
    assert len(_ep_index(events, "finish_train_step")) == expected

    # every begin gets gbs == minibatch_size; train sizes match.
    gbs_values = [e[2] for e in events if e[0] == "begin_train_step"]
    train_sizes = [e[2] for e in events if e[0] == "train_microbatch_from_meta"]
    assert gbs_values == [minibatch_size] * expected
    assert train_sizes == [minibatch_size] * expected
    # Σ gbs == the whole (aligned) batch.
    assert sum(gbs_values) == expected * minibatch_size


def test_single_optimizer_step_when_minibatch_equals_full_batch():
    """minibatch_size == full_batch_size ⇒ exactly one begin/train/finish over the
    whole batch (the num_minibatches==1 / base-controller case)."""
    events: list = []
    n = 10
    helper = _ArealHelper(
        minibatch_size=n,
        max_num_steps=1,
        batches=[_make_batch(n)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    begins = [e for e in events if e[0] == "begin_train_step"]
    trains = [e for e in events if e[0] == "train_microbatch_from_meta"]
    finishes = [e for e in events if e[0] == "finish_train_step"]
    assert len(begins) == len(trains) == len(finishes) == 1
    # the single minibatch is the whole batch.
    assert trains[0][2] == n
    # gbs at this single step == whole batch size.
    assert begins[0][2] == n


# ── tail-drop: full batch NOT a multiple of minibatch_size ──────────────────


@pytest.mark.parametrize(
    "full_batch_size,minibatch_size,expected_finishes",
    [
        (10, 3, 3),  # usable 9, drop 1
        (7, 3, 2),  # usable 6, drop 1
        (520, 256, 2),  # usable 512, drop 8
        (5, 3, 1),  # usable 3, drop 2
    ],
)
def test_tail_remainder_dropped_finish_count(
    full_batch_size, minibatch_size, expected_finishes
):
    """A full batch NOT a whole number of minibatches: _dp_align_batch trims the
    tail, so finish count == full_batch // minibatch (tail dropped) and every
    trained minibatch is exactly minibatch_size."""
    events: list = []
    helper = _ArealHelper(
        minibatch_size=minibatch_size,
        max_num_steps=1,
        batches=[_make_batch(full_batch_size)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    assert expected_finishes == full_batch_size // minibatch_size
    assert len(_ep_index(events, "finish_train_step")) == expected_finishes
    # no minibatch carries the dropped tail: each trained one is exactly the size.
    train_sizes = [e[2] for e in events if e[0] == "train_microbatch_from_meta"]
    assert train_sizes == [minibatch_size] * expected_finishes


def test_tail_drop_clear_samples_uses_full_unaligned_batch_ids():
    """The dropped tail rows are not trained, but ``clear_samples`` still frees the
    FULL (un-aligned) batch: consumed_ids = batch_meta.sample_ids, not the trimmed
    train_meta. Here 10 rows / minibatch 3 -> train 9 rows over 3 minibatches but
    clear all 10."""
    events: list = []
    full = _make_batch(10)
    helper = _ArealHelper(
        minibatch_size=3,
        max_num_steps=1,
        batches=[full],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    # 3 minibatches trained (s0..s8); s9 dropped from training.
    train_sizes = [e[2] for e in events if e[0] == "train_microbatch_from_meta"]
    assert train_sizes == [3, 3, 3]  # 9 rows trained, s9 dropped

    # clear_samples frees the FULL batch — all 10 ids, including the dropped tail.
    assert len(helper.clear_samples_calls) == 1
    assert helper.clear_samples_calls[0]["sample_ids"] == full.sample_ids
    assert len(helper.clear_samples_calls[0]["sample_ids"]) == 10


# NOTE (intentionally NOT tested here): a collected batch SMALLER than one
# minibatch is trimmed to zero by ``_dp_align_batch``, leaving ``step_results``
# empty; the publish path then evaluates ``aggregate_step_metrics(step_results[-1])``
# (single_controller_areal.py:544), which raises IndexError on an empty list.
# This is a latent edge in the controller, not a property to assert as "graceful";
# in production ``train_global_batch_size`` divides ``rl_step_samples`` so the
# common path is exact and this never trips. Flagged in the report, not encoded
# as an expectation. Tail-drop with a surviving minibatch is covered above.


# ── per-minibatch begin → train → finish ordering & pairing ─────────────────


def test_each_minibatch_is_begin_train_finish_in_order():
    """Per minibatch: begin then train then finish, all with the SAME episode id,
    and minibatches run in index order (single pass, no shuffle)."""
    events: list = []
    minibatch_size = 3
    n = 9  # -> 3 minibatches
    m = n // minibatch_size
    helper = _ArealHelper(
        minibatch_size=minibatch_size,
        max_num_steps=1,
        batches=[_make_batch(n)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    # collapse to the per-minibatch triples in event order.
    triples = [
        e
        for e in events
        if e[0]
        in (
            "begin_train_step",
            "train_microbatch_from_meta",
            "finish_train_step",
        )
    ]
    assert [e[0] for e in triples] == (
        ["begin_train_step", "train_microbatch_from_meta", "finish_train_step"] * m
    )
    # episode ids derived from step id, indexed, in order.
    eps = [e[1] for e in triples]
    expected_eps = []
    for j in range(m):
        ep = f"sc-step-000000-mb{j:03d}"
        expected_eps += [ep, ep, ep]
    assert eps == expected_eps


def test_gbs_per_minibatch_is_constant_and_sums_to_aligned_batch():
    """Per-minibatch gbs == train_global_batch_size (constant), so Σ gbs == the
    aligned batch and the LR schedule advances by one global batch per step."""
    events: list = []
    minibatch_size = 4
    n = 16  # exact multiple -> 4 minibatches, no tail
    m = n // minibatch_size
    helper = _ArealHelper(
        minibatch_size=minibatch_size,
        max_num_steps=1,
        batches=[_make_batch(n)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    gbs_values = [e[2] for e in events if e[0] == "begin_train_step"]
    train_sizes = [e[2] for e in events if e[0] == "train_microbatch_from_meta"]
    # every minibatch is exactly train_gbs (size-based; never a ragged tail).
    assert gbs_values == [minibatch_size] * m
    assert sum(gbs_values) == n
    # gbs matches the minibatch the optimizer step trains on.
    assert gbs_values == train_sizes


# ── publish: clear_samples + version bump + sync ONCE per step, after loop ───


def test_publish_happens_once_after_minibatch_loop():
    events: list = []
    minibatch_size = 3
    n = 12  # -> 4 minibatches
    helper = _ArealHelper(
        minibatch_size=minibatch_size,
        max_num_steps=1,
        batches=[_make_batch(n)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    # clear_samples and sync_weights happen exactly once.
    assert len(helper.clear_samples_calls) == 1
    assert helper.sync_weights_calls == 1
    # version bumped once.
    assert helper._trainer_version == 1
    assert helper._train_steps == 1

    # ordering: every finish_train_step precedes clear_samples, which precedes
    # sync_weights.
    finishes = _ep_index(events, "finish_train_step")
    clear = [i for i, e in enumerate(events) if e == ("call_dp", "clear_samples")]
    sync = _ep_index(events, "sync_weights")
    assert len(clear) == 1 and len(sync) == 1
    assert max(finishes) < clear[0] < sync[0]


def test_clear_samples_uses_full_batch_sample_ids():
    """clear_samples frees exactly the batch's DP rows, with the controller's
    partition id — once, after the minibatch loop."""
    events: list = []
    batch = _make_batch(12)
    helper = _ArealHelper(
        minibatch_size=3,  # 12 / 3 -> 4 minibatches
        max_num_steps=1,
        batches=[batch],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    assert len(helper.clear_samples_calls) == 1
    call = helper.clear_samples_calls[0]
    assert call["sample_ids"] == batch.sample_ids
    assert call["partition_id"] == _PARTITION_ID


def test_version_bump_before_sync_weights():
    """_trainer_version is incremented BEFORE _sync_weights publishes it (AReaL
    i ← i+1 then signal refit)."""
    events: list = []
    helper = _ArealHelper(
        minibatch_size=4,  # 8 / 4 -> 2 minibatches
        max_num_steps=1,
        batches=[_make_batch(8)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    # sync saw the bumped version (1), not the pre-bump 0.
    assert helper.trainer_versions_at_sync == [1]


# ── multi-step: phases repeat once per step, in the right order ──────────────


def test_multi_step_phase_order_and_counts():
    """Three steps: each is collect -> get_logprobs(once) -> advantage ->
    M finishes -> clear -> sync. Counts scale by step count."""
    events: list = []
    minibatch_size = 4
    n = 8  # -> 2 minibatches per step
    m = n // minibatch_size
    steps = 3
    helper = _ArealHelper(
        minibatch_size=minibatch_size,
        max_num_steps=steps,
        batches=[_make_batch(n) for _ in range(steps)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    assert len(_ep_index(events, "get_logprobs_from_meta")) == steps
    assert len(_ep_index(events, "finish_train_step")) == steps * m
    assert helper.sync_weights_calls == steps
    assert len(helper.clear_samples_calls) == steps
    assert helper._train_steps == steps
    assert helper._trainer_version == steps

    # Per-step structure: get_logprobs before that step's begins, and exactly one
    # sync between consecutive steps' collects.
    collects = _ep_index(events, "collect_full_batch")
    syncs = _ep_index(events, "sync_weights")
    assert len(collects) == steps
    assert len(syncs) == steps
    # each sync comes after its step's collect and before the next collect.
    for i in range(steps):
        assert collects[i] < syncs[i]
        if i + 1 < steps:
            assert syncs[i] < collects[i + 1]


# ── clean break on rollout exhaustion (collect returns None) ────────────────


def test_clean_break_on_exhaustion_no_training():
    """When _collect_full_batch returns None the loop breaks cleanly: no
    get_logprobs, no begin/finish, no clear/sync, version unchanged."""
    events: list = []
    helper = _ArealHelper(
        minibatch_size=4,
        max_num_steps=5,
        batches=[],  # collect returns None immediately
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    assert _ep_index(events, "get_logprobs_from_meta") == []
    assert _ep_index(events, "begin_train_step") == []
    assert _ep_index(events, "finish_train_step") == []
    assert helper.clear_samples_calls == []
    assert helper.sync_weights_calls == 0
    assert helper._train_steps == 0
    assert helper._trainer_version == 0


def test_exhaustion_after_some_steps_stops_cleanly():
    """Two batches then exhaustion (within a 10-step cap): trains 2 steps then
    breaks; no publish for the empty 3rd collect."""
    events: list = []
    minibatch_size = 4
    n = 8  # -> 2 minibatches per step
    m = n // minibatch_size
    helper = _ArealHelper(
        minibatch_size=minibatch_size,
        max_num_steps=10,
        batches=[_make_batch(n), _make_batch(n)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    assert helper._train_steps == 2
    assert helper.sync_weights_calls == 2
    assert len(_ep_index(events, "finish_train_step")) == 2 * m
    # 3 collects: 2 productive + 1 that returned None (then break).
    assert helper.collect_calls == 3


def test_max_num_steps_caps_training():
    """The loop stops at max_num_steps even with batches still available."""
    events: list = []
    helper = _ArealHelper(
        minibatch_size=4,  # 4 / 4 -> 1 minibatch
        max_num_steps=2,
        batches=[_make_batch(4) for _ in range(5)],
        trainer=_RecordingTrainer(events),
        events=events,
    )
    _run(helper)

    assert helper._train_steps == 2
    assert helper.sync_weights_calls == 2
    # only 2 collects consumed (cap hit before the 3rd).
    assert helper.collect_calls == 2


# ── exception path: abort then reraise, no publish ──────────────────────────


def test_minibatch_exception_aborts_and_reraises_no_publish():
    """If train_microbatch_from_meta raises, abort_train_step fires for that
    episode and the error propagates — no finish for that minibatch, no clear, no
    sync, no version bump."""
    events: list = []
    helper = _ArealHelper(
        minibatch_size=3,  # 12 / 3 -> 4 minibatches
        max_num_steps=1,
        batches=[_make_batch(12)],
        trainer=_FailingTrainer(events, fail_on=1),  # second minibatch fails
        events=events,
    )

    with pytest.raises(RuntimeError, match="boom in train_microbatch"):
        _run(helper)

    aborts = [e for e in events if e[0] == "abort_train_step"]
    finishes = [e for e in events if e[0] == "finish_train_step"]
    # the failing (2nd) minibatch aborted; the first one finished before it.
    assert len(aborts) == 1
    assert aborts[0][1] == "sc-step-000000-mb001"
    assert len(finishes) == 1
    assert finishes[0][1] == "sc-step-000000-mb000"
    # publish did NOT run.
    assert helper.clear_samples_calls == []
    assert helper.sync_weights_calls == 0
    assert helper._trainer_version == 0
    assert helper._train_steps == 0
