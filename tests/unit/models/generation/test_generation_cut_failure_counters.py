"""Exercise cut-failure counters without starting Ray, Torch or vLLM.

Load the production control methods in isolation, as the prefix-cut diagnostics
tests do. Acknowledgements use Gym's actual wire schema; only token buffers and
the staging transport are test doubles.
"""

import ast
import json
import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.utils import prefix_cut_diagnostics as diagnostics


@pytest.fixture(scope="module")
def implementation():
    pytest.importorskip("nemo_gym._checkpoint.generation_cut")
    source = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/generation_cut_capture.py"
    )
    tree = ast.parse(source.read_text())
    owner = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "GenerationCutCaptureMixin"
    )
    methods = {
        "_completed_generation_cut_ack",
        "_generation_cut_transaction",
        "_checkpoint_generation_cut_batched",
    }
    owner.body = [n for n in owner.body if getattr(n, "name", None) in methods]
    selected = [
        n
        for n in tree.body
        if getattr(n, "name", None)
        in {
            "_failed_generation_cut_ack",
            "_RouteSnapshotRequest",
            "_ends_inside_character",
            "GenerationCutCaptureMixin",
        }
    ]
    selected.extend(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_PREFIX_BOUNDARY_TAIL_TOKENS"
            for target in n.targets
        )
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = dict(
        dataclass=dataclass,
        time=time,
        LOGGER=logging.getLogger(__name__),
        prefix_cut_count=diagnostics.prefix_cut_count,
        prefix_cut_wait=diagnostics.prefix_cut_wait,
        prefix_cut_timing=diagnostics.prefix_cut_timing,
        checkpoint_phase=lambda *a, **kw: nullcontext(),
        _ROUTE_SNAPSHOT_GROUP_MAX_CALLS=64,
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *selected], type_ignores=[])
            ),
            str(source),
            "exec",
        ),
        namespace,
    )
    return SimpleNamespace(**namespace)


@pytest.fixture
def worker(implementation):
    worker = implementation.GenerationCutCaptureMixin()
    worker._capture_registry_lock = threading.Lock()
    worker._capture_calls_by_model_call_id = {}
    worker._completed_capture_calls = {}
    worker._generation_prefix_batch_max_rows = 2
    worker._generation_prefix_batch_max_tokens = 1024
    worker._capture_sink = SimpleNamespace()
    worker.token_capture = SimpleNamespace()
    worker._generation_cut_tokenizer = SimpleNamespace(
        decode=lambda ids, **kw: bytes(ids).decode(errors="replace")
    )
    return worker


def inventory(count=1):
    from nemo_gym._checkpoint.generation_cut import (
        GenerationCutInventory,
        GenerationCutPrefix,
    )

    return GenerationCutInventory.build(
        checkpoint_id="checkpoint",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id=f"ticket-{i}",
                rollout_id="rollout",
                attempt=0,
                model_call_id=f"call-{i}",
                admitted_at=1,
            )
            for i in range(count)
        ],
    )


def measured(caplog, operation, *, enabled=True):
    caplog.set_level(logging.INFO)
    with diagnostics.prefix_cut_probe(
        enabled=enabled,
        checkpoint_id="checkpoint",
        inventory_digest="inventory",
        calls=3,
        deadline=None,
    ):
        result = operation()
    summaries = [
        json.loads(r.message.split(" ", 1)[1])
        for r in caplog.records
        if r.message.startswith("prefix_cut_probe ")
    ]
    return result, summaries[-1]["counts"] if summaries else {}


def assert_failures(acks, counts, expected):
    categories = {
        k.removeprefix("failed_model_calls_"): v
        for k, v in counts.items()
        if k.startswith("failed_model_calls_")
    }
    assert categories == expected
    assert counts.get("failed_model_calls", 0) == sum(expected.values())
    assert sum(a.disposition == "durable_failure" for a in acks) == sum(
        expected.values()
    )


@pytest.mark.parametrize(
    "reason",
    [
        "no_capture_or_completed_state",
        "completed_not_staged",
        "completed_missing_output_limit",
        "completed_missing_finish_reason",
        None,
    ],
)
def test_completed_fallback_categories(worker, caplog, reason):
    if reason != "no_capture_or_completed_state":
        worker._completed_capture_calls["call-0"] = SimpleNamespace(
            coords=SimpleNamespace(
                rollout_id="rollout",
                disposition="failed" if reason == "completed_not_staged" else "staged",
                staging_key="canonical-key",
                digest="a" * 64,
            ),
            effective_output_limit=None
            if reason == "completed_missing_output_limit"
            else 128,
            terminal_finish_reason=None
            if reason == "completed_missing_finish_reason"
            else "stop",
            terminal_stop_reason=None,
            generation_token_count=4,
        )
    acks, counts = measured(
        caplog, lambda: worker._checkpoint_generation_cut_batched(inventory())
    )
    assert_failures(acks, counts, {} if reason is None else {reason: 1})
    assert counts["capture_lookup_missing"] == 1
    if reason is None:
        assert counts["completed_fallback_success"] == 1
        assert acks[0].cut_kind == "terminal_completion"


@pytest.mark.parametrize(
    "reason",
    [
        "uncuttable_request",
        "observation_error",
        "missing_output_limit",
        "zero_generated_tokens",
        "incomplete_utf8_boundary",
        "route_snapshot_timeout",
        "route_validation_error",
    ],
)
def test_active_failure_categories_and_rollback(worker, caplog, reason):
    tokens = (
        []
        if reason == "zero_generated_tokens"
        else [0xC3]
        if reason == "incomplete_utf8_boundary"
        else [65]
    )
    rollbacks = []
    state = SimpleNamespace(
        lifecycle_lock=threading.Lock(),
        lock=threading.Lock(),
        terminal_started=False,
        call=SimpleNamespace(
            rollout_id="rollout", admission=SimpleNamespace(generation_cut=None)
        ),
        uncuttable_reason="unsupported" if reason == "uncuttable_request" else None,
        observation_error="dropped delta" if reason == "observation_error" else None,
        effective_output_limit=None if reason == "missing_output_limit" else 128,
        terminal_finish_reason=None,
        terminal_stop_reason=None,
        resumed_generation_token_ids=[],
        frozen_cut=("checkpoint", len(tokens)),
        freeze_for_checkpoint=lambda checkpoint: (
            "buffer",
            0,
            tokens,
            [-0.1] * len(tokens),
            tokens,
            [-0.1] * len(tokens),
        ),
        rollback_frozen_cut=rollbacks.append,
    )
    worker._capture_calls_by_model_call_id["call-0"] = state

    def snapshot(*args, **kwargs):
        if reason == "route_snapshot_timeout":
            raise TimeoutError("owner unavailable")
        raise ValueError("short routes")

    worker._frozen_cut_engine_routed_experts = snapshot

    def run():
        transaction = worker._generation_cut_transaction(
            inventory().active_prefixes[0], state, "checkpoint"
        )
        with pytest.raises(StopIteration) as finished:
            next(transaction)
        return [finished.value.value]

    acks, counts = measured(caplog, run)
    assert_failures(acks, counts, {reason: 1})
    assert not state.lifecycle_lock.locked()
    assert rollbacks == (
        []
        if reason in {"uncuttable_request", "observation_error", "missing_output_limit"}
        else ["checkpoint"]
    )


@pytest.mark.parametrize(
    "reason",
    [
        "staging_error",
        "acknowledgement_error",
        "seal_error",
        "unexpected_seal_yield",
        "transaction_prepare_error",
        "control_deadline",
        None,
    ],
)
def test_batch_categories_count_once_per_failed_ack(
    worker, implementation, caplog, reason
):
    inv = inventory(count=3)
    states = {
        p.model_call_id: SimpleNamespace(lock=threading.Lock(), frozen_cut=None)
        for p in inv.active_prefixes
    }
    worker._capture_calls_by_model_call_id.update(states)

    def transaction(prefix, state, checkpoint, **kwargs):
        if reason == "transaction_prepare_error":
            raise ValueError("prepare error")
        key = yield SimpleNamespace(token_ids_delta=[65]), 0, None
        if reason == "acknowledgement_error":
            raise ValueError("ack error")
        yield SimpleNamespace(
            ticket_id=prefix.ticket_id,
            disposition="durable_prefix",
            staging_keys=(key,),
        )
        if reason == "seal_error":
            raise ValueError("seal error")
        if reason == "unexpected_seal_yield":
            yield None

    def stage(records, **kwargs):
        if reason == "staging_error":
            raise ValueError("put error")
        return [f"key-{i}" for i in range(len(records))]

    worker._generation_cut_transaction = transaction
    worker._capture_sink = SimpleNamespace(
        stage_generation_prefix_batch=stage, clear=lambda keys: None
    )
    acks, counts = measured(
        caplog,
        lambda: worker._checkpoint_generation_cut_batched(
            inv,
            deadline=time.monotonic() - 1 if reason == "control_deadline" else None,
        ),
    )
    assert_failures(acks, counts, {} if reason is None else {reason: 3})
    assert [a.ticket_id for a in acks] == [p.ticket_id for p in inv.active_prefixes]


def test_disabled_probe_preserves_failed_ack_without_logging(worker, caplog):
    acks, counts = measured(
        caplog,
        lambda: worker._checkpoint_generation_cut_batched(inventory()),
        enabled=False,
    )
    assert acks[0].disposition == "durable_failure"
    assert counts == {}


def test_terminal_race_counts_the_fallback_failure_only_once(worker, caplog):
    state = SimpleNamespace(lifecycle_lock=threading.Lock(), terminal_started=True)
    worker._capture_calls_by_model_call_id["call-0"] = state
    acks, counts = measured(
        caplog, lambda: worker._checkpoint_generation_cut_batched(inventory())
    )
    assert_failures(acks, counts, {"no_capture_or_completed_state": 1})
    assert counts["capture_lookup_active"] == 1
    assert counts["capture_lifecycle_races"] == 1


def test_recovered_batch_error_does_not_count_successful_rows_as_failures(
    worker, caplog
):
    inv = inventory(count=3)
    worker._capture_calls_by_model_call_id.update(
        {p.model_call_id: object() for p in inv.active_prefixes}
    )

    def transaction(prefix, state, checkpoint, **kwargs):
        key = yield SimpleNamespace(token_ids_delta=[65]), 0, None
        yield SimpleNamespace(
            ticket_id=prefix.ticket_id,
            disposition="durable_prefix",
            staging_keys=(key,),
        )

    def stage(records, **kwargs):
        if len(records) > 1:
            raise ValueError("retry individually")
        return ["canonical-key"]

    worker._generation_cut_transaction = transaction
    worker._capture_sink = SimpleNamespace(
        stage_generation_prefix_batch=stage, clear=lambda keys: None
    )
    acks, counts = measured(
        caplog, lambda: worker._checkpoint_generation_cut_batched(inv)
    )
    assert_failures(acks, counts, {})


def test_route_resume_error_has_its_own_failure_category(
    worker, implementation, caplog
):
    inv = inventory(count=3)
    worker._capture_calls_by_model_call_id.update(
        {p.model_call_id: object() for p in inv.active_prefixes}
    )

    def transaction(prefix, state, checkpoint, **kwargs):
        yield implementation._RouteSnapshotRequest(prefix.model_call_id, 1)
        raise RuntimeError("resume error")

    worker._generation_cut_transaction = transaction
    worker._snapshot_engine_routed_experts_many = lambda request_ids, **kw: {
        request_id: object() for request_id in request_ids
    }
    acks, counts = measured(
        caplog, lambda: worker._checkpoint_generation_cut_batched(inv)
    )
    assert_failures(acks, counts, {"route_resume_error": 3})
