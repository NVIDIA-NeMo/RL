"""CPU tests independent of the repository's Ray/PyTorch test fixtures."""

import threading
import time
from array import array
from dataclasses import replace

import pytest

from tools.http_profile.capture import (
    CallRecord,
    CaptureQueue,
    PrefixTree,
    ShardedCaptureQueue,
)


def record(call_id: str, tokens: list[int], prompt_tokens: int) -> CallRecord:
    return CallRecord(
        "rollout",
        call_id,
        array("I", tokens).tobytes(),
        prompt_tokens,
        array("f", [-0.5] * (len(tokens) - prompt_tokens)).tobytes(),
        time.perf_counter(),
    )


def test_tree_reordered_branches_and_distinct_identical_generations() -> None:
    tree = PrefixTree(block_tokens=2)
    calls = [
        record("child", [1, 2, 3, 4, 5, 6, 7], 5),
        record("parent", [1, 2, 3, 4], 2),
        record("retry", [1, 2, 3, 4], 2),
        record("compacted", [8, 9, 10], 2),
    ]
    for call in calls:
        tree.insert(call)
    for call in calls:
        assert tree.materialize(call.rollout_id, call.call_id) == call.token_bytes
    assert tree.stats()["stored_calls"] == 4
    assert tree.stats()["generated_tokens"] == 7
    assert tree.stats()["unique_token_bytes"] < sum(
        len(call.token_bytes) for call in calls
    )
    with pytest.raises(ValueError, match="duplicate"):
        tree.insert(calls[0])


def test_thread_submission_does_not_wait_for_tree_and_overflow_is_explicit(
    monkeypatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    original = PrefixTree.insert

    def gated_insert(self, item):
        entered.set()
        assert release.wait(10)
        original(self, item)

    monkeypatch.setattr(PrefixTree, "insert", gated_insert)
    capture = CaptureQueue("thread", max_records=1, max_bytes=1024)
    try:
        assert capture.submit(record("first", [1, 2, 3], 2))
        assert entered.wait(10)  # insertion is still blocked after submit returned
        assert not capture.submit(record("overflow", [1, 2, 4], 2))
        release.set()
        stats = capture.drain()
        assert stats["completed"] == stats["stored_calls"] == 1
        assert stats["pending"] == stats["pending_bytes"] == 0
        assert stats["rejected"] == 1
        assert stats["invalid_rollouts"] == ["rollout"]
    finally:
        release.set()
        capture.close()


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_drain_accounts_for_errors_and_reset(mode) -> None:
    capture = CaptureQueue(mode, max_records=16, max_bytes=1024)
    try:
        assert not capture.submit(record("oversized", list(range(1024)), 1023))
        assert capture.submit(record("same", [1, 2, 3], 2))
        assert capture.submit(record("same", [1, 2, 3], 2))
        stats = capture.drain()
        assert stats["completed"] == 2
        assert stats["stored_calls"] == 1
        assert len(stats["failures"]) == 1
        assert stats["pending"] == stats["pending_bytes"] == 0
        capture.reset()
        stats = capture.drain()
        assert stats["submitted"] == stats["stored_calls"] == stats["rejected"] == 0
        assert stats["failures"] == []
    finally:
        capture.close()


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_trace_links_submission_to_resolver_without_payload_contents(mode) -> None:
    capture = CaptureQueue(mode, tracing=True)
    try:
        item = replace(record("first", [1, 2, 3], 2), trace_id="rollout/first")
        assert capture.submit(item)
        events = capture.drain()["trace_events"]
        flow_ends = [event for event in events if event["ph"] == "f"]
        assert len(flow_ends) == 1
        assert flow_ends[0]["bp"] == "e"
        spans = [event for event in events if event["ph"] == "X"]
        assert {event["name"] for event in spans} == {"queue wait", "resolver work"}
        assert all(event["dur"] >= 0 for event in spans)
        assert all(event["args"]["bytes"] == item.nbytes for event in spans)
        assert all(event["tid"] >= 1_000_000 for event in spans)
        capture.reset()
        assert all(event["ph"] == "M" for event in capture.drain()["trace_events"])
    finally:
        capture.close()


def test_shards_keep_rollout_ownership_and_aggregate_failures() -> None:
    capture = ShardedCaptureQueue(2, max_records=64, max_bytes=8192)
    try:
        for index in range(16):
            for call_id in ("parent", "child"):
                assert capture.submit(
                    replace(
                        record(call_id, [1, 2, 3], 2), rollout_id=f"rollout-{index}"
                    )
                )
        stats = capture.drain()
        assert stats["stored_calls"] == stats["completed"] == stats["submitted"] == 32
        assert stats["pending"] == 0
        assert all(shard["stored_calls"] > 0 for shard in stats["shards"])
        # Repeated identity must reach its original owner and be rejected there.
        assert capture.submit(
            replace(record("parent", [1, 2, 3], 2), rollout_id="rollout-0")
        )
        assert len(capture.drain()["failures"]) == 1
        capture.reset()
        assert capture.drain()["completed"] == 0
    finally:
        capture.close()
