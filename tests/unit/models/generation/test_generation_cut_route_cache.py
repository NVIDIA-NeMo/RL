"""CPU-only cut/restore regressions using real Torch and Gym integrity code.

Load the production mixin without Ray/telemetry startup. Only telemetry
decorators are replaced; route verification, capture records and digests are
the actual implementations, not fake hash functions or tensor operations.
"""

import ast
import asyncio
import contextvars
import importlib.metadata
import json
import logging
import sys
import threading
import time
import weakref
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from packaging.version import Version

from nemo_rl.models.generation.vllm.packed_logprobs import PackedSelectedLogprobs

pytest.importorskip("nemo_gym.token_id_capture.staging.capture")

from nemo_gym._checkpoint.generation_cut import (  # noqa: E402
    GenerationCutInventory,
    GenerationCutPrefix,
)
from nemo_gym.episode_types import EpisodeId  # noqa: E402
from nemo_gym.token_id_capture.staging.capture import (  # noqa: E402
    CaptureError,
    RolloutTokenCapture,
)
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    StagedCallBaseSnapshot,
)

from nemo_rl.data_plane.generation_prefix_write_batcher import (  # noqa: E402
    GenerationPrefixWriteBatcher,
)
from nemo_rl.utils.prefix_cut_diagnostics import (  # noqa: E402
    prefix_cut_probe,
    run_engine_snapshot,
)
from nemo_rl.utils.routed_experts_codec import (  # noqa: E402
    decode_routed_experts,
    encode_routed_experts,
)

pytestmark = pytest.mark.nemo_gym


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_limit", [150, 1024 * 1024])
async def test_grouped_original_receipts_persist_and_restore_two_cuts(
    grouped_implementation, owner_loop, monkeypatch, payload_limit
):
    from nemo_rl.models.generation.generation_cut_grouping import GenerationCutGrouper

    m = grouped_implementation
    worker, states, processor, whole = enable_engine_tokens(
        m, owner_loop, monkeypatch, count=16, packed=True
    )
    worker._generation_cut_receipts = {}
    worker._generation_cut_control = m.GenerationCutControl(max_workers=1)
    worker._generation_cut_grouper = GenerationCutGrouper(
        max_groups=1,
        max_inventories=8,
        max_calls=256,
        wait_seconds=0.01,
        max_payload_bytes=payload_limit,
        execute=worker._checkpoint_generation_cut_group,
    )
    puts = []
    original = worker._capture_sink.stage_generation_prefix_batch

    def put(records, **kwargs):
        puts.append(len(records))
        return original(records, **kwargs)

    worker._capture_sink.stage_generation_prefix_batch = put

    def inputs(checkpoint):
        return [
            GenerationCutInventory.build(
                checkpoint_id=checkpoint,
                server_name=f"participant-{i}",
                active_prefixes=list(whole.active_prefixes[i * 2 : i * 2 + 2]),
            )
            for i in range(8)
        ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker._generation_cut_control_executor = executor
        first_inputs = inputs("grouped-first")
        first = await asyncio.gather(
            *(
                worker._checkpoint_generation_cut_control(item, deadline=None)
                for item in first_inputs
            )
        )
        for item, answer in zip(first_inputs, first, strict=True):
            answer.validate_for(item)
            assert all(ack.disposition == "durable_prefix" for ack in answer.prefixes)
        assert len(puts) == (1 if payload_limit > 150 else 8)
        first_count = len(puts)
        retry = await worker._checkpoint_generation_cut_control(
            first_inputs[0], deadline=None
        )
        assert retry is first[0] and len(puts) == first_count
        _, restored = restore(
            worker,
            first[0].prefixes[0],
            chunks(m, worker._capture_sink, first[0].prefixes[0]),
        )
        assert restored.snapshot.token_ids_delta == [65, 66, 67, 68]
        for engine in processor.request_states.values():
            engine.detokenizer.output_token_ids.append(69)
            engine.logprobs_processor.logprobs.append_arrays(
                np.array([[69]]), np.array([[-0.3]]), np.array([1])
            )
            engine.routed_experts_chunks.append(routes(9))
        second_inputs = inputs("grouped-second")
        second = await asyncio.gather(
            *(
                worker._checkpoint_generation_cut_control(item, deadline=None)
                for item in second_inputs
            )
        )
        for item, answer in zip(second_inputs, second, strict=True):
            answer.validate_for(item)
            assert all(ack.disposition == "durable_prefix" for ack in answer.prefixes)
        _, restored = restore(
            worker,
            second[0].prefixes[0],
            chunks(m, worker._capture_sink, second[0].prefixes[0]),
        )
        assert restored.routed_experts.flatten().tolist() == [1, 2, 3, 9, -1]
        assert all(state.staged_token_count == 3 for state in states)
        assert_group_released(states)
        assert worker._generation_cut_grouper.stats()["groups"] == 2
        worker._generation_cut_grouper.close()
        worker._generation_cut_control.close()


@pytest.fixture(scope="module")
def implementation():
    root = Path(__file__).resolve().parents[4] / "nemo_rl"
    route_source = ast.parse((root / "experience/route_assembly.py").read_text())
    verify = next(
        n
        for n in route_source.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "verify_route_fragment_integrity"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ModuleType("_route_cache_cut_test")
    sys.modules[module.__name__] = module
    module.__dict__.update(json=json, ROUTE_ENCODING_ENVELOPE=1, ROUTE_ENCODING_LIST=2)
    # Read the real wire constants without importing data_plane's Ray factory.
    schema = ast.parse((root / "data_plane/schema.py").read_text())
    constants = [
        n
        for n in schema.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id
            in {
                "ROUTE_ENCODING_ENVELOPE",
                "ROUTE_ENCODING_LIST",
                "ROUTED_EXPERTS_FIELD",
            }
            for t in n.targets
        )
    ]
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *constants, verify], type_ignores=[])
            ),
            str(root / "experience/route_assembly.py"),
            "exec",
        ),
        module.__dict__,
    )
    module.__dict__.update(
        ROUTE_MISSING_SENTINEL=-1,
        checkpoint_phase=lambda *a, **kw: nullcontext(),
        telemetry_metrics_exporting=lambda: True,
        checkpoint_phase_sync=lambda *a, **kw: lambda function: function,
    )
    path = root / "models/generation/generation_cut_capture.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        n
        for n in tree.body
        if not (
            isinstance(n, ast.ImportFrom)
            and n.module
            in {
                "nemo_rl.experience.route_assembly",
                "nemo_rl.telemetry.rollout_checkpoint",
            }
        )
    ]
    exec(compile(tree, str(path), "exec"), module.__dict__)
    control_path = root / "models/generation/generation_cut_control.py"
    exec(compile(control_path.read_text(), str(control_path), "exec"), module.__dict__)
    yield module
    sys.modules.pop(module.__name__, None)


class Sink:
    def __init__(self):
        self.records = {}
        self.fail = False

    def stage_generation_prefix_batch(
        self, records, *, checkpoint_id, chunk_sequences, attachments=None
    ):
        if self.fail:
            raise RuntimeError("injected staging failure")
        if attachments is not None:
            from nemo_rl.utils.routed_experts_codec import (
                ROUTED_EXPERTS_STAGING_ATTACHMENT,
            )

            for record, attachment in zip(records, attachments, strict=True):
                if attachment is not None:
                    routes = attachment[ROUTED_EXPERTS_STAGING_ATTACHMENT]
                    assert torch.equal(
                        routes.tensor_for_payload(record.extras["routed_experts"]),
                        decode_routed_experts(record.extras["routed_experts"]),
                    )
        keys = []
        for record, sequence in zip(records, chunk_sequences, strict=True):
            key = f"{checkpoint_id}/{record.model_call_id}/{sequence}"
            self.records[key] = record
            keys.append(key)
        return keys

    def clear(self, keys):
        for key in keys:
            self.records.pop(key, None)


def make_worker(m, *, sink=None, admission=None, resolved=None, prompt=(65, 66)):
    class Worker(m.GenerationCutCaptureMixin):
        def _return_routed_experts_enabled(self):
            return True

        def _snapshot_engine_routed_experts(self, request_id, *, timeout_s):
            # Assembly must not block HTTP token observation on its state lock.
            assert not self.state.lock.locked()
            return self.pieces

    worker = Worker()
    worker._generation_prefix_cuts_enabled = False
    worker._capture_sink = sink or Sink()
    worker.token_capture = RolloutTokenCapture(
        sink=worker._capture_sink, weight_version_fn=lambda: 7
    )
    admission = admission or CaptureAdmission(
        rollout_id="r0", model_call_id="c0", mode="text"
    )
    call = worker.token_capture.begin_call(
        admission,
        generation_cut=None if resolved is None else resolved.snapshot,
        generation_cut_staging_keys=None
        if admission.generation_cut is None
        else admission.generation_cut.staging_keys,
    )
    worker.state = m._RequestCaptureState(
        call=call,
        prompt_token_ids=list(prompt),
        capture_routed_experts=True,
        engine_request_id="engine",
        effective_output_limit=128,
        restored_routed_experts=None if resolved is None else resolved.routed_experts,
        generation_cut_staging_keys=[]
        if admission.generation_cut is None
        else list(admission.generation_cut.staging_keys),
        generation_cut_chunk_digests=[]
        if resolved is None
        else (None if resolved.chunk_digests is None else list(resolved.chunk_digests)),
    )
    worker._capture_registry_lock = threading.Lock()
    worker._capture_calls_by_model_call_id = {admission.model_call_id: worker.state}
    worker._generation_prefix_batch_max_rows = 256
    worker._generation_prefix_batch_max_tokens = 4_194_304
    worker._generation_cut_tokenizer = SimpleNamespace(
        decode=lambda ids, **kw: bytes(ids).decode()
    )
    worker._rollout_weight_version = 7
    worker.pieces = []
    return worker


def routes(*values):
    return np.asarray(values, dtype=np.int16).reshape(-1, 1, 1)


def cut(worker, checkpoint):
    a = worker.state.call.admission
    episode = EpisodeId.from_capture_key(a.rollout_id)
    inventory = GenerationCutInventory.build(
        checkpoint_id=checkpoint,
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id=checkpoint,
                rollout_id=episode.rollout_id,
                attempt=episode.attempt,
                model_call_id=a.model_call_id,
                admitted_at=1.0,
            )
        ],
    )
    return worker._checkpoint_generation_cut_batched(inventory)[0]


def chunks(m, sink, ack):
    result = []
    for key in ack.staging_keys:
        record = sink.records[key]
        extras = dict(record.extras or {})
        tensor = decode_routed_experts(extras.pop("routed_experts"))
        result.append(
            SimpleNamespace(
                staging_key=key,
                snapshot=StagedCallBaseSnapshot.model_validate(
                    record.model_dump(exclude={"extras"})
                ),
                extras=extras,
                routed_len=len(tensor),
                fragment=SimpleNamespace(
                    routes=tensor,
                    encoding=m.ROUTE_ENCODING_ENVELOPE,
                    extras_metadata_json=json.dumps(
                        extras, sort_keys=True, separators=(",", ":")
                    ).encode(),
                ),
            )
        )
    return result


def restore(worker, ack, fetched=None):
    old = worker.state.call.admission
    admission = CaptureAdmission(
        rollout_id="r0-a1",
        model_call_id="c0-resumed",
        mode="text",
        generation_cut=dict(
            source_capture_key=old.rollout_id,
            source_model_call_id=old.model_call_id,
            staging_keys=ack.staging_keys,
            prefix_token_count=ack.prefix_token_count,
            prefix_digest=ack.prefix_digest,
            effective_output_limit=128,
        ),
    )
    resolved = worker._resolve_generation_cut(admission, [], fetched)
    worker.token_capture.begin_call(
        admission,
        generation_cut=resolved.snapshot,
        generation_cut_staging_keys=ack.staging_keys,
    )
    assert resolved.snapshot.digest == ack.prefix_digest
    return admission, resolved


def test_cache_only_concatenates_new_pieces_and_reuses_capacity(
    implementation, monkeypatch
):
    state = implementation._RequestCaptureState(call=None, prompt_token_ids=[])
    pieces = [routes(1, 2), routes(3)]
    sizes = []
    original = np.concatenate

    def concatenate(values, **kwargs):
        sizes.append(len(values))
        return original(values, **kwargs)

    monkeypatch.setattr(np, "concatenate", concatenate)
    assert state.absorb_engine_route_pieces(pieces).flatten().tolist() == [1, 2, 3]
    state.absorb_engine_route_pieces(pieces)
    pieces.append(routes(4))
    state.absorb_engine_route_pieces(pieces)
    buffer = state.engine_route_buffer
    pieces.append(routes(5))
    assert state.absorb_engine_route_pieces(pieces).flatten().tolist() == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert sizes == [2, 1, 1]
    assert state.engine_route_buffer is buffer


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [np.zeros((2, 1))],
        [routes(1), np.zeros((1, 2, 1), dtype=np.int16)],
        [np.zeros((1, 1, 1), dtype=np.float32)],
    ],
)
def test_cache_rejects_malformed_pieces_without_advancing(implementation, bad):
    state = implementation._RequestCaptureState(call=None, prompt_token_ids=[])
    with pytest.raises(ValueError):
        state.absorb_engine_route_pieces(bad)
    assert state.engine_route_piece_count == 0


def test_cache_rejects_shrink_replacement_and_dimension_change(implementation):
    state = implementation._RequestCaptureState(call=None, prompt_token_ids=[])
    pieces = [routes(1), routes(2)]
    state.absorb_engine_route_pieces(pieces)
    for bad, message in [
        (pieces[:1], "shorter"),
        ([pieces[0], routes(9)], "replaced"),
        ([*pieces, np.zeros((1, 2, 1), dtype=np.int16)], "dimensions changed"),
    ]:
        with pytest.raises(ValueError, match=message):
            state.absorb_engine_route_pieces(bad)
    assert state.engine_route_piece_count == 2


def test_repeated_cuts_restore_and_cut_again_without_cumulative_route_encoding(
    implementation, monkeypatch
):
    m = implementation
    worker = make_worker(m)
    worker.state.observe([67, 68], [-0.1, -0.2])
    worker.pieces = [routes(1, 2), routes(3)]
    encoded_sizes = []
    original = m.encode_routed_experts

    def track_encode(tensor):
        encoded_sizes.append(len(tensor))
        return original(tensor)

    monkeypatch.setattr(
        m,
        "encode_routed_experts",
        track_encode,
    )
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_codec.encode_routed_experts", track_encode
    )
    first = cut(worker, "cut1")
    worker.state.observe([69], [-0.3])
    worker.pieces.append(routes(4))
    second = cut(worker, "cut2")
    assert encoded_sizes == [4, 1]
    assert (
        first.prefix_digest
        != worker._capture_sink.records[first.staging_keys[0]].digest
    )
    fetched = chunks(m, worker._capture_sink, second)
    admission, resolved = restore(worker, second, fetched)
    assert resolved.routed_experts.flatten().tolist() == [1, 2, 3, 4, -1]
    assert len(resolved.chunk_digests) == 2
    resumed = make_worker(
        m,
        sink=worker._capture_sink,
        admission=admission,
        resolved=resolved,
        prompt=resolved.snapshot.token_ids_delta,
    )
    resumed.state.observe([70], [-0.4])
    resumed.pieces = [routes(90, 91, 92, 93, 5)]
    third = cut(resumed, "cut3")
    _, again = restore(resumed, third, chunks(m, resumed._capture_sink, third))
    assert again.routed_experts.flatten().tolist() == [1, 2, 3, 4, 5, -1]


def test_failed_staging_keeps_cache_but_not_durable_digest_progress(implementation):
    worker = make_worker(implementation)
    worker.state.observe([67], [-0.1])
    worker.pieces = [routes(1, 2)]
    worker._capture_sink.fail = True
    assert cut(worker, "failed").disposition == "durable_failure"
    assert worker.state.engine_route_piece_count == 1
    assert worker.state.generation_cut_chunk_digests == []
    assert worker.state.staged_token_count == 0
    worker._capture_sink.fail = False
    ack = cut(worker, "retry")
    assert ack.disposition == "durable_prefix"
    restore(worker, ack, chunks(implementation, worker._capture_sink, ack))


def test_routes_and_order_are_bound_to_the_advertised_digest(implementation):
    m = implementation
    worker = make_worker(m)
    worker.state.observe([67], [-0.1])
    worker.pieces = [routes(1, 2)]
    cut(worker, "cut1")
    worker.state.observe([68], [-0.2])
    worker.pieces.append(routes(3))
    ack = cut(worker, "cut2")
    fetched = chunks(m, worker._capture_sink, ack)
    fetched[0].fragment.routes[0, 0, 0] = 99
    with pytest.raises(RuntimeError, match="integrity check"):
        restore(worker, ack, fetched)
    fetched = chunks(m, worker._capture_sink, ack)
    with pytest.raises(RuntimeError):
        restore(worker, ack, fetched[::-1])
    # Even a self-consistent replacement row must fail the cumulative root.
    record = worker._capture_sink.records[ack.staging_keys[0]]
    extras = dict(record.extras)
    extras["routed_experts"] = encode_routed_experts(
        torch.tensor([[[99]], [[2]], [[-1]]], dtype=torch.int16)
    )
    replacement = worker.token_capture.build_prefix_record(
        worker.state.call,
        prompt_token_ids=[65, 66],
        generated_token_ids=[67],
        generated_logprobs=[-0.1],
        extras=extras,
    )
    worker._capture_sink.records[ack.staging_keys[0]] = replacement
    with pytest.raises(CaptureError):
        restore(worker, ack, chunks(m, worker._capture_sink, ack))


def test_legacy_route_digest_still_restores_and_extends(implementation):
    m = implementation
    worker = make_worker(m)
    worker.state.generation_cut_chunk_digests = None
    worker.state.observe([67, 68], [-0.1, -0.2])
    worker.pieces = [routes(1, 2, 3)]
    first = cut(worker, "legacy")
    admission, resolved = restore(worker, first, chunks(m, worker._capture_sink, first))
    assert resolved.chunk_digests is None
    resumed = make_worker(
        m,
        sink=worker._capture_sink,
        admission=admission,
        resolved=resolved,
        prompt=resolved.snapshot.token_ids_delta,
    )
    resumed.state.observe([69], [-0.3])
    resumed.pieces = [routes(90, 91, 92, 4)]
    second = cut(resumed, "legacy-resumed")
    _, again = restore(resumed, second, chunks(m, resumed._capture_sink, second))
    assert again.routed_experts.flatten().tolist() == [1, 2, 3, 4, -1]


def test_cache_can_include_post_freeze_routes_without_staging_them(implementation):
    worker = make_worker(implementation)
    worker.state.observe([67, 68], [-0.1, -0.2])
    worker.pieces = [routes(1, 2, 3, 4, 5)]
    ack = cut(worker, "freeze")
    _, resolved = restore(
        worker, ack, chunks(implementation, worker._capture_sink, ack)
    )
    assert resolved.routed_experts.flatten().tolist() == [1, 2, 3, -1]
    assert worker.state.engine_route_rows == 5


def test_recut_without_new_tokens_reuses_the_same_digest_and_rows(implementation):
    worker = make_worker(implementation)
    worker.state.observe([67], [-0.1])
    worker.pieces = [routes(1, 2)]
    first = cut(worker, "cut1")
    again = cut(worker, "cut2")
    assert again.prefix_digest == first.prefix_digest
    assert again.staging_keys == first.staging_keys
    assert len(worker._capture_sink.records) == 1
    restore(worker, again, chunks(implementation, worker._capture_sink, again))


def test_first_chunk_preserves_call_extras_and_rejects_unknown_layout(implementation):
    m = implementation
    worker = make_worker(m)
    worker._generation_cut_record_payload = lambda state: {
        "call_metadata": {"example": 3}
    }
    worker.state.observe([67], [-0.1])
    worker.pieces = [routes(1, 2)]
    ack = cut(worker, "cut1")
    fetched = chunks(m, worker._capture_sink, ack)
    assert fetched[0].extras["call_metadata"] == {"example": 3}
    restore(worker, ack, fetched)
    fetched[0].extras[m._CUT_DIGEST_LAYOUT] = 99
    with pytest.raises(RuntimeError, match="unsupported.*layout"):
        restore(worker, ack, fetched)


def test_canonical_terminal_row_retains_its_original_digest(implementation):
    worker = make_worker(implementation)
    record = worker.token_capture.build_prefix_record(
        worker.state.call,
        prompt_token_ids=[65, 66],
        generated_token_ids=[67],
        generated_logprobs=[-0.1],
        extras={
            "routed_experts": encode_routed_experts(
                torch.tensor([[[1]], [[2]], [[-1]]], dtype=torch.int16)
            )
        },
    )
    worker._capture_sink.records["canonical"] = record
    ack = SimpleNamespace(
        staging_keys=("canonical",), prefix_token_count=1, prefix_digest=record.digest
    )
    _, resolved = restore(
        worker, ack, chunks(implementation, worker._capture_sink, ack)
    )
    assert resolved.chunk_digests is None
    assert resolved.routed_experts.flatten().tolist() == [1, 2, -1]


def test_later_cut_failure_does_not_advance_the_chunk_commitment(implementation):
    worker = make_worker(implementation)
    worker.state.observe([67], [-0.1])
    worker.pieces = [routes(1, 2)]
    first = cut(worker, "cut1")
    digests = list(worker.state.generation_cut_chunk_digests)
    worker.state.observe([68], [-0.2])
    worker.pieces.append(routes(3))
    worker._capture_sink.fail = True
    assert cut(worker, "failed").disposition == "durable_failure"
    assert worker.state.generation_cut_chunk_digests == digests
    assert worker.state.generation_cut_staging_keys == list(first.staging_keys)
    assert worker.state.engine_route_piece_count == 2
    worker._capture_sink.fail = False
    ack = cut(worker, "retry")
    _, resolved = restore(
        worker, ack, chunks(implementation, worker._capture_sink, ack)
    )
    assert resolved.routed_experts.flatten().tolist() == [1, 2, 3, -1]


@pytest.mark.parametrize("legacy", [False, True])
def test_real_worker_admission_preserves_the_restored_digest_layout(
    implementation, legacy
):
    m = implementation
    worker = make_worker(m)
    if legacy:
        worker.state.generation_cut_chunk_digests = None
    worker.state.observe([67], [-0.1])
    worker.pieces = [routes(1, 2)]
    ack = cut(worker, "cut1")
    admission, resolved = restore(worker, ack, chunks(m, worker._capture_sink, ack))
    path = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VllmAsyncGenerationWorkerImpl"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_begin_request_capture"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        m.__dict__,
    )
    resumed = make_worker(m, sink=worker._capture_sink)
    resumed.routed_experts_dtype = torch.int16
    resumed._capture_calls = {}
    resumed._capture_calls_by_model_call_id = {}
    resumed._restored_generation_prefix_calls = 0
    resumed._restored_generation_prefix_tokens = 0
    request = SimpleNamespace(stream=False)
    m._begin_request_capture(
        resumed,
        request,
        resolved.snapshot.token_ids_delta,
        admission=admission,
        generation_cut=resolved.snapshot,
        generation_cut_routed_experts=resolved.routed_experts,
        generation_cut_chunk_digests=resolved.chunk_digests,
        resumed_generation_token_ids=[67],
    )
    state = resumed._capture_calls[id(request)]
    assert state.generation_cut_chunk_digests == (
        None if legacy else list(resolved.chunk_digests)
    )
    assert state.generation_cut_staging_keys == list(ack.staging_keys)
    # A cut immediately after restore, before another token, needs no engine lookup.
    resumed.state = state
    same = cut(resumed, "no-new-tokens")
    restore(resumed, same, chunks(m, resumed._capture_sink, same))


@pytest.fixture(scope="module")
def grouped_implementation(implementation):
    """Use actual vLLM adapter/worker methods without importing GPU dependencies."""
    m = implementation
    path = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    )
    tree = ast.parse(path.read_text())
    client = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "_AsyncLLMHTTPClient"
    )
    client.body = [
        n
        for n in client.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name
        in {
            "__init__",
            "generate",
            "_generate",
            "_run_on_engine_loop",
            "check_route_snapshot_support",
            "_install_route_retention",
            "close_route_snapshots",
            "_direct_route_snapshot_enabled",
            "snapshot_routed_experts",
            "snapshot_routed_experts_many",
            "_routed_expert_chunks",
            "enable_token_snapshots",
            "snapshot_generation_tokens",
        }
    ]
    worker = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "VllmAsyncGenerationWorkerImpl"
    )
    bridges = [
        n
        for n in worker.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "_snapshot_engine_routed_experts_many",
            "_observe_request_capture",
            "_refresh_generation_cut_tokens",
        }
    ]
    utilities = ast.parse((path.parent / "utils.py").read_text())
    extraction = [
        n
        for n in utilities.body
        if isinstance(n, ast.FunctionDef)
        and n.name
        in {
            "extract_published_generation",
            "_flat_logprob_arrays",
            "_extract_selected_token_logprobs",
            "own_routed_expert_fragments",
        }
    ]
    m.PackedSelectedLogprobs = PackedSelectedLogprobs
    m.VLLM_LOGPROB_FLOOR = -9999.0
    terminal = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "_RestoredPrefixTerminal"
    )
    m.run_engine_snapshot = run_engine_snapshot
    m.sys = sys
    m.defaultdict = defaultdict
    m.importlib = importlib
    m.Version = Version
    m._ROUTE_SNAPSHOT_VLLM_RELEASE = (0, 29, 0)
    m._G_ENGINE_CAPTURE_STATE = contextvars.ContextVar(
        "test_capture_state", default=None
    )
    snapshots = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in {"_EngineTokenSnapshot", "_RetainedEngineTokens"}
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[future, terminal, *snapshots, client, *bridges, *extraction],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        m.__dict__,
    )
    return m


@pytest.fixture
def owner_loop():
    loop = asyncio.new_event_loop()
    started = threading.Event()
    loop.call_soon(started.set)
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    assert started.wait(5)
    try:
        yield loop
    finally:
        # Drain cancellation of a timed-out snapshot before stopping the owner.
        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        assert not thread.is_alive()
        loop.close()


def make_group_worker(m, loop, *, count=3):
    worker = make_worker(m)
    states = [worker.state]
    for i in range(1, count):
        call = worker.token_capture.begin_call(
            CaptureAdmission(rollout_id=f"r{i}", model_call_id=f"c{i}", mode="text")
        )
        state = m._RequestCaptureState(
            call=call,
            prompt_token_ids=[65, 66],
            capture_routed_experts=True,
            engine_request_id=f"engine-{i}",
            effective_output_limit=128,
        )
        worker._capture_calls_by_model_call_id[call.model_call_id] = state
        states.append(state)
    external = {}
    requests = {}
    for i, state in enumerate(states):
        state.engine_request_id = f"engine-{i}"
        state.observe([67, 68], [-0.1, -0.2])
        external[f"engine-{i}"] = {f"internal-{i}"}
        requests[f"internal-{i}"] = SimpleNamespace(
            request_id=f"internal-{i}",
            external_req_id=f"engine-{i}",
            routed_experts_chunks=[routes(1, 2), routes(3)],
        )
    processor = SimpleNamespace(external_req_ids=external, request_states=requests)

    def finish(state):
        processor.request_states.pop(state.request_id)
        ids = processor.external_req_ids[state.external_req_id]
        ids.remove(state.request_id)
        if not ids:
            processor.external_req_ids.pop(state.external_req_id)

    processor._finish_request = finish
    worker._http_engine_client = m._AsyncLLMHTTPClient(
        SimpleNamespace(
            output_processor=processor,
            model_config=None,
            renderer=None,
            input_processor=None,
            vllm_config=None,
        ),
        loop,
        flat_logprobs=False,
    )
    worker._snapshot_engine_routed_experts_many = (
        m._snapshot_engine_routed_experts_many.__get__(worker, type(worker))
    )
    inventory = GenerationCutInventory.build(
        checkpoint_id="group-cut",
        server_name="policy",
        active_prefixes=[
            GenerationCutPrefix(
                ticket_id=f"t{i}",
                rollout_id=f"r{i}",
                attempt=0,
                model_call_id=f"c{i}",
                admitted_at=1.0,
            )
            for i in range(count)
        ],
    )
    return worker, states, processor, inventory


def assert_group_released(states):
    for state in states:
        assert state.frozen_cut is None
        assert state.lifecycle_lock.acquire(blocking=False)
        state.lifecycle_lock.release()


def enable_engine_tokens(m, loop, monkeypatch, *, count=1, packed=False):
    worker, states, processor, inventory = make_group_worker(m, loop, count=count)
    client = verified_direct_client(m, worker, processor, monkeypatch)
    processor.process_outputs = lambda updates: [update.apply() for update in updates]
    client.enable_token_snapshots()
    worker._refresh_generation_cut_tokens = m._refresh_generation_cut_tokens.__get__(
        worker, type(worker)
    )
    worker._capture_calls = {}
    for i, state in enumerate(states):
        state.engine_token_snapshot = True
        state.generated_token_ids = []
        state.generated_logprobs = []
        engine = processor.request_states[f"internal-{i}"]
        engine.detokenizer = SimpleNamespace(output_token_ids=[67, 68])
        engine.logprobs_processor = SimpleNamespace(
            logprobs=[
                {67: SimpleNamespace(logprob=-0.1)},
                {68: SimpleNamespace(logprob=-0.2)},
            ]
        )
        if packed:
            history = PackedSelectedLogprobs(SimpleNamespace)
            history.append_arrays(
                np.array([[67], [68]]), np.array([[-0.1], [-0.2]]), np.array([1, 2])
            )
            engine.logprobs_processor.logprobs = history
        engine.queue = SimpleNamespace(output=None)
    return worker, states, processor, inventory


@pytest.mark.parametrize("packed", [False, True])
def test_owned_fragments_preserve_two_cut_restore_integrity(
    grouped_implementation, owner_loop, monkeypatch, packed
):
    m = grouped_implementation
    worker, states, processor, inventory = enable_engine_tokens(
        m, owner_loop, monkeypatch, packed=packed
    )
    engine = processor.request_states["internal-0"]
    engine.routed_experts_chunks = [routes(1, 2), routes(3)]
    first = worker._checkpoint_generation_cut_batched(inventory)[0]
    assert first.disposition == "durable_prefix"
    restore(worker, first, chunks(m, worker._capture_sink, first))
    old_piece = engine.routed_experts_chunks[-1]
    backing = bytearray(routes(9).tobytes())
    update = SimpleNamespace(
        routed_experts=np.frombuffer(backing, dtype=np.int16).reshape(1, 1, 1)
    )

    def publish():
        engine.routed_experts_chunks.append(update.routed_experts)
        engine.detokenizer.output_token_ids.append(69)
        if packed:
            engine.logprobs_processor.logprobs.append_arrays(
                np.array([[69]]), np.array([[-0.3]]), np.array([1])
            )
        else:
            engine.logprobs_processor.logprobs.append(
                {69: SimpleNamespace(logprob=-0.3)}
            )

    update.apply = publish
    processor.process_outputs([update])
    assert update.routed_experts.flags.owndata
    assert engine.routed_experts_chunks[-2] is old_piece
    np.frombuffer(backing, dtype=np.int16)[:] = 42
    second_inventory = GenerationCutInventory.build(
        checkpoint_id="owned-cut-2",
        server_name=inventory.server_name,
        active_prefixes=inventory.active_prefixes,
    )
    second = worker._checkpoint_generation_cut_batched(second_inventory)[0]
    assert second.disposition == "durable_prefix"
    _, resolved = restore(worker, second, chunks(m, worker._capture_sink, second))
    assert resolved.routed_experts.flatten().tolist() == [1, 2, 3, 9, -1]
    assert states[0].generated_token_ids == [67, 68, 69]
    assert_group_released(states)


@pytest.mark.parametrize("packed", [False, True])
def test_direct_token_cut_and_restore_without_http_progress(
    grouped_implementation, owner_loop, monkeypatch, packed
):
    m = grouped_implementation
    worker, states, processor, inventory = enable_engine_tokens(
        m, owner_loop, monkeypatch, packed=packed
    )
    entered, release = threading.Event(), threading.Event()
    owner_loop.call_soon_threadsafe(lambda: (entered.set(), release.wait(5)))
    assert entered.wait(5)
    try:
        # The engine owner is deliberately unavailable. A direct cut needs
        # neither its event loop nor intermediate HTTP observations.
        ack = worker._checkpoint_generation_cut_batched(inventory)[0]
        assert ack.disposition == "durable_prefix"
        assert states[0].generated_token_ids == [67, 68]
        assert states[0].generated_logprobs == [-0.1, -0.2]
        _, restored = restore(worker, ack, chunks(m, worker._capture_sink, ack))
        assert restored.snapshot.token_ids_delta == [65, 66, 67, 68]
        assert states[0].staged_token_count == 2
        assert processor.request_states
        assert_group_released(states)
    finally:
        release.set()


def test_direct_token_probe_waits_for_complete_publication_and_honors_deadline(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, states, processor, inventory = enable_engine_tokens(
        m, owner_loop, monkeypatch
    )
    engine = processor.request_states["internal-0"]
    entered, release = threading.Event(), threading.Event()

    def partial_update():
        engine.detokenizer.output_token_ids.append(69)
        entered.set()
        assert release.wait(5)
        engine.logprobs_processor.logprobs.append({69: SimpleNamespace(logprob=-0.3)})
        engine.routed_experts_chunks.append(routes(4))

    writer = threading.Thread(
        target=processor.process_outputs,
        args=([SimpleNamespace(routed_experts=None, apply=partial_update)],),
    )
    writer.start()
    assert entered.wait(5)
    try:
        ack = worker._checkpoint_generation_cut_batched(
            inventory, deadline=time.monotonic() + 0.02
        )[0]
        assert ack.disposition == "durable_failure"
        assert not worker._capture_sink.records
        assert states[0].frozen_cut is None
        assert states[0].generated_token_ids == []
    finally:
        release.set()
        writer.join(5)
        assert not writer.is_alive()
    ack = worker._checkpoint_generation_cut_batched(inventory)[0]
    assert ack.disposition == "durable_prefix"
    assert states[0].generated_token_ids == [67, 68, 69]
    restore(worker, ack, chunks(m, worker._capture_sink, ack))
    assert_group_released(states)


@pytest.mark.parametrize(
    "finish_reason,stop_reason", [("stop", "END"), ("length", None)]
)
def test_direct_token_retention_preserves_terminal_boundary(
    grouped_implementation, owner_loop, monkeypatch, finish_reason, stop_reason
):
    m = grouped_implementation
    worker, states, processor, inventory = enable_engine_tokens(
        m, owner_loop, monkeypatch
    )
    client = worker._http_engine_client
    engine = processor.request_states["internal-0"]
    final_output = SimpleNamespace(
        request_id="engine-0",
        outputs=[
            SimpleNamespace(
                token_ids=engine.detokenizer.output_token_ids,
                logprobs=engine.logprobs_processor.logprobs,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
                routed_experts=np.concatenate(engine.routed_experts_chunks),
            )
        ],
    )
    request = object()
    worker._capture_calls[id(request)] = states[0]

    async def generate(*args, **kwargs):
        # The engine finishes before any output is handed to HTTP.
        def publish_final():
            engine.queue.output = final_output
            processor._finish_request(engine)

        processor.process_outputs(
            [SimpleNamespace(routed_experts=None, apply=publish_final)]
        )
        yield final_output

    client._engine_client.generate = generate

    async def consume():
        stream = client.generate(None, None, "engine-0")
        assert await anext(stream) is final_output
        assert not processor.request_states
        assert (
            client._completed_token_states["engine-0"]["internal-0"].request_state
            is engine
        )
        ack = worker._checkpoint_generation_cut_batched(inventory)[0]
        assert ack.disposition == "durable_prefix"
        assert ack.terminal_finish_reason == finish_reason
        assert ack.terminal_stop_reason == stop_reason
        restore(worker, ack, chunks(m, worker._capture_sink, ack))
        m._observe_request_capture(worker, request, final_output)
        assert states[0].generated_token_ids == [67, 68]  # No duplicate append.
        assert not states[0].engine_token_snapshot
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert not client._completed_token_states
        retry = worker._checkpoint_generation_cut_batched(inventory)[0]
        assert retry.disposition == "durable_prefix"
        assert retry.terminal_finish_reason == finish_reason
        assert_group_released(states)

    asyncio.run(consume())


def test_direct_token_reads_only_new_delta_after_previous_cut(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, states, processor, inventory = enable_engine_tokens(
        m, owner_loop, monkeypatch
    )
    first = worker._checkpoint_generation_cut_batched(inventory)[0]
    assert first.disposition == "durable_prefix"
    engine = processor.request_states["internal-0"]

    def update():
        engine.detokenizer.output_token_ids.append(69)
        engine.logprobs_processor.logprobs.append({69: SimpleNamespace(logprob=-0.3)})
        engine.routed_experts_chunks.append(routes(4))

    processor.process_outputs([SimpleNamespace(routed_experts=None, apply=update)])
    second_inventory = GenerationCutInventory.build(
        checkpoint_id="next-cut",
        server_name="policy",
        active_prefixes=list(inventory.active_prefixes),
    )
    worker._capture_sink.fail = True
    failed = worker._checkpoint_generation_cut_batched(second_inventory)[0]
    assert failed.disposition == "durable_failure"
    assert states[0].staged_token_count == 2
    assert_group_released(states)
    worker._capture_sink.fail = False
    second = worker._checkpoint_generation_cut_batched(second_inventory)[0]
    assert second.disposition == "durable_prefix"
    assert states[0].staged_token_count == 3
    records = chunks(m, worker._capture_sink, second)
    assert records[-1].snapshot.token_ids_delta == [69]
    _, restored = restore(worker, second, records)
    assert restored.snapshot.token_ids_delta == [65, 66, 67, 68, 69]


def test_final_http_handoff_wins_race_with_older_token_probe(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, states, processor, _ = enable_engine_tokens(m, owner_loop, monkeypatch)
    state = states[0]
    request = object()
    worker._capture_calls[id(request)] = state
    engine = processor.request_states["internal-0"]
    output = SimpleNamespace(
        request_id="engine-0",
        outputs=[
            SimpleNamespace(
                token_ids=[67, 68],
                logprobs=engine.logprobs_processor.logprobs,
                finish_reason="length",
                stop_reason=None,
                routed_experts=np.concatenate(engine.routed_experts_chunks),
            )
        ],
    )

    def snapshot(*args, **kwargs):
        m._observe_request_capture(worker, request, output)
        return m._EngineTokenSnapshot([67], [-0.1], None, None)

    monkeypatch.setattr(
        worker._http_engine_client, "snapshot_generation_tokens", snapshot
    )
    worker._refresh_generation_cut_tokens(state, deadline=None)
    assert state.generated_token_ids == [67, 68]
    assert state.terminal_finish_reason == "length"


def test_engine_request_id_is_bound_before_first_output(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, states, _, _ = enable_engine_tokens(m, owner_loop, monkeypatch, count=2)
    client = worker._http_engine_client
    for state in states:
        state.engine_request_id = None

    async def generate(prompt, sampling_params, request_id, **kwargs):
        index = int(request_id[-1])
        assert states[index].engine_request_id == request_id
        yield "final"

    client._engine_client.generate = generate

    async def consume(index):
        token = m._G_ENGINE_CAPTURE_STATE.set(states[index])
        try:
            assert [
                x async for x in client.generate(None, None, f"engine-{index}")
            ] == ["final"]
        finally:
            m._G_ENGINE_CAPTURE_STATE.reset(token)

    async def run():
        await asyncio.gather(consume(0), consume(1))

    asyncio.run(run())
    assert m._G_ENGINE_CAPTURE_STATE.get() is None
    assert not client._completed_token_states


def test_token_publication_hook_is_idempotent_and_restored(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, _, processor, _ = enable_engine_tokens(m, owner_loop, monkeypatch)
    client = worker._http_engine_client
    original = client._token_process_original
    wrapper = processor.process_outputs
    client.enable_token_snapshots()
    assert processor.process_outputs is wrapper
    client.close_route_snapshots()
    client.close_route_snapshots()
    assert processor.process_outputs is original
    assert not client._token_snapshots_enabled
    assert not client._completed_token_states


def test_direct_token_reader_uses_its_own_diagnostics_imports(
    grouped_implementation, owner_loop, monkeypatch
):
    """The worker must not borrow names imported by the capture-mixin fixture."""
    m = grouped_implementation
    worker, _, _, _ = enable_engine_tokens(m, owner_loop, monkeypatch)
    path = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    )
    tree = ast.parse(path.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "snapshot_generation_tokens"
    )
    imports = [
        n
        for n in tree.body
        if isinstance(n, ast.ImportFrom)
        and n.module == "nemo_rl.utils.prefix_cut_diagnostics"
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    namespace = dict(
        time=time,
        _EngineTokenSnapshot=m._EngineTokenSnapshot,
        _extract_selected_token_logprobs=m._extract_selected_token_logprobs,
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *imports, method], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    snapshot = namespace["snapshot_generation_tokens"](
        worker._http_engine_client, "engine-0", deadline=None
    )
    assert snapshot.token_ids == [67, 68]
    assert snapshot.logprobs == [-0.1, -0.2]


def test_token_probe_releases_publication_lock_when_diagnostics_raise(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, _, _, _ = enable_engine_tokens(m, owner_loop, monkeypatch)
    client = worker._http_engine_client

    def broken_timing(*args, **kwargs):
        raise RuntimeError("timing failure")

    monkeypatch.setattr(m, "prefix_cut_wait", broken_timing)
    with pytest.raises(RuntimeError, match="timing failure"):
        client.snapshot_generation_tokens("engine-0", deadline=None)

    def try_lock():
        acquired = client._token_snapshot_lock.acquire(timeout=0.1)
        if acquired:
            client._token_snapshot_lock.release()
        return acquired

    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(try_lock).result(timeout=2)


@pytest.mark.parametrize("stream", [False, True])
def test_production_sampling_params_use_final_only_without_changing_sampling(
    grouped_implementation, monkeypatch, stream
):
    m = grouped_implementation
    path = (
        Path(__file__).resolve().parents[4]
        / "nemo_rl/models/generation/vllm/vllm_worker_async.py"
    )
    tree = ast.parse(path.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "to_sampling_params"
    )
    params = SimpleNamespace(
        output_kind="delta" if stream else "final_only",
        max_tokens=128,
        temperature=0.9,
        top_p=0.95,
        seed=42,
    )

    class Base:
        def to_sampling_params(self, *args, **kwargs):
            return params

    capture_state = m._RequestCaptureState(call=None, prompt_token_ids=[])
    recorded = []
    worker = SimpleNamespace(
        _generation_prefix_cuts_enabled=True,
        _get_request_capture=lambda request: capture_state,
        _record_request_effective_output_limit=lambda request, limit: recorded.append(
            limit
        ),
        _record_request_cut_constraints=lambda *args: None,
    )
    cls = ast.ClassDef(
        name="Request",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    namespace = dict(
        Base=Base,
        worker_self=worker,
        _G_ENGINE_CAPTURE_STATE=m._G_ENGINE_CAPTURE_STATE,
        RequestOutputKind=SimpleNamespace(FINAL_ONLY="final_only"),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    request = namespace["Request"]()
    request.stream = stream
    request.ng_capture = {"model_call_id": "c0"}
    token = m._G_ENGINE_CAPTURE_STATE.set(None)
    try:
        assert request.to_sampling_params() is params
        assert m._G_ENGINE_CAPTURE_STATE.get() is capture_state
    finally:
        m._G_ENGINE_CAPTURE_STATE.reset(token)
    assert params.output_kind == ("delta" if stream else "final_only")
    assert (params.temperature, params.top_p, params.seed, params.max_tokens) == (
        0.9,
        0.95,
        42,
        128,
    )
    assert recorded == [128]


@pytest.mark.parametrize(
    "failure", ["missing", "mismatched_logprobs", "unknown_terminal"]
)
def test_direct_token_probe_rejects_incomplete_evidence(
    grouped_implementation, owner_loop, monkeypatch, failure
):
    m = grouped_implementation
    worker, states, processor, inventory = enable_engine_tokens(
        m, owner_loop, monkeypatch
    )
    if failure == "missing":
        processor.request_states.clear()
        processor.external_req_ids.clear()
    elif failure == "mismatched_logprobs":
        processor.request_states["internal-0"].logprobs_processor.logprobs.pop()
    else:
        client = worker._http_engine_client
        client._route_retention_requests.add("engine-0")
        processor._finish_request(processor.request_states["internal-0"])
    ack = worker._checkpoint_generation_cut_batched(inventory)[0]
    assert ack.disposition == "durable_failure"
    assert not worker._capture_sink.records
    assert states[0].staged_token_count == 0
    assert_group_released(states)


def test_control_queue_timer_includes_guarded_admission(implementation, caplog):
    m = implementation
    worker = make_worker(m)
    worker._generation_cut_control = m.GenerationCutControl(max_workers=1)
    worker._generation_cut_receipts = {}
    entered, release = threading.Event(), threading.Event()

    def operation(inventory, *, deadline):
        if inventory.inventory_digest == "blocker":
            entered.set()
            assert release.wait(5)
        return inventory.inventory_digest

    worker._checkpoint_generation_cut = operation
    inventories = [
        SimpleNamespace(
            checkpoint_id="timer-cut",
            inventory_digest=name,
            active_prefixes=[SimpleNamespace(model_call_id=name)],
        )
        for name in ("blocker", "target")
    ]
    caplog.set_level(logging.INFO)

    async def scenario():
        blocker = asyncio.create_task(
            worker._checkpoint_generation_cut_control(inventories[0], deadline=None)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            with prefix_cut_probe(
                enabled=True,
                checkpoint_id="timer-cut",
                inventory_digest="target",
                calls=1,
                deadline=None,
            ):
                target = asyncio.create_task(
                    worker._checkpoint_generation_cut_control(
                        inventories[1], deadline=None
                    )
                )
                await asyncio.sleep(0.03)
                release.set()
                assert await asyncio.wait_for(target, 2) == "target"
            await blocker
        finally:
            release.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker._generation_cut_control_executor = executor
        try:
            asyncio.run(scenario())
        finally:
            worker._generation_cut_control.close()
    report = next(
        json.loads(record.message.split("prefix_cut_probe ", 1)[1])
        for record in caplog.records
        if record.message.startswith("prefix_cut_probe ")
    )
    phases = report["phases"]
    assert phases["control_admission_wait"]["wall_s"] >= 0.02
    assert (
        phases["control_queue_wait"]["wall_s"]
        >= phases["control_admission_wait"]["wall_s"]
    )
    assert phases["control_execution"]["calls"] == 1


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("fail_first", [False, True])
def test_guarded_control_stages_and_seals_independent_real_cuts(
    grouped_implementation, owner_loop, workers, fail_first
):
    """Use production admission, executor, records, digests and cut transactions."""
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop, count=2)
    worker._generation_cut_control = m.GenerationCutControl(max_workers=workers)
    worker._generation_cut_receipts = {}
    worker._completed_capture_calls = {}
    first_entered, second_staged, release_first = (threading.Event() for _ in range(3))
    writes = []
    sink = worker._capture_sink
    original_put = sink.stage_generation_prefix_batch

    def put(records, **kwargs):
        call_id = records[0].model_call_id
        writes.append(call_id)
        if call_id == "c0":
            first_entered.set()
            assert release_first.wait(5)
            if fail_first:
                raise RuntimeError("injected first inventory write failure")
        result = original_put(records, **kwargs)
        if call_id == "c1":
            second_staged.set()
        return result

    sink.stage_generation_prefix_batch = put
    inventories = [
        GenerationCutInventory.build(
            checkpoint_id=inventory.checkpoint_id,
            server_name=f"policy-{i}",
            active_prefixes=[prefix],
        )
        for i, prefix in enumerate(inventory.active_prefixes)
    ]

    async def scenario():
        first = asyncio.create_task(
            worker._checkpoint_generation_cut_control(inventories[0], deadline=None)
        )
        try:
            assert await asyncio.to_thread(first_entered.wait, 2)
            duplicate = asyncio.create_task(
                worker._checkpoint_generation_cut_control(inventories[0], deadline=None)
            )
            second = asyncio.create_task(
                worker._checkpoint_generation_cut_control(inventories[1], deadline=None)
            )
            if workers == 2:
                assert await asyncio.to_thread(second_staged.wait, 2)
                second_receipt = await asyncio.wait_for(asyncio.shield(second), 2)
                second_receipt.validate_for(inventories[1])
                assert states[1].staged_token_count == 2
            else:
                assert not await asyncio.to_thread(second_staged.wait, 0.02)
            # The first record is not durable yet, so no caller may receive its
            # acknowledgement, nor may its cut be sealed ahead of staging.
            assert not first.done() and not duplicate.done()
            assert states[0].staged_token_count == 0
            assert states[0].lifecycle_lock.locked()
        finally:
            release_first.set()
        first_receipt, duplicate_receipt, second_receipt = await asyncio.wait_for(
            asyncio.gather(first, duplicate, second), 3
        )
        assert first_receipt is duplicate_receipt
        first_receipt.validate_for(inventories[0])
        second_receipt.validate_for(inventories[1])
        return first_receipt.prefixes[0], second_receipt.prefixes[0]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        worker._generation_cut_control_executor = executor
        try:
            first_ack, second_ack = asyncio.run(scenario())
        finally:
            release_first.set()
            worker._generation_cut_control.close()
    assert writes.count("c0") == 1 and writes.count("c1") == 1
    assert first_ack.disposition == (
        "durable_failure" if fail_first else "durable_prefix"
    )
    assert second_ack.disposition == "durable_prefix"
    assert states[0].staged_token_count == (0 if fail_first else 2)
    assert states[1].staged_token_count == 2
    assert_group_released(states)
    # The staged fragment is still a valid route-covered record, not just an ack.
    record = sink.records[second_ack.staging_keys[0]]
    fragment = chunks(m, sink, second_ack)[0].fragment
    assert m.verify_route_fragment_integrity(
        fragment,
        extras_digest_version=record.extras_digest_version,
        expected_extras_digest=record.extras_digest,
    )
    restore_host = make_worker(m, sink=sink, admission=states[1].call.admission)
    _, restored = restore(restore_host, second_ack, chunks(m, sink, second_ack))
    assert restored.snapshot.digest == second_ack.prefix_digest
    assert restored.routed_experts.flatten().tolist() == [1, 2, 3, -1]


def test_group_uses_one_real_owner_handoff_after_every_call_is_frozen(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop)
    handoffs = []

    def snapshot(callback, **kwargs):
        handoffs.append(1)
        assert all(s.frozen_cut == ("group-cut", 2) for s in states)
        assert all(not s.lock.locked() for s in states)
        assert all(s.lifecycle_lock.locked() for s in states)
        return run_engine_snapshot(callback, **kwargs)

    monkeypatch.setattr(m, "run_engine_snapshot", snapshot)
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert handoffs == [1]
    assert [ack.ticket_id for ack in acks] == ["t0", "t1", "t2"]
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    for ack, state in zip(acks, states, strict=True):
        assert ack.prefix_token_count == 2
        assert state.staged_token_count == 2
        record = worker._capture_sink.records[ack.staging_keys[0]]
        assert decode_routed_experts(
            record.extras["routed_experts"]
        ).flatten().tolist() == [1, 2, 3, -1]
    assert_group_released(states)


@pytest.mark.parametrize(
    "failure", ["missing", "multiple_ids", "missing_chunks", "short_routes"]
)
def test_group_bad_request_does_not_fail_healthy_calls(
    grouped_implementation, owner_loop, failure
):
    m = grouped_implementation
    worker, states, processor, inventory = make_group_worker(m, owner_loop)
    if failure == "missing":
        processor.external_req_ids.pop("engine-1")
    elif failure == "multiple_ids":
        processor.external_req_ids["engine-1"].add("other-internal")
    elif failure == "missing_chunks":
        processor.request_states["internal-1"].routed_experts_chunks = None
    else:
        processor.request_states["internal-1"].routed_experts_chunks = [routes(1)]
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert [ack.disposition for ack in acks] == [
        "durable_prefix",
        "durable_failure",
        "durable_prefix",
    ]
    assert states[1].staged_token_count == 0
    assert states[1].generation_cut_chunk_digests == []
    assert_group_released(states)


def test_group_completion_and_observation_can_arrive_during_lookup(
    grouped_implementation, owner_loop
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop)
    client = worker._http_engine_client
    original = client._routed_expert_chunks

    def lookup(request_id):
        index = int(request_id.rsplit("-", 1)[1])
        state = states[index]
        assert not state.lock.locked()
        with state.lock:
            state.observe([69], [-0.3])
            if index == 1:
                # Engine state disappeared, but the final output was observed.
                state.terminal_routed_experts = torch.from_numpy(routes(1, 2, 3, 4))
        return None if index == 1 else [*original(request_id), routes(4)]

    client._routed_expert_chunks = lookup
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    assert all(ack.prefix_token_count == 2 for ack in acks)
    assert all(s.generated_token_ids[s.staged_token_count :] == [69] for s in states)
    assert_group_released(states)


@pytest.mark.parametrize("bound", ["calls", "rows", "tokens"])
def test_route_group_limits_do_not_change_tq_row_limits(
    grouped_implementation, owner_loop, monkeypatch, bound
):
    m = grouped_implementation
    count = 65 if bound == "calls" else 5
    worker, states, _, inventory = make_group_worker(m, owner_loop, count=count)
    if bound == "rows":
        worker._generation_prefix_batch_max_rows = 2
    if bound == "tokens":
        worker._generation_prefix_batch_max_tokens = 8  # Two four-token first rows.
    handoffs = []
    puts = []
    original_put = worker._capture_sink.stage_generation_prefix_batch

    def snapshot(callback, **kwargs):
        # Count actual owner reads inside each handoff, not frozen calls.
        def read():
            result = callback()
            handoffs.append(len(result))
            return result

        return run_engine_snapshot(read, **kwargs)

    def put(records, **kwargs):
        puts.append(len(records))
        return original_put(records, **kwargs)

    monkeypatch.setattr(m, "run_engine_snapshot", snapshot)
    worker._capture_sink.stage_generation_prefix_batch = put
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert handoffs == ([64, 1] if bound == "calls" else [2, 2, 1])
    assert puts == ([65] if bound == "calls" else [2, 2, 1])
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    assert_group_released(states)


def test_route_group_owner_timeout_rolls_back_all_calls_and_can_retry(
    grouped_implementation, owner_loop
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop)
    entered, release = threading.Event(), threading.Event()

    def block_owner():
        entered.set()
        assert release.wait(5)

    owner_loop.call_soon_threadsafe(block_owner)
    assert entered.wait(5)
    try:
        acks = worker._checkpoint_generation_cut_batched(
            inventory, deadline=m.time.monotonic() + 0.05
        )
        assert all(ack.disposition == "durable_failure" for ack in acks)
        assert not worker._capture_sink.records
        assert all(s.staged_token_count == 0 for s in states)
        assert_group_released(states)
    finally:
        release.set()
    # Drain the cancelled callback; it must not seal any already rolled-back cut.
    asyncio.run_coroutine_threadsafe(asyncio.sleep(0), owner_loop).result(timeout=5)
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    assert_group_released(states)


def test_group_wait_keeps_observation_running_but_serializes_terminal_capture(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop)
    owner_entered, release_owner = threading.Event(), threading.Event()
    submitted, terminal_done = threading.Event(), threading.Event()

    def block_owner():
        owner_entered.set()
        assert release_owner.wait(5)

    def snapshot(callback, **kwargs):
        submitted.set()
        return run_engine_snapshot(callback, **kwargs)

    def terminal():
        with states[0].lifecycle_lock:
            terminal_done.set()

    monkeypatch.setattr(m, "run_engine_snapshot", snapshot)
    owner_loop.call_soon_threadsafe(block_owner)
    assert owner_entered.wait(5)
    with ThreadPoolExecutor(max_workers=2) as executor:
        cut_future = executor.submit(
            worker._checkpoint_generation_cut_batched, inventory
        )
        try:
            assert submitted.wait(5)
            terminal_future = executor.submit(terminal)
            assert not terminal_done.wait(0.05)
            for state in states:
                with state.lock:
                    state.observe([69], [-0.3])
        finally:
            release_owner.set()
        acks = cut_future.result(timeout=5)
        terminal_future.result(timeout=5)
    assert all(ack.prefix_token_count == 2 for ack in acks)
    assert all(s.generated_token_ids[s.staged_token_count :] == [69] for s in states)
    assert_group_released(states)


def test_closing_transactions_waiting_for_routes_rolls_back_and_releases_locks(
    grouped_implementation, owner_loop
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop)
    transactions = []
    for state, prefix in zip(states, inventory.active_prefixes, strict=True):
        transaction = worker._generation_cut_transaction(
            prefix, state, inventory.checkpoint_id, batch_route_lookup=True
        )
        assert isinstance(next(transaction), m._RouteSnapshotRequest)
        transactions.append(transaction)
    for transaction in transactions:
        transaction.close()
    assert_group_released(states)
    assert not worker._capture_sink.records


def test_group_transport_failure_does_not_advance_durable_progress(
    grouped_implementation, owner_loop
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop)
    worker._capture_sink.fail = True
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert all(ack.disposition == "durable_failure" for ack in acks)
    assert all(
        s.staged_token_count == 0 and not s.generation_cut_chunk_digests for s in states
    )
    assert_group_released(states)


def test_group_snapshot_copies_lists_but_not_route_arrays(
    grouped_implementation, owner_loop
):
    m = grouped_implementation
    worker, _, processor, _ = make_group_worker(m, owner_loop)
    result = worker._http_engine_client.snapshot_routed_experts_many(
        ["engine-0", "engine-1"], timeout_s=5
    )
    original = processor.request_states["internal-0"].routed_experts_chunks
    assert result["engine-0"].routes is not original
    assert result["engine-0"].routes[0] is original[0]


def test_group_terminal_routes_and_replay_disabled_skip_owner_lookup(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop)
    for state in states:
        state.terminal_routed_experts = torch.from_numpy(routes(1, 2, 3))
    states[1].capture_routed_experts = False

    def unexpected(*args, **kwargs):
        pytest.fail("no call needs live engine routes")

    monkeypatch.setattr(m, "run_engine_snapshot", unexpected)
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    assert_group_released(states)


@pytest.mark.parametrize("when", ["before_lookup", "after_lookup", "during_put"])
def test_group_deadline_releases_pending_routes_and_records(
    grouped_implementation, owner_loop, monkeypatch, when
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop, count=5)
    clock = {"now": 0.0}
    monkeypatch.setattr(
        m,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock["now"], perf_counter=m.time.perf_counter
        ),
    )
    if when == "before_lookup":
        original_freeze = states[0].freeze_for_checkpoint

        def freeze(checkpoint_id):
            result = original_freeze(checkpoint_id)
            clock["now"] = 10.0
            return result

        states[0].freeze_for_checkpoint = freeze
    elif when == "after_lookup":

        def snapshot(callback, **kwargs):
            result = run_engine_snapshot(callback, **kwargs)
            clock["now"] = 10.0
            return result

        monkeypatch.setattr(m, "run_engine_snapshot", snapshot)
    else:
        worker._generation_prefix_batch_max_rows = 2
        original_put = worker._capture_sink.stage_generation_prefix_batch

        def put(records, **kwargs):
            result = original_put(records, **kwargs)
            clock["now"] = 10.0
            return result

        worker._capture_sink.stage_generation_prefix_batch = put
    acks = worker._checkpoint_generation_cut_batched(inventory, deadline=5.0)
    expected = (
        ["durable_prefix"] * 2 + ["durable_failure"] * 3
        if when == "during_put"
        else ["durable_failure"] * 5
    )
    assert [ack.disposition for ack in acks] == expected
    assert len(worker._capture_sink.records) == (2 if when == "during_put" else 0)
    assert_group_released(states)


@pytest.mark.parametrize("when", ["lookup", "record", "token_bound"])
def test_group_interruption_closes_every_owned_transaction(
    grouped_implementation, owner_loop, monkeypatch, when
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop)
    if when == "record":
        original = worker.token_capture.build_prefix_record

        def build(call, **kwargs):
            if call.model_call_id == "c1":
                raise KeyboardInterrupt("injected record interruption")
            return original(call, **kwargs)

        worker.token_capture.build_prefix_record = build
    else:
        if when == "token_bound":
            # A second call is already frozen when the previous group flushes.
            worker._generation_prefix_batch_max_tokens = 5

        def snapshot(*args, **kwargs):
            raise KeyboardInterrupt("injected lookup interruption")

        monkeypatch.setattr(m, "run_engine_snapshot", snapshot)
    with pytest.raises(KeyboardInterrupt, match="injected"):
        worker._checkpoint_generation_cut_batched(inventory)
    assert_group_released(states)
    assert all(s.staged_token_count == 0 for s in states)
    assert not worker._capture_sink.records


@pytest.mark.parametrize("shared_writer", [False, True])
def test_group_failed_batch_put_retries_individual_rows(
    grouped_implementation, owner_loop, shared_writer: bool
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop)
    sink = worker._capture_sink
    original_put = sink.stage_generation_prefix_batch
    puts = []

    def put(records, **kwargs):
        puts.append(len(records))
        keys = original_put(records, **kwargs)
        if len(records) > 1:
            sink.clear(keys)  # Match the production sink's failed-put cleanup.
            raise RuntimeError("injected multi-row failure")
        return keys

    sink.stage_generation_prefix_batch = put
    writer = None
    if shared_writer:
        writer = GenerationPrefixWriteBatcher(
            sink, max_rows=256, max_tokens=4_194_304, wait_seconds=0
        )
        worker._generation_prefix_write_batcher = writer
    try:
        acks = worker._checkpoint_generation_cut_batched(inventory)
        assert puts == [3, 1, 1, 1]
        assert all(ack.disposition == "durable_prefix" for ack in acks)
        assert len(sink.records) == 3
        assert_group_released(states)
    finally:
        if writer is not None:
            writer.close()


def test_group_snapshot_does_not_submit_empty_or_expired_lookups(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, _, _, _ = make_group_worker(m, owner_loop)

    def unexpected(*args, **kwargs):
        pytest.fail("must not submit an empty/expired handoff")

    monkeypatch.setattr(m, "run_engine_snapshot", unexpected)
    client = worker._http_engine_client
    assert client.snapshot_routed_experts_many([], timeout_s=5) == {}
    with pytest.raises(TimeoutError):
        client.snapshot_routed_experts_many(["engine-0"], timeout_s=0)


def test_group_consecutive_cuts_restore_each_calls_exact_prefix(
    grouped_implementation, owner_loop
):
    m = grouped_implementation
    worker, states, processor, inventory = make_group_worker(m, owner_loop)
    first = worker._checkpoint_generation_cut_batched(inventory)
    for i, state in enumerate(states):
        state.observe([69], [-0.3])
        processor.request_states[f"internal-{i}"].routed_experts_chunks.append(
            routes(4)
        )
    second_inventory = GenerationCutInventory.build(
        checkpoint_id="second-group-cut",
        server_name="policy",
        active_prefixes=inventory.active_prefixes,
    )
    second = worker._checkpoint_generation_cut_batched(second_inventory)
    for state, before, after in zip(states, first, second, strict=True):
        assert after.staging_keys[:1] == before.staging_keys
        worker.state = state
        _, resolved = restore(worker, after, chunks(m, worker._capture_sink, after))
        assert resolved.routed_experts.flatten().tolist() == [1, 2, 3, 4, -1]
    assert_group_released(states)


def test_group_probe_counts_handoffs_separately_from_requests(
    grouped_implementation, owner_loop, caplog
):
    m = grouped_implementation
    worker, _, _, inventory = make_group_worker(m, owner_loop, count=65)
    caplog.set_level("INFO")
    with prefix_cut_probe(
        enabled=True,
        checkpoint_id=inventory.checkpoint_id,
        inventory_digest=inventory.inventory_digest,
        calls=65,
        deadline=None,
    ):
        worker._checkpoint_generation_cut_batched(inventory)
    messages = [
        r.message for r in caplog.records if r.message.startswith("prefix_cut_probe ")
    ]
    assert len(messages) == 1
    result = json.loads(messages[0].split(" ", 1)[1])
    assert result["counts"]["engine_snapshot_groups"] == 2
    assert result["counts"]["engine_snapshot_requests"] == 65
    assert result["counts"]["max_engine_snapshot_group_calls"] == 64
    for phase in [
        "engine_snapshot_wait",
        "engine_snapshot_dispatch_wait",
        "engine_snapshot_callback",
    ]:
        assert result["phases"][phase]["calls"] == 2
    assert result["phases"]["transaction_route_resume"]["calls"] == 65


def test_replay_off_preserves_existing_1000_call_transport_batches(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, states, _, inventory = make_group_worker(m, owner_loop, count=1000)
    for state in states:
        state.capture_routed_experts = False
    puts = []
    original_put = worker._capture_sink.stage_generation_prefix_batch

    def put(records, **kwargs):
        puts.append(len(records))
        return original_put(records, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("replay-off path must not fetch routes")

    monkeypatch.setattr(m, "run_engine_snapshot", unexpected)
    worker._capture_sink.stage_generation_prefix_batch = put
    acks = worker._checkpoint_generation_cut_batched(inventory)
    assert puts == [256, 256, 256, 232]
    assert all(ack.disposition == "durable_prefix" for ack in acks)
    assert_group_released(states)


def verified_direct_client(m, worker, processor, monkeypatch):
    # vLLM's actual external-ID values are lists, not the sets used by the
    # earlier owner-loop fixtures. Verify the production setup gate too.
    processor.external_req_ids = {
        key: list(value) for key, value in processor.external_req_ids.items()
    }
    monkeypatch.setattr(
        m,
        "importlib",
        SimpleNamespace(metadata=SimpleNamespace(version=lambda name: "0.29.0")),
    )
    client = worker._http_engine_client
    client.check_route_snapshot_support()
    assert client._direct_route_snapshot_enabled()
    return client


def test_direct_group_cuts_and_restores_while_engine_loop_is_blocked(
    grouped_implementation, owner_loop, monkeypatch, caplog
):
    m = grouped_implementation
    worker, states, processor, inventory = make_group_worker(m, owner_loop)
    verified_direct_client(m, worker, processor, monkeypatch)
    entered, release = threading.Event(), threading.Event()

    def block_owner():
        entered.set()
        release.wait(5)

    def unexpected(*args, **kwargs):
        pytest.fail("verified reference copy must not schedule an owner callback")

    monkeypatch.setattr(m, "run_engine_snapshot", unexpected)
    owner_loop.call_soon_threadsafe(block_owner)
    assert entered.wait(2)
    try:
        caplog.set_level("INFO")
        with prefix_cut_probe(
            enabled=True,
            checkpoint_id="group-cut",
            inventory_digest="direct",
            calls=3,
            deadline=None,
        ):
            acks = worker._checkpoint_generation_cut_batched(inventory)
        assert all(ack.disposition == "durable_prefix" for ack in acks)
        assert not release.is_set()
        for state, ack in zip(states, acks, strict=True):
            worker.state = state
            restore(worker, ack, chunks(m, worker._capture_sink, ack))
        summary = next(
            json.loads(record.message.split("prefix_cut_probe ", 1)[1])
            for record in caplog.records
            if "prefix_cut_probe " in record.message
        )
        assert summary["counts"]["engine_snapshot_direct_reads"] == 1
        assert summary["phases"]["engine_snapshot_direct_read"]["calls"] == 1
        assert "engine_snapshot_dispatch_wait" not in summary["phases"]
        assert_group_released(states)
    finally:
        release.set()


def test_direct_snapshot_is_a_stable_prefix_during_append_and_removal(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, _, processor, _ = make_group_worker(m, owner_loop, count=1)
    client = verified_direct_client(m, worker, processor, monkeypatch)
    published = [routes(i) for i in range(128)]
    state = processor.request_states["internal-0"]
    state.routed_experts_chunks = published[:2]
    captured = [client.snapshot_routed_experts("engine-0", timeout_s=5)]
    start, done = threading.Event(), threading.Event()
    writer_errors = []

    def publish():
        try:
            assert start.wait(2)
            for piece in published[2:]:
                state.routed_experts_chunks.append(piece)
                done.wait(0.0005)
            processor.request_states.pop("internal-0")
            processor.external_req_ids.pop("engine-0")
        except BaseException as error:
            writer_errors.append(error)
        finally:
            done.set()

    writer = threading.Thread(target=publish)
    writer.start()
    start.set()
    try:
        for _ in range(200):
            snapshot = client.snapshot_routed_experts("engine-0", timeout_s=5)
            if snapshot is not None:
                assert all(piece is published[i] for i, piece in enumerate(snapshot))
                if len(captured) < 16:
                    captured.append(snapshot)
            if done.wait(0.0005):
                break
    finally:
        writer.join(timeout=3)
    assert not writer.is_alive() and not writer_errors
    assert client.snapshot_routed_experts("engine-0", timeout_s=5) is None
    assert captured[0] == published[:2]
    assert all(snapshot is not state.routed_experts_chunks for snapshot in captured)


@pytest.mark.parametrize(
    "failure",
    ["missing_state", "missing_ids", "multiple_ids", "missing_chunks", "custom_chunks"],
)
def test_direct_group_preserves_individual_lookup_failures(
    grouped_implementation, owner_loop, monkeypatch, failure
):
    m = grouped_implementation
    worker, _, processor, _ = make_group_worker(m, owner_loop)
    client = verified_direct_client(m, worker, processor, monkeypatch)
    if failure == "missing_state":
        processor.request_states.pop("internal-1")
    elif failure == "missing_ids":
        processor.external_req_ids.pop("engine-1")
    elif failure == "multiple_ids":
        processor.external_req_ids["engine-1"].append("another")
    elif failure == "missing_chunks":
        processor.request_states["internal-1"].routed_experts_chunks = None
    else:
        processor.request_states["internal-1"].routed_experts_chunks = iter([routes(1)])
    result = client.snapshot_routed_experts_many(
        ["engine-0", "engine-1", "engine-2"], timeout_s=5
    )
    assert result["engine-0"].routes and result["engine-2"].routes
    if failure.startswith("missing_") and failure != "missing_chunks":
        assert result["engine-1"].routes is None and result["engine-1"].error is None
    else:
        assert isinstance(result["engine-1"].error, ValueError)


@pytest.mark.parametrize(
    "guard",
    [
        "disabled",
        "unverified",
        "other_python",
        "no_gil",
        "unknown_gil",
        "custom_mapping",
    ],
)
def test_direct_route_guards_keep_the_owner_loop_fallback(
    grouped_implementation, owner_loop, monkeypatch, guard
):
    m = grouped_implementation
    worker, _, processor, _ = make_group_worker(m, owner_loop)
    client = verified_direct_client(m, worker, processor, monkeypatch)
    if guard == "disabled":
        client = m._AsyncLLMHTTPClient(
            client._engine_client,
            owner_loop,
            disable_direct_route_snapshot=True,
            flat_logprobs=False,
        )
        client.check_route_snapshot_support()
    elif guard == "unverified":
        client._route_snapshot_state_verified = False
    elif guard == "other_python":
        monkeypatch.setattr(
            m,
            "sys",
            SimpleNamespace(
                implementation=SimpleNamespace(name="pypy"),
                _is_gil_enabled=lambda: True,
            ),
        )
    elif guard == "no_gil":
        monkeypatch.setattr(
            m,
            "sys",
            SimpleNamespace(
                implementation=SimpleNamespace(name="cpython"),
                _is_gil_enabled=lambda: False,
            ),
        )
    elif guard == "unknown_gil":
        monkeypatch.setattr(
            m, "sys", SimpleNamespace(implementation=SimpleNamespace(name="cpython"))
        )
    else:
        processor.request_states = type("CustomRequests", (dict,), {})(
            processor.request_states
        )
    assert not client._direct_route_snapshot_enabled()
    readers = []
    original = client._routed_expert_chunks

    def read(request_id):
        readers.append(threading.get_ident())
        return original(request_id)

    monkeypatch.setattr(client, "_routed_expert_chunks", read)
    result = client.snapshot_routed_experts_many(["engine-0", "engine-1"], timeout_s=5)
    assert result["engine-0"].routes and result["engine-1"].routes
    assert len(readers) == 2 and all(
        reader != threading.get_ident() for reader in readers
    )


def test_direct_lookup_keeps_version_and_deadline_checks(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, _, processor, _ = make_group_worker(m, owner_loop)
    client = worker._http_engine_client
    monkeypatch.setattr(
        m,
        "importlib",
        SimpleNamespace(metadata=SimpleNamespace(version=lambda name: "0.30.0")),
    )
    with pytest.raises(RuntimeError, match="verified on vLLM"):
        client.check_route_snapshot_support()
    assert not client._route_snapshot_state_verified
    verified_direct_client(m, worker, processor, monkeypatch)
    assert client.snapshot_routed_experts_many([], timeout_s=0) == {}
    with pytest.raises(TimeoutError):
        client.snapshot_routed_experts_many(["engine-0"], timeout_s=0)
    with pytest.raises(TimeoutError):
        client.snapshot_routed_experts("engine-0", timeout_s=-1)


@pytest.mark.parametrize("owner_read", [False, True], ids=["direct", "owner-loop"])
@pytest.mark.parametrize("finish_at", ["before_cut", "after_freeze", "after_lookup"])
def test_completed_routes_bridge_http_handoff_and_restore(
    grouped_implementation, owner_loop, monkeypatch, caplog, owner_read, finish_at
):
    """Run real adapter, observation, cut and restore across engine removal."""
    m = grouped_implementation
    worker, states, processor, inventory = make_group_worker(m, owner_loop, count=1)
    client = verified_direct_client(m, worker, processor, monkeypatch)
    client._disable_direct_route_snapshot = owner_read
    engine_state = processor.request_states["internal-0"]
    pieces = engine_state.routed_experts_chunks
    request = object()
    worker._capture_calls = {id(request): states[0]}
    terminal_routes = torch.from_numpy(np.concatenate(pieces))
    output = SimpleNamespace(
        request_id="engine-0",
        outputs=[
            SimpleNamespace(
                token_ids=[67, 68],
                logprobs=[
                    {67: SimpleNamespace(logprob=-0.1)},
                    {68: SimpleNamespace(logprob=-0.2)},
                ],
                finish_reason="length",
                stop_reason=None,
                routed_experts=terminal_routes,
            )
        ],
    )

    async def generate(*args, **kwargs):
        yield "progress"
        yield output

    async def finish():
        processor._finish_request(engine_state)

    client._engine_client.generate = generate

    async def consume():
        stream = client.generate(None, None, "engine-0")
        assert await anext(stream) == "progress"
        assert client._route_retention_requests == {"engine-0"}
        if finish_at == "before_cut":
            await client._run_on_engine_loop(finish)
        elif finish_at == "after_freeze":
            original = client._routed_expert_chunks

            def read(request_id):
                if request_id in processor.external_req_ids:
                    if asyncio.get_running_loop() is owner_loop:
                        processor._finish_request(engine_state)
                    else:
                        asyncio.run_coroutine_threadsafe(finish(), owner_loop).result(5)
                return original(request_id)

            monkeypatch.setattr(client, "_routed_expert_chunks", read)
        else:
            original_many = client.snapshot_routed_experts_many

            def read_many(*args, **kwargs):
                result = original_many(*args, **kwargs)
                asyncio.run_coroutine_threadsafe(finish(), owner_loop).result(5)
                return result

            monkeypatch.setattr(client, "snapshot_routed_experts_many", read_many)

        assert states[0].terminal_routed_experts is None
        caplog.set_level("INFO")
        with prefix_cut_probe(
            enabled=True,
            checkpoint_id="group-cut",
            inventory_digest="handoff",
            calls=1,
            deadline=None,
        ):
            ack = worker._checkpoint_generation_cut_batched(inventory)[0]
        assert ack.disposition == "durable_prefix"
        assert not processor.request_states and not processor.external_req_ids
        retained = client._completed_route_pieces["engine-0"]["internal-0"]
        assert retained is pieces  # No route tensor copy on engine completion.
        restore(worker, ack, chunks(m, worker._capture_sink, ack))
        assert states[0].terminal_routed_experts is None
        assert await anext(stream) is output
        m._observe_request_capture(worker, request, output)
        assert states[0].observation_error is None
        assert torch.equal(states[0].terminal_routed_experts, terminal_routes)
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        assert not client._completed_route_pieces
        assert not client._route_retention_requests
        assert client._routed_expert_chunks("engine-0") is None
        # Capture now owns the routes; no engine lookup or retained list needed.
        retry = worker._checkpoint_generation_cut_batched(inventory)[0]
        assert retry.disposition == "durable_prefix"
        assert_group_released(states)

    asyncio.run(consume())
    summary = next(
        json.loads(record.message.split("prefix_cut_probe ", 1)[1])
        for record in caplog.records
        if "prefix_cut_probe " in record.message
    )
    assert summary["counts"].get("engine_snapshot_retained_reads", 0) == (
        0 if finish_at == "after_lookup" else 1
    )


@pytest.mark.parametrize("abort_error", [False, True])
@pytest.mark.parametrize("cancel", [False, True], ids=["close", "cancel"])
@pytest.mark.parametrize("direct_tokens", [False, True])
def test_completed_route_retention_cleans_up_aborted_http_consumer(
    grouped_implementation, owner_loop, monkeypatch, abort_error, cancel, direct_tokens
):
    m = grouped_implementation
    worker, _, processor, _ = (
        enable_engine_tokens(m, owner_loop, monkeypatch)
        if direct_tokens
        else make_group_worker(m, owner_loop, count=1)
    )
    client = verified_direct_client(m, worker, processor, monkeypatch)
    state = processor.request_states["internal-0"]
    aborts = []

    async def generate(*args, **kwargs):
        yield "progress"
        await asyncio.Event().wait()

    async def abort(request_id):
        aborts.append(request_id)
        if abort_error:
            raise RuntimeError("injected abort failure")

    async def finish():
        processor._finish_request(state)

    client._engine_client.generate = generate
    client._engine_client.abort = abort

    async def consume():
        stream = client.generate(None, None, "engine-0")
        assert await anext(stream) == "progress"
        await client._run_on_engine_loop(finish)
        assert client._completed_route_pieces
        if cancel:
            task = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await stream.aclose()
        assert not client._completed_route_pieces
        assert not client._route_retention_requests
        assert not client._completed_token_states

    asyncio.run(consume())
    assert aborts == ["engine-0"]


def test_route_retention_hook_is_scoped_idempotent_and_released_on_shutdown(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, _, processor, _ = make_group_worker(m, owner_loop, count=1)
    original = processor._finish_request
    client = verified_direct_client(m, worker, processor, monkeypatch)
    wrapper = processor._finish_request
    client.check_route_snapshot_support()
    assert processor._finish_request is wrapper
    # A non-adapter request is removed normally, without retaining its arrays.
    processor._finish_request(processor.request_states["internal-0"])
    assert not client._completed_route_pieces
    assert not processor.request_states
    client._route_retention_requests.add("active")
    client._completed_route_pieces["active"] = {"internal": [routes(1)]}
    held = client._completed_route_pieces["active"]["internal"][:]
    client.close_route_snapshots()
    client.close_route_snapshots()
    assert processor._finish_request is original
    assert not client._route_retention_requests and not client._completed_route_pieces
    assert not client._route_snapshot_state_verified
    assert held[0].tolist() == [[[1]]]  # Existing readers keep valid references.


def test_retained_cut_does_not_wait_for_engine_completion_callback(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, states, processor, inventory = make_group_worker(m, owner_loop, count=1)
    removed, release = threading.Event(), threading.Event()
    original = processor._finish_request

    def finish(state):
        original(state)
        removed.set()
        assert release.wait(5)

    processor._finish_request = finish
    client = verified_direct_client(m, worker, processor, monkeypatch)
    client._route_retention_requests.add("engine-0")
    state = processor.request_states["internal-0"]

    async def complete():
        processor._finish_request(state)

    completion = asyncio.run_coroutine_threadsafe(complete(), owner_loop)
    try:
        assert removed.wait(2)
        assert not processor.request_states
        ack = worker._checkpoint_generation_cut_batched(inventory)[0]
        assert ack.disposition == "durable_prefix"
        assert not completion.done() and not release.is_set()
        restore(worker, ack, chunks(m, worker._capture_sink, ack))
        assert_group_released(states)
    finally:
        release.set()
        completion.result(5)
        client.close_route_snapshots()


def test_route_retention_releases_the_actual_array_references(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, _, processor, _ = make_group_worker(m, owner_loop, count=1)
    client = verified_direct_client(m, worker, processor, monkeypatch)
    client._route_retention_requests.add("engine-0")
    state = processor.request_states["internal-0"]
    array = weakref.ref(state.routed_experts_chunks[0])
    processor._finish_request(state)
    del state
    assert array() is not None
    client.close_route_snapshots()
    assert array() is None


def test_retention_does_not_hide_multiple_internal_requests(
    grouped_implementation, owner_loop, monkeypatch
):
    m = grouped_implementation
    worker, _, processor, _ = make_group_worker(m, owner_loop, count=1)
    client = verified_direct_client(m, worker, processor, monkeypatch)
    client._route_retention_requests.add("engine-0")
    processor.external_req_ids["engine-0"].append("other-internal")
    processor.request_states["other-internal"] = SimpleNamespace(
        request_id="other-internal",
        external_req_id="engine-0",
        routed_experts_chunks=[routes(7)],
    )
    processor._finish_request(processor.request_states["internal-0"])
    result = client.snapshot_routed_experts_many(["engine-0"], timeout_s=5)
    assert isinstance(result["engine-0"].error, ValueError)
    assert "found 2" in str(result["engine-0"].error)
    client.close_route_snapshots()


def test_shared_inventory_writer_preserves_two_cut_restore_integrity(
    implementation: Any,
) -> None:
    class SharedSink(Sink):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes = []

        def stage_generation_prefix_batch(
            self, records: Any, **kwargs: Any
        ) -> list[str]:
            self.batch_sizes.append(len(records))
            return super().stage_generation_prefix_batch(records, **kwargs)

    sink = SharedSink()
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=4_194_304, wait_seconds=0.1
    )
    workers = [
        make_worker(
            implementation,
            sink=sink,
            admission=CaptureAdmission(
                rollout_id=f"r{index}", model_call_id=f"c{index}", mode="text"
            ),
        )
        for index in range(2)
    ]
    for worker in workers:
        worker._generation_prefix_write_batcher = writer
        worker.state.observe([67, 68], [-0.1, -0.2])
        worker.pieces = [routes(1, 2), routes(3)]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for checkpoint in ["cut1", "cut2"]:
                start = threading.Barrier(2)

                def run(worker: Any) -> Any:
                    start.wait()
                    return cut(worker, checkpoint)

                acks = list(pool.map(run, workers))
                assert all(ack.disposition == "durable_prefix" for ack in acks)
                if checkpoint == "cut1":
                    for worker in workers:
                        worker.state.observe([69], [-0.3])
                        worker.pieces.append(routes(4))
        assert sink.batch_sizes == [2, 2]
        for worker, ack in zip(workers, acks, strict=True):
            assert len(ack.staging_keys) == 2
            _, resolved = restore(worker, ack, chunks(implementation, sink, ack))
            assert resolved.routed_experts.flatten().tolist() == [1, 2, 3, 4, -1]
            assert len(resolved.chunk_digests) == 2
    finally:
        writer.close()


def test_shared_put_failure_does_not_advance_unrelated_call_digests(
    implementation: Any,
) -> None:
    class RejectOneSink(Sink):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes = []

        def stage_generation_prefix_batch(
            self, records: Any, **kwargs: Any
        ) -> list[str]:
            self.batch_sizes.append(len(records))
            if any(record.model_call_id == "bad" for record in records):
                raise RuntimeError("injected bad record")
            return super().stage_generation_prefix_batch(records, **kwargs)

    sink = RejectOneSink()
    writer = GenerationPrefixWriteBatcher(
        sink, max_rows=256, max_tokens=4_194_304, wait_seconds=0.1
    )
    workers = [
        make_worker(
            implementation,
            sink=sink,
            admission=CaptureAdmission(
                rollout_id=f"r{index}", model_call_id=name, mode="text"
            ),
        )
        for index, name in enumerate(["good", "bad"])
    ]
    for worker in workers:
        worker._generation_prefix_write_batcher = writer
        worker.state.observe([67, 68], [-0.1, -0.2])
        worker.pieces = [routes(1, 2), routes(3)]
    start = threading.Barrier(2)

    def run(worker: Any) -> Any:
        start.wait()
        return cut(worker, "cut1")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            good, bad = pool.map(run, workers)
        assert good.disposition == "durable_prefix"
        assert bad.disposition == "durable_failure"
        assert sink.batch_sizes == [2, 1, 1]
        assert len(workers[0].state.generation_cut_chunk_digests) == 1
        assert workers[1].state.generation_cut_chunk_digests == []
        assert workers[1].state.generation_cut_staging_keys == []
        assert workers[1].state.frozen_cut is None
        _, resolved = restore(workers[0], good, chunks(implementation, sink, good))
        assert resolved.routed_experts.flatten().tolist() == [1, 2, 3, -1]
    finally:
        writer.close()
