"""CPU regressions for local route attachments with real Gym integrity checks.

Load the production sink and terminal helpers without importing Ray/vLLM's
GPU dependency graph. Only transport is a recording client; tensor encoding,
capture records, digests, and sink/source conversion use production code.
"""

import ast
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging.capture")

from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter  # noqa: E402
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture  # noqa: E402
from nemo_gym.token_id_capture.staging.records import CaptureAdmission  # noqa: E402

from nemo_rl.utils.prefix_cut_diagnostics import prefix_cut_probe  # noqa: E402
from nemo_rl.utils.routed_experts_codec import (  # noqa: E402
    ROUTED_EXPERTS_STAGING_ATTACHMENT,
    RoutedExpertsStagingPayload,
    decode_routed_experts,
    encode_routed_experts,
)

pytestmark = pytest.mark.nemo_gym


@pytest.fixture(scope="module")
def implementation():
    root = Path(__file__).resolve().parents[3] / "nemo_rl"
    module = ModuleType("_route_tensor_reuse_test")
    sys.modules[module.__name__] = module
    exec(
        compile((root / "data_plane/schema.py").read_text(), "schema.py", "exec"),
        module.__dict__,
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    route_tree = ast.parse((root / "experience/route_assembly.py").read_text())
    codec_tree = ast.parse((root / "data_plane/codec.py").read_text())
    helpers = [
        node
        for node in route_tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name
        in {"RouteFragment", "boundary_route_rows", "verify_route_fragment_integrity"}
    ] + [
        node
        for node in codec_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "stack_or_nest"
    ]
    import dataclasses

    module.__dict__.update(torch=torch, dataclass=dataclasses.dataclass, json=json)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *helpers], type_ignores=[])
            ),
            "helpers.py",
            "exec",
        ),
        module.__dict__,
    )
    path = root / "data_plane/tq_token_sink.py"
    tree = ast.parse(path.read_text())
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Import)
            and any(alias.name == "ray" for alias in node.names)
            or isinstance(node, ast.ImportFrom)
            and node.module
            in {
                "nemo_rl.data_plane.codec",
                "nemo_rl.data_plane.schema",
                "nemo_rl.experience.route_assembly",
            }
        )
    ]
    exec(compile(tree, str(path), "exec"), module.__dict__)
    path = root / "models/generation/vllm/vllm_worker_async.py"
    tree = ast.parse(path.read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "VllmAsyncGenerationWorkerImpl"
    )
    worker.bases = []
    worker.decorator_list = []
    worker.body = [
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_delta_align_routed_experts",
            "_merge_generation_cut_routed_experts",
            "_finish_request_capture_with_lifecycle_owned",
        }
    ]
    # A separate namespace lets the sink's decode trap distinguish the one
    # unavoidable response decode from a redundant decode at staging.
    worker_ns = {**module.__dict__, "LOGGER": module.logging.getLogger(__name__)}
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, worker], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        worker_ns,
    )
    module.Worker = worker_ns["VllmAsyncGenerationWorkerImpl"]
    yield module
    sys.modules.pop(module.__name__, None)


class Client:
    def __init__(self):
        self.puts = []
        self.clears = []
        self.fail = False

    def put_samples(self, **kwargs):
        if self.fail:
            raise RuntimeError("injected put failure")
        self.puts.append(kwargs)

    def clear_samples(self, **kwargs):
        self.clears.extend(kwargs["sample_ids"])


def record_and_attachment(
    sink, *, dtype=torch.int16, call_id="call", route_length=4, routed=True
):
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=lambda: 7)
    call = capture.begin_call(
        CaptureAdmission(rollout_id="rollout", model_call_id=call_id, mode="text")
    )
    routes = torch.arange(route_length * 4).reshape(route_length, 2, 2).to(dtype)
    payload = RoutedExpertsStagingPayload(routes)
    record = capture.build_prefix_record(
        call,
        prompt_token_ids=[1, 2],
        generated_token_ids=[3, 4],
        generated_logprobs=[-0.25, -0.5],
        extras={"routed_experts": payload.encoded} if routed else None,
    )
    return record, payload


def unexpected_decode(*args, **kwargs):
    pytest.fail("the sink must reuse the route tensor, not decode its envelope")


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("media", [False, True])
def test_route_attachment_is_byte_exact_and_preserves_media(
    implementation, monkeypatch, caplog, dtype, media
):
    m = implementation
    sink = m.TQTokenSink(
        Client(),
        staging_partition="test",
        capture_media=media,
        media_pixel_dtype=torch.bfloat16 if media else None,
    )
    record, payload = record_and_attachment(sink, dtype=dtype)
    attachments = (
        {
            "imgs": torch.zeros(1, 1, 12, dtype=torch.bfloat16),
            "imgs_sizes": torch.tensor([[2, 2]], dtype=torch.int32),
        }
        if media
        else {}
    )
    expected, expected_tags = sink._encode_record(
        record, attachments=attachments or None
    )
    monkeypatch.setattr(m, "decode_routed_experts", unexpected_decode)
    caplog.set_level("INFO")
    with prefix_cut_probe(
        enabled=True,
        checkpoint_id="cp",
        inventory_digest="test",
        calls=1,
        deadline=None,
    ):
        actual, tags = sink._encode_record(
            record,
            attachments={**attachments, ROUTED_EXPERTS_STAGING_ATTACHMENT: payload},
        )
    assert tags == expected_tags
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in actual)
    assert m._row_to_base_snapshot(actual).digest == record.digest
    fragment = m._row_to_route_fragment(actual)
    assert m.verify_route_fragment_integrity(
        fragment,
        extras_digest_version=record.extras_digest_version,
        expected_extras_digest=record.extras_digest,
    )
    summary = json.loads(
        next(
            item.message.split(" ", 1)[1]
            for item in caplog.records
            if item.message.startswith("prefix_cut_probe ")
        )
    )
    assert summary["counts"]["sink_route_reuse_rows"] == 1
    assert "sink_route_decode" not in summary["phases"]


@pytest.mark.parametrize("failure", ["mismatch", "mutated", "missing", "shape", "type"])
def test_bad_attachment_fails_before_any_put(implementation, failure):
    m = implementation
    client = Client()
    sink = m.TQTokenSink(client, staging_partition="test")
    record, payload = record_and_attachment(
        sink, route_length=3 if failure == "shape" else 4
    )
    if failure == "mismatch":
        payload = RoutedExpertsStagingPayload(torch.ones(4, 2, 2, dtype=torch.int16))
    elif failure == "mutated":
        payload.tensor_for_payload(payload.encoded).add_(1)
    elif failure == "missing":
        record = record.model_copy(update={"extras": None})
    elif failure == "type":
        payload = torch.zeros(4, 2, 2)
    result = sink.stage(
        record, attachments={ROUTED_EXPERTS_STAGING_ATTACHMENT: payload}
    )
    assert not result.ok
    assert not client.puts


def test_mixed_prefix_batch_reuses_routes_and_retry_keeps_attachment(
    implementation, monkeypatch
):
    m = implementation
    client = Client()
    sink = m.TQTokenSink(client, staging_partition="test")
    first, payload = record_and_attachment(sink)
    second, _ = record_and_attachment(sink, call_id="plain", routed=False)
    # A route-free row keeps the legacy path and is written in its own schema group.
    attachments = [{ROUTED_EXPERTS_STAGING_ATTACHMENT: payload}, None]
    monkeypatch.setattr(m, "decode_routed_experts", unexpected_decode)
    client.fail = True
    with pytest.raises(RuntimeError, match="injected put failure"):
        sink.stage_generation_prefix_batch(
            [first, second],
            checkpoint_id="cp",
            chunk_sequences=[0, 0],
            attachments=attachments,
        )
    assert len(client.clears) == 2
    client.fail = False
    keys = sink.stage_generation_prefix_batch(
        [first, second],
        checkpoint_id="cp",
        chunk_sequences=[0, 0],
        attachments=attachments,
    )
    assert len(keys) == 2
    assert len(client.puts) == 2
    routes = client.puts[0]["fields"]["routed_experts"]
    assert torch.equal(routes[0], decode_routed_experts(first.extras["routed_experts"]))


def test_terminal_capture_passes_attachment_through_real_gym(
    implementation, monkeypatch
):
    m = implementation
    client = Client()
    sink = m.TQTokenSink(client, staging_partition="test")
    capture = RolloutTokenCapture(
        sink=sink, weight_version_fn=lambda: 7, adapter=VLLMCaptureAdapter()
    )
    call = capture.begin_call(
        CaptureAdmission(rollout_id="rollout", model_call_id="terminal", mode="text")
    )
    state = SimpleNamespace(
        call=call,
        prompt_token_ids=[1, 2],
        media=None,
        restored_routed_experts=None,
        lock=threading.Lock(),
        effective_output_limit=8,
        terminal_finish_reason="stop",
        terminal_stop_reason=None,
        generation_cut_staging_keys=[],
    )
    worker = m.Worker()
    worker.token_capture = capture
    worker._remember_completed_capture = lambda *args, **kwargs: None
    worker._pop_request_capture = lambda request: None
    content = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "generation_token_ids": [3, 4],
                    "generation_log_probs": [-0.25, -0.5],
                    "routed_experts": encode_routed_experts(
                        torch.zeros(4, 2, 2, dtype=torch.int16)
                    ),
                },
            }
        ]
    }
    monkeypatch.setattr(m, "decode_routed_experts", unexpected_decode)
    result, obsolete = worker._finish_request_capture_with_lifecycle_owned(
        state, object(), content
    )
    assert result["ng_commit_coords"]["disposition"] == "staged"
    assert not obsolete
    assert len(client.puts) == 1
    assert "routed_experts" not in result["choices"][0]["message"]


def test_restored_terminal_routes_keep_saved_prefix_and_current_boundary(
    implementation,
):
    worker = implementation.Worker()
    current = torch.arange(1, 7, dtype=torch.int16).reshape(6, 1, 1)
    content = {
        "choices": [{"message": {"routed_experts": encode_routed_experts(current)}}]
    }
    aligned = worker._delta_align_routed_experts(
        content, prev_len=1, prompt_len=4, generated_len=2
    )
    restored = torch.tensor([10, 11, -1], dtype=torch.int16).reshape(3, 1, 1)
    merged, payload = worker._merge_generation_cut_routed_experts(
        content, restored=restored, prev_len=1, generated_len=2, route_payload=aligned
    )
    assert merged.flatten().tolist() == [10, 11, 4]
    assert payload.tensor_for_payload(
        content["choices"][0]["message"]["routed_experts"]
    ).flatten().tolist() == [1, 10, 11, 4, 5, 6]
    # Merging must not mutate an attachment that could still be referenced.
    assert aligned.tensor_for_payload(aligned.encoded).flatten().tolist() == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_concurrent_terminal_attachments_are_request_local(implementation):
    m = implementation
    sink = m.TQTokenSink(Client(), staging_partition="test")

    def encode(index):
        record, payload = record_and_attachment(sink, call_id=str(index))
        fields, _ = sink._encode_record(
            record, attachments={ROUTED_EXPERTS_STAGING_ATTACHMENT: payload}
        )
        return m._row_to_base_snapshot(fields).model_call_id

    with ThreadPoolExecutor(max_workers=4) as executor:
        assert list(executor.map(encode, range(16))) == [
            str(index) for index in range(16)
        ]
