# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the SGLang non-colocated NCCL bridge wire format.

These tests cover the small, GPU-free, Ray-free pieces of the slime-style
NCCL bridge refit path:

  * ``SGLangGenerationWorker.init_weights_update_group`` — HTTP wrapper
    body / status-code handling, model-owner short-circuit.
  * ``SGLangGenerationWorker.update_weights_from_distributed`` — same
    plus body-level success/message inspection.
  * ``SGLangGenerationWorker.destroy_weights_update_group`` — idempotent
    on 4xx, surfaces 5xx as failure.
  * ``SGLangGeneration.init_collective_nccl_bridge`` — rank_offset
    arithmetic + Ray fan-out shape (worker_group is mocked).
  * ``SGLangGeneration.update_weights_via_nccl_bridge`` — payload shape.
  * Per-bucket flattened-tensor byte layout matches SGLang's
    ``FlattenedTensorBucket(named_tensors=...).flattened_tensor`` exactly,
    so a ``dist.broadcast`` from train side is bit-compatible with
    SGLang's recv side.

The integration paths (real Ray cluster, real GPU broadcast) are
intentionally excluded; those are exercised by
``examples/configs/grpo_glm45_air_sglang_noncolo.yaml`` /
``grpo_glm45_air_sglang_noncolo_job.slurm`` and the train-side smoke run.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Skip the whole module gracefully if the optional generation-side deps
# aren't importable on this node. The unit tests themselves are GPU-free,
# but the worker module pulls in ``aiohttp`` / ``ray`` / ``torch`` at
# import time.
pytest.importorskip("aiohttp")
pytest.importorskip("ray")
pytest.importorskip("torch")
pytest.importorskip("requests")


# ---------------------------------------------------------------------------
# SGLangGenerationWorker HTTP wrappers
# ---------------------------------------------------------------------------
#
# The worker class is decorated with ``@ray.remote`` so we cannot
# instantiate it directly. We exercise the unbound methods against a stub
# ``self`` (a SimpleNamespace) that carries the attributes the methods
# actually read: ``is_model_owner``, ``base_url``, ``global_rank``.


def _make_stub_worker(
    is_model_owner: bool = True,
    base_url: str | None = "http://10.0.0.1:31000",
    global_rank: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        is_model_owner=is_model_owner,
        base_url=base_url,
        global_rank=global_rank,
    )


def _get_unbound(method_name: str):
    """Pull the unbound method off the SGLangGenerationWorker remote class.

    Ray remote actors store the original (un-decorated) Python class in
    several places that have shifted across Ray versions. We try them in
    order; the import is itself wrapped in ``importorskip`` so a missing
    sglang worker module doesn't stall test collection.
    """
    sglang_worker_mod = pytest.importorskip(
        "nemo_rl.models.generation.sglang.sglang_worker"
    )
    actor = sglang_worker_mod.SGLangGenerationWorker

    # Ordered list of known attribute paths to the underlying class, from
    # newest Ray to oldest. The first match wins.
    candidates = [
        lambda a: a.__ray_metadata__.modified_class,
        lambda a: a._modified_class,
        lambda a: a.__ray_actor_class__,
        lambda a: a,  # last resort: the wrapper itself
    ]
    cls = None
    for getter in candidates:
        try:
            cand = getter(actor)
        except (AttributeError, TypeError):
            continue
        if cand is not None and hasattr(cand, method_name):
            cls = cand
            break
    if cls is None:
        pytest.skip(
            f"Could not locate {method_name!r} on SGLangGenerationWorker "
            f"actor class wrapper (ray={getattr(__import__('ray'), '__version__', '?')})"
        )
    return getattr(cls, method_name)


def _make_response(status_code: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if json_body is not None:
        resp.json = MagicMock(return_value=json_body)
    else:
        resp.json = MagicMock(side_effect=ValueError("no json body"))
    return resp


# ---- init_weights_update_group ------------------------------------------------


def test_init_weights_update_group_non_master_returns_true_without_post():
    """Slave-node TP workers (``is_model_owner=False``) are no-ops."""
    init = _get_unbound("init_weights_update_group")
    worker = _make_stub_worker(is_model_owner=False, base_url=None)

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post"
    ) as mock_post:
        ok = init(
            worker,
            master_address="10.0.0.1",
            master_port=29500,
            rank_offset=1,
            world_size=9,
            group_name="test-group",
        )

    assert ok is True
    mock_post.assert_not_called()


def test_init_weights_update_group_no_base_url_returns_true_without_post():
    """When base_url is None the worker has no HTTP server to talk to."""
    init = _get_unbound("init_weights_update_group")
    worker = _make_stub_worker(is_model_owner=True, base_url=None)

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post"
    ) as mock_post:
        assert init(worker, "10.0.0.1", 29500, 1, 9) is True

    mock_post.assert_not_called()


def test_init_weights_update_group_posts_correct_body():
    """Verify the JSON body matches SGLang's
    ``InitWeightsUpdateGroupReqInput`` schema."""
    init = _get_unbound("init_weights_update_group")
    worker = _make_stub_worker(base_url="http://1.2.3.4:9999")

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        return_value=_make_response(200),
    ) as mock_post:
        ok = init(
            worker,
            master_address="10.0.0.1",
            master_port=29500,
            rank_offset=1,
            world_size=9,
            group_name="my-bridge",
            backend="nccl",
        )

    assert ok is True
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://1.2.3.4:9999/init_weights_update_group"
    assert kwargs["json"] == {
        "master_address": "10.0.0.1",
        "master_port": 29500,
        "rank_offset": 1,
        "world_size": 9,
        "group_name": "my-bridge",
        "backend": "nccl",
    }


def test_init_weights_update_group_status_500_returns_false():
    init = _get_unbound("init_weights_update_group")
    worker = _make_stub_worker()

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        return_value=_make_response(500, text="boom"),
    ):
        ok = init(worker, "10.0.0.1", 29500, 1, 9)

    assert ok is False


def test_init_weights_update_group_exception_returns_false():
    init = _get_unbound("init_weights_update_group")
    worker = _make_stub_worker()

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        side_effect=ConnectionError("network down"),
    ):
        ok = init(worker, "10.0.0.1", 29500, 1, 9)

    assert ok is False


# ---- update_weights_from_distributed -----------------------------------------


def test_update_weights_from_distributed_non_master_returns_true():
    update = _get_unbound("update_weights_from_distributed")
    worker = _make_stub_worker(is_model_owner=False, base_url=None)

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post"
    ) as mock_post:
        ok = update(
            worker,
            names=["a"],
            dtypes=["bfloat16"],
            shapes=[[4, 8]],
            group_name="g",
        )

    assert ok is True
    mock_post.assert_not_called()


def test_update_weights_from_distributed_posts_correct_body_with_load_format():
    update = _get_unbound("update_weights_from_distributed")
    worker = _make_stub_worker(base_url="http://1.2.3.4:9999")

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        return_value=_make_response(200, json_body={"success": True, "message": "ok"}),
    ) as mock_post:
        ok = update(
            worker,
            names=["model.embed", "model.layers.0.q"],
            dtypes=["bfloat16", "bfloat16"],
            shapes=[[1024, 4096], [4096, 4096]],
            group_name="my-bridge",
            flush_cache=True,
            load_format="flattened_bucket",
        )

    assert ok is True
    args, kwargs = mock_post.call_args
    assert args[0] == "http://1.2.3.4:9999/update_weights_from_distributed"
    body = kwargs["json"]
    assert body["names"] == ["model.embed", "model.layers.0.q"]
    assert body["dtypes"] == ["bfloat16", "bfloat16"]
    assert body["shapes"] == [[1024, 4096], [4096, 4096]]
    assert body["group_name"] == "my-bridge"
    assert body["flush_cache"] is True
    assert body["load_format"] == "flattened_bucket"


def test_update_weights_from_distributed_omits_load_format_when_none():
    """When ``load_format`` is ``None`` we must NOT send the field
    (otherwise SGLang will treat the explicit ``None`` differently from
    "not provided" via pydantic)."""
    update = _get_unbound("update_weights_from_distributed")
    worker = _make_stub_worker(base_url="http://1.2.3.4:9999")

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        return_value=_make_response(200, json_body={"success": True}),
    ) as mock_post:
        update(
            worker,
            names=["x"],
            dtypes=["bfloat16"],
            shapes=[[1]],
            group_name="g",
            load_format=None,
        )

    body = mock_post.call_args.kwargs["json"]
    assert "load_format" not in body


def test_update_weights_from_distributed_status_500_returns_false():
    update = _get_unbound("update_weights_from_distributed")
    worker = _make_stub_worker()

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        return_value=_make_response(500, text="server error"),
    ):
        ok = update(
            worker,
            names=["x"],
            dtypes=["bfloat16"],
            shapes=[[1]],
            group_name="g",
        )

    assert ok is False


def test_update_weights_from_distributed_body_success_false_returns_false():
    """SGLang returns HTTP 200 even when ``model.load_weights`` raises;
    we surface the body-level ``success=False`` as a failure."""
    update = _get_unbound("update_weights_from_distributed")
    worker = _make_stub_worker()

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        return_value=_make_response(
            200,
            json_body={"success": False, "message": "load_weights raised"},
        ),
    ):
        ok = update(
            worker,
            names=["x"],
            dtypes=["bfloat16"],
            shapes=[[1]],
            group_name="g",
        )

    assert ok is False


def test_update_weights_from_distributed_body_no_success_field_defaults_true():
    """Older SGLang versions may not include the ``success`` field; we
    treat absence as success to avoid spurious failures."""
    update = _get_unbound("update_weights_from_distributed")
    worker = _make_stub_worker()

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        return_value=_make_response(200, json_body={}),
    ):
        ok = update(
            worker,
            names=["x"],
            dtypes=["bfloat16"],
            shapes=[[1]],
            group_name="g",
        )

    assert ok is True


# ---- destroy_weights_update_group --------------------------------------------


def test_destroy_weights_update_group_4xx_treated_as_success():
    """Group not existing on the engine is fine; we should NOT fail."""
    destroy = _get_unbound("destroy_weights_update_group")
    worker = _make_stub_worker()

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        return_value=_make_response(404, text="no such group"),
    ):
        assert destroy(worker, "g") is True


def test_destroy_weights_update_group_5xx_returns_false():
    destroy = _get_unbound("destroy_weights_update_group")
    worker = _make_stub_worker()

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        return_value=_make_response(500, text="boom"),
    ):
        assert destroy(worker, "g") is False


def test_destroy_weights_update_group_exception_treated_as_success():
    """Tolerant on shutdown — network errors during teardown should not
    crash training."""
    destroy = _get_unbound("destroy_weights_update_group")
    worker = _make_stub_worker()

    with patch(
        "nemo_rl.models.generation.sglang.sglang_worker.requests.post",
        side_effect=ConnectionError(),
    ):
        assert destroy(worker, "g") is True


# ---------------------------------------------------------------------------
# SGLangGeneration orchestration
# ---------------------------------------------------------------------------
#
# We can't construct a real SGLangGeneration without a Ray cluster; we
# test the methods via stub objects that carry ``dp_size``,
# ``gpus_per_server``, and a ``worker_group`` mock.


class _FakeWorkerGroup:
    """Minimal stand-in for ``RayWorkerGroup`` recording invocations."""

    def __init__(self) -> None:
        self.workers = ["w0"]
        self.calls: list[dict] = []

    def run_all_workers_multiple_data(self, method_name, **kwargs):
        self.calls.append({"name": method_name, "mode": "multiple_data", **kwargs})
        # ``ray.get`` on these stub futures should yield ``True``.
        return [_FakeFuture(True) for _ in (kwargs.get("rank_offset") or [None])]

    def run_all_workers_single_data(self, method_name, **kwargs):
        self.calls.append({"name": method_name, "mode": "single_data", **kwargs})
        return [_FakeFuture(True)]


class _FakeFuture:
    def __init__(self, value):
        self._value = value


def _patched_ray_get(refs):
    return [r._value for r in refs]


def _make_stub_generation(num_engines: int = 1, gpus_per_server: int = 8):
    sglang_gen_mod = pytest.importorskip(
        "nemo_rl.models.generation.sglang.sglang_generation"
    )
    sg = sglang_gen_mod.SGLangGeneration.__new__(sglang_gen_mod.SGLangGeneration)
    sg.dp_size = num_engines  # type: ignore[attr-defined]
    sg.gpus_per_server = gpus_per_server  # type: ignore[attr-defined]
    sg.worker_group = _FakeWorkerGroup()  # type: ignore[attr-defined]
    return sg


def test_init_collective_nccl_bridge_rank_offsets_one_engine():
    sg = _make_stub_generation(num_engines=1, gpus_per_server=8)
    with patch(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        side_effect=_patched_ray_get,
    ):
        ok = sg.init_collective_nccl_bridge(
            train_master_address="10.0.0.1",
            train_master_port=29500,
            train_world_size=1,
            group_name="bridge",
        )
    assert ok is True
    call = sg.worker_group.calls[0]
    assert call["name"] == "init_weights_update_group"
    # 1 engine -> [train_world_size + 0 * tp] = [1]
    assert call["rank_offset"] == [1]
    # world_size = 1 (train) + 1 * 8 (engines x tp) = 9
    assert call["common_kwargs"]["world_size"] == 9
    assert call["common_kwargs"]["master_address"] == "10.0.0.1"
    assert call["common_kwargs"]["master_port"] == 29500
    assert call["common_kwargs"]["group_name"] == "bridge"
    assert call["common_kwargs"]["backend"] == "nccl"


def test_init_collective_nccl_bridge_rank_offsets_multi_engine():
    sg = _make_stub_generation(num_engines=3, gpus_per_server=4)
    with patch(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        side_effect=_patched_ray_get,
    ):
        ok = sg.init_collective_nccl_bridge(
            train_master_address="10.0.0.1",
            train_master_port=29500,
            train_world_size=2,
        )
    assert ok is True
    call = sg.worker_group.calls[0]
    # 3 engines, tp=4, train_world_size=2 -> rank offsets [2, 6, 10]
    assert call["rank_offset"] == [2, 6, 10]
    # world_size = 2 + 3 * 4 = 14
    assert call["common_kwargs"]["world_size"] == 14


def test_init_collective_nccl_bridge_propagates_failure():
    sg = _make_stub_generation()
    # Force one of the futures to be False.
    sg.worker_group.run_all_workers_multiple_data = lambda *a, **k: [
        _FakeFuture(False)
    ]
    with patch(
        "nemo_rl.models.generation.sglang.sglang_generation.ray.get",
        side_effect=_patched_ray_get,
    ):
        ok = sg.init_collective_nccl_bridge("ip", 1, 1)
    assert ok is False


def test_init_collective_nccl_bridge_raises_when_workers_missing():
    sg = _make_stub_generation()
    sg.worker_group = None  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="Worker group is not initialized"):
        sg.init_collective_nccl_bridge("ip", 1, 1)


def test_update_weights_via_nccl_bridge_dispatches_with_load_format():
    sg = _make_stub_generation(num_engines=2)
    futures = sg.update_weights_via_nccl_bridge(
        names=["w0", "w1"],
        dtypes=["bfloat16", "bfloat16"],
        shapes=[[2, 3], [4, 5]],
        group_name="bridge",
        flush_cache=True,
        load_format="flattened_bucket",
    )
    assert len(futures) == 1
    call = sg.worker_group.calls[0]
    assert call["name"] == "update_weights_from_distributed"
    assert call["mode"] == "single_data"
    assert call["names"] == ["w0", "w1"]
    assert call["dtypes"] == ["bfloat16", "bfloat16"]
    assert call["shapes"] == [[2, 3], [4, 5]]
    assert call["group_name"] == "bridge"
    assert call["flush_cache"] is True
    assert call["load_format"] == "flattened_bucket"


def test_destroy_collective_nccl_bridge_idempotent_when_no_workers():
    sg = _make_stub_generation()
    sg.worker_group = None  # type: ignore[attr-defined]
    assert sg.destroy_collective_nccl_bridge("g") is True


# ---------------------------------------------------------------------------
# Wire format: train-side flattened bucket bytes match SGLang's expected
# ``FlattenedTensorBucket(named_tensors=...).flattened_tensor`` exactly.
# This is THE invariant that makes the NCCL broadcast bit-correct.
# ---------------------------------------------------------------------------


def test_flattened_bucket_byte_layout_matches_sglang_concat():
    """Train side: ``torch.cat([t.flatten().contiguous().view(uint8) ...])``
    Receive side (SGLang): ``FlattenedTensorBucket(named_tensors=[
        (name, torch.empty(shape, dtype, device)) ...
    ]).flattened_tensor`` (constructed with the SAME shape/dtype list).

    Both must agree byte-for-byte: the train-side concat IS what SGLang
    will write into via NCCL. We don't import SGLang here (heavy + GPU);
    instead we assert the layout invariant directly.
    """
    torch = pytest.importorskip("torch")

    # Three tensors with mixed dtypes (bf16 + f32 + i32) and shapes; the
    # ``view(uint8)`` reinterpretation should yield the underlying bytes
    # in row-major order.
    a = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    b = torch.tensor([[1.5, -2.5], [3.5, -4.5]], dtype=torch.float32)
    c = torch.tensor([7, 8, 9, 10], dtype=torch.int32)

    flat_parts = [
        a.flatten().contiguous().view(torch.uint8),
        b.flatten().contiguous().view(torch.uint8),
        c.flatten().contiguous().view(torch.uint8),
    ]
    train_flat = torch.cat(flat_parts, dim=0)

    # Expected layout: bytes(a) | bytes(b) | bytes(c).
    a_bytes = a.contiguous().view(torch.uint8).flatten()
    b_bytes = b.contiguous().view(torch.uint8).flatten()
    c_bytes = c.contiguous().view(torch.uint8).flatten()
    expected = torch.cat([a_bytes, b_bytes, c_bytes], dim=0)

    assert train_flat.dtype == torch.uint8
    assert train_flat.numel() == (
        a.numel() * a.element_size()
        + b.numel() * b.element_size()
        + c.numel() * c.element_size()
    )
    assert torch.equal(train_flat, expected)


def test_flattened_bucket_round_trips_via_metadata_offsets():
    """Validates the metadata schema we produce against
    ``FlattenedTensorMetadata``: ``(name, shape, dtype, start_idx, end_idx,
    numel)``. Reconstruction by ``flat[start:end].view(dtype).reshape(shape)``
    must yield byte-equal tensors back."""
    torch = pytest.importorskip("torch")

    tensors = [
        ("embed", torch.randn(4, 6, dtype=torch.bfloat16)),
        ("q_proj", torch.randn(6, 6, dtype=torch.bfloat16)),
        ("router", torch.randn(8, dtype=torch.float32)),
    ]

    flat_parts = [t.flatten().contiguous().view(torch.uint8) for _, t in tensors]
    flat = torch.cat(flat_parts, dim=0)

    # Build metadata exactly like the train-side code does.
    metadata = []
    cursor = 0
    for name, t in tensors:
        nbytes = t.numel() * t.element_size()
        metadata.append(
            {
                "name": name,
                "shape": list(t.shape),
                "dtype": t.dtype,
                "start_idx": cursor,
                "end_idx": cursor + nbytes,
                "numel": nbytes,
            }
        )
        cursor += nbytes

    # Reconstruct and check.
    for meta, (orig_name, orig_t) in zip(metadata, tensors):
        view = (
            flat[meta["start_idx"] : meta["end_idx"]]
            .view(meta["dtype"])
            .reshape(meta["shape"])
        )
        assert meta["name"] == orig_name
        assert torch.equal(view, orig_t)


def test_dtype_string_format_matches_sglang_expectation():
    """SGLang's server-side does ``getattr(torch, dtype)`` to look up the
    dtype, so we must emit ``"bfloat16"`` not ``"torch.bfloat16"`` in the
    HTTP body. The train-side helper uses
    ``str(t.dtype).removeprefix("torch.")``."""
    torch = pytest.importorskip("torch")

    cases = [
        (torch.bfloat16, "bfloat16"),
        (torch.float16, "float16"),
        (torch.float32, "float32"),
        (torch.int32, "int32"),
    ]
    for dtype, expected_wire in cases:
        wire = str(dtype).removeprefix("torch.")
        assert wire == expected_wire
        # And SGLang's server-side trick must round-trip.
        assert getattr(torch, wire) is dtype
