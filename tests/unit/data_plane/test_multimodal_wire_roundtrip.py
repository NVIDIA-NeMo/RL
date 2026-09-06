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
"""End-to-end VLM multimodal wire roundtrip through the data plane.

Runs a realistic VLM rollout write → TQ (NoOp adapter) → materialize →
trainer read, exercising every layer the multimodal fix touches:

  * ``PackedTensor.to_wire`` on the write side
  * ``kv_first_write`` wire-type filter
  * ``codec.pack_jagged_fields`` (write-time layout transform)
  * ``codec.materialize`` (read-time padded conversion), including the
    ``PACKED_MULTIMODAL_FIELDS`` pad_to_seqlen exclusion
  * ``PackedTensor.from_wire`` on the read side
  * ``BatchedDataDict.get_multimodal_dict`` dispatch

Guards the silent-drop regression class that motivated the PR.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from nemo_rl.data.multimodal_utils import (
    PACKED_MULTIMODAL_FIELDS,
    PER_TOKEN_MULTIMODAL_FIELDS,
    PackedTensor,
    encode_multimodal_for_wire,
    multimodal_row_tags,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import kv_first_write, read_columns
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import GLOBAL_FORWARD_PAD_SEQLEN
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from ._rollout_shapes import keys_from_uids, register_train_partition


def _make_vlm_rollout_output(
    n: int = 4, seqlen: int = 32
) -> tuple[BatchedDataDict, dict[str, PackedTensor], dict[str, torch.Tensor]]:
    """Build the ``final_batch_cpu`` a VLM rollout would hand to
    ``kv_first_write``, plus the per-key ground truth for round-trip
    assertions.

    Ground truth is captured as ``as_tensor()``-form so the read side
    can compare after concat, regardless of jagged reshuffle.
    """
    torch.manual_seed(0)
    input_lengths = torch.tensor([seqlen, seqlen, seqlen, seqlen], dtype=torch.long)

    # PackedTensor multimodal — sample 2 is text-only (None) to
    # exercise the placeholder / lengths[i]==0 path.
    pixel_values = PackedTensor(
        [
            torch.randn(3, 4, 4),  # sample 0: 3 patches
            torch.randn(5, 4, 4),  # sample 1: 5 patches
            None,  # sample 2: no image
            torch.randn(2, 4, 4),  # sample 3: 2 patches
        ],
        dim_to_pack=0,
    )
    image_grid_thw = PackedTensor(
        [
            torch.tensor([[1, 2, 2]], dtype=torch.int64),
            torch.tensor([[1, 3, 3], [1, 2, 2]], dtype=torch.int64),
            None,
            torch.tensor([[1, 2, 1]], dtype=torch.int64),
        ],
        dim_to_pack=0,
    )
    # Per-token multimodal (rectangular) — model-specific type map.
    mm_token_type_ids = torch.randint(0, 3, (n, seqlen), dtype=torch.int64)

    fb = BatchedDataDict()
    fb["input_ids"] = torch.arange(n * seqlen, dtype=torch.long).reshape(n, seqlen)
    fb["input_lengths"] = input_lengths
    fb["token_mask"] = torch.ones((n, seqlen), dtype=torch.long)
    fb["sample_mask"] = torch.ones((n,), dtype=torch.long)
    fb["generation_logprobs"] = torch.zeros((n, seqlen), dtype=torch.float32)
    fb["pixel_values"] = pixel_values
    fb["image_grid_thw"] = image_grid_thw
    fb["mm_token_type_ids"] = mm_token_type_ids

    packed_truth = {
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }
    plain_truth = {"mm_token_type_ids": mm_token_type_ids}
    return fb, packed_truth, plain_truth


def _sync_rollout_write_loop(fb: BatchedDataDict, bulk_batch: BatchedDataDict) -> None:
    """Replay the write loop in ``sync_rollout_actor.rollout_to_tq``.

    Copies non-multimodal fields verbatim, then hands every multimodal
    field to the *production* encoder ``encode_multimodal_for_wire`` —
    the same call the actor makes. Reimplementing the encode branches
    here would make this test blind to exactly the drift (new key added,
    one branch forgotten) that the silent-drop regression was.
    """
    for k, v in fb.items():
        if k in PACKED_MULTIMODAL_FIELDS or k in PER_TOKEN_MULTIMODAL_FIELDS:
            continue  # handled below via get_multimodal_dict
        if isinstance(v, PackedTensor):
            continue  # multimodal PackedTensor; also handled below
        bulk_batch[k] = v

    multimodal = fb.get_multimodal_dict(as_tensors=False)
    for k, v in multimodal.items():
        wire_value = encode_multimodal_for_wire(k, v)
        if wire_value is not None:
            bulk_batch[k] = wire_value


def test_vlm_wire_roundtrip_through_noop_data_plane():
    """Full trip: build a VLM rollout batch → write via kv_first_write
    → materialize on read → verify ``get_multimodal_dict`` on the
    reassembled batch matches the pre-wire ``.as_tensor()`` output.

    Uses the NoOp data-plane adapter so the test runs without TQ
    installed but still exercises the real ABC contract.
    """
    # Registry sanity — if a rename is landed without updating the
    # module-level sets, the failure surfaces here, not in production.
    assert "pixel_values" in PACKED_MULTIMODAL_FIELDS
    assert "image_grid_thw" in PACKED_MULTIMODAL_FIELDS
    assert "mm_token_type_ids" in PER_TOKEN_MULTIMODAL_FIELDS

    n = 4
    seqlen = 32
    fb, packed_truth, plain_truth = _make_vlm_rollout_output(n=n, seqlen=seqlen)

    # ── Write path ──────────────────────────────────────────────────
    client = NoOpDataPlaneClient()
    fields = list(fb.keys())
    register_train_partition(client, num_samples=n, fields=fields)

    bulk_batch: BatchedDataDict = BatchedDataDict()
    _sync_rollout_write_loop(fb, bulk_batch)
    meta = kv_first_write(
        bulk_batch,
        sample_ids=keys_from_uids(["a", "b", "c", "d"]),
        dp_client=client,
        partition_id="train",
        tags=multimodal_row_tags(fb.get_multimodal_dict(as_tensors=False), n),
    )
    # kv_first_write ships one wire field per logical field — the nested
    # parent carries its own row boundaries, so there is no companion.
    for k in packed_truth:
        assert k in meta.fields
        assert f"{k}__lengths" not in meta.fields
    assert "mm_token_type_ids" in meta.fields

    # ── Read path ───────────────────────────────────────────────────
    fetched = read_columns(
        client,
        meta,
        select_fields=meta.fields,
        layout="padded",
    )

    # ── Contract checks on the materialized batch ───────────────────
    # Packed multimodal fields are never rectangularized to
    # ``[B, max_per_sample, ...]`` — no zero padding on the wire or in
    # worker memory, and the row boundaries never have to be recovered
    # from a companion column.
    #
    # ``materialize`` hands them back as ``PackedTensor``, not as the raw
    # nested wire value: generic ``BatchedDataDict`` consumers dispatch on
    # ``PackedTensor`` and otherwise fall through to ``tensor[start:end]``
    # (``slice``) or ``tuple(v.shape)`` (``_broadcast_batched_data_dict``),
    # neither of which torch.nested supports on the ragged dim 0.
    assert isinstance(fetched["pixel_values"], PackedTensor)
    assert isinstance(fetched["image_grid_thw"], PackedTensor)
    assert [t.numel() for t in fetched["pixel_values"].to_wire()[0].unbind()] == [
        t.numel() for t in packed_truth["pixel_values"].to_wire()[0].unbind()
    ]
    assert "pixel_values__lengths" not in fetched
    assert "image_grid_thw__lengths" not in fetched

    # ── Reassembly via BatchedDataDict.get_multimodal_dict ──────────
    mm = fetched.get_multimodal_dict(as_tensors=True)
    assert set(mm.keys()) == {"pixel_values", "image_grid_thw", "mm_token_type_ids"}
    # Packed fields concatenated back to their pre-wire ``.as_tensor()``.
    assert torch.equal(mm["pixel_values"], packed_truth["pixel_values"].as_tensor())
    assert torch.equal(mm["image_grid_thw"], packed_truth["image_grid_thw"].as_tensor())
    # Per-token field passes through untouched.
    assert torch.equal(mm["mm_token_type_ids"], plain_truth["mm_token_type_ids"])

    # ``as_tensors=False`` returns ``PackedTensor`` wrappers for the
    # packed keys so callers can slice per-sample if needed.
    mm_wrapped = fetched.get_multimodal_dict(as_tensors=False)
    assert isinstance(mm_wrapped["pixel_values"], PackedTensor)
    assert isinstance(mm_wrapped["image_grid_thw"], PackedTensor)
    # Non-multimodal keys (input_ids, token_mask, ...) must NOT leak
    # into the multimodal dict.
    assert "input_ids" not in mm_wrapped
    assert "token_mask" not in mm_wrapped


def test_materialize_forward_pad_skips_packed_multimodal_fields():
    """``codec.materialize`` right-pads dim 1 up to the cross-DP forward pad
    target, which is a *token* seqlen. For a packed multimodal field dim 1 is
    patch/image count, so padding it there would inflate ``pixel_values`` by
    the ratio seqlen/patches — the ~40x blow-up the exclusion averts.

    Those fields are now excluded one step earlier: they skip
    ``to_padded_tensor`` entirely and stay nested, so they are never
    rectangular for ``pad_to_seqlen`` to extend. This pins that the
    token-aligned fields still pad while the multimodal ones are untouched.

    The pad only fires when ``GLOBAL_FORWARD_PAD_SEQLEN`` is stamped on the
    meta (``TQPolicy._stamp_pad_seqlen`` does this in production). The plain
    roundtrip above never stamps it, so without this test the exclusion
    branch in ``codec.materialize`` is never entered by any unit test.
    """
    n, seqlen = 4, 32
    pad_to = 64  # cross-DP forward pad target, > seqlen
    fb, _, _ = _make_vlm_rollout_output(n=n, seqlen=seqlen)

    client = NoOpDataPlaneClient()
    fields = list(fb.keys())
    register_train_partition(client, num_samples=n, fields=fields)

    bulk_batch: BatchedDataDict = BatchedDataDict()
    _sync_rollout_write_loop(fb, bulk_batch)
    meta = kv_first_write(
        bulk_batch,
        sample_ids=keys_from_uids(["a", "b", "c", "d"]),
        dp_client=client,
        partition_id="train",
        tags=multimodal_row_tags(fb.get_multimodal_dict(as_tensors=False), n),
    )
    meta = replace(
        meta, extra_info={**(meta.extra_info or {}), GLOBAL_FORWARD_PAD_SEQLEN: pad_to}
    )

    fetched = read_columns(client, meta, select_fields=meta.fields, layout="padded")

    # Token-aligned fields DO get padded up to the forward target.
    assert fetched["input_ids"].shape == (n, pad_to)
    assert fetched["mm_token_type_ids"].shape == (n, pad_to)
    # Packed multimodal fields are never rectangularized, so the forward pad
    # target cannot reach them. ``materialize`` returns them as PackedTensor;
    # ``to_wire()`` recovers the per-row view to check the row counts.
    assert isinstance(fetched["pixel_values"], PackedTensor)
    assert isinstance(fetched["image_grid_thw"], PackedTensor)
    assert (
        max(t.shape[0] for t in fetched["pixel_values"].tensors if t is not None) == 5
    )
    assert (
        max(t.shape[0] for t in fetched["image_grid_thw"].tensors if t is not None) == 2
    )

    # Reassembly still reproduces the pre-wire values under forward padding.
    mm = fetched.get_multimodal_dict(as_tensors=True)
    assert torch.equal(mm["pixel_values"], fb["pixel_values"].as_tensor())
    assert torch.equal(mm["image_grid_thw"], fb["image_grid_thw"].as_tensor())


# ── Dispatch field selection (logprob vs train parity) ──────────────────


def _stub_tq_policy(monkeypatch, captured: dict[str, KVBatchMeta]):
    """A ``TQPolicy`` with only the surface the dispatch bodies touch.

    ``shard_meta_for_dp`` is stubbed to record the meta each dispatch
    builds, so the assertions are on the real field-selection code
    rather than a reimplementation of it.
    """
    from nemo_rl.models.policy import tq_policy as tq_policy_mod

    def fake_shard(meta, **kwargs):
        captured[meta.task_name] = meta
        return [meta], None

    monkeypatch.setattr(tq_policy_mod, "shard_meta_for_dp", fake_shard)

    # ``Policy.shutdown`` short-circuits on a policy with no ``worker_group``,
    # but the dispatch bodies need one, so that escape hatch is unavailable
    # here. Override the destructor instead: this object never owned Ray
    # workers or a TQ client, so running live teardown on it at GC is simply
    # wrong. Faking ``shutdown``/``dp_client`` would work too, but it would
    # report a successful shutdown that never happened and would silently
    # absorb any future change to the real teardown contract.
    #
    # It must be a real no-op method, not ``__del__ = None``: CPython
    # installs ``tp_finalize`` whenever ``__del__`` is present in the class
    # dict, then calls it — ``None()`` raises ``TypeError`` at GC and pytest
    # reports it as a ``PytestUnraisableExceptionWarning`` charged to
    # whichever test happens to be running.
    class _StubTQPolicy(tq_policy_mod.TQPolicy):
        def __del__(self) -> None:
            pass

    pol = object.__new__(_StubTQPolicy)
    pol.cfg = {}
    pol._router_replay_enabled = False
    pol.flops_tracker = None
    pol.sharding_annotations = SimpleNamespace(get_axis_size=lambda _axis: 1)
    pol.worker_group = SimpleNamespace(
        run_all_workers_sharded_data=lambda *a, **k: [],
        get_all_worker_results=lambda _futures: [
            {"global_loss": 0.0, "grad_norm": 0.0, "all_mb_metrics": {}}
        ],
    )
    return pol


def test_train_dispatch_ships_the_same_multimodal_fields_as_logprob(monkeypatch):
    """The training forward must see the images the logprob forwards saw.

    ``DP_TRAIN_FIELDS`` is a static text-only schema, so without the
    per-batch multimodal add-on the GRPO update would run image-blind
    against prev/ref logprobs that were computed *with* images — a
    silent objective mismatch, not a crash.
    """
    mm_fields = [
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    ]
    meta = KVBatchMeta(
        partition_id="train",
        task_name="rollout",
        sample_ids=["a", "b"],
        fields=["input_ids", "input_lengths", "token_mask", *mm_fields],
        sequence_lengths=[8, 8],
    )

    captured: dict[str, KVBatchMeta] = {}
    pol = _stub_tq_policy(monkeypatch, captured)

    pol.get_logprobs_from_meta(meta)
    pol.train_from_meta(meta, loss_fn=None, gbs=2, mbs=1)

    lp_fields = set(captured["prev_lp"].fields)
    train_fields = set(captured["train"].fields)
    # Parity across the FULL registry, not just the hand-listed mm_fields:
    # the two supersets below would be satisfied by a dispatch that shipped
    # extra multimodal columns to one side only.
    registry = PACKED_MULTIMODAL_FIELDS | PER_TOKEN_MULTIMODAL_FIELDS
    assert set(mm_fields) <= lp_fields
    assert set(mm_fields) <= train_fields
    assert lp_fields & registry == train_fields & registry


def test_text_only_dispatch_requests_no_multimodal_fields(monkeypatch):
    """Text-only runs never write the multimodal columns; requesting
    them would raise at the adapter, so the add-on must stay empty."""
    meta = KVBatchMeta(
        partition_id="train",
        task_name="rollout",
        sample_ids=["a", "b"],
        fields=["input_ids", "input_lengths", "token_mask"],
        sequence_lengths=[8, 8],
    )

    captured: dict[str, KVBatchMeta] = {}
    pol = _stub_tq_policy(monkeypatch, captured)

    pol.get_logprobs_from_meta(meta)
    pol.train_from_meta(meta, loss_fn=None, gbs=2, mbs=1)

    all_mm = PACKED_MULTIMODAL_FIELDS | PER_TOKEN_MULTIMODAL_FIELDS
    assert not (set(captured["prev_lp"].fields) & all_mm)
    assert not (set(captured["train"].fields) & all_mm)


def test_ref_logprob_dispatch_ships_multimodal_fields(monkeypatch):
    """``get_reference_policy_logprobs_from_meta`` shares ``_logprob_dispatch``
    with the prev-logprob path, but it is the ref forward whose output ends up
    in the KL term — an image-blind ref logprob is a silent objective skew, so
    pin the ref task explicitly rather than trusting the shared body."""
    mm_fields = [
        "pixel_values",
        "mm_token_type_ids",
    ]
    meta = KVBatchMeta(
        partition_id="train",
        task_name="rollout",
        sample_ids=["a", "b"],
        fields=["input_ids", "input_lengths", "token_mask", *mm_fields],
        sequence_lengths=[8, 8],
    )

    captured: dict[str, KVBatchMeta] = {}
    pol = _stub_tq_policy(monkeypatch, captured)

    pol.get_reference_policy_logprobs_from_meta(meta)

    assert set(mm_fields) <= set(captured["ref_lp"].fields)


def test_sc_microbatch_dispatch_ships_multimodal_fields(monkeypatch):
    """The single-controller split-API path (``train_microbatches_from_meta``)
    carries its own copy of the multimodal add-on, separate from
    ``train_from_meta``. Without a test here the SC path could regress to an
    image-blind forward while the sync path stays green."""
    mm_fields = [
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    ]
    meta = KVBatchMeta(
        partition_id="train",
        task_name="rollout",
        sample_ids=["a", "b"],
        fields=["input_ids", "input_lengths", "token_mask", *mm_fields],
        sequence_lengths=[8, 8],
    )

    # One policy, one patch: ``_stub_tq_policy`` patches the module-level
    # ``shard_meta_for_dp``, so two stubs would share the last patch and the
    # first capture dict would stay empty.
    captured: dict[str, KVBatchMeta] = {}
    pol = _stub_tq_policy(monkeypatch, captured)

    pol.train_from_meta(meta, loss_fn=None, gbs=2, mbs=1)
    sync_fields = set(captured["train"].fields)
    pol.train_microbatches_from_meta(meta)
    sc_fields = set(captured["train"].fields)

    assert set(mm_fields) <= sc_fields
    # Both train entrypoints must request the identical multimodal set.
    all_mm = PACKED_MULTIMODAL_FIELDS | PER_TOKEN_MULTIMODAL_FIELDS
    assert sc_fields & all_mm == sync_fields & all_mm


def test_materialized_multimodal_batch_survives_microbatch_slicing():
    """A materialized batch must be sliceable before any multimodal reassembly.

    The training path slices the fetched batch long before anything calls
    ``get_multimodal_dict``::

        get_logprobs -> make_processed_microbatch_iterator
                     -> make_microbatch_iterator_with_dynamic_shapes
                     -> BatchedDataDict.slice -> self.data[k][start:end]

    ``slice`` dispatches on ``PackedTensor`` and otherwise falls through to
    plain indexing. A raw nested value satisfies ``isinstance(v, Tensor)`` and
    has a valid ``shape[0]``, so it clears both guards and only fails at the
    indexing itself with ``slice(): not supported for NestedTensor on dim=0``.
    That took down a Qwen3.5 VLM GRPO run at the first logprob step.
    """
    n = 4
    fb, packed_truth, _ = _make_vlm_rollout_output(n=n, seqlen=32)

    client = NoOpDataPlaneClient()
    register_train_partition(client, num_samples=n, fields=list(fb.keys()))
    bulk_batch: BatchedDataDict = BatchedDataDict()
    _sync_rollout_write_loop(fb, bulk_batch)
    meta = kv_first_write(
        bulk_batch,
        sample_ids=keys_from_uids(["a", "b", "c", "d"]),
        dp_client=client,
        partition_id="train",
        tags=multimodal_row_tags(fb.get_multimodal_dict(as_tensors=False), n),
    )
    fetched = read_columns(client, meta, select_fields=meta.fields, layout="padded")

    # The regression: this raised RuntimeError when the packed fields were
    # left as raw nested tensors.
    sliced = fetched.slice(1, 3)

    for key in ("pixel_values", "image_grid_thw"):
        assert isinstance(sliced[key], PackedTensor)
        expected = packed_truth[key].slice([1, 2]).as_tensor()
        assert torch.equal(sliced.get_multimodal_dict(as_tensors=True)[key], expected)


def test_from_wire_keeps_packed_multimodal_fields_nested():
    """``_from_wire`` must not densify packed multimodal fields.

    It stacks any nested field whose rows all share a shape, to restore the
    dense representation of ordinary batched inputs. For a packed multimodal
    field, uniform rows are a *data-dependent accident* -- every sample simply
    happened to carry one image -- and stacking discards the row boundaries.

    The dense value then fails the ``is_nested`` check in
    ``codec.materialize``, so it is never reassembled, and reaches
    ``get_multimodal_dict`` as a dense tensor::

        ValueError: 'image_grid_thw' is still in data-plane wire form (dense).
        Packed multimodal fields must be rebuilt by
        multimodal_utils.reassemble_packed_multimodal ...

    That killed a Qwen3.5 VLM GRPO run on a batch where every sample had
    exactly one image.
    """
    from tensordict import TensorDict

    from nemo_rl.data_plane.adapters.transfer_queue import _from_wire

    # Rows that flatten to equal lengths -- one image per sample.
    uniform_grid = PackedTensor(
        [
            torch.tensor([[1, 2, 2]], dtype=torch.int64),
            torch.tensor([[1, 3, 3]], dtype=torch.int64),
            torch.tensor([[1, 2, 1]], dtype=torch.int64),
        ],
        dim_to_pack=0,
    )
    # A non-multimodal field with equally uniform rows, to pin that the
    # densification still happens for everything else.
    ordinary = torch.nested.as_nested_tensor(
        [torch.ones(4), torch.ones(4), torch.ones(4)], layout=torch.jagged
    )

    td = TensorDict(
        {"image_grid_thw": uniform_grid.to_wire()[0], "input_ids": ordinary},
        batch_size=(3,),
    )
    out = _from_wire(td)

    assert out["image_grid_thw"].is_nested, (
        "packed multimodal field was densified; row boundaries are lost"
    )
    assert not out["input_ids"].is_nested, (
        "ordinary uniform field should still be densified"
    )
    # Still reassembles into the original per-row values.
    rebuilt = PackedTensor.from_wire(out["image_grid_thw"], uniform_grid.to_wire()[1])
    assert torch.equal(rebuilt.as_tensor(), uniform_grid.as_tensor())
