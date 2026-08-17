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

  * ``PackedTensor.to_nested_wire`` on the write side
  * ``kv_first_write`` wire-type filter
  * ``codec.pack_jagged_fields`` (write-time layout transform)
  * ``codec.materialize`` (read-time padded conversion), including the
    ``PACKED_MULTIMODAL_FIELDS`` pad_to_seqlen exclusion
  * ``PackedTensor.from_nested_wire`` on the read side
  * ``BatchedDataDict.get_multimodal_dict`` dispatch

Guards the silent-drop regression class that motivated the PR.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from nemo_rl.data.multimodal_utils import (
    PACKED_MULTIMODAL_FIELDS,
    PER_TOKEN_MULTIMODAL_FIELDS,
    PackedTensor,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import kv_first_write, read_columns
from nemo_rl.data_plane.interfaces import KVBatchMeta
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

    Copies non-multimodal fields verbatim, encodes ``PackedTensor``
    multimodal via ``to_nested_wire``, and passes per-token multimodal
    through as plain tensors. Any drift here (new key added, one branch
    forgotten) is exactly what the silent-drop regression looked like.
    """
    for k, v in fb.items():
        if k in PACKED_MULTIMODAL_FIELDS or k in PER_TOKEN_MULTIMODAL_FIELDS:
            continue  # handled below via get_multimodal_dict
        if isinstance(v, PackedTensor):
            continue  # multimodal PackedTensor; also handled below
        bulk_batch[k] = v

    for k, v in fb.get_multimodal_dict(as_tensors=False).items():
        if isinstance(v, PackedTensor):
            assert k in PACKED_MULTIMODAL_FIELDS, (
                f"unregistered packed multimodal field {k!r}"
            )
            nested, lengths = v.to_nested_wire()
            if nested is None:
                continue
            bulk_batch[k] = nested
            bulk_batch[PackedTensor.lengths_key(k)] = lengths
        else:
            bulk_batch[k] = v


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
    fields = list(fb.keys()) + [PackedTensor.lengths_key(k) for k in packed_truth]
    register_train_partition(client, num_samples=n, fields=fields)

    bulk_batch: BatchedDataDict = BatchedDataDict()
    _sync_rollout_write_loop(fb, bulk_batch)
    meta = kv_first_write(
        bulk_batch,
        sample_ids=keys_from_uids(["a", "b", "c", "d"]),
        dp_client=client,
        partition_id="train",
    )
    # kv_first_write ships the wire-form parents AND the __lengths
    # companions — both are needed by the read side for reassembly.
    for k in packed_truth:
        assert k in meta.fields
        assert PackedTensor.lengths_key(k) in meta.fields
    assert "mm_token_type_ids" in meta.fields

    # ── Read path ───────────────────────────────────────────────────
    fetched = read_columns(
        client,
        meta,
        select_fields=meta.fields,
        layout="padded",
    )

    # ── Contract checks on the materialized batch ───────────────────
    # Packed multimodal parent survived as a rectangular tensor of
    # shape ``[B, max_per_sample, ...]``; ``codec.materialize`` did NOT
    # over-pad dim 1 to ``pad_to_seqlen`` (the fix that averts multi-GB
    # zero blow-up for VLM).
    assert fetched["pixel_values"].shape == (n, 5, 4, 4)  # max patches = 5
    assert fetched["image_grid_thw"].shape == (n, 2, 3)  # max images = 2
    assert PackedTensor.lengths_key("pixel_values") in fetched
    assert PackedTensor.lengths_key("image_grid_thw") in fetched

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
    # here. Drop the destructor instead: this object never owned Ray workers
    # or a TQ client, so running live teardown on it at GC is simply wrong.
    # Faking ``shutdown``/``dp_client`` would work too, but it would report a
    # successful shutdown that never happened and would silently absorb any
    # future change to the real teardown contract.
    class _StubTQPolicy(tq_policy_mod.TQPolicy):
        __del__ = None  # type: ignore[assignment]

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
        PackedTensor.lengths_key("pixel_values"),
        "image_grid_thw",
        PackedTensor.lengths_key("image_grid_thw"),
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
    assert set(mm_fields) <= lp_fields
    assert set(mm_fields) <= train_fields
    assert lp_fields & set(mm_fields) == train_fields & set(mm_fields)


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
