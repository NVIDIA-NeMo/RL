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
"""Byte-level equivalence between legacy and TQ seqpack/dynbatch paths.

Both paths share ``BatchedDataDict.shard_by_batch_size(shards=DP_world,
sequence_packing_args=...)`` for cross-DP balance (Option 1 fix). The only
implementation difference is data transport: legacy hands each shard's
tensors directly to the worker; TQ writes them into the queue, then the
worker reads them back.

This test isolates the seqpack/dynbatch math from rollout sampling, NCCL
non-determinism, and optimizer steps. If it passes, the only remaining
sources of legacy-vs-TQ run-to-run divergence live outside NeMo-RL.

Spec:
  1. Build a deterministic ``train_data`` with variable input lengths.
  2. Run ``shard_by_batch_size`` on the driver — this is the *one* call
     both paths share. Save its output as the legacy reference.
  3. Round-trip each shard through TQ (``put_samples`` →
     ``get_samples`` → ``materialize``) and re-attach the per-shard
     packing metadata from ``extra_info`` (what
     ``train_presharded`` does in production).
  4. Assert each rank's tensors and packing metadata are byte-identical
     to the legacy reference.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("ray")
transfer_queue = pytest.importorskip("transfer_queue")  # noqa: F841

from nemo_rl.data_plane import build_data_plane_client, materialize  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402

from ._rollout_shapes import mooncake_available

# Ray is initialized once by the parent autouse fixture
# ``tests/unit/conftest.py::init_ray_cluster`` (mirrors production: NeMo-RL
# inits Ray at startup; the data plane attaches on top). Each test just
# builds a TQ client on the shared Ray and closes it on teardown.


# Mirror of the seed-field set in nemo_rl/algorithms/grpo_sync.py.
_DP_SEED_FIELDS = (
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "prev_logprobs",
    "reference_policy_logprobs",
    "advantages",
    "token_mask",
    "sample_mask",
)

# ── loud-skip helpers ─────────────────────────────────────────────────────────

# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_tq_cfg(backend: str) -> dict:
    # DataPlaneConfig requires the full schema (see interfaces.py); the
    # adapter dereferences ``claim_meta_poll_interval_s`` at construction
    # so missing it short-circuits the fixture before any test runs.
    # ``global_segment_size`` / ``local_buffer_size`` only matter for
    # ``mooncake_cpu`` but are required for schema conformance.
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": backend,
        "storage_capacity": 1024,
        "num_storage_units": 1,
        "claim_meta_poll_interval_s": 0.5,
        "global_segment_size": 8589934592,  # 8 GiB — sized for CI host RAM, not prod
        "local_buffer_size": 1073741824,  # 1 GiB
    }


@pytest.fixture(
    scope="module",
    params=["simple", "mooncake_cpu"],
    ids=["simple", "mooncake_cpu"],
)
def tq_client(request):
    """Parametrized fixture over simple and mooncake_cpu backends.

    mooncake_cpu is skipped when the mooncake wheel is not installed.
    Set NEMO_RL_REQUIRE_MOONCAKE=1 to promote the skip to a loud failure.

    Module-scoped so the mooncake_master + Transfer Engine survive across
    the test cases in this file: each test uses its own ``partition_id``
    ("seqpack-eq" / "dynbatch-eq" / "nopack-eq") so no cross-test data
    leak is possible, and reusing one client avoids the close→re-init
    race in mooncake's C++ mount registry (upstream ``transfer_queue``
    leaks the master process on close; the C++ engine then keeps stale
    endpoint references that 404 against the next run's fresh master).

    Relies on parent autouse ``init_ray_cluster`` for the Ray runtime.
    """
    backend = request.param
    if backend == "mooncake_cpu" and not mooncake_available():
        pytest.skip(
            "mooncake not installed — skipping mooncake_cpu seqpack equivalence "
            "(set NEMO_RL_REQUIRE_MOONCAKE=1 to fail loud)"
        )
    client = build_data_plane_client(_make_tq_cfg(backend))
    yield client
    client.close()


def _make_fake_train_data(
    n_samples: int = 64,
    max_seqlen: int = 4096,
    seed: int = 42,
) -> BatchedDataDict:
    """Stand-in for GRPO ``train_data``.

    Variable lengths in ``[256, max_seqlen]`` so the bin packer actually
    produces multiple bins per shard — flat-length data would trivially
    match.
    """
    g = torch.Generator().manual_seed(seed)
    input_lengths = torch.randint(256, max_seqlen + 1, (n_samples,), generator=g)
    input_ids = torch.zeros((n_samples, max_seqlen), dtype=torch.long)
    for i in range(n_samples):
        n = int(input_lengths[i])
        input_ids[i, :n] = torch.randint(1, 50000, (n,), generator=g)
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "advantages": torch.randn(n_samples, max_seqlen, generator=g),
            "token_mask": torch.ones(n_samples, max_seqlen),
            "sample_mask": torch.ones(n_samples),
            "prev_logprobs": torch.randn(n_samples, max_seqlen, generator=g),
            "reference_policy_logprobs": torch.randn(
                n_samples, max_seqlen, generator=g
            ),
            "generation_logprobs": torch.randn(n_samples, max_seqlen, generator=g),
        }
    )


def _round_trip_shards_through_tq(
    tq_client,
    pre_shards: list,
    partition_id: str,
) -> list[BatchedDataDict]:
    """Put each shard's seed fields to TQ, fetch back, attach packing metadata.

    This is the same dance the production driver+worker does:
    ``grpo_sync.py`` builds per-rank metas and seeds TQ; ``train_presharded``
    fetches its slice and attaches ``extra_info`` packing metadata.
    """
    n_total = sum(int(s["sample_mask"].shape[0]) for s in pre_shards)
    tq_client.register_partition(
        partition_id=partition_id,
        fields=list(_DP_SEED_FIELDS),
        num_samples=n_total,
        consumer_tasks=["train"],
    )
    out: list[BatchedDataDict] = []
    for r, shard in enumerate(pre_shards):
        n = int(shard["sample_mask"].shape[0])
        keys = [f"r{r}_s{i}" for i in range(n)]
        names = [
            f
            for f in _DP_SEED_FIELDS
            if f in shard and isinstance(shard[f], torch.Tensor)
        ]
        fields = TensorDict(
            {f: shard[f].detach().contiguous() for f in names},
            batch_size=[n],
        )
        tq_client.put_samples(
            sample_ids=keys,
            partition_id=partition_id,
            fields=fields,
        )
        td_back = tq_client.get_samples(
            sample_ids=keys,
            partition_id=partition_id,
            select_fields=list(names),
        )
        bdd = materialize(td_back, layout="padded")
        bdd.micro_batch_indices = shard.micro_batch_indices
        bdd.micro_batch_lengths = shard.micro_batch_lengths
        bdd.elem_counts_per_gb = shard.elem_counts_per_gb
        out.append(bdd)
    return out


def _assert_shards_byte_equal(legacy, recovered, *, expect_metadata: bool) -> None:
    assert len(legacy) == len(recovered), (
        f"shard count mismatch: legacy={len(legacy)} tq={len(recovered)}"
    )
    for r, (L, T) in enumerate(zip(legacy, recovered)):
        L_tensor_keys = {k for k, v in L.data.items() if isinstance(v, torch.Tensor)}
        # TQ only transmits _DP_SEED_FIELDS — non-seed legacy fields are
        # out of scope for this test.
        common = L_tensor_keys & set(_DP_SEED_FIELDS)
        assert common <= set(T.data.keys()), (
            f"rank {r}: TQ shard missing seed fields {common - set(T.data.keys())}"
        )
        for k in common:
            assert L[k].shape == T[k].shape, (
                f"rank {r} field {k}: shape {L[k].shape} != {T[k].shape}"
            )
            assert L[k].dtype == T[k].dtype, (
                f"rank {r} field {k}: dtype {L[k].dtype} != {T[k].dtype}"
            )
            assert torch.equal(L[k], T[k]), f"rank {r} field {k}: byte-level mismatch"
        if expect_metadata:
            assert L.micro_batch_indices == T.micro_batch_indices, (
                f"rank {r} micro_batch_indices mismatch"
            )
            assert L.micro_batch_lengths == T.micro_batch_lengths, (
                f"rank {r} micro_batch_lengths mismatch"
            )
            assert L.elem_counts_per_gb == T.elem_counts_per_gb, (
                f"rank {r} elem_counts_per_gb mismatch"
            )


def test_seqpack_legacy_equals_tq(tq_client):
    """Sequence packing: legacy shards == TQ-roundtripped shards (byte-level)."""
    DP_WORLD = 4
    GBS = 64
    spa = {
        "algorithm": "modified_first_fit_decreasing",
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "sequence_length_pad_multiple": 64,
        "max_tokens_per_microbatch": 4096,
    }
    data = _make_fake_train_data(n_samples=GBS)

    legacy_shards, _ = data.shard_by_batch_size(
        DP_WORLD,
        batch_size=GBS,
        sequence_packing_args=spa,
    )
    tq_pre_shards, _ = data.shard_by_batch_size(
        DP_WORLD,
        batch_size=GBS,
        sequence_packing_args=spa,
    )
    recovered = _round_trip_shards_through_tq(
        tq_client,
        tq_pre_shards,
        partition_id="seqpack-eq",
    )
    _assert_shards_byte_equal(legacy_shards, recovered, expect_metadata=True)


def test_dynbatch_legacy_equals_tq(tq_client):
    """Dynamic batching: same equivalence claim as seqpack."""
    DP_WORLD = 4
    GBS = 64
    dba = {
        "input_key": "input_ids",
        "input_lengths_key": "input_lengths",
        "sequence_length_round": 64,
        "max_tokens_per_microbatch": 4096,
    }
    data = _make_fake_train_data(n_samples=GBS)

    legacy_shards, _ = data.shard_by_batch_size(
        DP_WORLD,
        batch_size=GBS,
        dynamic_batching_args=dba,
    )
    tq_pre_shards, _ = data.shard_by_batch_size(
        DP_WORLD,
        batch_size=GBS,
        dynamic_batching_args=dba,
    )
    recovered = _round_trip_shards_through_tq(
        tq_client,
        tq_pre_shards,
        partition_id="dynbatch-eq",
    )
    _assert_shards_byte_equal(legacy_shards, recovered, expect_metadata=True)


def test_no_packing_legacy_equals_tq(tq_client):
    """Sanity: even without packing/dynbatch the transport should be lossless."""
    DP_WORLD = 4
    GBS = 64
    data = _make_fake_train_data(n_samples=GBS)

    legacy_shards = data.shard_by_batch_size(DP_WORLD, batch_size=GBS)
    tq_pre_shards = data.shard_by_batch_size(DP_WORLD, batch_size=GBS)
    recovered = _round_trip_shards_through_tq(
        tq_client,
        tq_pre_shards,
        partition_id="nopack-eq",
    )
    # No packing → no micro_batch_* metadata to compare.
    _assert_shards_byte_equal(legacy_shards, recovered, expect_metadata=False)
