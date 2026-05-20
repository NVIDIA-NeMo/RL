# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Round-trip tests for FP8 KV-scale transport through the data plane.

The legacy sync path returns scales via a Ray dict; this module routes
them through TQ so the async path can decouple producer (calibrator)
from consumer (vLLM refit worker). Verifying:

  1. ``pack_kv_scales`` → ``unpack_kv_scales`` is a pure round-trip
     (value-equality for active layers; gaps drop).
  2. ``put_kv_scales`` → ``get_kv_scales`` round-trip via the NoOp
     adapter (purely in-process; smokes the adapter contract without
     spinning up TQ).
"""

from __future__ import annotations

import math

import pytest

from nemo_rl.data_plane.kv_scales import (
    KV_SCALES_FIELDS,
    KV_SCALES_PARTITION_ID,
    get_kv_scales,
    pack_kv_scales,
    put_kv_scales,
    unpack_kv_scales,
)


def _make_scales(n_layers: int = 4) -> dict[str, dict[str, float]]:
    return {
        f"layer_{i}": {
            "q_scale": 0.1 * (i + 1),
            "k_scale": 0.2 * (i + 1),
            "v_scale": 0.3 * (i + 1),
        }
        for i in range(n_layers)
    }


def test_pack_unpack_round_trip():
    scales = _make_scales(n_layers=4)
    packed = pack_kv_scales(scales)
    assert set(packed.keys()) == set(KV_SCALES_FIELDS)
    for f in KV_SCALES_FIELDS:
        assert packed[f].shape == (4,)
    unpacked = unpack_kv_scales(packed)
    assert set(unpacked.keys()) == set(scales.keys())
    for k in scales:
        for s in ("q_scale", "k_scale", "v_scale"):
            assert math.isclose(unpacked[k][s], scales[k][s], rel_tol=1e-6)


def test_pack_handles_gaps():
    # Sparse layer indices: only 0 and 3. Layers 1, 2 should fill 0.0
    # in the tensor and DROP from the unpacked dict (all-zero rule).
    scales = {
        "layer_0": {"q_scale": 1.0, "k_scale": 2.0, "v_scale": 3.0},
        "layer_3": {"q_scale": 4.0, "k_scale": 5.0, "v_scale": 6.0},
    }
    packed = pack_kv_scales(scales)
    assert packed[KV_SCALES_FIELDS[0]].shape == (4,)  # max_idx + 1
    assert packed[KV_SCALES_FIELDS[0]][1].item() == 0.0
    unpacked = unpack_kv_scales(packed)
    assert set(unpacked.keys()) == {"layer_0", "layer_3"}


def test_empty_round_trip():
    assert pack_kv_scales({}) == {f: pack_kv_scales({})[f] for f in KV_SCALES_FIELDS}
    assert unpack_kv_scales(pack_kv_scales({})) == {}


@pytest.mark.parametrize("n_layers", [1, 8, 24])
def test_put_get_round_trip_via_noop_adapter(n_layers):
    """End-to-end transport: pack → put → get → unpack returns the
    same scales.

    Flat tensors cross the TQ boundary; dict-of-dict lives only at the
    caller (calibrate output / refit input). Uses the NoOp adapter so
    the test stays in-process.
    """
    from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient

    client = NoOpDataPlaneClient()
    scales = _make_scales(n_layers=n_layers)
    put_kv_scales(client, pack_kv_scales(scales))
    round_tripped = unpack_kv_scales(get_kv_scales(client))
    assert set(round_tripped.keys()) == set(scales.keys())
    for k in scales:
        for s in ("q_scale", "k_scale", "v_scale"):
            assert math.isclose(
                round_tripped[k][s], scales[k][s], rel_tol=1e-6
            ), f"mismatch at {k}/{s}: {round_tripped[k][s]} vs {scales[k][s]}"


def test_put_idempotent_partition_id():
    """``put_kv_scales`` registers / writes / returns the partition_id
    every call. Default ``KV_SCALES_PARTITION_ID`` returned unchanged."""
    from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient

    client = NoOpDataPlaneClient()
    pid = put_kv_scales(client, pack_kv_scales(_make_scales(2)))
    assert pid == KV_SCALES_PARTITION_ID
    pid2 = put_kv_scales(
        client, pack_kv_scales(_make_scales(2)), partition_id="custom_pid"
    )
    assert pid2 == "custom_pid"
