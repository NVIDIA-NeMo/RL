# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FP8 KV-cache scale transport through the data plane.

The legacy sync path computes per-layer Q/K/V scales on workers and
returns them as a Python dict via Ray to the driver, which then
broadcasts them to vLLM workers via another Ray call. That keeps the
driver on the critical path and doesn't compose with the async story —
multiple training steps in flight can't share scales without driver
serialization.

This module routes scales through the data plane instead:

  worker.calibrate_qkv_fp8_scales(data)
      ─► returns {"layer_<i>": {"q_scale": ..., "k_scale": ..., "v_scale": ...}}
      ─► ``put_kv_scales`` packs into a single-sample TQ partition
          (fields: ``q_scales``, ``k_scales``, ``v_scales`` tensors of
          shape ``(n_layers,)``)
  vLLM refit worker
      ─► ``get_kv_scales`` reads back, unpacks, applies
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nemo_rl.data_plane.interfaces import DataPlaneClient


# Single canonical partition id for scale exchange across the training
# step. Cleared and re-registered each calibration cycle.
KV_SCALES_PARTITION_ID = "kv_scales"
KV_SCALES_FIELDS = ("q_scales", "k_scales", "v_scales")


def _layer_index(layer_name: str) -> int:
    """``layer_<i>`` → ``i``. Falls back to a hash for unparsed names."""
    if layer_name.startswith("layer_"):
        try:
            return int(layer_name.removeprefix("layer_"))
        except ValueError:
            pass
    raise ValueError(
        f"unrecognized layer name {layer_name!r}; expected ``layer_<i>``"
    )


def pack_kv_scales(
    scales: dict[str, dict[str, float]],
) -> dict[str, torch.Tensor]:
    """Pack a ``{layer_<i>: {q_scale, k_scale, v_scale}}`` dict into
    fixed-shape tensors keyed by ``KV_SCALES_FIELDS``.

    Layer ordering is dense ``layer_0 .. layer_{n-1}``; gaps fill with 0.0.
    Missing q/k/v entries for a layer also fill with 0.0.
    """
    if not scales:
        return {k: torch.zeros(0) for k in KV_SCALES_FIELDS}
    indices = [_layer_index(k) for k in scales.keys()]
    n_layers = max(indices) + 1
    out: dict[str, torch.Tensor] = {
        f: torch.zeros(n_layers, dtype=torch.float32) for f in KV_SCALES_FIELDS
    }
    for name, entry in scales.items():
        i = _layer_index(name)
        for f, key in zip(KV_SCALES_FIELDS, ("q_scale", "k_scale", "v_scale")):
            v = entry.get(key)
            if v is not None:
                out[f][i] = float(v)
    return out


def unpack_kv_scales(
    packed: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Inverse of :func:`pack_kv_scales`.

    Layers whose q/k/v are all 0.0 are omitted (treated as unset).
    """
    if not packed or all(
        t.numel() == 0 for t in packed.values() if torch.is_tensor(t)
    ):
        return {}
    q = packed[KV_SCALES_FIELDS[0]].tolist()
    k = packed[KV_SCALES_FIELDS[1]].tolist()
    v = packed[KV_SCALES_FIELDS[2]].tolist()
    n_layers = max(len(q), len(k), len(v))
    out: dict[str, dict[str, float]] = {}
    for i in range(n_layers):
        qi = q[i] if i < len(q) else 0.0
        ki = k[i] if i < len(k) else 0.0
        vi = v[i] if i < len(v) else 0.0
        if qi == 0.0 and ki == 0.0 and vi == 0.0:
            continue
        out[f"layer_{i}"] = {"q_scale": qi, "k_scale": ki, "v_scale": vi}
    return out


def put_kv_scales(
    client: "DataPlaneClient",
    packed: dict[str, torch.Tensor],
    *,
    partition_id: str = KV_SCALES_PARTITION_ID,
) -> str:
    """Write packed FP8 scales (flat tensors) to the data plane.

    ``packed`` is ``{q_scales, k_scales, v_scales}`` — three 1-D tensors
    of equal length ``n_layers``. Use :func:`pack_kv_scales` to convert
    from the dict-of-dict shape that ``calibrate_qkv_fp8_scales`` returns.

    Re-registers ``partition_id`` (single sample) idempotently. Returns
    the partition_id so the reader can address it.
    """
    from tensordict import TensorDict

    # Idempotent registration. Single sample, three packed fields, no
    # consumer-task accounting (scales are read-many, not consumed).
    client.register_partition(
        partition_id=partition_id,
        fields=list(KV_SCALES_FIELDS),
        num_samples=1,
        consumer_tasks=[],
    )
    td = TensorDict(
        {k: v.unsqueeze(0) for k, v in packed.items()},  # (1, n_layers)
        batch_size=[1],
    )
    meta = client.put_samples(
        sample_ids=["scales_v0"],
        partition_id=partition_id,
        fields=td,
    )
    del meta  # single known sample_id; not needed downstream
    return partition_id


def get_kv_scales(
    client: "DataPlaneClient",
    *,
    partition_id: str = KV_SCALES_PARTITION_ID,
) -> dict[str, torch.Tensor]:
    """Read packed FP8 scales (flat tensors) from the data plane.

    Returns ``{q_scales, k_scales, v_scales}`` — feed to
    :func:`unpack_kv_scales` to recover the dict-of-dict shape that
    refit consumers (e.g. ``broadcast_weights_for_collective``) expect.
    """
    td = client.get_samples(
        sample_ids=["scales_v0"],
        partition_id=partition_id,
        select_fields=list(KV_SCALES_FIELDS),
    )
    # td is shape (1, n_layers) per field; squeeze the sample dim.
    return {f: td[f].squeeze(0) for f in KV_SCALES_FIELDS}
