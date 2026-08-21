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
"""Real-NIC smoke test for register mode with GDR.

The unit tests fake the Transfer Engine, so they cannot answer the question this
backend actually rests on: does ``ibv_reg_mr`` over a CUDA allocation succeed on
this host, and does a one-sided READ land the producer's HBM bytes in the
consumer's HBM? This runs two real engines in one process — distinct segments,
so the pull goes through the NIC rather than the local-copy shortcut — and
compares the bytes.

    python tools/smoke_register_mode_gdr.py [--host-only]

Exit code is non-zero on any mismatch or transfer failure.
"""

from __future__ import annotations

import argparse
import sys

import torch

from nemo_rl.data_plane.factory import maybe_configure_data_plane_env
from nemo_rl.data_plane.interfaces import TransferEngineConfig

# Deliberately no module-level import of the adapters: they load
# transfer_queue, and mooncake snapshots its MC_* / WITH_NVIDIA_PEERMEM
# configuration as its extension loads. Importing here would fix the engine's
# settings before _engine_env() could choose them, which is how this tool used
# to fail with ERR_CONTEXT (-202) when run standalone.


def _engine_env() -> None:
    """Apply the same engine configuration a real run gets, before any import."""
    maybe_configure_data_plane_env(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "transfer_engine",
            "claim_meta_poll_interval_s": 0.5,
        }
    )


def _client(use_gdr: bool, device_name: str):
    """Build a client from the config model, so new fields cannot be missed.

    Hand-writing this dict is how the tool silently broke once already: the
    client reads every key it declares, and a field added to
    TransferEngineConfig left the literal short by one.
    """
    from nemo_rl.data_plane.adapters.tq_register_mode import TransferEngineClient

    return TransferEngineClient(
        {
            **TransferEngineConfig(use_gdr=use_gdr).model_dump(),
            "local_hostname": "",  # resolved per process
            "protocol": "rdma",
            "device_name": device_name,
            "metadata_server": "P2PHANDSHAKE",
        }
    )


def main() -> int:
    """Publish from one engine, pull from another, and check the bytes."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host-only",
        action="store_true",
        help="register host memory instead of HBM (isolates GDR from RDMA)",
    )
    args = parser.parse_args()

    _engine_env()
    from nemo_rl.data_plane.adapters.transfer_queue import rdma_devices

    devices = rdma_devices()
    print(f"rdma_devices: {devices or '(none)'}")
    if not devices:
        print("FAIL: no RDMA device visible; register mode cannot run here")
        return 1

    use_gdr = not args.host_only
    if use_gdr:
        torch.cuda.init()
        print(f"cuda: {torch.cuda.get_device_name(0)}")

    producer = _client(use_gdr, devices)
    consumer = _client(use_gdr, devices)
    print(f"producer endpoint: {producer.endpoint}")
    print(f"consumer endpoint: {consumer.endpoint}")
    print(f"receive device:    {consumer.receive_device}")

    device = "cuda" if use_gdr else "cpu"
    # One allocation, four row views — the shape TQ hands put, and the reason
    # the pin table registers a base rather than a key.
    batch = torch.arange(4 * 4096, dtype=torch.float32, device=device).reshape(4, 4096)
    rows = [batch[i] for i in range(4)]
    keys = [f"{i}@smoke" for i in range(4)]
    payload = {"note": "non-tensor payloads ride the host path"}

    meta = producer.put(keys + ["4@smoke_obj"], rows + [payload])
    print(f"put: {len(meta)} keys published, base=0x{meta[0]['base']:x}")
    if len({entry["base"] for entry in meta[:4]}) != 1:
        print("FAIL: row views should share one registered base")
        return 1

    values = consumer.get(
        keys + ["4@smoke_obj"],
        shapes=[tuple(row.shape) for row in rows] + [[]],
        dtypes=[row.dtype for row in rows] + [None],
        custom_backend_meta=meta,
    )

    failures = 0
    for i, (row, value) in enumerate(zip(rows, values[:4], strict=True)):
        on_device = value.device.type
        same = torch.equal(value.to(row.device), row)
        print(f"  key {i}: device={on_device} bytes_match={same}")
        if not same:
            failures += 1
        if use_gdr and on_device != "cuda":
            print(f"FAIL: key {i} landed on {on_device}, expected cuda")
            failures += 1
    if values[4] != payload:
        print(f"FAIL: non-tensor payload round-trip mismatch: {values[4]!r}")
        failures += 1

    producer.clear(keys + ["4@smoke_obj"], custom_backend_meta=meta)
    producer.close()
    consumer.close()

    print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
