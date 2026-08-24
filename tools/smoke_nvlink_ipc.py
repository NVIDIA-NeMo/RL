# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Verify register mode over Mooncake's NVLink IPC transport, across processes.

``smoke_register_mode_gdr.py`` builds both engines in one process, which is
fine for RDMA but cannot answer the IPC question: a handle exported and
imported inside one process does not exercise ``cudaIpcOpenMemHandle`` the way
two processes do. This runs producer and consumer as separate processes on
separate GPUs and moves the published metadata over a queue, which is what the
real deployment does.

Two things this is built to catch, both of which fail *quietly*:

* Under fabric memory (``HANDLE_TYPE_FABRIC_SUPPORTED=1``, true on GB200/GB300),
  ``NvlinkTransport::registerLocalMemory`` calls ``cuMemRetainAllocationHandle``
  and, when the address did not come from ``cuMemCreate``, logs a warning and
  returns 0 -- success, having published nothing. Torch's default caching
  allocator uses ``cudaMalloc``, so a put would report success and leave the
  key unreadable. ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` routes
  torch through the VMM APIs instead; this tool is meant to be run both ways.
* A transport that silently degrades to a local copy would pass a correctness
  check while proving nothing about NVLink, so the read is always cross-GPU and
  the achieved bandwidth is printed for comparison against the RDMA baseline.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import traceback
from time import perf_counter

# maybe_configure_data_plane_env must run before the engine extension loads, so
# every adapter import below is function-local. See smoke_register_mode_gdr.py.


def _engine_env() -> None:
    from nemo_rl.data_plane.factory import maybe_configure_data_plane_env

    maybe_configure_data_plane_env(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "transfer_engine",
            "claim_meta_poll_interval_s": 0.5,
        }
    )


def _client(protocol: str, device_name: str):
    from nemo_rl.data_plane.adapters.tq_register_mode import TransferEngineClient
    from nemo_rl.data_plane.interfaces import TransferEngineConfig

    return TransferEngineClient(
        {
            **TransferEngineConfig(use_gdr=True).model_dump(),
            "local_hostname": "",
            "protocol": protocol,
            "device_name": device_name,
            "metadata_server": "P2PHANDSHAKE",
        }
    )


def producer(protocol: str, device_name: str, mb: int, rows: int, out, gate) -> None:
    try:
        import torch

        _engine_env()
        torch.cuda.set_device(0)
        client = _client(protocol, device_name)
        elems = (mb * 2**20) // (rows * 4)
        batch = torch.arange(rows * elems, dtype=torch.float32, device="cuda:0")
        batch = batch.reshape(rows, elems)
        views = [batch[i] for i in range(rows)]
        keys = [f"{i}@nvlink" for i in range(rows)]

        started = perf_counter()
        meta = client.put(keys, views)
        put_ms = (perf_counter() - started) * 1e3

        published = sum(int(e.get("size", 0)) for e in meta)
        out.put(
            {
                "role": "producer",
                "endpoint": client.endpoint,
                "put_ms": put_ms,
                "published_bytes": published,
                "meta": meta,
                "keys": keys,
                "shape": (rows, elems),
                "checksum": float(batch.double().sum().item()),
            }
        )
        gate.wait()  # hold the registration alive until the consumer has read
        client.clear(keys, custom_backend_meta=meta)
        client.close()
    except Exception:
        out.put({"role": "producer", "error": traceback.format_exc()})
        gate.set()


def consumer(protocol: str, device_name: str, inbox, out, gate) -> None:
    try:
        import torch

        published = inbox.get(timeout=90)
        if "error" in published:
            out.put({"role": "consumer", "error": "producer failed first"})
            gate.set()
            return

        _engine_env()
        torch.cuda.set_device(1)
        client = _client(protocol, device_name)
        rows, elems = published["shape"]

        started = perf_counter()
        values = client.get(
            published["keys"],
            shapes=[(elems,)] * rows,
            dtypes=[torch.float32] * rows,
            custom_backend_meta=published["meta"],
        )
        get_ms = (perf_counter() - started) * 1e3

        got = torch.stack([v.to("cuda:1") for v in values])
        checksum = float(got.double().sum().item())
        out.put(
            {
                "role": "consumer",
                "endpoint": client.endpoint,
                "get_ms": get_ms,
                "checksum": checksum,
                "device": values[0].device.type,
            }
        )
        client.close()
    except Exception:
        out.put({"role": "consumer", "error": traceback.format_exc()})
    finally:
        gate.set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="nvlink")
    parser.add_argument("--mb", type=int, default=64)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument(
        "--device-name",
        default=None,
        help="engine device filter; a name no HCA carries (e.g. no_such_nic0) "
        "leaves topology discovery empty, so a transfer that still succeeds "
        "cannot have gone over RDMA",
    )
    args = parser.parse_args()

    print(f"protocol={args.protocol} payload={args.mb}MB rows={args.rows}")
    print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '(unset)')}")

    device_name = args.device_name or ""
    if args.device_name is None and args.protocol == "rdma":
        _engine_env()
        from nemo_rl.data_plane.adapters.transfer_queue import rdma_devices

        device_name = rdma_devices()
        print(f"rdma_devices: {device_name or '(none)'}")
    elif args.device_name:
        print(f"device filter: {args.device_name!r} (overriding discovery)")

    ctx = mp.get_context("spawn")
    to_consumer, results, gate = ctx.Queue(), ctx.Queue(), ctx.Event()
    procs = [
        ctx.Process(target=producer,
                    args=(args.protocol, device_name, args.mb, args.rows,
                          to_consumer, gate)),
        ctx.Process(target=consumer,
                    args=(args.protocol, device_name, to_consumer, results, gate)),
    ]
    # The producer's message is both the handshake and the consumer's input, so
    # it is forwarded rather than duplicated: read it, report it, pass it on.
    procs[0].start()
    first = to_consumer.get(timeout=90)
    to_consumer.put(first)
    procs[1].start()

    if "error" in first:
        print(f"\nPRODUCER FAILED:\n{first['error']}")
        gate.set()
        for p in procs:
            p.join(timeout=60)
        return 1

    print(f"\nput:  {first['put_ms']:.2f}ms  published={first['published_bytes'] / 2**20:.1f} MB "
          f"endpoint={first['endpoint']}")
    if first["published_bytes"] == 0:
        print("  WARNING: nothing published -- the fabric export path no-ops on "
              "cudaMalloc memory; expect the get to fail")

    outcome = results.get(timeout=90)
    for p in procs:
        p.join(timeout=120)

    if "error" in outcome:
        print(f"\nCONSUMER FAILED:\n{outcome['error']}")
        return 1

    mbps = (args.mb / (outcome["get_ms"] / 1e3))
    match = abs(outcome["checksum"] - first["checksum"]) < 1e-3
    print(f"get:  {outcome['get_ms']:.2f}ms  {mbps:.0f} MB/s  device={outcome['device']} "
          f"endpoint={outcome['endpoint']}")
    print(f"bytes_match={match}  (producer={first['checksum']:.1f} "
          f"consumer={outcome['checksum']:.1f})")
    print("RESULT:", "PASS" if match else "FAIL")
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
