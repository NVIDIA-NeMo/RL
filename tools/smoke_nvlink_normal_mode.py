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
"""Does NVLink IPC work in *normal* mode, where Mooncake owns the buffers?

Register mode fails on the NVLink transport and the reason is structural, not a
misconfiguration. ``NvlinkTransport::allocatePinnedLocalMemory`` builds its
buffers with::

    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC

so the allocation is fabric-exportable *by construction*. Register mode instead
hands the transport whatever torch already allocated, and neither torch mode
qualifies: the default caching allocator uses ``cudaMalloc`` (no cuMem handle at
all, so ``cuMemRetainAllocationHandle`` fails and registration publishes
nothing), while ``expandable_segments`` uses cuMem but never requests a FABRIC
handle type.

Normal mode goes through ``allocate_managed_buffer``, which calls that
allocator, so it should be the supported path. This checks that directly
against the Mooncake API -- no TransferQueue, no NeMo-RL adapter -- so a failure
lands on Mooncake rather than on our layers.

Producer and consumer are separate processes on separate GPUs, since a handle
exported and imported inside one process proves nothing about IPC.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import multiprocessing as mp
import os
import traceback
from time import perf_counter


def _engine(protocol: str, device_name: str):
    from mooncake.engine import TransferEngine

    from nemo_rl.data_plane.adapters.transfer_queue_env import local_node_ip

    engine = TransferEngine()
    # "<ip>:0" exactly as TransferEngineClient does, not "": under P2PHANDSHAKE
    # this string *is* the segment name peers address, so an empty one leaves
    # the process unaddressable and every read fails with rc=-1 -- including
    # over RDMA, which is how this harness was caught being wrong. Port 0 lets
    # the engine choose, which co-located clients need.
    rc = engine.initialize(
        f"{local_node_ip()}:0", "P2PHANDSHAKE", protocol, device_name
    )
    if rc != 0:
        raise RuntimeError(f"initialize failed rc={rc} protocol={protocol!r}")
    return engine


def _endpoint(engine) -> str:
    from nemo_rl.data_plane.adapters.transfer_queue_env import local_node_ip

    return f"{local_node_ip()}:{engine.get_rpc_port()}"


_CUDA_MEMCPY_DEFAULT = 4  # cudaMemcpyDefault: infers direction, needs UVA


def _memcpy(dst: int, src: int, nbytes: int) -> None:
    """Move bytes to or from an engine-managed buffer.

    Not ``write_bytes_to_buffer``/``read_bytes_from_buffer``: those are a plain
    host ``memcpy`` (transfer_engine_py.h:173). Under protocol="nvlink" the
    allocator hands back *device* memory, so the host memcpy segfaults and takes
    the process down with no Python exception -- it looks like a hang, then a
    queue timeout. cudaMemcpyDefault serves host and device buffers alike, so
    the same harness works for the rdma control and the nvlink arms.
    """
    cudart = ctypes.CDLL("libcudart.so")
    rc = cudart.cudaMemcpy(
        ctypes.c_void_p(dst),
        ctypes.c_void_p(src),
        ctypes.c_size_t(nbytes),
        ctypes.c_int(_CUDA_MEMCPY_DEFAULT),
    )
    if rc != 0:
        raise RuntimeError(f"cudaMemcpy failed rc={rc} ({nbytes} bytes)")


def producer(protocol: str, device_name: str, nbytes: int, out, gate) -> None:
    try:
        import torch

        torch.cuda.set_device(0)
        engine = _engine(protocol, device_name)
        ptr = engine.allocate_managed_buffer(nbytes)
        if not ptr:
            raise RuntimeError(f"allocate_managed_buffer({nbytes}) returned 0")

        # One full-size buffer, tiled in place. Building `bytes(...) * n` and
        # then copying it into a ctypes buffer allocates 2x the payload (1GB at
        # --mb 512) and costs ~494ms; this is ~188ms and half the peak RSS.
        host = ctypes.create_string_buffer(nbytes)
        mv = memoryview(host).cast("B")
        tile = bytes(range(256)) * 4096
        for off in range(0, nbytes, len(tile)):
            mv[off:off + len(tile)] = tile
        started = perf_counter()
        _memcpy(ptr, ctypes.addressof(host), nbytes)
        write_ms = (perf_counter() - started) * 1e3

        out.put(
            {
                "role": "producer",
                "endpoint": _endpoint(engine),
                "ptr": ptr,
                "nbytes": nbytes,
                "write_ms": write_ms,
                "digest": hashlib.sha256(host).hexdigest()[:16],
            }
        )
        gate.wait()  # keep the buffer alive until the consumer has read it
        engine.free_managed_buffer(ptr, nbytes)
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

        torch.cuda.set_device(1)
        engine = _engine(protocol, device_name)
        nbytes = published["nbytes"]
        dst = engine.allocate_managed_buffer(nbytes)
        if not dst:
            raise RuntimeError(f"allocate_managed_buffer({nbytes}) returned 0")

        # Repeat, and report every iteration. The first read pays openSegment,
        # the handle import and the address mapping; only later ones show steady
        # state. Cross-node measured 7.25 GB/s against 104 GB/s same-node on a
        # single read, and a one-time setup cost inside the timed region is one
        # of the few explanations that does not require a hidden fallback path.
        # cudaDeviceSynchronize after each read, or the timing is fiction:
        # the NVLink transport submits cudaMemcpyAsync and transfer_sync_read
        # returns before completion. Unsynchronized, repeats measured 0.00-0.02ms
        # for 512MB -- 33 to 149 TB/s, which no fabric does. Digests still
        # matched only because the verification copy synchronizes the device.
        repeats = int(os.environ.get("NVLINK_REPEAT", "5"))
        laps = []
        for _ in range(repeats):
            started = perf_counter()
            rc = engine.transfer_sync_read(
                published["endpoint"], dst, published["ptr"], nbytes
            )
            torch.cuda.synchronize()
            laps.append((perf_counter() - started) * 1e3)
            if rc < 0:
                raise RuntimeError(f"transfer_sync_read failed rc={rc}")
        # Report the steady-state lap, not laps[0]. The first read pays
        # openSegment, the fabric handle import and the address mapping -- 76ms
        # cross-node against 0.67ms warm -- so quoting it as "the" bandwidth
        # understates the transport by ~100x. Both numbers are carried out:
        # the cold lap is the per-peer setup cost and is real, just not the
        # throughput.
        cold_ms = laps[0]
        warm_ms = min(laps[1:]) if len(laps) > 1 else laps[0]

        host = ctypes.create_string_buffer(nbytes)
        _memcpy(ctypes.addressof(host), dst, nbytes)
        # hashlib over the ctypes buffer directly: ``.raw`` materialises a
        # second full-size copy (~254ms for 512MB), which costs about as much
        # as the hash it feeds.
        out.put(
            {
                "role": "consumer",
                "endpoint": _endpoint(engine),
                "cold_ms": cold_ms,
                "warm_ms": warm_ms,
                "laps_ms": laps,
                "digest": hashlib.sha256(host).hexdigest()[:16],
            }
        )
        engine.free_managed_buffer(dst, nbytes)
    except Exception:
        out.put({"role": "consumer", "error": traceback.format_exc()})
    finally:
        gate.set()


def _report(first: dict, outcome: dict, mb: int) -> int:
    """One reporting path for both the same-node and cross-node arms.

    Written once because the two used to drift: the split-role branch dropped
    ``write_ms`` and formatted the same numbers differently, so the two arms of
    one experiment were not directly comparable.
    """
    match = outcome["digest"] == first["digest"]
    warm_MBps = mb / (outcome["warm_ms"] / 1e3)
    cold_MBps = mb / (outcome["cold_ms"] / 1e3)
    print(f"put(alloc+write): {first['write_ms']:.2f}ms  ptr=0x{first['ptr']:x}")
    print(f"read warm:        {outcome['warm_ms']:.2f}ms  {warm_MBps:.0f} MB/s"
          "   <- steady state, the transport's throughput")
    print(f"read cold:        {outcome['cold_ms']:.2f}ms  {cold_MBps:.0f} MB/s"
          "   <- first touch: openSegment + handle import + mapping, per peer")
    print("laps_ms=" + " ".join(f"{v:.2f}" for v in outcome["laps_ms"]))
    print(f"producer_endpoint={first['endpoint']} "
          f"consumer_endpoint={outcome['endpoint']}")
    print(f"digest_match={match}  ({first['digest']} vs {outcome['digest']})")
    print("RESULT:", "PASS" if match else "FAIL")
    return 0 if match else 1


class _FileQueue:
    """put/get over a shared file, so the two roles can live on different nodes.

    Same surface as the multiprocessing queue the single-node path uses, which
    is the point: ``producer`` and ``consumer`` are unchanged, and the only
    difference between a same-node and a cross-node run is which queue they get.
    Cross-node is the measurement that matters -- both processes on one host
    exercise NVLink within a chassis, while MNNVL fabric is supposed to span the
    whole NVL72 domain, and only a two-node run can tell those apart.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def put(self, payload: dict) -> None:
        import json
        import pathlib

        tmp = f"{self._path}.tmp"
        pathlib.Path(tmp).write_text(json.dumps(payload))
        os.replace(tmp, self._path)  # atomic, so a reader never sees a partial write

    def get(self, timeout: float = 90):
        import json
        import pathlib
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            if pathlib.Path(self._path).exists():
                return json.loads(pathlib.Path(self._path).read_text())
            time.sleep(0.25)
        raise TimeoutError(f"no payload at {self._path} after {timeout}s")


class _FileGate:
    """The producer must hold its buffer until the consumer has read it."""

    def __init__(self, path: str) -> None:
        self._path = path

    def set(self) -> None:
        import pathlib

        pathlib.Path(self._path).touch()

    def wait(self, timeout: float = 120) -> None:
        import pathlib
        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            if pathlib.Path(self._path).exists():
                return
            time.sleep(0.25)
        # Raise, like _FileQueue.get: a producer whose consumer never started
        # would otherwise free its buffer, print "published ..." and exit 0.
        raise TimeoutError(f"gate {self._path} never opened after {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="nvlink")
    parser.add_argument("--mb", type=int, default=64)
    parser.add_argument("--device-name", default="")
    parser.add_argument(
        "--role",
        default="both",
        choices=("both", "producer", "consumer"),
        help="'both' forks two local processes; the split roles are for a "
        "two-node srun, where --rendezvous carries the published metadata",
    )
    parser.add_argument("--rendezvous", default=None,
                        help="shared-filesystem prefix, required unless --role both")
    args = parser.parse_args()

    if args.role != "both":
        if not args.rendezvous:
            parser.error("--role producer/consumer requires --rendezvous")
        inbox = _FileQueue(f"{args.rendezvous}.meta")
        gate = _FileGate(f"{args.rendezvous}.done")
        nbytes = args.mb * 2**20
        print(f"role={args.role} protocol={args.protocol} payload={args.mb}MB")
        if args.role == "producer":
            producer(args.protocol, args.device_name, nbytes, inbox, gate)
            posted = inbox.get(timeout=5)
            if "error" in posted:
                print(f"PRODUCER FAILED:\n{posted['error']}")
                return 1
            print(f"published ptr=0x{posted['ptr']:x} endpoint={posted['endpoint']}")
            return 0
        results = _FileQueue(f"{args.rendezvous}.result")
        consumer(args.protocol, args.device_name, inbox, results, gate)
        outcome = results.get(timeout=5)
        if "error" in outcome:
            print(f"CONSUMER FAILED:\n{outcome['error']}")
            return 1
        return _report(inbox.get(timeout=5), outcome, args.mb)

    nbytes = args.mb * 2**20
    print(f"protocol={args.protocol} payload={args.mb}MB "
          f"device_name={args.device_name or '(auto)'}")
    for var in ("MC_FORCE_MNNVL", "MC_INTRANODE_NVLINK", "PYTORCH_CUDA_ALLOC_CONF"):
        print(f"  {var}={os.environ.get(var, '(unset)')}")

    ctx = mp.get_context("spawn")
    to_consumer, results, gate = ctx.Queue(), ctx.Queue(), ctx.Event()
    procs = [
        ctx.Process(target=producer,
                    args=(args.protocol, args.device_name, nbytes, to_consumer, gate)),
        ctx.Process(target=consumer,
                    args=(args.protocol, args.device_name, to_consumer, results, gate)),
    ]
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

    print(f"\nalloc+write: {first['write_ms']:.2f}ms  ptr=0x{first['ptr']:x}  "
          f"endpoint={first['endpoint']}")

    outcome = results.get(timeout=90)
    for p in procs:
        p.join(timeout=120)
    if "error" in outcome:
        print(f"\nCONSUMER FAILED:\n{outcome['error']}")
        return 1

    return _report(first, outcome, args.mb)


if __name__ == "__main__":
    raise SystemExit(main())
