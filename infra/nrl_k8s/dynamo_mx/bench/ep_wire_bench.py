"""EP4 device-RDMA wire benchmark — true GPU->GPU transfer time for an EP-sharded
Megatron trainer, pulled with arena registration (one region/source) + 4-rail stripe.

Contrast with ep_tp_receiver.py, whose per-source ~48 Gbps is a HOST-comparison
artifact: it uses receive_weights_scratch (per-tensor NIXL registration + thousands
of tiny per-tensor RDMA ops + sequential per source), which can't saturate the rails.
This harness mirrors wire_bench.py (which hit 900 Gbps): register each source's
buffers as ONE arena region and issue one batched read, so we measure the wire.

Pairs with ep_publisher.py (EP=N Megatron publisher, N EP sources). For each source
we fetch its NIXL metadata + descriptors from the MX server, allocate matching device
buffers in a VMM arena, register the arena once, and time a single receive_from_source.
Reports per-source Gbps, the SEQUENTIAL total (sum of per-source device transfers), and
a CONCURRENT run (N threads, one NIXL agent + arena each) to show aggregate rail sharing.

Run on the vLLM WORKER image (full CUDA+IB NIXL) with the striped RDMA env
(MX_RDMA_NIC_PIN=stripe, UCX GPUDirect, UCX_CUDA_COPY_REG_WHOLE_ALLOC=off). Env:
MODEL_EXPRESS_URL, EP_SIZE (default 4).
"""
from __future__ import annotations
import os, socket, threading, time
import torch

from modelexpress import MxV2RefitReceiver
from modelexpress.nixl_transfer import NixlTransferManager, is_nixl_available
from modelexpress.types import TensorDescriptor
from modelexpress.refit_receiver import _DTYPE_MAP
from modelexpress.vmm import VmmArena, CudaVmmBackend, use_arena, install_pluggable_allocator

MODEL = os.environ.get("MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")
EP_SIZE = int(os.environ.get("EP_SIZE", "4"))
DEV = "cuda:0"
os.environ.setdefault("UCX_CUDA_COPY_REG_WHOLE_ALLOC", "off")


def _descriptors_for(client, ref):
    """Fetch a source's NIXL metadata + real (non-sidecar) tensor descriptors."""
    mr = client.get_metadata(mx_source_id=ref.mx_source_id, worker_id=ref.worker_id)
    if not mr.found:
        raise RuntimeError(f"source {ref.mx_source_id}/{ref.worker_id} not found")
    w = mr.worker
    desc = [TensorDescriptor(name=t.name, addr=t.addr, size=t.size,
                             device_id=t.device_id, dtype=t.dtype)
            for t in w.tensors if not t.name.startswith("__mx_") and t.size > 0]
    return w.nixl_metadata, desc


def _pull_one(ep, meta, desc, out):
    """Arena-register matching buffers for one EP source and time one batched read."""
    mgr = NixlTransferManager(agent_name=f"ep{ep}-wire-{socket.gethostname()}",
                              device_id=0, listen_port=0)
    mgr.initialize()
    arena = VmmArena(total_bytes=40 * (1024 ** 3), device=0, backend=CudaVmmBackend(device=0))
    bufs = {}
    with use_arena(arena, torch.device(DEV)):
        for td in desc:
            dt = _DTYPE_MAP.get(td.dtype, torch.bfloat16)
            numel = td.size // torch.tensor([], dtype=dt).element_size()
            bufs[td.name] = torch.empty(numel, dtype=dt, device=DEV)
    torch.cuda.synchronize()
    mgr.register_arena(arena, bufs)
    torch.cuda.synchronize()
    nb, nt, dur = mgr.receive_from_source(meta, desc, timeout_seconds=600)
    torch.cuda.synchronize()
    out[ep] = (nb, nt, dur)


def main() -> int:
    assert is_nixl_available(), "NIXL not available"
    rcv = MxV2RefitReceiver(
        agent_name=f"{socket.gethostname()}-ep{EP_SIZE}-wire", device_id=0,
        mx_server_url=os.environ["MODEL_EXPRESS_URL"], worker_rank=0)
    rcv.initialize(model_tensors=None)
    client = rcv._receiver._client

    print(f"[wire] discovering {EP_SIZE} EP sources for {MODEL} ...", flush=True)
    deadline = time.time() + 120
    by_ep = {}
    while time.time() < deadline:
        cands = rcv.discover_v2_sources(model_name=MODEL, min_version=1,
                                        same_rank_only=False, include_replicas=True)
        by_ep = {}
        for c in sorted([c for c in cands if c.megatron_meta is not None],
                        key=lambda x: -x.updated_at):
            by_ep.setdefault(c.megatron_meta.ep_rank, c)
        if len(by_ep) >= EP_SIZE:
            break
        time.sleep(3)
    if len(by_ep) < EP_SIZE:
        print(f"[wire] ERROR: only {len(by_ep)}/{EP_SIZE} EP sources"); return 3

    sources = []
    for ep in sorted(by_ep):
        meta, desc = _descriptors_for(client, by_ep[ep].ref)
        nbytes = sum(d.size for d in desc)
        sources.append((ep, meta, desc))
        print(f"  ep{ep}: {len(desc)} tensors, {nbytes/1e9:.2f} GB", flush=True)

    install_pluggable_allocator()

    # ---- SEQUENTIAL: one source at a time (each arena-registered + one batched read) ----
    print(f"\n[wire] SEQUENTIAL device->device pulls (arena + stripe) ...", flush=True)
    seq = {}
    t0 = time.perf_counter()
    for ep, meta, desc in sources:
        _pull_one(ep, meta, desc, seq)
        nb, nt, dur = seq[ep]
        print(f"  ep{ep}: {nb/1e9:.2f} GB in {dur:.3f}s -> {nb*8/dur/1e9:.1f} Gbps", flush=True)
    seq_wall = time.perf_counter() - t0
    seq_bytes = sum(v[0] for v in seq.values())
    seq_dev = sum(v[2] for v in seq.values())
    print(f"[wire] SEQUENTIAL: {seq_bytes/1e9:.2f} GB; device transfer sum {seq_dev:.3f}s "
          f"= {seq_bytes*8/seq_dev/1e9:.1f} Gbps; wall {seq_wall:.3f}s", flush=True)

    # ---- CONCURRENT: N threads, one NIXL agent + arena each (rail sharing) ----
    print(f"\n[wire] CONCURRENT device->device pulls ({EP_SIZE} threads) ...", flush=True)
    conc = {}
    ths = [threading.Thread(target=_pull_one, args=(ep, meta, desc, conc))
           for ep, meta, desc in sources]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    conc_wall = time.perf_counter() - t0
    conc_bytes = sum(v[0] for v in conc.values())
    for ep in sorted(conc):
        nb, nt, dur = conc[ep]
        print(f"  ep{ep}: {nb/1e9:.2f} GB in {dur:.3f}s -> {nb*8/dur/1e9:.1f} Gbps", flush=True)
    print(f"[wire] CONCURRENT: {conc_bytes/1e9:.2f} GB in {conc_wall:.3f}s wall "
          f"= {conc_bytes*8/conc_wall/1e9:.1f} Gbps aggregate", flush=True)

    print("\n==== EP4 WIRE SUMMARY ====", flush=True)
    print(f"total model bytes across {EP_SIZE} EP sources: {seq_bytes/1e9:.2f} GB", flush=True)
    print(f"sequential device-transfer time : {seq_dev:.3f}s "
          f"({seq_bytes*8/seq_dev/1e9:.1f} Gbps)", flush=True)
    print(f"concurrent wall time            : {conc_wall:.3f}s "
          f"({conc_bytes*8/conc_wall/1e9:.1f} Gbps aggregate)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
