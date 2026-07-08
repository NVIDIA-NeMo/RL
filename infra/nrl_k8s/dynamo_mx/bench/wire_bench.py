"""Cross-host NIXL wire benchmark on the REAL Qwen3-30B-A3B tensor set.

publisher: allocate + fill + register 61 GB, publish NIXL metadata to a PVC file, hold.
receiver : allocate + register matching buffers, pull all via one batched RDMA read,
           report GB/s + Gbps (striped across 4 RDMA NICs when MX_RDMA_NIC_PIN=stripe).

Two RDMA pods on different nodes (pod anti-affinity), each 1 GPU. Set EP_SIZE>1 on
the receiver to measure expert byte-pruning (pull only local experts).

Verified 2026-07-08 GB200, cross-node, 4-rail stripe, GPU->GPU:
  full 61 GB in 0.54s = 900 Gbps; EP=8 -> 10.33 GB in 0.170s (5.9x less, 3.2x faster).
"""
import os, sys, json, time, glob, base64, pickle, re, torch
from safetensors import safe_open
from modelexpress.nixl_transfer import NixlTransferManager, is_nixl_available

ROLE = sys.argv[1]
EP_SIZE = int(os.environ.get("EP_SIZE", "1"))
EP_RANK = int(os.environ.get("EP_RANK", "0"))
D = "/mnt/rl-workspace/kavink/nixl_wire_bench"
META, READY, DONE = f"{D}/pub_meta.pkl", f"{D}/pub_ready", f"{D}/recv_done"
DEV = "cuda:0"
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
      "F8_E4M3": torch.float8_e4m3fn, "I64": torch.int64, "I32": torch.int32,
      "U8": torch.uint8, "BOOL": torch.bool}


def _local_expert(name):
    m = re.search(r"experts\.(\d+)\.", name)
    if m is None:
        return True
    return int(m.group(1)) % EP_SIZE == EP_RANK


def build_specs():
    snap = glob.glob("/mnt/rl-workspace/kavink/hf-cache/hub/"
                     "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*/")[0]
    idx = json.load(open(snap + "model.safetensors.index.json"))
    specs = {}
    for sh in sorted(set(idx["weight_map"].values())):
        with safe_open(snap + sh, framework="pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                specs[k] = (list(sl.get_shape()), DT.get(sl.get_dtype(), torch.bfloat16))
    return specs


assert is_nixl_available(), "NIXL not available"
os.makedirs(D, exist_ok=True)
specs = build_specs()
buffers = {n: torch.empty(s, dtype=d, device=DEV) for n, (s, d) in specs.items()}
total = sum(t.numel() * t.element_size() for t in buffers.values())
print(f"[{ROLE}] {len(buffers)} tensors, {total/1e9:.2f} GB on {DEV}", flush=True)
mgr = NixlTransferManager(agent_name=f"wire-{ROLE}", device_id=0, listen_port=0)
mgr.initialize()

if ROLE == "publisher":
    with torch.no_grad():
        for t in buffers.values():
            t.normal_()
    torch.cuda.synchronize()
    mgr.register_tensors(buffers)
    meta = {"agent_metadata": base64.b64encode(mgr.nixl_metadata).decode(),
            "descriptors": mgr.tensor_descriptors}
    pickle.dump(meta, open(META, "wb"))
    open(READY, "w").write("1")
    print("[publisher] registered + published; holding for receiver ...", flush=True)
    for _ in range(1200):
        if os.path.exists(DONE):
            break
        time.sleep(1)
    print("[publisher] done", flush=True)
else:
    mgr.register_tensors(buffers)
    print("[receiver] waiting for publisher ready ...", flush=True)
    for _ in range(1200):
        if os.path.exists(READY):
            break
        time.sleep(1)
    meta = pickle.load(open(META, "rb"))
    src_meta = base64.b64decode(meta["agent_metadata"])
    src_desc = meta["descriptors"]
    if EP_SIZE > 1:
        full = len(src_desc)
        src_desc = [d for d in src_desc if _local_expert(d.name)]
        print(f"[receiver] EP={EP_SIZE} rank={EP_RANK}: pulling local subset "
              f"{len(src_desc)}/{full} tensors", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nbytes, ntensors, dur = mgr.receive_from_source(src_meta, src_desc, timeout_seconds=600)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(f"RESULT wire: {nbytes/1e9:.2f} GB, {ntensors} tensors in {dur:.3f}s "
          f"-> {nbytes*8/dur/1e9:.1f} Gbps ({nbytes/dur/1e9:.1f} GB/s); wall {wall:.3f}s",
          flush=True)
    open(DONE, "w").write("1")
