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
from pathlib import Path
from safetensors import safe_open
from modelexpress.nixl_transfer import NixlTransferManager, is_nixl_available

ROLE = sys.argv[1]
EP_SIZE = int(os.environ.get("EP_SIZE", "1"))
EP_RANK = int(os.environ.get("EP_RANK", "0"))
D = os.environ.get("RESULT_DIR", "/mnt/rl-workspace/kavink/nixl_wire_bench")
META, READY, DONE = f"{D}/pub_meta.pkl", f"{D}/pub_ready", f"{D}/recv_done"
DEV = "cuda:0"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CHECKPOINT_BYTES = 61_064_245_248
STAGE_NAMES = (
    "control_discovery", "source_preparation", "setup_registration",
    "transfer_planning", "wire_transfer", "receive_sync", "transformation",
    "installation", "post_install", "rollout_readiness",
)
HF_SNAPSHOT = os.environ.get("HF_SNAPSHOT")
if not HF_SNAPSHOT:
    raise RuntimeError("HF_SNAPSHOT must name an immutable Qwen3-30B-A3B snapshot")
SNAPSHOT = Path(HF_SNAPSHOT)
RESULT_JSON = Path(os.environ.get("RESULT_JSON", f"{D}/receiver.json"))
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
      "F8_E4M3": torch.float8_e4m3fn, "I64": torch.int64, "I32": torch.int32,
      "U8": torch.uint8, "BOOL": torch.bool}


def _local_expert(name):
    m = re.search(r"experts\.(\d+)\.", name)
    if m is None:
        return True
    return int(m.group(1)) % EP_SIZE == EP_RANK


def resolve_snapshot():
    snap = SNAPSHOT.resolve()
    if "Qwen3-30B-A3B" not in str(snap):
        raise RuntimeError(f"Refusing non-30B snapshot: {snap}")
    index_path = snap / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(f"Missing safetensors index under {snap}")
    return snap, json.loads(index_path.read_text())


def build_specs(snap, idx):
    specs = {}
    for sh in sorted(set(idx["weight_map"].values())):
        with safe_open(str(snap / sh), framework="pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                dtype = sl.get_dtype()
                if dtype not in DT:
                    raise RuntimeError(f"Unsupported safetensors dtype {dtype}: {k}")
                specs[k] = (list(sl.get_shape()), DT[dtype])
    return specs


def load_checkpoint(snap, idx):
    buffers = {}
    for sh in sorted(set(idx["weight_map"].values())):
        with safe_open(str(snap / sh), framework="pt", device="cpu") as f:
            for name in f.keys():
                buffers[name] = f.get_tensor(name).to(DEV)
    return buffers


def artifact(tensor_source):
    return {"schema_version": "refit-stage-v1",
            "model": MODEL_ID, "checkpoint": snapshot.name,
            "checkpoint_bytes": CHECKPOINT_BYTES, "tensor_source": tensor_source}


def stages(wire_seconds):
    result = {
        name: {"status": "unavailable", "seconds": None} for name in STAGE_NAMES
    }
    result["wire_transfer"] = {
        "status": "available", "seconds": wire_seconds,
        "source": "NixlTransferManager.receive_from_source",
    }
    return result


assert is_nixl_available(), "NIXL not available"
os.makedirs(D, exist_ok=True)
snapshot, checkpoint_index = resolve_snapshot()
specs = build_specs(snapshot, checkpoint_index)
buffers = (
    load_checkpoint(snapshot, checkpoint_index)
    if ROLE == "publisher"
    else {n: torch.empty(s, dtype=d, device=DEV) for n, (s, d) in specs.items()}
)
total = sum(t.numel() * t.element_size() for t in buffers.values())
if total != CHECKPOINT_BYTES:
    raise RuntimeError(f"Expected {CHECKPOINT_BYTES} checkpoint bytes, got {total}")
print(f"[{ROLE}] {len(buffers)} tensors, {total/1e9:.2f} GB on {DEV}", flush=True)
mgr = NixlTransferManager(agent_name=f"wire-{ROLE}", device_id=0, listen_port=0)
mgr.initialize()

if ROLE == "publisher":
    torch.cuda.synchronize()
    mgr.register_tensors(buffers)
    meta = {"agent_metadata": base64.b64encode(mgr.nixl_metadata).decode(),
            "descriptors": mgr.tensor_descriptors, **artifact("safetensors")}
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
    result = {"bytes": nbytes, "tensors": ntensors, "pull_dur_s": dur,
              "wall_seconds": wall, "gbps": nbytes * 8 / dur / 1e9,
              "ep_size": EP_SIZE, "ep_rank": EP_RANK,
              "stages": stages(dur),
              **artifact("received_safetensors")}
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True))
    open(DONE, "w").write("1")
