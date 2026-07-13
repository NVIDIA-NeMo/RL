"""Elastic / straggler demo for MX refit (real Qwen3-30B-A3B tensor set).

Contrast with NCCL: a collective broadcast has an implicit barrier — every rank
must arrive for the op to complete. MX refit is per-receiver P2P RDMA reads, so:
  - a receiver completes its refit independent of whether others have started
    (straggler isolation: a slow/late worker cannot delay a ready one), and
  - a worker can JOIN late and still pull the current weights (elastic).

Roles (each an RDMA pod on its own node; N receivers + 1 publisher):
  publisher: register 61 GB, publish NIXL metadata, hold until all receivers done.
  receiver : wait START_DELAY_S, pull, record its own timeline to a PVC json.
Env: N_RECV, RECV_ID, START_DELAY_S (stagger to inject stragglers / late joiners).

Verified 2026-07-08 GB200: r0/r1 (delay 0) + r2 (delay 20s) each pulled the full
61 GB at ~850-940 Gbps independently — completions fully staggered, no barrier,
the 20s-late worker got full bandwidth without perturbing the earlier finishers.
"""
import os, sys, json, time, glob, base64, pickle, torch
from pathlib import Path
from safetensors import safe_open
from modelexpress.nixl_transfer import NixlTransferManager, is_nixl_available

ROLE = sys.argv[1]
D = os.environ.get("RESULT_DIR", "/mnt/rl-workspace/kavink/elastic_bench")
META, READY = f"{D}/pub_meta.pkl", f"{D}/pub_ready"
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
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
      "F8_E4M3": torch.float8_e4m3fn, "I64": torch.int64, "I32": torch.int32,
      "U8": torch.uint8, "BOOL": torch.bool}
N_RECV = int(os.environ.get("N_RECV", "4"))
RECV_ID = os.environ.get("RECV_ID", "0")
START_DELAY_S = float(os.environ.get("START_DELAY_S", "0"))


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


assert is_nixl_available()
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
mgr = NixlTransferManager(agent_name=f"elastic-{ROLE}-{RECV_ID}", device_id=0, listen_port=0)
mgr.initialize()

if ROLE == "publisher":
    torch.cuda.synchronize()
    mgr.register_tensors(buffers)
    meta = {"agent_metadata": base64.b64encode(mgr.nixl_metadata).decode(),
            "descriptors": mgr.tensor_descriptors, **artifact("safetensors")}
    pickle.dump(meta, open(META, "wb"))
    open(READY, "w").write("1")
    print(f"[publisher] {len(buffers)} tensors / {total/1e9:.1f} GB published; holding", flush=True)
    for _ in range(1800):
        if len(glob.glob(f"{D}/result_*.json")) >= N_RECV:
            break
        time.sleep(1)
    print(f"[publisher] {len(glob.glob(f'{D}/result_*.json'))} receivers done; exit", flush=True)
else:
    mgr.register_tensors(buffers)
    while not os.path.exists(READY):
        time.sleep(0.5)
    t_launch = time.time()
    if START_DELAY_S > 0:
        print(f"[recv {RECV_ID}] late/straggler start: sleeping {START_DELAY_S}s", flush=True)
        time.sleep(START_DELAY_S)
    meta = pickle.load(open(META, "rb"))
    src_meta = base64.b64decode(meta["agent_metadata"])
    src_desc = meta["descriptors"]
    torch.cuda.synchronize()
    t_pull_start = time.time()
    nbytes, ntensors, dur = mgr.receive_from_source(src_meta, src_desc, timeout_seconds=600)
    torch.cuda.synchronize()
    t_pull_end = time.time()
    res = {"id": RECV_ID, "receiver_id": int(RECV_ID),
           "delay_s": START_DELAY_S, "launch_epoch": t_launch,
           "pull_start_epoch": t_pull_start, "pull_end_epoch": t_pull_end,
           "pull_dur_s": round(dur, 3), "bytes": nbytes,
           "gb": round(nbytes / 1e9, 2),
           "gbps": round(nbytes * 8 / dur / 1e9, 1), "tensors": ntensors,
           "stages": stages(dur),
           **artifact("received_safetensors")}
    json.dump(res, open(f"{D}/result_{RECV_ID}.json", "w"))
    print(f"RESULT recv {RECV_ID}: {res}", flush=True)
