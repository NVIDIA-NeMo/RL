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
from safetensors import safe_open
from modelexpress.nixl_transfer import NixlTransferManager, is_nixl_available

ROLE = sys.argv[1]
D = "/mnt/rl-workspace/kavink/elastic_bench"
META, READY = f"{D}/pub_meta.pkl", f"{D}/pub_ready"
DEV = "cuda:0"
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
      "F8_E4M3": torch.float8_e4m3fn, "I64": torch.int64, "I32": torch.int32,
      "U8": torch.uint8, "BOOL": torch.bool}
N_RECV = int(os.environ.get("N_RECV", "4"))
RECV_ID = os.environ.get("RECV_ID", "0")
START_DELAY_S = float(os.environ.get("START_DELAY_S", "0"))


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


assert is_nixl_available()
os.makedirs(D, exist_ok=True)
specs = build_specs()
buffers = {n: torch.empty(s, dtype=d, device=DEV) for n, (s, d) in specs.items()}
total = sum(t.numel() * t.element_size() for t in buffers.values())
mgr = NixlTransferManager(agent_name=f"elastic-{ROLE}-{RECV_ID}", device_id=0, listen_port=0)
mgr.initialize()

if ROLE == "publisher":
    for t in buffers.values():
        t.normal_()
    torch.cuda.synchronize()
    mgr.register_tensors(buffers)
    meta = {"agent_metadata": base64.b64encode(mgr.nixl_metadata).decode(),
            "descriptors": mgr.tensor_descriptors}
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
    res = {"id": RECV_ID, "delay_s": START_DELAY_S, "launch_epoch": t_launch,
           "pull_start_epoch": t_pull_start, "pull_end_epoch": t_pull_end,
           "pull_dur_s": round(dur, 3), "gb": round(nbytes / 1e9, 2),
           "gbps": round(nbytes * 8 / dur / 1e9, 1), "tensors": ntensors}
    json.dump(res, open(f"{D}/result_{RECV_ID}.json", "w"))
    print(f"RESULT recv {RECV_ID}: {res}", flush=True)
