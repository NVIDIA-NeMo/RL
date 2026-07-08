"""Buffer-registration microbenchmark on the REAL Qwen3-30B-A3B tensor set.

Mirrors modelexpress weight_update.py: per-tensor register (old) vs a single
arena region (new). Run with mode=pertensor|arena as argv[1] (one 1-GPU pod).

Verified 2026-07-08 GB200 (18,867 tensors / 61 GB):
  per-tensor 0.90s  ->  arena 0.016s  (~56x, one region).
"""
import sys, json, glob, time, torch
from safetensors import safe_open

MODE = sys.argv[1] if len(sys.argv) > 1 else "arena"
DEV = "cuda:0"
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
      "BOOL": torch.bool, "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8,
      "F8_E4M3": torch.float8_e4m3fn}

snap = glob.glob("/mnt/rl-workspace/kavink/hf-cache/hub/"
                 "models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/*/")[0]
idx = json.load(open(snap + "model.safetensors.index.json"))
specs = {}
for sh in sorted(set(idx["weight_map"].values())):
    with safe_open(snap + sh, framework="pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            specs[k] = (list(sl.get_shape()), DT.get(sl.get_dtype(), torch.bfloat16))
print(f"[{MODE}] real tensor set: {len(specs)} tensors")

from modelexpress.nixl_transfer import NixlTransferManager, is_nixl_available
assert is_nixl_available()
nixl = NixlTransferManager(agent_name=f"regbench-{MODE}", device_id=0, listen_port=0)
nixl.initialize()

if MODE == "pertensor":
    buffers = {n: torch.empty(s, dtype=d, device=DEV) for n, (s, d) in specs.items()}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nixl.register_tensors(buffers)
    torch.cuda.synchronize()
    print(f"RESULT pertensor register: {time.perf_counter()-t0:.3f}s  ({len(buffers)} calls)")
else:
    from modelexpress.vmm import VmmArena, CudaVmmBackend, use_arena, install_pluggable_allocator
    install_pluggable_allocator()
    arena = VmmArena(total_bytes=80 * (1024 ** 3), device=0, backend=CudaVmmBackend(device=0))
    buffers = {}
    with use_arena(arena, torch.device(DEV)):
        for n, (s, d) in specs.items():
            buffers[n] = torch.empty(s, dtype=d, device=DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    nixl.register_arena(arena, buffers)
    torch.cuda.synchronize()
    print(f"RESULT arena register: {time.perf_counter()-t0:.3f}s  "
          f"(1 region, {len(buffers)} tensors, {arena.used_bytes/1e9:.2f} GB)")
