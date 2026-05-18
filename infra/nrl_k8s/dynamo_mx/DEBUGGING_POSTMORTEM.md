# MX Refit on Dynamo VllmDecodeWorker — Debugging Postmortem

**Date:** 2026-05-13
**Branch context:** `dynamo-k8s-integration` (nemo-rl) + `jthomson04/tokenize-endpoint-merge-main-05-07` (dynamo)
**Status at write time:** Blocker isolated to UCX-CUDA memory-type classification inside vLLM v1's EngineCore Worker subprocess; cluster state cleaned back to baseline; one validated code change kept in place.

---

## TL;DR

We wired ModelExpress v2 mid-training weight refit into the Dynamo vLLM backend so a GRPO trainer can stream policy weights to the inference worker over NIXL RDMA without going through NCCL/IPC. The trainer side works end-to-end. The DGD worker side fails at NIXL VRAM registration with:

```
ucx_utils.cpp:581] memory is detected as host, check that UCX is configured with CUDA support
nixl_cu12._bindings.nixlNotFoundError: NIXL_ERR_NOT_FOUND
```

After exhausting plausible env-var / package-version workarounds, a `cuPointerGetAttribute` probe inserted directly in the worker extension's refit path proved that CUDA itself reports the buffer as `DEVICE`, but UCX's `ucp_mem_map` returns `HOST`. The mismatch is entirely inside UCX, and it only manifests inside vLLM v1's EngineCore Worker — a standalone NIXL-register-CUDA-buffer test in the same pod, same image, same UCX install, passes cleanly.

One code change is validated and kept (`receive_weights_scratch` migration that fixes a separate HF↔vLLM naming bug — see §6). Everything else is reverted.

---

## 1. Architecture under test

| Side | Image | Role |
|---|---|---|
| Trainer | `jwillthomson/nemo-rl-mx:latest` (nemo-rl base, nixl-cu12==1.1.0 pip wheel + modelexpress baked in) | `DTensorPolicyWorker.stream_weights_via_mx` publishes each rank's DTensor `to_local()` shards via NIXL to a ModelExpress server (`modelexpress-server.default.svc.cluster.local:8001`, Redis-backed) |
| MX server | `nvcr.io/nvidia/ai-dynamo/modelexpress-server:0.3.0` + Redis sidecar | Stores per-version source metadata; matches receivers to publishers |
| Inference | `jwillthomson/dynamo-arm-rl-tokenize-endpoint-91cdc71:latest` (dynamo, source-built UCX 1.20.1 + NIXL 0.10.1 at `/usr/local/ucx` and `/opt/nvidia/nvda_nixl`) | Custom `MxRefitWorkerExtension` injected into vLLM via `engine_args.worker_extension_cls`; polls MX server for new versions, calls `MxV2RefitReceiver.receive_weights_scratch` to RDMA-pull weights into temp CUDA buffers, then `model.load_weights` to merge |

Both pods sit on GB300 nodes with 8 RoCE NICs each. The trainer NIXL setup works (publishes 400 tensors per step, confirmed in MX server logs). The blocker is exclusively on the DGD worker side.

---

## 2. The bug, as seen from the trainer log on each refit cycle

```
[trainer] ========================= Step 1/2 =========================
[trainer] (DTensorPolicyWorker) Initialized NIXL agent: nemo-rl-trainer-r0
[trainer] [mx] publish version=1                                    ← trainer publishes 400 tensors

[worker]  [mx-poller] new version detected: 1 (last=0)              ← poller sees it
[worker]  [mx] rank=0 chosen role=trainer src_rank=0 version=1
[worker]  Allocated 400 scratch buffers (8.82 GB)                   ← receive_weights_scratch allocates
[worker]  W ucx_utils.cpp:581] memory is detected as host, check that UCX is configured with CUDA support
[worker]  ERROR ... NIXL_ERR_NOT_FOUND                              ← FAIL
[worker]  [mx-poller] refit failed for version 1; will retry        ← infinite retry loop
```

Trainer side is happy throughout. Worker side fails every refit attempt with the same `VRAM detected as host` warning followed by `NIXL_ERR_NOT_FOUND` from `prep_xfer_dlist` (or the equivalent in NIXL 1.1.0 at `ucx_utils.cpp:592`).

---

## 3. Initial debugging — incorrect path (the qkv_proj red herring)

First failure mode encountered was actually **different** from the UCX one above. Initial attempts hit a `KeyError: 'layers.0.self_attn.qkqkv_proj.weight'` inside vLLM's `model.load_weights`.

Root cause: the trainer publishes tensors with **HF state_dict naming** (`q_proj`, `k_proj`, `v_proj` separate) but vLLM's internal params use **fused naming** (`qkv_proj`). The original extension registered vLLM's `named_parameters()` as receive buffers (Kavin's reference does the same) and then passed those vLLM-internal names back into `model.load_weights`, which has its own HF→fused merging logic — that took the input `q_proj` substring rule, applied it to a name that was already `qkv_proj`, and produced `qkqkv_proj` via `name.replace("q_proj", "qkv_proj")`.

**Fix (kept in place):** switched `MxRefitWorkerExtension.update_weights_via_mx` to `MxRefitReceiver.receive_weights_scratch`. The scratch path allocates temp CUDA buffers sized to the publisher's tensor list, RDMA-pulls into them, and yields `(hf_name, tensor)` pairs that `model.load_weights` consumes correctly. From modelexpress's docstring:
> This is the correct approach when the source (trainer) publishes HuggingFace-format weights but the target (vLLM) uses fused internal parameter names.

Shape info comes from the v2 candidate's `registry["tensors"]` (the publisher's `TensorDescriptorV2` list with `global_shape`).

This fix is structurally validated — every subsequent refit attempt allocates the 8.82GB scratch buffers cleanly and gets as far as NIXL transport before failing. The `qkqkv_proj` KeyError never recurs.

---

## 4. Real bug discovery — UCX rejects VRAM registration

After the qkv_proj fix, the next failure surfaced: NIXL `prep_xfer_dlist` returns `NIXL_ERR_NOT_FOUND`, preceded by:

```
ucx_utils.cpp:581] memory is detected as host, check that UCX is configured with CUDA support
```

This refers to NIXL's own helper in `ai-dynamo/nixl/src/plugins/ucx/ucx_utils.cpp`:

```cpp
ucp_mem_query(mem.memh, &attr);
if (attr.mem_type == UCS_MEMORY_TYPE_HOST) {
    NIXL_ERROR << "VRAM memory is detected as host by UCX. "
                  "UCX is likely not configured with CUDA support. "
                  "VRAM registration cannot proceed.";
    return -1;
}
```

So UCX maps a CUDA pointer, then UCX itself classifies it as HOST. NIXL bails.

---

## 5. The investigation that went sideways (UCX-install layer)

### 5a. LD_DEBUG=libs trace of the EngineCore subprocess

Showed multiple UCX plugin modules failing to dlopen:

```
/usr/local/ucx/lib/ucx/libucm_cuda.so.0: error: symbol lookup error: undefined symbol: ucs_module_global_init (fatal)
/usr/local/ucx/lib/ucx/libucm_cuda.so.0: error: symbol lookup error: undefined symbol: cuMemFreeHost_v2 (fatal)
/usr/local/ucx/lib/ucx/libuct_ib_efa.so.0: error: symbol lookup error: undefined symbol: ucs_module_global_init (fatal)
/usr/local/ucx/lib/ucx/libuct_ib.so.0: error: symbol lookup error: undefined symbol: ucs_module_global_init (fatal)
/usr/local/ucx/lib/ucx/libuct_rdmacm.so.0: error: symbol lookup error: undefined symbol: ucs_module_global_init (fatal)
/usr/local/ucx/lib/ucx/libuct_cma.so.0: error: symbol lookup error: undefined symbol: ucs_module_global_init (fatal)
```

`ucs_module_global_init` is defined in `libuct_cuda.so` + `libuct_ib_mlx5.so` but NOT in `libucs.so` — and `libucm_cuda.so`'s DT_NEEDED doesn't list `libuct_cuda.so`. With `RTLD_NOW`, the dynamic linker can't resolve the symbol → libucm_cuda fails fatally → UCX has no CUDA memhook installed → UCX defaults to HOST classification.

This looked like THE answer. It was a red herring.

### 5b. Fixes tried, all unsuccessful

| Attempt | Outcome |
|---|---|
| `UCX_TLS=cuda_copy,cuda_ipc,rc,sm,self` | no change |
| `UCX_MEMTYPE_CACHE=n` | no change |
| `UCX_MODULES_DIR` pointing at wheel UCX dir | no change |
| `LD_PRELOAD=libuct_cuda.so.0` (force RTLD_GLOBAL so libucm_cuda finds the symbol) | LD_DEBUG confirms libuct_cuda loaded RTLD_GLOBAL; libucm_cuda still fails — UCX module loader uses RTLD_DEEPBIND-style isolation |
| patchelf libucm_cuda.so to add `libuct_cuda.so.0` to DT_NEEDED + compile a `cuMemFreeHost_v2 → cuMemFreeHost` shim .so, overlay all of `/usr/local/ucx/lib/ucx/` via Lustre volumeMount | DGD CRD v1alpha1 annotation schema accepts top-level `volumeMounts` (no subPath) but silently drops `extraPodSpec.mainContainer.volumeMounts` (subPath-capable), so the overlay couldn't be delivered |
| `pip uninstall nixl nixl-cu12 && pip install nixl==1.1.0 nixl-cu12==1.1.0` (replace with the trainer-side wheel) | Wheel installs cleanly, UCX backend instantiates without error — but the `VRAM detected as host` failure persists at the same callsite |
| Same wheel + `UCX_MODULES_DIR` at wheel's UCX dir + strip `/usr/local/ucx` from `LD_LIBRARY_PATH` + `UCX_MEMTYPE_CACHE=n` | Same failure |

Wheel-swap experiment was the most interesting: the trainer-side image uses the exact same `nixl-cu12==1.1.0` pip wheel without issue. Installing the same wheel on the worker side produced the same failure mode. So:

**Conclusion from §5: the UCX install is not the only thing wrong.** Even a known-working UCX (the wheel's) fails the same way once vLLM is in the picture.

---

## 6. Isolation: standalone NIXL works, vLLM-hosted NIXL doesn't

To separate "is this a dynamo-image problem" from "is this a vLLM problem", patched the DGD worker container `command` to run a small Python diagnostic instead of `python -m dynamo.vllm`, then `sleep infinity`. Same image, same env, same nixl install, no vLLM:

```python
# mx_diagnostic.py — distilled essence
import torch, nixl_cu12
torch.cuda.set_device(0); _ = torch.zeros(8, device="cuda:0"); torch.cuda.synchronize()
from nixl_cu12 import nixl_agent, nixl_agent_config
agent = nixl_agent("mx-diag-r0", nixl_agent_config(backends=["UCX"]))
buf = torch.empty(1024, dtype=torch.float32, device="cuda:0")
descs = agent.register_memory([buf], backends=["UCX"])
print("PASS")
```

Result:
```
[mx-diag] T1: PASS (38.4ms) descs=nixlRegDList
```

Then a stage-2 diagnostic added more cases all in the same process:

| Test | Description | Result |
|---|---|---|
| T1 | `torch.cuda` init + NIXL register CUDA tensor | **PASS (38ms)** |
| T2 | T1 + `import vllm` before register | **PASS** |
| T3 | T1 in fresh `multiprocessing.spawn` child (mimics vLLM v1 EngineCore subprocess startup) | **PASS** |
| T4 | T3 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (vLLM's allocator config) | **PASS** (pointers 0x340000000/0x340400000 — expandable segments active) |

All four pass. The dynamo image's UCX install — even with the libucm_cuda dlopen failure that the LD_DEBUG trace showed — does successfully register CUDA buffers via some fallback path in this code path.

So the bug only manifests **inside the actual vLLM v1 EngineCore Worker subprocess** that's spawned by `python -m dynamo.vllm` after loading the model weights, the KV cache, and all of vLLM's CUDA infrastructure. Our scratch buffer (allocated by `torch.empty(..., device="cuda:0")` in `MxRefitReceiver.receive_weights_scratch`) lives in vLLM's torch caching-allocator pool, but our standalone test buffer does too, and that works.

---

## 7. Decisive probe: CUDA's view vs UCX's view, inside the real vLLM context

Inserted a `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_MEMORY_TYPE)` probe directly in `MxRefitWorkerExtension.update_weights_via_mx`, right before the receiver call, using ctypes against `libcuda.so.1`. The probe allocates its own fresh `torch.empty(1024, device=self.device)` and queries CUDA directly.

```
[mx-probe] fresh tensor ptr=0xed9b7c855c00 cuPointerGetAttribute(MEMORY_TYPE):
    rc=0 memtype=DEVICE; context rc=0 ctx_handle=0x28caa8b0; torch.device=cuda:0
```

Microseconds later, on essentially the same allocator pool:
```
W ucx_utils.cpp:581] memory is detected as host
ERROR ... NIXL_ERR_NOT_FOUND
```

**This is the root cause as observable from outside UCX:** the CUDA driver itself returns `memtype=DEVICE` (return code 0, valid context handle), but UCX's `ucp_mem_query` after `ucp_mem_map` returns `UCS_MEMORY_TYPE_HOST`. The mismatch lives entirely inside UCX.

`UCX_MEMTYPE_CACHE=n` does not fix it — UCX's `ucp_mem_map` path that NIXL hits doesn't honor that flag (it uses internal classification based on its memhooks; `MEMTYPE_CACHE` only affects a different `ucp_memtype_lookup` path).

---

## 8. Active hypothesis on why standalone works and vLLM doesn't

Both contexts have the same broken `/usr/local/ucx` install. The diff likely comes down to which UCX code path `ucp_mem_map` takes when classifying memory:

- In the standalone case (clean process state), `ucp_mem_map` likely falls back to calling `cuPointerGetAttribute` directly → returns DEVICE → registration succeeds.
- In the vLLM EngineCore Worker case, after vLLM has done extensive CUDA allocations through torch's caching allocator, KV cache setup (potentially with `cuMemMap`/`cuMemAddressReserve` for expandable segments), and possibly its own NIXL connector init for KV-cache transfer (`dynamo.nixl_connect`) — UCX's internal state has been populated such that `ucp_mem_map` takes a memhook-only path and never falls back to a CUDA-driver query.

Without working memhooks (libucm_cuda failed to load), the memhook-only path returns HOST for everything.

This is a UCX-build problem first, vLLM-interaction problem second. Fixing the UCX build to make libucm_cuda dlopen successfully would almost certainly fix this — and is needed anyway for other reasons (libuct_ib, libuct_ib_efa, libuct_rdmacm all fail the same way).

---

## 9. What was kept after cleanup

| File | State |
|---|---|
| `dynamo/components/src/dynamo/vllm/mx_refit/extension.py` (and `dynamo-dev/` Lustre mirror) | **Kept**: `receive_weights_scratch` migration with `tensor_shapes` from V2 registry. This is the qkv_proj fix. Debug probe removed. |
| `nemo-rl/infra/nrl_k8s/dynamo_mx/bootstrap_mx.sh` | **Reverted to PYTHONPATH-only**. Pip swap, LD_PRELOAD, UCX env overrides all removed. |
| `nemo-rl/examples/nemo_gym/grpo_workplace_assistant_dynamo_mx.yaml` + paired infra YAML + DGD manifest | **Kept**, unchanged from earlier session work — all part of plan typed-swimming-ladybug |
| DGD `jothomson-dynamo-wpa-mx` env vars | **Reverted to 4 baseline entries** (DYN_HEALTH_CHECK_ENABLED, HF_HOME, DYN_MX_REFIT_ENABLED, MODEL_EXPRESS_URL) |
| `modelexpress/.../nixl_transfer.py` | **Reverted** to upstream (bytes→str decode patch removed) |
| `/mnt/rl-workspace/jothomson/dynamo_mx_ucx_patch/`, `mx_diagnostic*.py`, `dynamo_mx_ucx_overlay/` | **Deleted** |

Worker pod is alive and idle (no fresh source to poll). Trainer is stopped. RayCluster, DGD, MX server all up and waiting.

---

## 10. Recommended next steps (in priority order)

1. **Fix the dynamo worker image's UCX build.** Pin `NIXL_UCX_REF` to a known-good tag (probably the one the `nixl-cu12==1.1.0` pip wheel uses internally), or fix the configure flags so libucm_cuda's DT_NEEDED lists libuct_cuda. This is needed regardless — the libucm_cuda load failure is a real bug that masks UCX-CUDA capability for many workloads, not just ours.
2. **If image rebuild isn't immediate, sidestep UCX entirely**: reuse `dynamo.nixl_connect` (vLLM's built-in NIXL connector for KV-cache transfer) for refit too. It's known to work in this image+vLLM combo. Architecturally it's also cleaner since both refit and KV-cache transfer would use the same NIXL agent.
3. **If neither is feasible**: fall back to an IPC ZMQ refit path (the colocated transport NeMo-RL already supports). Loses RDMA throughput but unblocks correctness. Acceptable for a smoke; not for the bigger MoE runs.

---

## 11. Relevant commits and files

- Code fix (qkv_proj): `dynamo/components/src/dynamo/vllm/mx_refit/extension.py:267-318` — `receive_weights_scratch` migration with `tensor_shapes` built from `chosen.registry["tensors"]`
- Trainer side (cherry-pick): nemo-rl @ `2e5307d6 feat: ModelExpress + NIXL RDMA weight-sync (v2 path) for non-colocated refit`
- Plan of record: `/mnt/rl-workspace/jothomson/.claude/plans/typed-swimming-ladybug.md`
- NIXL `ucx_utils.cpp` lines: `571-585` in 0.10.1, `571-592` in 1.1.0
- Dynamo Dockerfile (UCX build): `dynamo/container/dynamo-planner-cuda12.9-arm64-rendered.Dockerfile:322-385` (`NIXL_UCX_REF=v1.20.x`, `NIXL_REF=0.10.1`)
- LD_DEBUG capture of the failing dlopen chain: `/tmp/ld-debug-pull/ld-debug.953` (EngineCore PID 953 at the time of capture; file lives only on the dev pod)
