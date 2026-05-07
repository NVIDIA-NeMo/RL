# Mooncake-cpu backend — status

## TL;DR

`data_plane.backend = "mooncake_cpu"` (TCP transport) is **validated 1-node
in nemo-rl end-to-end with jagged wire** (smoke job 11633071, 5/5 steps
clean, FLOPS climb 12.94 → 264.26 — within noise of the padded baseline).
On multi-node it works in `data-plane-bench` (32→32 P2P at 13.96 GB/s,
48→16 reshard validated) but nemo-rl still has one **latent multi-node
gap** to close before flipping production runs to mooncake_cpu — see
"What's still fragile" below.

nemo-rl ships these changes:

1. `mooncake-transfer-engine` is a base dep (worker venvs auto-include it).
2. Adapter prepends the wheel's package dir to `$PATH` so `mooncake_master`
   is discoverable.
3. `MC_TCP_BIND_ADDRESS` set per-process to head IP.
4. **`MC_STORE_MEMCPY=0` set per-process** — bypasses Mooncake upstream
   issue [#1986](https://github.com/kvcache-ai/Mooncake/issues/1986)
   (`isLocalTransfer()` regression cross-process-derefs another actor's
   virtual address under TCP). Without it, the first `kv_batch_put`
   segfaults inside `mooncake::MemcpyWorkerPool::workerThread()`.
   PR #1995 is the upstream fix; not yet in our wheel.
5. `protocol: tcp` (the working transport — RDMA has separate native-IB
   issues; see Issue 1b).
6. `global_segment_size: 128 GiB`, `local_buffer_size: 16 GiB` (bench
   validated sizes).
7. **`ray.sub` runs the bench's `NETWORK_INIT_CMDS` block** at SLURM
   container startup as root, killing `avahi-autoipd`, telling
   NetworkManager to drop usb0, and looping a 2 s `ip addr flush` as a
   failsafe.
8. **`mooncake_cpu` keeps the jagged wire** — the original "nested-tensor
   pointer-arithmetic segfault" was actually #1986 in disguise. With
   `MC_STORE_MEMCPY=0` in place, jagged round-trips fine. All backends
   share `_PACK_JAGGED=True` and the Phase 1B bandwidth win.
9. **1D field round-trip** (KV-path-only): writer-side `_to_wire`
   unsqueezes any 1D tensor field to `(N, 1)`; `materialize` squeezes
   trailing 1 back. TQ's `extract_field_schema`
   (transfer_queue/metadata.py:171) silently unsqueezes 1D fields to
   record per-row shape `(1,)` in metadata, while `_generate_values`
   row-iterates the actual 1D tensor producing 0-dim per-row tensors —
   mooncake stores under metadata shape `(1,)` and returns `(1,)` on
   get, stack-merging to `(N, 1)` instead of `(N,)`. Simple backend
   uses a different ZMQ-routed path so the bug doesn't surface there.
   Both halves of the fix are gated on `_KV_PROMOTE_1D` (an
   independent flag from `_PACK_JAGGED`); flipped on by the mooncake_cpu
   adapter and by any future backend that goes through TQ's
   `KVStorageManager` (yuanrong, ray_storage_manager all inherit it).

## What's still fragile

**Multi-node `MC_TCP_BIND_ADDRESS` propagation.** Even with our `ray.sub`
network-init block, smoke job 11630793 showed Ray-spawned
`MegatronPolicyWorker` actors **still binding to 169.254.3.1** for their
Mooncake TCP RPC listener. The 1-node smoke worked because all 8 ranks
were loopback-routable on the same host. On 2+-node jobs, peers across
hosts cannot reach each other's 169.254 RPC address and the run will
hang / 404 / segfault.

**Fix path** (~5 LoC): extend
`_patch_tq_actor_runtime_env` (in
`nemo_rl/data_plane/adapters/transfer_queue.py`) to inject
`env_vars={"MC_TCP_BIND_ADDRESS": <head_ip>, ...}` alongside the existing
`pip` injection. Mooncake's `engine.so` honors `MC_TCP_BIND_ADDRESS` for
client *registration* even when the C++ listener still scans
`getifaddrs()`. Per the bench's debug doc, that's enough on the
registration side to avoid the 169.254 bind for the addresses other peers
will look up.

This is **not needed for 1-node mooncake_cpu**. It IS needed before any
multi-node mooncake_cpu job.

## What's broken upstream (out of nemo-rl's scope)

- **Issue 1b**: Mooncake's RDMA transport doesn't handle native-IB GID
  routing (this cluster has native IB, not RoCE). RDMA mode hangs on
  `Failed to complete transfers after 60 seconds`. **TCP is the working
  path; RDMA stays parked.**

For the full debugging history see
`data-plane-bench/DEBUG_TQ_BACKENDS.md` (Issues 1, 1b) and
`data-plane-bench/PLAN_MOONCAKE_RDMA_FIX.md`.

## What's fixed in nemo-rl (committed)

1. **`mooncake-transfer-engine` is a base dep** in `pyproject.toml`, next to
   `TransferQueue==0.1.6` and `tensordict`. Worker venvs built by
   `nemo_rl.utils.venvs.create_local_venv` (no extras) automatically pull it.

2. **`mooncake_master` discovery** — `nemo_rl/data_plane/adapters/transfer_queue.py`,
   `mooncake_cpu` branch:
   - Imports `mooncake`, resolves `<site-packages>/mooncake/` (where the
     wheel puts the binary).
   - Restores the `+x` bit if pip stripped it on extract.
   - Prepends that dir to `os.environ["PATH"]` before `tq.init()` so TQ's
     `subprocess.Popen(["mooncake_master", ...])` resolves.

3. **Configurable transport** — `_mooncake_transport_config()` defaults to
   TCP; RDMA via `MC_MOONCAKE_PROTOCOL=rdma`, optional `MC_MOONCAKE_DEVICE`.
   Bench notes RDMA is non-functional on this cluster's native InfiniBand
   fabric (Issue 1b); TCP is the working path.

4. **`_usb0_down()` retained for reference but documented as a no-op
   from Python** (Ray actors lack `CAP_NET_ADMIN`; APIPA is re-assigned by
   `avahi-autoipd` / NetworkManager within seconds). See its docstring.

## How the SLURM `NETWORK_INIT_CMDS` block works

Lifted from `data-plane-bench/ray.sub` and now in `ray.sub`. Runs at
container start in both `head_cmd` and `worker_cmd`:

```bash
# Kill avahi-autoipd: it reassigns 169.254.3.1 to usb0 even after flush.
pkill avahi-autoipd 2>/dev/null || true
if [ -f /run/avahi-autoipd.usb0.pid ]; then kill $(cat /run/avahi-autoipd.usb0.pid) 2>/dev/null || true; fi
# Tell NetworkManager to stop managing usb0 (so it doesn't re-bring it up).
nmcli device set usb0 managed no 2>/dev/null || true
# Bring usb0 down + remove its IP entirely (Mooncake's getifaddrs
# doesn't filter by IFF_UP — it picks any interface with an IP).
ifconfig usb0 0.0.0.0 2>/dev/null || true
ifconfig usb0 down 2>/dev/null || true
ip link set usb0 down 2>/dev/null || true
ip addr flush dev usb0 2>/dev/null || true
# Belt-and-suspenders: 2 s flush loop in case NM/avahi resurrects it.
{ while :; do
    pkill avahi-autoipd 2>/dev/null || true
    ifconfig usb0 0.0.0.0 2>/dev/null || true
    ifconfig usb0 down 2>/dev/null || true
    ip link set usb0 down 2>/dev/null || true
    ip addr flush dev usb0 2>/dev/null || true
    sleep 2
  done; } &
```

Each step is necessary; the bench's debug log
(`data-plane-bench/DEBUG_TQ_BACKENDS.md` Issue 1) walks through several
weaker attempts that all failed. ifconfig + ip variants both attempted
because the container set varies.

## Reproducer

```bash
# Cluster wrapper now ships NETWORK_INIT_CMDS in ray.sub.
sbatch run_mooncake_cpu_smoke.sh
# Inspect the smoke log; success = step 1 reached with non-NaN loss.
```

If the smoke still fails after this commit, the next likely failure is
inside Mooncake's wire codec when it sees a `torch.nested.nested_tensor`
(the bench validated mooncake_cpu against rectangular tensors only).
Mitigation in that case: either fall back to padded wire just for the
mooncake_cpu backend, or copy verl's
`(layout, [list_of_tensors])`-style encoder pattern from
`verl/protocol.py:247-293`.

## References

- `data-plane-bench/DEBUG_TQ_BACKENDS.md` — Issues 1 & 1b, full debug log
- `data-plane-bench/ray.sub` — proven `NETWORK_INIT_CMDS` block
- `data-plane-bench/PLAN_MOONCAKE_RDMA_FIX.md` — RDMA-side debugging (parked)
- `nemo_rl/data_plane/adapters/transfer_queue.py:_init_tq` — our mooncake_cpu branch
- `run_mooncake_cpu_smoke.sh` — minimal repro for the cluster-wrapper gap
- Smoke runs that confirmed each layer:
  - `11630039` — PATH fix verified (`mooncake_master` exec succeeds)
  - `11630086` — usb0 / 169.254.x failure mode (this is the cluster-wrapper TODO)
  - `11631109` — `MC_TCP_BIND_ADDRESS` per-process eliminates 169.254 binds
  - `11632698` — `MC_STORE_MEMCPY=0` resolves MemcpyWorkerPool segfault;
    surfaces the (N,1) shape mismatch in `extract_field_schema`
  - `11632821` — both fixes landed (padded wire): 5/5 steps clean,
    FLOPS 12.80 → 278.09, no segfaults, no shape errors
  - `11633071` — jagged wire re-enabled on mooncake_cpu: 5/5 steps clean,
    FLOPS 12.94 → 264.26 (within noise of padded). Confirms original
    "nested-tensor segfault" was Mooncake #1986, not jagged-specific
  - `11633583` — Llama 8B dtensor + seqpack 1-node: 5/5 steps clean,
    FLOPS 509.65 → 700.94. Validates a different framework (dtensor)
    on mooncake_cpu

## Multi-node + qwen3 30B fixes (all green)

The qwen3 30B + TP=2+SP + 2-node failure at step 3 was traced to **two
independent bugs** that surface together on this config:

1. **MC_TCP_BIND_ADDRESS env-var inheritance.** Driver set the env var
   via `os.environ.setdefault(...)`; Ray actor processes inherit env
   vars from the driver, so `setdefault` was a no-op on worker nodes
   and they announced the driver's IP. Peers connecting to the
   announced address hit a host where no such mooncake port existed
   ("Connection refused"). Fix: force-assign with
   `os.environ[...] = local_ip` per process, plus rename the helper
   to `_get_local_node_ip` to make the per-process semantic obvious.
2. **Worker write-back shape divergence under mcore SP.** mcore SP
   rounds the forward output's seq dim up to a multiple of TP, so
   `prev_logprobs` / `reference_policy_logprobs` arrive at the
   write-back site 1+ tokens wider than `max(meta.sequence_lengths)`.
   The strict shape check in `maybe_pack_jagged` left them rectangular
   at the SP-padded width while `input_ids` re-materialized to the
   lengths-derived width — the seq-dim validator at training time then
   crashed on the cross-field shape divergence. Fix: a separate
   `pack_per_token_field` helper that's explicitly invoked by the
   write-back site (which knows the field is per-token) and accepts
   `val.shape[1] >= max_len`; `to_nested_by_length` slices each row to
   its own length and drops the trailing SP padding. The conservative
   `maybe_pack_jagged` heuristic stays untouched so 3D extras like
   image features still round-trip correctly.

Validated 5/5 steps end-to-end on qwen3 30B mcore + TP=2 + SP + 2-node
(job 11635431, FLOPS 140.61 → 568.89, within noise of the simple
backend control). All 96 data-plane unit tests pass.
