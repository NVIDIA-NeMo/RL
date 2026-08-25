# Reproducing register mode on CPU RDMA and GPU RDMA (GDR)

Register mode publishes a buffer's address at `put` and moves bytes only on
`get`. This is how to run it end to end on both fabrics, what to expect, and
how to tell a real result from a broken harness.

Everything here was run on GB300, 6 nodes x 4 GPU, RoCE (`mlx5_8`; `mlx5_17`
present with no active port), wheel
`mooncake-transfer-engine-cuda13 == 0.3.11.post1`.

---

## 1. Smoke test first (1 node, ~2 minutes)

Never debug an end-to-end run before this passes — it isolates the data plane
from Ray, Megatron and vLLM, so a failure here is the data plane's.

```bash
# GPU RDMA (GDR): sources stay in HBM, one-sided read HBM -> HBM
python tools/smoke_register_mode_gdr.py

# CPU RDMA: same design, host-resident sources
python tools/smoke_register_mode_gdr.py --host-only
```

Expect `RESULT: PASS`, four tensor keys plus a non-tensor payload, byte-exact,
and under GDR every key returning `device=cuda`.

Cross-process variant (two processes, separate GPUs), which the single-process
smoke cannot cover:

```bash
python tools/smoke_nvlink_ipc.py --protocol rdma --mb 64
```

---

## 2. End to end

Three recipes. The first two differ only in `use_gdr`; the third is the store
baseline. All three are overlays on the first, so the shared shape cannot drift:

| recipe | fabric |
|---|---|
| `grpo-qwen3-30ba3b-6n4g-gb200-single-controller-tq_register.yaml` | GPU RDMA (GDR) |
| `grpo-qwen3-30ba3b-6n4g-gb200-single-controller-tq_register_cpu.yaml` | CPU RDMA |
| `grpo-qwen3-30ba3b-6n4g-gb200-single-controller-tq_mooncake.yaml` | store baseline |

```bash
cd <checkout>

DRY_RUN=0 EXP_NAME=register-gdr \
  RECIPE=examples/configs/recipes/llm/grpo-qwen3-30ba3b-6n4g-gb200-single-controller-tq_register.yaml \
  bash examples/nemo_gym/nemotron-3-ultra/qwen30b_sc_tq_register_gb200.sh

DRY_RUN=0 EXP_NAME=register-cpu \
  RECIPE=examples/configs/recipes/llm/grpo-qwen3-30ba3b-6n4g-gb200-single-controller-tq_register_cpu.yaml \
  bash examples/nemo_gym/nemotron-3-ultra/qwen30b_sc_tq_register_gb200.sh
```

`DRY_RUN` defaults to 1: without it the launcher prints the resolved config and
exits.

`CONTAINER` is **required** and has no default — a date-stamped squashfs under
one person's scratch is unreadable to everyone else, and the failure would
surface as an enroot error deep inside `srun`. Set it to a squashfs path or an
`nvcr.io` ref (see `docs/cluster.md`).

Set `HF_HOME` and `PERSISTENT_CACHE` to paths you own; the defaults point at
one user's scratch and will fail on permissions or write into someone else's
cache. `CODE_DIR` derives from the launcher's own location and decides what
gets bind-mounted, so override it only to test a different checkout. Other
knobs: `SBATCH_QOS`, `WALLTIME`, `EXP_NAME`.

On QOS: `short` has a **64-node per-user cap**, so one large job of your own
starves every other short job you submit — including 6-node ones, which sit in
`QOSMaxNodePerUserLimit` indefinitely. Use the default `normal` for these.

---

## 3. Reading the result

Per-step, from the metrics client (`observability.enabled: true` in both
recipes):

```
data plane: 1207ms, 473.7 MB moved
```

Per call, from the adapter's own log lines:

```bash
python tools/parse_register_mode_log.py \
  workspace/ray_logs/<exp>/<jobid>-logs/ray-driver.log <total_step_seconds>
```

which breaks `get` into `alloc` / `register` / `move` and reports
distributions rather than means — the mean is misleading here, `get` spans
2.8–78.9 ms in one run.

The measured comparison lives in `docs/design-docs/tq-register-mode.md`
(section "End to end…"), with job IDs. Summary: register mode's data plane cost
~31% less than the store's for the same volume, and both were under 1% of step
time.

Those numbers were taken **driver-scope**, before the metrics hookup used the
worker fan-out. `TQPolicy.collect_data_plane_snapshots` aggregates every DP
rank, and the driver sees roughly a sixth of a step's traffic — so a re-measure
on current code will report a `data_plane/cluster` prefix and larger absolute
numbers. The ratio is what the comparison rests on, not the absolute values.

`parse_register_mode_log.py` sums per-process prints multiplexed into one driver
log, so its total overlaps in wall-clock and is an upper bound, not a serial
total. It also reports how many lines Ray collapsed into `[repeated Nx]` — about
40% of puts in a typical run.

---

## 4. Failure modes worth recognising

| symptom | cause |
|---|---|
| `ERR_CONTEXT` (-202) on every CUDA registration | `WITH_NVIDIA_PEERMEM` defaults true but the module is absent; `configure_engine_env` sets it to 0 to select the DMA-BUF route |
| every cross-rail transfer dies, same-rail always works | RoCE rail isolation; needs `MC_ENABLE_DEST_DEVICE_AFFINITY=1`, which mooncake reads **presence-only** (`=0` enables it) |
| `RuntimeError: MC_FORCE_MNNVL is set...` | deliberate: no backend can feed mooncake's NVLink transport today, and setting it was measured to end the run in a host OOM |
| host OOM at setup with `mooncake_cpu` | store buffers are **per client process**: `global_segment_size` 64 GiB + `local_buffer_size` 4 GiB each, so a 4-GPU node pins 4 x 68 GiB = 272 GiB. Register mode mounts no store and never hits this |
| `batch_get_tensor returned None` | store under-sized; raise `global_segment_size` |

---

## 5. What this does not cover

NVLink. It is fast on this hardware (765 GB/s warm same-node, 768 cross-node within one
clique, vs 18.2 GB/s for RDMA) and no shipped backend can reach it today —
the store rejects `protocol=nvlink`, and register mode's torch-allocated
sources fail in at least two ways, only one of which is understood. See
`docs/design-docs/tq-register-mode.md`. Reaching it likely needs a mooncake
rebuild (`-DUSE_INTRA_NVLINK=ON`, `-DENABLE_MULTI_PROTOCOL=ON`) rather than a
version bump: the newer source still has both known blockers, and a third is
unexplained.
