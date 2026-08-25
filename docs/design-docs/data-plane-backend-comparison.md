# Data-plane backends compared: store vs register mode, host staging vs GDR

Four configurations, one recipe, measured end to end. The question is what the
choice of data-plane backend and fabric actually costs a training step.

**Setup.** Qwen3-30B-A3B, GRPO, single controller, 6 nodes x 4 GPU GB300, RoCE
(`mlx5_8`; `mlx5_17` present with no active port), 4096-token sequences, five
steps each. Wheel `mooncake-transfer-engine-cuda13 == 0.3.11.post1`. All four
ran on equivalent code with `observability.enabled: true`, after the
measurement fixes described under [Measurement validity](#measurement-validity).

Reproduce with `docs/guides/register-mode-repro.md`.

---

## Results

| config | backend | `use_gdr` | data plane | vs best | bytes | step total |
|---|---|---|---|---|---|---|
| **CPU RDMA + register** | `transfer_engine` | false | **0.79 s** | — | 529.4 MB | 546.9 s |
| CPU RDMA store | `mooncake_cpu` | false | 0.95 s | +20% | 527.2 MB | 573.7 s |
| GDR + register | `transfer_engine` | true | 1.00 s | +27% | 527.9 MB | 549.8 s |
| GDR store | `mooncake_cpu` | true | 1.35 s | +71% | 528.6 MB | 560.8 s |

Volumes agree within 0.4% (527.2–529.4 MB), so the four are directly
comparable: they moved the same bytes and differ only in how.

Two effects, each consistent on both axes:

**Register mode beats the store**, −17% on CPU RDMA and −26% on GDR. That is
the put side. The store copies every tensor into a staging buffer it owns;
register mode publishes the producer's own address and moves nothing until a
consumer reads.

**GDR loses to host staging**, −21% with register mode and −30% with the store.
This is the counterintuitive result and it held on both backends. See
[Why GDR loses here](#why-gdr-loses-here).

**Do not read the step-time column as a backend effect.** Its spread is ~27 s
while the entire data-plane difference is ~0.5 s. The rest is generation and
training variance, and a step-time ranking would invert on a re-run.

---

## Per-step breakdown

Driver-side cluster-scope totals, per step, in the order the steps ran:

| config | step 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| CPU RDMA + register | 204 ms | 141 | 157 | 139 | 151 |
| CPU RDMA store | 306 ms | 157 | 128 | 228 | 127 |
| GDR + register | 349 ms | 180 | 136 | 173 | 163 |
| GDR store | 415 ms | 254 | 250 | 268 | 164 |

Step 1 is the most expensive everywhere — first-touch registration, buffer
pools not yet warm — and the gap between backends is widest there (204 ms vs
415 ms, 2x). By step 5 the spread narrows to 127–164 ms. A benchmark that runs
one step would overstate the difference by roughly double.

## Per-call breakdown, register mode

`tools/parse_register_mode_log.py` splits `get` into its phases. Both register
cells, all five steps:

| | CPU RDMA + register | GDR + register |
|---|---|---|
| `put` mean | 1.64 ms (n=118, +104 collapsed) | 1.56 ms (n=125, +97 collapsed) |
| `get` mean | 18.07 ms (n=20) | 21.64 ms (n=20) |
| — of which `alloc` | 1.39 ms | **2.85 ms** |
| — of which `register` | 0.86 ms | 0.64 ms |
| — of which `move` | 15.81 ms | 18.15 ms |
| `clear` mean | 2.56 ms (n=7) | 1.48 ms (n=7) |

`move` dominates `get` in both — 87% and 84% respectively. The GDR arm pays
**2x more in `alloc`** (2.85 ms vs 1.39 ms): allocating and registering device
receive buffers costs more than host ones, and at these payload sizes that is
not repaid.

Read the `get` mean with care. The distribution is wide — CPU RDMA spans
2.62–79.89 ms with a p50 of 10.56 — because payload per call varies by an order
of magnitude. The p50/p90 columns in the tool's output are the honest summary;
the mean is dragged by a few large reads.

`put` counts are understated: Ray collapses identical consecutive log lines
into `[repeated Nx]`, about 45% of puts here. The tool reports the collapse
rather than hiding it. Per-call means are unaffected.

## Why GDR loses here

GDR exists to avoid a host round-trip on large transfers. These transfers are
not large: ~527 MB per five steps across ~64 keys per put, i.e. roughly 100 KB
per key. Against that, GDR adds device-buffer allocation and registration on
the receive side — visible directly as the 2x `alloc` cost above — and the
round-trip it saves is small enough not to cover it.

This is a payload-size effect, not a verdict on GDR. The prediction it implies
is that GDR should win once payloads are large enough for `move` to dominate
the fixed per-call costs, and
`examples/configs/recipes/llm/grpo-qwen2.5-1.5B-8n4g-gb300-yarn-200k*.yaml`
exists to test exactly that at 48x the sequence length. **That run has not been
completed**; treat "GDR loses" as scoped to 4096 tokens.

## Measurement validity

Everything above was measured after fixing two bugs in the observability
wrapper that biased precisely this comparison:

1. **`_tensor_bytes` read `offsets[-1]` from device memory** to size jagged
   fields — a blocking D2H sync on every jagged leaf of every op. Free for host
   payloads, a stall for GDR ones. It penalised the GDR arms for being on GPU.
2. **The sizing walk ran inside the op's timing window**, so observability's own
   cost was billed to the transfer.

Both are fixed and covered by tests. Numbers taken before those fixes — in
particular an earlier 3.87 s vs 5.62 s register-vs-store pair — are not
comparable to this table and should not be quoted alongside it.

Scope: these are **cluster-scope** counters, aggregated over the driver and
every DP rank via `TQPolicy.collect_data_plane_snapshots`. Earlier driver-scope
numbers cover only the SingleController's own client, roughly a sixth of a
step's traffic.

## Known issue: store sizing on the head node

Both store cells run with the store sized down:

```yaml
data_plane:
  mooncake_cpu:
    global_segment_size: 8589934592   # 8 GiB, default is 64 GiB
    local_buffer_size: 2147483648     # 2 GiB, default is 4 GiB
```

`global_segment_size` and `local_buffer_size` are **per client process**, and
the head node runs the SingleControllerActor alongside policy workers, each
mounting its own segment. At the defaults that is ~5 x 68 GiB pinned on one
node. Ray's OOM report from a failing run:

```
task name=SingleControllerActor.__init__, actual memory used=68.43GB
Memory on the node was 880.57GB / 920.00GB (95.7%)
```

95.7% is Ray's kill threshold. The configuration sits at the edge before
anything goes wrong, so whether a run survives is close to a coin flip: this
branch OOMed 4 of 5 attempts at the defaults while `main` completed 3 of 3.

The branch's specific contribution to that few-GB delta was **not identified** —
a five-point bisect and seven refuted hypotheses did not name it, and each
bisect point was a single run of a probabilistic failure, so that ladder is not
trustworthy evidence. Sizing the store removes ~230 GiB of pinned memory that
existed to move 500 MB per step, which is worth doing on its own merits and
takes the node well clear of the threshold. Recorded as tracked, not fixed.
