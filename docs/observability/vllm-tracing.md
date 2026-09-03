# vLLM Tracing

Generation is where most of an RL step's wall-clock goes, so NeMo-RL instruments vLLM at **two independent layers**. They answer different questions, ship over different transports, and are enabled independently.

| | Layer 1 — RL-side spans | Layer 2 — vLLM native OTLP |
|---|---|---|
| What | `rl.vllm.generate` / `rl.vllm.generate_text` spans + token/latency metrics, emitted by NeMo-RL around the vLLM call | vLLM's own internal engine spans (scheduling, prefill/decode, ...) |
| Where | driver, `nemo_rl/models/generation/vllm/vllm_generation.py` | vLLM engine, enabled in `vllm_worker.py` |
| Enabled by | `generation` span group (on by default in `per_step`/`all`) | opt-in: `telemetry.vllm_native_tracing: true` |
| Transport | rides the normal lens OTLP path (`http/protobuf` OK) | **gRPC-only** (needs an OTLP/gRPC endpoint / collector) |
| Correlation | nested under the rollout span (parent-child) | via shared `run_id` / resource attributes (not parent-child) |

## Layer 1 — RL-side generation spans (default)

`VllmGeneration.generate` / `generate_text` on the driver are wrapped with `trace_fn(RLSpanGroup.GENERATION, ...)`, emitting `rl.vllm.generate` and `rl.vllm.generate_text` spans. These nest under the active `rl.<algo>.generation` span, so a rollout waterfall shows exactly how long generation took and how it fits inside the step. They also emit the `gen_ai.*` token/latency metrics (see [Metrics](metrics.md)).

Because these are ordinary lens spans, they travel the same OTLP transport as everything else — including a direct-to-backend `http/protobuf` export path. **Nothing extra is required**: enable the `generation` group (it is in the `per_step` and `all` presets) and they appear.

This covers the synchronous rollout path only. Async runs drive generation through `generate_async`, which carries no span today, so Layer 1 shows no generate spans there — and the `rl.grpo.generation` span they would nest under comes from the collector actor rather than the driver. See [span groups — coverage gaps](span-groups.md#coverage-gaps).

## Layer 2 — vLLM native OTLP tracing (opt-in)

vLLM can emit its own OpenTelemetry spans for the engine internals. Enable it in your run config:

```yaml
telemetry:
  enabled: true
  vllm_native_tracing: true
```

and point the exporter at a gRPC endpoint:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector-host>:4317   # gRPC!
```

Under the hood, `_maybe_enable_vllm_native_tracing()` (in `vllm_worker.py`, called from `_load_model`) sets `otlp_traces_endpoint` on the vLLM engine args. It reads `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` if set, otherwise `OTEL_EXPORTER_OTLP_ENDPOINT`.

### Caveat 1 — one span per generation request

vLLM traces at **request** granularity: every prompt it serves produces a span. An RL step generates one request per rollout, so a run doing 10k rollouts a step emits 10k+ spans per step from this layer alone — orders of magnitude more than the ~20 `rl.*` spans a step emits otherwise. That is enough to saturate a collector, and the cost lands on the generation workers.

Treat Layer 2 as a **debugging tool you switch on for a few steps**, not as something to leave on for a training run. There is no sampling knob: vLLM builds its own `TracerProvider` and the only lever is the process-global `OTEL_TRACES_SAMPLER`, which would also thin out the `rl.*` spans in the same process.

If what you want is engine behaviour in aggregate — token throughput, queue time, sequence lengths, preemptions, finish reasons — the `vllm/*` metrics are teed to OTel by default and cost one RPC per step, with no per-request spans. See [Metrics](metrics.md). Reach for Layer 2 only when an aggregate number has already told you *something* is wrong and you need per-request detail to find out what.

`collect_detailed_traces` is deliberately **not** set. vLLM documents it as "possibly costly and or blocking", and it adds per-request timing inside the engine, so it slows generation rather than just adding spans. Pass it through `vllm_kwargs` if you specifically want it.

### Caveat 2 — vLLM's exporter is gRPC-only

vLLM's OTLP span exporter speaks **OTLP/gRPC only**. It needs a gRPC OTLP endpoint — a collector on `:4317` or a gRPC-capable backend. It will **not** ride an `http/protobuf` OTLP endpoint, including a direct-to-backend `http/protobuf` path like the one Layer 1 uses.

So to get vLLM's native spans you need a gRPC OTLP receiver in the picture (e.g. an OTel Collector on `:4317` that forwards to your backend). This is why native tracing is left **off** by default when exporting to an `http/protobuf` endpoint with no collector. See [Observability Stack](observability-stack.md).

### Caveat 3 — offline generation cannot carry a trace context

NeMo-RL drives vLLM through the offline `LLM.generate()` API, which does not accept a per-request trace context. So vLLM's native spans **cannot** nest as children of the RL rollout span. Instead they correlate to the RL run through the **shared `run_id` and resource attributes** that every process in the job carries — you line them up by run, not by parent-child edges in one waterfall.

Practically: Layer 1 gives you generation timing *inside* the RL step tree; Layer 2 gives you vLLM engine internals as a separate set of spans tagged with the same `run_id`. Use both when you need to see why generation was slow at the engine level.

### Graceful degradation

If the installed vLLM does not support `otlp_traces_endpoint` (older versions), `_maybe_enable_vllm_native_tracing` logs a warning and skips — it never breaks the run. If the flag is set but no OTLP endpoint is configured, it logs a warning and does nothing.

## Which layer do I want?

- **Just want to see generation cost per rollout?** Layer 1 — enable the `generation` group. Works over any transport, including a direct-to-backend `http/protobuf` path.
- **Want engine behaviour over a whole run (throughput, queue time, preemptions, finish reasons)?** The `vllm/*` metrics, on by default — no per-request spans, no collector needed. See [Metrics](metrics.md).
- **Debugging vLLM engine internals (scheduling, batching, prefill/decode) on a specific step?** Add Layer 2 — but stand up a gRPC OTLP collector first, correlate by `run_id`, and turn it off again: it emits one span per request (see Caveat 1).
