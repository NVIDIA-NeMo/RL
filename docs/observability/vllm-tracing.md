# vLLM Tracing

Generation is where most of an RL step's wall-clock goes, so NeMo-RL instruments vLLM at **two independent layers**. They answer different questions, ship over different transports, and are enabled independently.

| | Layer 1 — RL-side spans | Layer 2 — vLLM native OTLP |
|---|---|---|
| What | `rl.vllm.generate` / `rl.vllm.generate_text` spans + token/latency metrics, emitted by NeMo-RL around the vLLM call | vLLM's own internal engine spans (scheduling, prefill/decode, ...) |
| Where | driver, `nemo_rl/models/generation/vllm/vllm_generation.py` | vLLM engine, enabled in `vllm_worker.py` |
| Enabled by | `generation` span group (on by default in `per_step`/`all`) | opt-in: `NEMO_RL_OTEL_VLLM_NATIVE_TRACING=1` |
| Transport | rides the normal lens OTLP path (`http/protobuf` OK) | **gRPC-only** (needs an OTLP/gRPC endpoint / collector) |
| Correlation | nested under the rollout span (parent-child) | via shared `run_id` / resource attributes (not parent-child) |

## Layer 1 — RL-side generation spans (default)

`VllmGeneration.generate` / `generate_text` on the driver are wrapped with `trace_fn(RLSpanGroup.GENERATION, ...)`, emitting `rl.vllm.generate` and `rl.vllm.generate_text` spans. These nest under the active `rl.<algo>.collect_rollouts` span, so a rollout waterfall shows exactly how long generation took and how it fits inside the step. They also emit the `gen_ai.*` token/latency metrics (see [Metrics](metrics.md)).

Because these are ordinary lens spans, they travel the same OTLP transport as everything else — including a direct-to-backend `http/protobuf` export path. **Nothing extra is required**: enable the `generation` group (it is in the `per_step` and `all` presets) and they appear.

## Layer 2 — vLLM native OTLP tracing (opt-in)

vLLM can emit its own OpenTelemetry spans for the engine internals. Enable it with:

```bash
export NEMO_RL_OTEL_VLLM_NATIVE_TRACING=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector-host>:4317   # gRPC!
```

Under the hood, `_maybe_enable_vllm_native_tracing()` (in `vllm_worker.py`, called from `_load_model`) sets `otlp_traces_endpoint` and `collect_detailed_traces=["all"]` on the vLLM engine args. It reads `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` if set, otherwise `OTEL_EXPORTER_OTLP_ENDPOINT`.

### Caveat 1 — vLLM's exporter is gRPC-only

vLLM's OTLP span exporter speaks **OTLP/gRPC only**. It needs a gRPC OTLP endpoint — a collector on `:4317` or a gRPC-capable backend. It will **not** ride an `http/protobuf` OTLP endpoint, including a direct-to-backend `http/protobuf` path like the one Layer 1 uses.

So to get vLLM's native spans you need a gRPC OTLP receiver in the picture (e.g. an OTel Collector on `:4317` that forwards to your backend). This is why native tracing is left **off** by default when exporting to an `http/protobuf` endpoint with no collector. See [Observability Stack](observability-stack.md).

### Caveat 2 — offline generation cannot carry a trace context

NeMo-RL drives vLLM through the offline `LLM.generate()` API, which does not accept a per-request trace context. So vLLM's native spans **cannot** nest as children of the RL rollout span. Instead they correlate to the RL run through the **shared `run_id` and resource attributes** that every process in the job carries — you line them up by run, not by parent-child edges in one waterfall.

Practically: Layer 1 gives you generation timing *inside* the RL step tree; Layer 2 gives you vLLM engine internals as a separate set of spans tagged with the same `run_id`. Use both when you need to see why generation was slow at the engine level.

### Graceful degradation

If the installed vLLM does not support `otlp_traces_endpoint` (older versions), `_maybe_enable_vllm_native_tracing` logs a warning and skips — it never breaks the run. `collect_detailed_traces` is only set when that engine arg is also supported. If the flag is set but no OTLP endpoint is configured, it logs a warning and does nothing.

## Which layer do I want?

- **Just want to see generation cost per rollout?** Layer 1 — enable the `generation` group. Works over any transport, including a direct-to-backend `http/protobuf` path.
- **Debugging vLLM engine internals (scheduling, batching, prefill/decode)?** Add Layer 2 — but stand up a gRPC OTLP collector first, and correlate by `run_id`.
