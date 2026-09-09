# vLLM Tracing

Generation is where most of an RL step's wall-clock goes, so NeMo-RL instruments vLLM at **two independent layers**. They answer different questions, ship over different transports, and are enabled independently.

| | Layer 1 — RL-side spans | Layer 2 — vLLM native OTLP |
|---|---|---|
| What | `rl.vllm.generate` / `rl.vllm.generate_text` spans + token/latency metrics, emitted by NeMo-RL around the vLLM call | vLLM's own internal engine spans (scheduling, prefill/decode, ...) |
| Where | driver, `nemo_rl/models/generation/vllm/vllm_generation.py` | vLLM engine, enabled in `vllm_worker.py` |
| Enabled by | `generation` span group (on by default in `per_step`/`all`) | opt-in: `telemetry.vllm_native_tracing: true` |
| Transport | rides the normal lens OTLP path (`http/protobuf` OK) | vLLM's own exporter, **gRPC by default** (`http/protobuf` needs `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`) |
| Correlation | nested under the rollout span (parent-child) | via shared `nemo.run.id` / resource attributes (not parent-child) |

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

and point the exporter at an endpoint. vLLM defaults to gRPC, so a collector on `:4317` needs nothing further:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://<collector-host>:4317   # gRPC
```

To send vLLM's spans to the same `http/protobuf` endpoint lens uses, name the protocol explicitly — vLLM reads only the traces-specific variable (see Caveat 2 below):

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://<backend-host>:4318
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
```

Under the hood, `_maybe_enable_vllm_native_tracing()` (in `vllm_worker.py`, called from `_load_model`) sets `otlp_traces_endpoint` on the vLLM engine args. It reads `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` if set, otherwise `OTEL_EXPORTER_OTLP_ENDPOINT`.

### Caveat 1 — one span per generation request

vLLM traces at **request** granularity: every prompt it serves produces a span. An RL step generates one request per rollout, so a run doing 10k rollouts a step emits 10k+ spans per step from this layer alone — orders of magnitude more than the ~20 `rl.*` spans a step emits otherwise. That is enough to saturate a collector, and the cost lands on the generation workers.

Treat Layer 2 as a **debugging tool you switch on for a few steps**, not as something to leave on for a training run. There is no sampling knob: vLLM builds its own `TracerProvider` and the only lever is the process-global `OTEL_TRACES_SAMPLER`, which would also thin out the `rl.*` spans in the same process.

If what you want is engine behaviour in aggregate — token throughput, sequence lengths, finish reasons — the `vllm/*` metrics are teed to OTel by default and cost one RPC per step, with no per-request spans. Queue time and preemptions are *not* among them: the engine exposes both, but `_KEPT_COUNTER_NAMES` does not keep them, so Layer 2 is the only way to see either today. See [Metrics](metrics.md). Reach for Layer 2 only when an aggregate number has already told you *something* is wrong and you need per-request detail to find out what.

`collect_detailed_traces` is deliberately **not** set. vLLM documents it as "possibly costly and or blocking", and it adds per-request timing inside the engine, so it slows generation rather than just adding spans. Pass it through `vllm_kwargs` if you specifically want it.

### Caveat 2 — vLLM's exporter picks its protocol separately

vLLM does not reuse lens's exporter; it builds its own and chooses the protocol from **`OTEL_EXPORTER_OTLP_TRACES_PROTOCOL`, defaulting to `grpc`**. It supports `grpc` and `http/protobuf`, and raises on anything else.

The trap is that it reads *only* that traces-specific variable — never the generic `OTEL_EXPORTER_OTLP_PROTOCOL`. So a run configured for a direct-to-backend `http/protobuf` path gets Layer 1 exported over HTTP and Layer 2 attempting **gRPC against the same HTTP port**, whose export failures surface only in the generation worker's own logs. Set `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` alongside it, or put a gRPC OTel Collector on `:4317` in the picture. Note also that vLLM constructs its gRPC exporter with `insecure=True`, so the gRPC path is plaintext. See [Observability Stack](observability-stack.md).

### Caveat 3 — the opt-in governs engine spans, not worker spans

`vllm_native_tracing` sets `otlp_traces_endpoint` on the engine args, which is the only thing that enables vLLM's **request/engine** tracing. vLLM's **worker** processes — `EngineCore`, `DPEngineCoreActor`, and the `multiproc_executor` workers — call `maybe_init_worker_tracer()` unconditionally and gate purely on `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` being present in their environment. They never look at the engine args.

That variable propagates cluster-wide on its own: `init_ray()` snapshots the whole driver environment into the Ray `runtime_env`, and nothing filters `OTEL_*`. So **exporting `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` in a job script turns on vLLM's worker-process tracers even with `vllm_native_tracing: false`** — bringing back the per-request span volume of Caveat 1 that the opt-in exists to keep off.

Prefer setting the generic `OTEL_EXPORTER_OTLP_ENDPOINT` for lens and leaving the traces-specific variable unset; `_maybe_enable_vllm_native_tracing` falls back to the generic one for the endpoint value, and vLLM's own `init_otel_tracer` then exports the traces-specific variable itself for its children. Clearing it inside the worker is not a fix: the OTel SDK prefers the traces-specific variable over the generic one, so that would retarget lens's exporter too.

### Caveat 4 — offline generation cannot carry a trace context

NeMo-RL drives vLLM through the offline `LLM.generate()` API, which does not accept a per-request trace context. So vLLM's native spans **cannot** nest as children of the RL rollout span. Instead they correlate to the RL run through the **shared `nemo.run.id` and resource attributes** that every process in the job carries — you line them up by run, not by parent-child edges in one waterfall.

Practically: Layer 1 gives you generation timing *inside* the RL step tree; Layer 2 gives you vLLM engine internals as a separate set of spans tagged with the same `nemo.run.id`. Use both when you need to see why generation was slow at the engine level.

### Graceful degradation

If the installed vLLM does not support `otlp_traces_endpoint` (older versions), `_maybe_enable_vllm_native_tracing` logs a warning and skips — it never breaks the run. If the flag is set but no OTLP endpoint is configured, it logs a warning and does nothing.

## Which layer do I want?

- **Just want to see generation cost per rollout?** Layer 1 — enable the `generation` group. Works over any transport, including a direct-to-backend `http/protobuf` path.
- **Want engine behaviour over a whole run (token throughput, sequence lengths, finish reasons)?** The `vllm/*` metrics, on by default — no per-request spans, no collector needed. See [Metrics](metrics.md).
- **Want queue time or preemptions?** Layer 2 — neither is teed as a metric today.
- **Debugging vLLM engine internals (scheduling, batching, prefill/decode) on a specific step?** Add Layer 2 — but settle the transport first (a gRPC collector, or `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf`), correlate by `nemo.run.id`, and turn it off again: it emits one span per request (see Caveat 1).
