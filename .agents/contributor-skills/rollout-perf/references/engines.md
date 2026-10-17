# Generation backends

How each backend's NeMo-RL config reaches the engine, how to prove what ran,
what metrics you get, and the known landmines. Select a backend with
`policy.generation.backend`.

Where a line below is quoted from an engine's own logging (not NeMo-RL's),
the exact text can change with the engine version. Grep for the key words.

## Cross-engine summary

| | vLLM | SGLang | TRT-LLM | Megatron | Dynamo |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Config block | `vllm_cfg` + `vllm_kwargs` | `sglang_cfg` | `trtllm_cfg` + `trtllm_kwargs` | `mcore_generation_config` | `dynamo_cfg` + `vllm_cfg` + `vllm_kwargs` |
| Pass-through | `vllm_kwargs` → engine args; repeating a key NeMo-RL already passes raises a duplicate-kwarg error | None; only whitelisted keys reach `ServerArgs` | `trtllm_kwargs` applied last; it overrides typed fields | Keys read one by one; optional keys forwarded when present | `vllm_kwargs` → worker CLI flags; duplicates are errors |
| Proof of effective config | Engine config line | `ServerArgs(...)` line | `LLM Args:` dump | Resolved config, MCore lines | Logged worker and frontend argv |
| Engine metrics in NeMo-RL | Yes (below) | None | None (`get_logger_metrics()` returns `{}`) | None | Yes, with `enable_vllm_metrics_logger` |

Engine metrics give you queue depth and KV use. Without them you cannot tell
engine-bound from feeder-bound directly; use GPU telemetry
(`logger.monitor_gpus`) and request timing instead.

## vLLM

- **Selected by** `backend: vllm`, the default. vLLM is pinned in
  `pyproject.toml`.
- **Config path.** Typed `vllm_cfg` keys plus `vllm_kwargs` go to `vllm.LLM`
  (sync) or `AsyncEngineArgs` (`vllm_cfg.async_engine: true`).
  - NeMo-RL itself passes: TP/PP, `enable_expert_parallel`,
    `gpu_memory_utilization`, `enable_prefix_caching`, `enforce_eager`,
    `max_model_len`, sleep mode, `disable_log_stats`, dtype, seed and load
    format. Don't repeat these in `vllm_kwargs`.
- **Proof** (vLLM's own lines in `ray-driver.log`):
  - `non-default args: {...}`: exactly what NeMo-RL passed. `max_num_seqs`
    and `async_scheduling` appear only here.
  - `Initializing a V1 LLM engine (vX.Y.Z) with config: ...`: shows
    `enforce_eager`, `enable_prefix_caching`, `enable_chunked_prefill`,
    `kv_cache_dtype`, `speculative_config`, and `compilation_config` with
    `'cudagraph_mode': <CUDAGraphMode.PIECEWISE: 1>`, the capture sizes and
    `pass_config`.
  - `Chunked prefill is enabled with max_num_batched_tokens=N`.
  - `GPU KV cache size: N tokens` and `Maximum concurrency for L tokens per
    request: Kx`.
  - `Capturing CUDA graphs (...)` and `Graph capturing finished in S secs`.
  - `Using <name> ... MoE backend`, and the attention backend line.
  - Periodic `Engine 000: Avg prompt throughput ..., Running: ..., Waiting:
    ..., GPU KV cache usage: ..., Prefix cache hit rate: ...`. Printed even
    with the sync engine, because NeMo-RL enables stats logging.
- **Metrics.**
  - `train/vllm/{prompt_tokens, generation_tokens, generations_failed, ...}`.
  - With speculative decoding: `train/vllm/spec_*`, including
    `spec_acceptance_rate`.
  - With `enable_vllm_metrics_logger` **and** `async_engine`: per-worker
    `generation_metrics/*` timelines (running and pending requests, KV use).
    These go to W&B, not to `metrics.json`.
- **Landmines.**
  - The prefix cache is reset at every `finish_generation`, and colocated
    runs also sleep the engine. Prefix reuse lasts within a step only.
  - Hybrid Mamba models: keep `max_num_seqs` at or below the Mamba state
    slots. The hybrid Nemotron Gym configs
    (`examples/nemo_gym/nemotron-3-ultra/*.yaml`) use 256; the vLLM default of 1024 can fail at
    startup.
  - `async_scheduling` has hung with data parallelism in some recipes; see
    their comments.
  - Graph mode and Inductor: see [levers.md §1](levers.md#1-cuda-graphs).
  - Profiling: target `vllm_generation_worker` or
    `vllm_async_generation_worker` with `NRL_NSYS_WORKER_PATTERNS`
    (`docs/nsys-profiling.md`).
  - Per-request tracing: `docs/observability/vllm-tracing.md`. It is
    debug-only, because it emits one span per request.

## SGLang

- **Selected by** `backend: sglang` (`examples/configs/grpo_math_1B_sglang.yaml`).
- **Config path.** NeMo-RL builds the SGLang `ServerArgs` from a fixed
  whitelist of `sglang_cfg` keys (`sglang_worker.py`):
  - `context_length`, `kv_cache_dtype`, `dtype`;
  - `max_running_requests`, `chunked_prefill_size`, `max_prefill_tokens`,
    `schedule_policy`, `schedule_conservativeness`;
  - `mem_fraction_static`, `cpu_offload_gb`;
  - `disable_cuda_graph`, `disable_cuda_graph_padding`,
    `cuda_graph_backend_{decode,prefill}`, `cuda_graph_max_bs_{decode,prefill}`,
    `cuda_graph_bs_{decode,prefill}`;
  - `tp_size`, `dp_size`, `pp_size`, `ep_size`.

  Declared but **not forwarded**: `disable_radix_cache`, `enable_nccl_nvls`,
  `disable_custom_all_reduce`, `enable_dp_attention`, `enable_mixed_chunk`,
  `num_continuous_decode_steps`, `enable_torch_compile` and `sglang_kwargs`.
  Setting them has no effect until the whitelist is extended.
- **Client concurrency.** `sglang_cfg.sglang_server_config.sglang_server_concurrency`
  caps in-flight HTTP requests per engine group.
- **Proof.** NeMo-RL logs `Launch HttpServerEngineAdapter at:` and `Router
  launched at`, but not the server arguments. Look for SGLang's own lines:
  - `server_args=ServerArgs(...)`;
  - `max_total_num_tokens=... chunked_prefill_size=... max_running_requests=...`;
  - `KV Cache is allocated`;
  - `Capture cuda graph begin/end`;
  - `Prefill batch ... #cached-token: ...` (prefix reuse);
  - `Decode batch ... cuda graph: True`.
- **Metrics.** None reach NeMo-RL metrics.
- **Landmines.**
  - Piecewise prefill graphs crash on torch 2.10, and the `breakable` backend
    is rejected under the memory saver.
  - BF16 only.
  - Weight refit rejects `pause_generation_mode: in_place`.
  - Not installable alongside the `nemo_gym` extra.
  - Check the Mamba radix strategy and overlap scheduling on hybrid models:
    an auto-resolved buffer mode under overlap scheduling lowered reward
    (see [levers.md §3](levers.md#3-prefixkv-reuse-and-session-affinity)).

## TensorRT-LLM

- **Selected by** `backend: trtllm` (`examples/configs/grpo_math_1B_trtllm.yaml`).
  It requires `trtllm_cfg.async_engine: true`.
  - Recipes named `*-trtllm` under qwen3.5 are vLLM recipes that use the
    FlashInfer TRTLLM MoE kernels; check `backend`.
- **Config path.** `trtllm_cfg` is applied in this order, each step able to
  override the one before:
  1. Typed fields go into `AsyncLLM`:
     - `max_num_tokens`, `max_batch_size`;
     - TP;
     - MoE TP/EP, which must satisfy MoE TP × MoE EP = TP;
     - `max_input_len` forced to `max_model_len`;
     - scheduler `MAX_UTILIZATION`;
     - `CudaGraphConfig(enable_padding=True, max_batch_size=max_batch_size)`.
  2. `trtllm_kwargs.kv_cache_config` becomes a `KvCacheConfig`.
     `trtllm_cfg.gpu_memory_utilization` fills `free_gpu_memory_fraction`
     only if it isn't set explicitly.
  3. `trtllm_kwargs` is applied last and overrides everything. That includes
     `cuda_graph_config`, `allreduce_strategy` and `enable_chunked_prefill`.
- **Proof.** NeMo-RL logs `[TrtllmAsyncWorker] bundle_indices=`, `AsyncLLM
  ready` and `HTTP server started`, but not the engine arguments. Look for
  TRT-LLM's own lines:
  - the `LLM Args:` dump (check `enable_block_reuse`, `max_num_tokens`,
    `enable_chunked_prefill`, `allreduce_strategy`, the MoE layout);
  - the paged KV cache allocation line;
  - the CUDA graph warmup batch sizes.

  For MNNVL claims, require the MNNVL fabric lines on every replica.
- **Metrics.** None: `get_logger_metrics()` returns `{}` (a TODO). Use GPU
  telemetry and request timing.
- **Version.** NeMo-RL pins `tensorrt-llm==1.3.0rc21`. Results from rc24
  (block reuse on hybrid models, `prefill_cuda_graph_backend: breakable`) may
  not apply to rc21; check that the pinned version accepts the key and
  survives.
- **Landmines.**
  - rc21 block reuse on hybrid Mamba models crashed (`Invalid recurrent state
    block index 2147483647`).
  - rc21 `num_postprocess_workers=4` failed with `RPCStreamingError`.
  - A requested Mamba state option can be disabled by the engine (stochastic
    rounding with float32 state); pin it to the effective value.
  - The prefix cache resets after generation when not colocated.
  - The native (non-HTTP) path round-robins per request, with no session
    affinity.
  - `NRL_TRTLLM_ASYNC_TIMEOUT_SECONDS` (default 900) bounds a request.
  - Profiling targets `trtllm_async_generation_worker`.

## Megatron inference

- **Selected by** `backend: megatron`. Defaults are in
  `examples/configs/grpo_math_1B.yaml` under `mcore_generation_config`.
- **Config path.** Keys are read into MCore's `InferenceConfig`:
  - Graphs:
    - `cuda_graph_impl: local`, `inference_cuda_graph_scope: block`,
      `num_cuda_graphs: 4`;
    - `cuda_graph_max_tokens: 512`,
      `cuda_graph_sizing_distribution: hybrid`;
    - `use_cuda_graphs_for_non_decode_steps: true`.
  - Memory and scheduling:
    - `buffer_size_gb: 10`, `block_size_tokens: 256`;
    - `max_tokens: 16384`, `enable_chunked_prefill: true`;
    - `kv_cache_management_mode: persist`, `async_sched_mode: async`.
  - Prefix caching: `enable_prefix_caching: false`, with
    `prefix_caching_coordinator_policy: longest_prefix`.

  Any `megatron_cfg` key can be overridden for the inference model.
  Colocated runs share the training model unless the layout differs, in which
  case a dedicated model is resharded on every wake.
- **Proof.**
  - `Initialized persistent inference engine`;
  - `Coordinator started` / `Starting HTTP Server`;
  - `mcore async scheduling steps (cumul): N` (async scheduling ran);
  - MCore's `[graph i/N] [T]: P P + D D` lines (N graphs, the largest
    covering T tokens) and `> built cuda graph(s) in S sec`.

  MCore does not log the effective prefix-caching, token-budget or
  admission settings. Treat the resolved `mcore_generation_config` values as
  requested only.
- **Metrics.** None.
- **Landmines.**
  - `cuda_graph_impl` and `inference_cuda_graph_scope` must be paired;
    colocated runs allow only `none`/`local`.
  - EP > 1 with graphs needs `moe_pad_experts_for_cuda_graph_inference`.
  - `inference_optimized` needs sequence parallelism when TP > 1.
  - `kv_cache_management_mode` must match
    `recompute_kv_cache_after_weight_updates`.
  - The real KV buffer is about 2× `buffer_size_gb`. Large `max_tokens` can
    OOM during graph warmup.
  - The `nvshmem` refit backend is broken (issue #3646).
  - Fleet health is unsupported.

## Dynamo

- **Selected by** `backend: dynamo` (`examples/configs/grpo_math_1B_dynamo.yaml`,
  `docs/guides/dynamo-generation.md`). It runs vLLM workers behind a Dynamo
  frontend.
  - The Dynamo package ships its own vLLM, which may differ from NeMo-RL's
    pinned vLLM. Compare versions before attributing a difference to Dynamo.
- **Config path.** `vllm_kwargs` become worker CLI flags. Duplicates and
  managed flags are errors.
  - `dynamo_cfg.frontend_args` set the router (`router_mode`: `round-robin`,
    `kv`, `least-loaded`, ...) and the tokenizer (`tokenizer`,
    `tokenizer_cache`, `tokenizer_cache_bytes`).
- **Proof.** `[Dynamo:<group>] launching argv=`, `[Dynamo] launching frontend
  argv=`, `frontend tokenizer environment=`, `frontend ready with N
  generation and RL workers`, plus the vLLM worker lines above.
- **Metrics.** With `vllm_cfg.enable_vllm_metrics_logger`, NeMo-RL polls each
  worker's `/metrics` endpoint into `generation_metrics/*` (in-flight
  requests, pending requests, KV use, generated tokens).
- **Landmines.**
  - Not supported: colocation, quantization, speculative decoding,
    multi-node engines.
  - The Gym adapter must match the frontend's token-ID protocol. A mismatched
    adapter got 404s, and every trajectory came back empty with reward 0 in a
    job that exited cleanly.
  - The router on the request path changes who owns the queue. Compare
    Dynamo as a deployed stack, not as an engine.

## Comparing backends

- **Label the comparison scope** (`measurement.md`). Backends differ in:
  - scheduler semantics;
  - cache scope;
  - the endpoints and routers on the request path;
  - default graph coverage.

  So most cross-backend results are deployed-stack comparisons, not
  engine-only ones.
- **Map before comparing.** Map every lever in [levers.md](levers.md) for each
  backend, pin each value explicitly, and record what ran.
- **Gate before speed.** Require reward or logprob parity before comparing
  speed. In the SWE benchmark, every backend comparison that looked
  surprisingly fast at request p50 either failed reward parity or was not
  matched on placement or harness.
- **Keep model families separate.** Never rank one model family's curve
  against another's.

## Onboarding a new backend

1. Map each lever in [levers.md](levers.md) to the backend's knobs, and record
   their semantics: per-iteration or per-step budget, cache scope, and
   whether the cache survives refit.
2. Find the log line or metric that proves each knob, and add it to
   `rollout_perf_report.py`.
3. Wire `get_logger_metrics()`, so that queue depth and KV use reach NeMo-RL.
4. Run the reference topology, and check parity against an existing backend
   on the same prompts.
5. Find the regime, then sweep levers in rank order, one factor at a time,
   with at least 3 runs per arm.
