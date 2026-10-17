# Rollout levers

Each lever lists:
- **Why** it matters;
- the **knob in each backend** as a NeMo-RL key under `policy.generation`;
- the measured **evidence**, graded per `measurement.md`;
- the **traps**.

Equal numbers do not mean equal semantics across engines, so sweep a lever
per engine and per topology instead of copying a value.

Evidence sources:
- **H100 nightly A/Bs.** Same-window eager vs. graph arms on the H100
  nightly CI cluster. vLLM 0.29, one run per arm, steps paired.
- **SWE rollout benchmark.** An internal multi-turn SWE-agent benchmark:
  - Nemotron Nano V3.5 (hybrid Mamba/MoE) on GB200;
  - 2 replicas × TP4;
  - 64 concurrent trajectories;
  - 20 SWE-bench instances × 16 samples.

  It compared vLLM 0.20/0.25, SGLang, TRT-LLM 1.3.0rc21/rc24, Megatron
  inference and Dynamo, with 3–5 runs per profile unless noted.

## Applying a lever to any engine

Every lever below is an engine-independent mechanism. The knob names in the
per-lever tables are only where that mechanism surfaces in today's backends.
To apply a lever to a backend not listed here, or after an engine upgrade
renames its knobs:

1. **Confirm the signal first**, using engine-independent evidence: NeMo-RL
   timing, the token shape, generation length vs. the cap, and GPU
   utilization (`logger.monitor_gpus`). A lever whose signal is absent will
   not pay off on any engine.
2. **Find the knob by concept, not by name.** Search the engine's arguments
   and docs for the concept terms in the table below.
3. **Write down its semantics** before choosing a value:
   - Is it per iteration or per request?
   - Does it cover prefill only, decode only, or both?
   - Is it per replica or per process?
   - Does the cache survive weight refit?

   Equal numbers rarely mean the same thing across engines.
4. **Find the proof**, meaning the log line or metric that shows the
   engine's effective value. If the engine prints none, record that the
   value is unproven rather than assuming the default.
5. **A/B it on that engine and topology.** Never port another engine's value.
   For example, a 4× larger token budget gained 44% on TRT-LLM, while a 2×
   larger one cost 15% on vLLM at a different topology.

| Lever | Mechanism (engine-independent) | Signal that it applies | Concept to look for in any engine | What differs across engines | Proof it took effect |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1. CUDA graphs | Replay captured kernel sequences instead of launching each kernel | Decode-tail-bound; small per-replica batch; long generations | "cuda graph", "eager", "capture sizes/batch sizes", "piecewise/full", prefill vs. decode graphs | Which phases are captured (decode only, prefill too); padding; capture-size list; compile backend | Engine graph-mode/capture log lines |
| 2. Scheduler token budget | Bound the tokens processed per engine iteration; chunk long prefills | Long or multi-turn prompts; requests waiting; timeouts | "max batched tokens", "max num tokens", "chunked prefill", "prefill chunk size" | Per-iteration vs. per-prefill limits; whether it also caps context; activation memory | Effective budget and chunked-prefill line |
| 3. Prefix/KV reuse and affinity | Skip prefill for a prefix already in cache, and route the next turn to the replica that holds it | Many generations per prompt, multi-turn, or a high raw prefill:generated ratio | "prefix caching", "radix cache", "block reuse", "KV-aware routing", "session affinity" | Cache scope (request, conversation, global); eviction; reset on refit or sleep; hybrid-state support | Cache-hit or cached-token counters; the routing policy on the request path |
| 4. Parallel layout | Trade per-replica latency against replica count and communication | Tail-bound decode; large models or MoE | "tensor/expert/data parallel size", "attention DP", replicas | Whether MoE layout is separate from dense TP; cross-node limits | Engine topology line; number of serving endpoints |
| 5. Admission cap and memory | Hold the known per-replica load without preemption, waste, or startup failure | Requests waiting or preempted; KV near full; hybrid models | "max running requests/seqs/batch size", "memory fraction/utilization", "KV blocks" | What counts toward memory (weights, graphs, activations); hybrid state slots | KV capacity or max-concurrency line vs. per-replica load |
| 6. Kernels and collectives | Choose faster attention, MoE and all-reduce implementations for the hardware | Profile shows kernel or collective time; new GPU type | "attention backend", "MoE backend", "all-reduce strategy", "custom all-reduce", "NVLS/MNNVL" | Hardware availability; silent fallbacks; refit compatibility | Selected-backend log lines; no fallback warnings |
| 7. Frontend, tokenizer, harness | Keep the engine fed: HTTP workers, tokenization, agent CPU and placement | GPUs idle, nothing waiting, low KV use | "workers", "tokenizer", HTTP server replicas, router mode | Token-ID vs. text interfaces; router fan-in | Worker processes started; requests spread across replicas |
| 8. Speculative decoding | Draft several tokens and verify them in one pass | Decode-bound at small batch | "speculative", "draft", "MTP", "EAGLE" | Whether the drafter is refit with the policy | Acceptance rate over the whole run |
| 9. Startup | Load weights and build kernels in parallel, ahead of time | Setup is a large share of wall time | "parallel load", "prefetch", precompiled kernels/caches | Refit safety of fast-load paths | Setup timing |

## 1. CUDA graphs

**Why.** In eager mode every decode step pays kernel-launch overhead for
every layer. Decode-bound rollouts (long generations, small batches per
replica) lose most of their time to it. Prefill in multi-turn workloads runs
medium-sized token chunks, so graph coverage for prefill matters too.

| Backend | Knobs |
| :-- | :-- |
| vLLM | `vllm_cfg.enforce_eager` (inherited default `False`); `vllm_kwargs.compilation_config.{cudagraph_mode, cudagraph_capture_sizes, backend}` |
| SGLang | `sglang_cfg.{disable_cuda_graph, cuda_graph_backend_decode, cuda_graph_backend_prefill, cuda_graph_max_bs_decode, cuda_graph_bs_decode, ...}` |
| TRT-LLM | NeMo-RL builds `CudaGraphConfig(enable_padding=True, max_batch_size=trtllm_cfg.max_batch_size)`; `trtllm_kwargs.cuda_graph_config` replaces it entirely |
| Megatron | `mcore_generation_config.{cuda_graph_impl, inference_cuda_graph_scope, num_cuda_graphs, cuda_graph_max_tokens, use_cuda_graphs_for_non_decode_steps, cuda_graph_sizing_distribution}` |
| Dynamo | Same as vLLM, passed as worker flags |

**Evidence.**

H100 nightly A/Bs, eager → graphs. Every listed arm passed its `check_metrics`
gates with `token_mult_prob_error` and `gen_kl_error` unchanged.

| Test | Generation per step | Wall time |
| :-- | :-- | :-- |
| `grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2-fp8` | −85% | 174 → 79 min |
| `grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2cp2` | −79% | 153 → 83 min |
| `vlm_grpo-nemotron-omni-30ba3b-clevr-2n8g-megatron-tp8ep8.v1` | −78% | 51 → 29 min |
| `grpo-math-qwen3-30ba3b-megatron-tp4-32k` | −74% | 123 → 57 min |
| `vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1` | −72% | 81 → 41 min |
| `grpo-qwen3-8b-base-dapo-2n8g-long-megatron-qa-nvfp4-w4a16` | −71% | 164 → 59 min |
| `dapo-nanov3.5-30BA3B-4n8g-automodel` | −69% | 152 → 65 min |
| `vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v2` | −65% | 95 → 79 min |
| `vlm_grpo-gemma4-e4b-geo3k-1n8g-automodel` | −60% | 59 → 45 min |

Other graph evidence:

| Change | Effect | Grade |
| :-- | :-- | :-: |
| Megatron inference `cuda_graph_max_tokens` 512 → 4,096 (SWE benchmark) | Fixed 76% of prefill steps running eager; model-call p50 144 → 127 s | C |
| TRT-LLM prefill CUDA graphs (rc24 `prefill_cuda_graph_backend: breakable`) | Part of a bundle that cut rollout time 5% | B |
| vLLM, dense extra small decode capture sizes (SWE benchmark, TP1) | Wall +7.5%: a regression | C |

**Choosing the graph mode** (vLLM):
- On vLLM 0.29, A/B the default FULL_AND_PIECEWISE against PIECEWISE. Across
  11 H100 nightly tests, FULL_AND_PIECEWISE won 6 and PIECEWISE won 5.
- FULL_AND_PIECEWISE lost in these cases:
  - `grpo-math-qwen3-30ba3b-megatron-tp4-32k` and `dapo-qwen2.5-7b.v2`:
    more `token_mult_prob_error` spikes; on the qwen2.5 test it was also
    slower.
  - `dapo-nanov3.5-30BA3B-4n8g-automodel`: the same speed, but 6/20 spike
    steps against 1/20 with PIECEWISE.
  - `grpo-qwen3.5-35ba3b-2n8g-megatron-ep16tp2-fp8`: the extra graph memory
    left 833 Mamba cache blocks for `max_num_seqs` 1024, so vLLM failed at
    startup. The driver then sat idle until the job was cancelled.
- FULL_AND_PIECEWISE won clearly on `grpo-nanov3-30BA3B-1n8g-fsdp2.v2` and
  `grpo-nemotron3-super-120BA12B-16n8g-megatron`: decode 2.0× and 1.5×
  faster than PIECEWISE, with the same gates.
- vLLM 0.20 had an accuracy bug with FULL graphs. In Nano V3.5 GRPO, 271–2,548
  of 8,192 sequences per step were masked, against 1–5 with PIECEWISE.
  PIECEWISE-only, with `pass_config.fuse_allreduce_rms: false`, was that
  version's workaround, not a general rule.
- If logprob gates regress with graphs, try
  `compilation_config.backend: eager`. It keeps graphs but replays vLLM's
  custom kernels instead of Inductor-compiled ones.
  - On `grpo-gspo-deepscaler-1.5b-8K`, Inductor raised the
    `token_mult_prob_error` median from 1.013 to 1.040 and failed the
    `gen_kl_error` gate.
  - `backend: eager` kept 1.013 and still cut generation by 48% and wall time
    by 40%.

**Traps.**
- **Treat decode batch sizes and prefill token buckets as separate factors.**
  Check that the runtime captured what you expect: graph sizes above the
  admission cap get pruned (SGLang), and some sizes may be missing.
- **SGLang:** piecewise prefill graphs (`tc_piecewise`) crash on torch 2.10,
  and the `breakable` backend is rejected under the memory saver. The example
  config sets `cuda_graph_backend_prefill: disabled`.
- **Megatron:**
  - `cuda_graph_impl` and `inference_cuda_graph_scope` must be paired;
  - colocated runs allow only `none`/`local`;
  - EP > 1 with graphs needs `moe_pad_experts_for_cuda_graph_inference`.
- **Setup cost.** Graph capture adds setup time: +21 s on the Nano DAPO test.
  Count it in the wall-time comparison.

## 2. Scheduler token budget

**Why.** The token budget caps how much prefill and decode work one engine
iteration schedules. Too small, and long or multi-turn prompts queue, which
inflates latency and agent timeouts. Too large, and activation memory grows
and latency suffers. The optimum depends on engine, topology and workload.

| Backend | Knobs |
| :-- | :-- |
| vLLM | `vllm_kwargs.{max_num_batched_tokens, enable_chunked_prefill}` (pass-through) |
| SGLang | `sglang_cfg.{chunked_prefill_size, max_prefill_tokens, schedule_policy, schedule_conservativeness}`; `chunked_prefill_size: -1` disables chunking |
| TRT-LLM | `trtllm_cfg.max_num_tokens`; chunked prefill via `trtllm_kwargs.enable_chunked_prefill` (not typed). The scheduler policy is hard-coded to `MAX_UTILIZATION` |
| Megatron | `mcore_generation_config.{max_tokens, enable_chunked_prefill}` |
| Dynamo | `vllm_kwargs.max_num_batched_tokens` |

**Evidence** (SWE benchmark).

| Change | Effect | Grade |
| :-- | :-- | :-: |
| TRT-LLM 8,480 → 32,768 | Timeouts 311/960 → 1/320; tokens/s 876 → 1,262 (+44%); trajectory p50 1,151 → 624 s | B |
| TRT-LLM 32,768 → 65,536 | One valid attempt; tokens +12%, but trajectory p99 +11.6%. Not promoted | C |
| Megatron inference 16,384 → 32,768 | Reported as the largest single win of its tuning series; no retained before/after | D |
| vLLM TP4×2, 2,048 → 4,096 | Throughput −15% | C |

**Traps.**
- **Chunked prefill must be on.** Without it, prompts longer than the budget
  are rejected (TRT-LLM returned HTTP 400).
- **The budget can cap context.** A TRT-LLM build with an 8,192 budget
  truncated long contexts until its scheduler was patched.
- **The budget costs activation memory.** A Megatron inference run on GB200
  ran out of memory during CUDA-graph warmup until both `max_tokens` and the
  Mamba prefix budget were cut.
- **Batch math rollouts rarely need this lever.** Short prompts and long
  decodes put them in the decode-tail regime.

## 3. Prefix/KV reuse and session affinity

**Why.** Multi-turn rollouts resend a growing conversation prefix every turn.
The raw prefill-to-generated ratio of 115:1 drops to 2.8:1, but only if the
previous turn's KV is still cached on the replica that receives the next turn.
GRPO with many generations per prompt also shares the prompt prefix.

| Backend | Knobs | NeMo-RL behaviour |
| :-- | :-- | :-- |
| vLLM | `vllm_cfg.enable_prefix_caching` (null means on for SM ≥ 8.0); hybrid models: `vllm_kwargs.{mamba_cache_mode, mamba_ssm_cache_dtype}` | The prefix cache is reset at every `finish_generation`, so reuse lasts within a step only |
| SGLang | Radix cache, always on: `sglang_cfg.disable_radix_cache` is declared but not forwarded | The cache is flushed at startup and on KV invalidation |
| TRT-LLM | `trtllm_kwargs.kv_cache_config.{enable_block_reuse, tokens_per_block, host_cache_size}` | The prefix cache is reset after generation when not colocated; colocated sleep drops the KV |
| Megatron | `mcore_generation_config.{enable_prefix_caching (default false), prefix_caching_coordinator_policy (default longest_prefix), prefix_caching_routing_alpha, prefix_caching_eviction_policy, prefix_cache_ttl_seconds, prefix_caching_mamba_gb}` | The coordinator routes requests to the replica with the longest cached prefix |
| Dynamo | `vllm_kwargs.enable_prefix_caching`; `dynamo_cfg.frontend_args.router_mode: kv` for KV-aware routing | One frontend URL; the router decides the replica |

**Routing.**
- **NeMo Gym's `vllm_model` response server** pins each session to one
  backend URL by hashing the session ID, so all turns of a trajectory reach
  the same replica. This is in the current Gym pin; older pins kept affinity
  only per Uvicorn worker.
- **The optional single-controller `async_rl.generation_router`** sends each
  request to the least-busy backend, which gives up that affinity. Measure
  prefix-hit rate before and after enabling it.
- **TRT-LLM's native (non-HTTP) path** round-robins per request.

**Evidence** (SWE benchmark).

| Change | Effect | Grade |
| :-- | :-- | :-: |
| TRT-LLM rc24 block reuse off → on | Rollout −18.7%; tokens/s +22.5%; request p50 −18.1% | B |
| TRT-LLM per-conversation KV/Mamba cache bundle | Request p99 −17%; tokens/s +3% (the harness also changed) | C |
| Megatron `longest_prefix` coordinator, in a training run | Prefill skip rate 59–63% → 98% | D |
| Dynamo frontend | About 88% prefix-hit rate observed | D |

**Traps.**
- **Hybrid (Mamba) state reuse is version-gated, and a correctness option
  first.**
  - TRT-LLM rc21 crashed with block reuse on a hybrid model (`Invalid
    recurrent state block index`); rc24 worked. NeMo-RL main pins
    `tensorrt-llm==1.3.0rc21`.
  - An SGLang Mamba radix strategy resolved to a buffer mode under overlap
    scheduling, and reward fell from about 0.29 to 0.13 while requests looked
    faster.
  - Gate every state-cache change on accuracy.
- **Retention scope can silently fall back.** TRT-LLM per-conversation reuse
  falls back to per-request when the caller does not pass conversation
  parameters. Check the scope the engine logs.
- **Router replay (R3) needs prefix caching off.**

## 4. Parallel layout

**Why.** Under a synchronous step, the slowest replica sets the step. For a
fixed GPU count, more and smaller replicas usually cut per-token decode
latency. MoE expert layout changes communication.

| Backend | Knobs |
| :-- | :-- |
| vLLM | `vllm_cfg.{tensor_parallel_size, pipeline_parallel_size, expert_parallel_size}`; replicas = generation GPUs / (TP × PP); EP > TP turns on vLLM DP |
| SGLang | `sglang_cfg.{tp_size, dp_size, pp_size, ep_size}`; PP must be 1 |
| TRT-LLM | `trtllm_cfg.{tensor_parallel_size, moe_tensor_parallel_size, moe_expert_parallel_size}` with MoE TP × MoE EP = TP; `trtllm_kwargs.enable_attention_dp`; PP must be 1 |
| Megatron | `mcore_generation_config.{tensor_model_parallel_size, pipeline_model_parallel_size, expert_model_parallel_size, expert_tensor_parallel_size}`; CP is forced to 1 |
| Dynamo | `vllm_cfg.{tensor_parallel_size, pipeline_parallel_size, expert_parallel_size}`; EP is 1 or equal to TP; one engine cannot span nodes |

The generation layout is independent of training parallelism. Colocated
generation shares GPUs with training. Non-colocated generation uses
`generation.colocated.resources`.

**Evidence.**

| Change | Effect | Grade |
| :-- | :-- | :-: |
| vLLM with graphs, Nano DAPO 4n8g: TP8×4 instead of TP4×8 | Generation about 25% slower; stopped after 4 steps | C |
| Megatron inference: TP4/EP4 × 2 → TP1 × 8 (with buffer cuts) | Trajectory p50 443 → 365 s | B |
| TRT-LLM: MoE TP4/EP1 → TP1/EP4, + NCCL all-reduce, + prefill graphs | Rollout −5.1%; tokens/s +3.8%; p50 flat | B |
| vLLM, Nemotron 3 Ultra: EP8 vs. EP1 | No difference | C |

**Traps.**
- **Smaller replicas leave less KV and state memory per GPU**; check the
  engine's concurrency line against the per-replica load.
- **Don't infer a TP effect from a run that also changed the node count or
  global concurrency.** A TP1×4 single-node run was 41% slower in trajectory
  p50, but its allocation and load also changed.
- **Check the layout that ran.** A profile named "TEP4" actually ran MoE
  TP4/EP1.

## 5. Admission cap and memory

**Why.** The admission cap (maximum concurrent sequences) and the KV/state
memory decide whether the engine can hold its load. In RL the load per
replica is known in advance:

> prompts × generations ÷ replicas (synchronous), or the live-trajectory
> budget ÷ replicas (agentic).

| Backend | Admission cap | Memory |
| :-- | :-- | :-- |
| vLLM | `vllm_kwargs.max_num_seqs` | `vllm_cfg.gpu_memory_utilization`, `vllm_kwargs.block_size` |
| SGLang | `sglang_cfg.max_running_requests`; client-side `sglang_cfg.sglang_server_config.sglang_server_concurrency` | `sglang_cfg.mem_fraction_static` |
| TRT-LLM | `trtllm_cfg.max_batch_size` (also the CUDA-graph batch cap) | `trtllm_cfg.gpu_memory_utilization`, or `trtllm_kwargs.kv_cache_config.free_gpu_memory_fraction` (wins if set) |
| Megatron | `mcore_generation_config.max_requests` (pass-through) | `buffer_size_gb` (the real buffer is about 2×), `block_size_tokens`, `kv_cache_management_mode` |

**Rules.**
- **Read the engine's capacity line.** vLLM's `Maximum concurrency for N
  tokens per request: Xx`. If X is far above the per-replica load, memory
  knobs cannot help. Many math tests use 1–5% of KV.
- **Hybrid models** keep one state slot per running sequence. A cap above the
  state slots fails at startup:
  - vLLM `max_num_seqs` 1024 against 833 Mamba blocks;
  - the hybrid Nemotron Gym configs (`examples/nemo_gym/nemotron-3-ultra/*.yaml`)
    use `max_num_seqs: 256`.
- **Keep the cap at or above the largest decode graph size.** SGLang pruned
  graph sizes above its cap.
- **Colocated generation shares memory with training.** Check training
  headroom before raising memory fractions.
- **KV dtype changes** (`kv_cache_dtype`, FP8) are numerics changes, not
  memory knobs.

## 6. Kernels and collectives

| Backend | Knobs |
| :-- | :-- |
| vLLM | `vllm_kwargs.moe_backend`; `vllm_kwargs.compilation_config.pass_config.fuse_allreduce_rms`; `vllm_kwargs.disable_custom_all_reduce`; NCCL environment via `vllm_cfg.env_vars` |
| SGLang | Not forwarded today (`disable_custom_all_reduce` and `enable_nccl_nvls` are declared but dropped) |
| TRT-LLM | `trtllm_kwargs.allreduce_strategy` (for example NCCL vs. MNNVL) |
| Megatron | Not exposed |

**Evidence.**

| Change | Effect | Grade |
| :-- | :-- | :-: |
| vLLM MoE backend `flashinfer_cutlass` → `auto` (FlashInfer TRTLLM) on GB200 | Per-turn p50 −12.8%; trajectory p50 −7% | C |
| vLLM `compilation_config.backend: eager` with graphs | Restored logprob gates lost under Inductor (§1) | C |
| TRT-LLM NCCL instead of MNNVL | Part of the 5% bundle in §4 | B |

**Traps.**
- **Backend availability depends on hardware.** Some MoE kernels target only
  Blackwell. Confirm the selected backend from the engine log (`Using ... MoE
  backend`) on every new GPU type.
- **Missing tuned kernel configs fall back silently** to default kernels;
  for example, SGLang fused-MoE tuning files for a new GPU. Look for the
  fallback warning.
- **Some kernel choices are refit-incompatible.** Example: vLLM 0.20's default
  FlashInfer TRTLLM MoE backend, which is why some recipes pin
  `moe_backend: triton`.

## 7. Frontend, tokenizer and harness

**Why.** In agentic rollouts, the HTTP layer, tokenization and the agent
harness can starve the engine. When GPUs sit idle and engine queues are empty,
engine knobs cannot help.

| Surface | Knob |
| :-- | :-- |
| Response-model workers (NeMo Gym) | `env.nemo_gym.policy_model.responses_api_models.vllm_model.num_workers` (16 in `examples/nemo_gym/nemotron-3.5-lightning/rlvr.yaml`) |
| HTTP frontends | Megatron `mcore_generation_config.http_server_num_replicas`; one in-process server per vLLM replica (`vllm_cfg.expose_http_server`) |
| Tokenizer | Dynamo `dynamo_cfg.frontend_args.{tokenizer, tokenizer_cache}`; fast-tokenizer settings must reach the driver, the Gym actor *and* the engine workers |
| Agent load | Concurrency per node, `agent_max_turns`, agent timeouts |
| Placement | Nodes in the same NVLink domain or scheduler block |

**Evidence** (SWE benchmark).

| Change | Effect | Grade |
| :-- | :-- | :-: |
| Fast tokenizer propagated into vLLM workers (it had silently stayed off there) | Wall −10%; token-normalized throughput +11% | C |
| Response-model `num_workers` unset → 16 | Tokens/s +6.2%; rollout −4.8%; request p99 +2.6% | C |
| Both nodes in one NVL72 block plus a matching topology segment | Tokens/s +5.1% (median of 5 pairs; 4 of 5 won) | B |
| 32 instead of 21 agent containers per node | Trajectory time about +22% (replica count also changed) | D |
| Node-local agent result transport | Tokens/s −14% | C |

**Rules.**
- **`max_turns` and agent timeouts change the workload, not the engine.**
  Freeze them across arms. Their safe values depend on concurrency: a turn cap
  that had no timeouts at 16–32 concurrent agents timed out at 64.
- **Prove a frontend setting from the runtime.** For example, check for
  Uvicorn multiprocess startup when `num_workers > 1`.

## 8. Speculative decoding

| Backend | Knobs |
| :-- | :-- |
| vLLM | `vllm_kwargs.speculative_config` (for example `{method: deepseek_mtp, num_speculative_tokens: 1}`) |
| Megatron | `mcore_generation_config.num_speculative_tokens` |
| SGLang | Draft weights are kept on CPU so training runs without MTP weights |
| Dynamo | Not supported |

**Rules.**
- **The drafter must follow the policy.**
  - When the policy does not train the MTP head, NeMo-RL loads the drafter
    from the checkpoint and never refits it
    (`load_mtp_weights_from_disk` in `vllm_worker.py`).
  - The frozen drafter drifts from the trained policy, and acceptance falls.
  - Track `train/vllm/spec_acceptance_rate` over the whole run, not the
    first steps.
- **Measure the trade-off.** Speculation trades decode latency for extra
  compute. Gains shrink at high batch sizes.

## 9. Startup

Startup affects wall time and GPU-hours, not rollout metrics; report it
separately (`timing/setup/total_setup_time_s`).

- **TRT-LLM:** parallel weight loading
  (`TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=False`). It was adopted for
  rollout-only benchmarks and is not validated with weight refit, so don't use
  it in training without testing refit.
- **Dependencies:** prebuild virtual environments and stage kernel caches
  (FlashInfer cubins) on CPU nodes. Don't compile on GPU nodes.
- **JIT caches:** keep per-rank JIT caches (Triton, Inductor, vLLM) at the
  container or node-local defaults, not on a shared filesystem.
- **Graph capture:** a bigger capture list costs setup time. Megatron
  `num_cuda_graphs`, vLLM capture sizes.
