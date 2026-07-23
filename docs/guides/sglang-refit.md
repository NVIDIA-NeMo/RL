# SGLang Generation and Weight Refit

NeMo RL GRPO can train with Megatron while SGLang serves rollouts, then stream
each new policy version directly into the SGLang engines. The supported paths
do not write an intermediate checkpoint:

| Topology | Policy backend | Transfer |
|---|---|---|
| Colocated | Megatron or DTensor | CUDA IPC through the SGLang weight-update API |
| Non-colocated | Megatron | NCCL broadcast from trainer rank 0 |

Non-colocated DTensor-to-SGLang refit is not supported. The non-colocated path
is currently wired through GRPO. SGLang refit also requires
`policy.generation.refit_transport: null`; the vLLM sparse and checkpoint-engine
transports are separate implementations.

## Install

The project container builds separate Megatron and SGLang environments from the
locked dependency groups. Use [`docker/Dockerfile`](../../docker/Dockerfile) so
the Ray actors select the matching environment automatically. Initialize the
repository submodules before building:

```bash
git submodule update --init --recursive
docker buildx build \
  -t nemo-rl-sglang-refit:local \
  --build-arg NEMO_GYM_PREFETCH_CONFIGS=examples/nemo_gym/prefetch_sglang_swe1.yaml \
  -f docker/Dockerfile .
```

For NeMo-Gym rollouts, the Gym submodule must include its native SGLang model
adapter. The adapter sends OpenAI Responses requests through SGLang's
`/generate` endpoint while preserving the exact sampled token prefix between
turns.

## Non-Colocated Configuration

The key distinction is the dedicated generation pool. This example creates one
TP1 SGLang engine per generation GPU:

```yaml
policy:
  generation:
    backend: sglang
    refit_transport: null
    vllm_cfg: null
    sglang_cfg:
      model_path: ${policy.model_name}
      dtype: ${policy.precision}
      context_length: ${policy.max_total_sequence_length}
      tp_size: 1
      pp_size: 1
      dp_size: 1
      ep_size: 1
      random_seed: 42
      skip_server_warmup: true
      refit_timeout_s: 1800
      quantization:
        scheme: bf16
      sglang_server_config:
        num_gpus: 4
        num_gpus_per_engine: 1
        needs_offload: false
        cpu_weight_backup: false
        pause_generation_mode: retract
        sglang_server_concurrency: 64
        weight_transfer_mode: broadcast
      sglang_router_config:
        use_external_router: false
    colocated:
      enabled: false
      resources:
        gpus_per_node: 4
        num_nodes: 1
```

`num_gpus` must equal the GPUs reserved for generation, and
`num_gpus_per_engine` must match `tp_size` while pipeline parallelism is 1. A
non-colocated generation pool should set `needs_offload: false` and
`cpu_weight_backup: false`; those memory-saver settings are for engines that
share GPUs with training.

The broadcast communicator contains trainer rank 0 plus every GPU rank in the
SGLang engines. Only trainer rank 0 broadcasts because Megatron restores full
Hugging Face tensors there; the other trainer ranks participate in the
Megatron-side tensor restoration but do not join the SGLang communicator.

## Refit Smoke Test

The two-node smoke recipe runs three synchronous GRPO steps on a public model.
It uses one 4-GPU training node and one 4-GPU generation node, producing a
five-rank SGLang refit communicator:

```bash
bash tests/test_suites/llm/grpo-qwen2.5-math-1.5b-instruct-2n4g-megatron-sglang-noncolocated-quick.sh
```

Run the script on an allocated two-node Ray cluster. The driver requires all of
the following:

1. `train/loss` reaches step 3.
2. `NRL_SGLANG_REFIT_GROUP_READY` appears.
3. At least two `NRL_SGLANG_REFIT_SUCCESS` markers appear.
4. No `NRL_SGLANG_REFIT_FAILURE` marker appears.

The recipe is intentionally synchronous: each completed step deterministically
exercises generation, training, refit, and resumed generation.

## Async GRPO with NeMo-Gym

The public SWE1 integration recipe exercises the complete async path:

```bash
bash tests/test_suites/llm/grpo-qwen3-30ba3b-thinking-swe1-16n8g-megatron-async-gym-sglang.sh
```

Prepare the public `swe1.jsonl` split under
`${HF_HOME}/superv3_data/swe1/` as described in
[Two-Stage SWE RL](swe-rl-qwen3.md). This recipe uses eight training nodes and
eight generation nodes. It is a three-step integration check, not a
convergence run.

The SGLang async configuration deliberately sets:

- `grpo.async_grpo.in_flight_weight_updates: false`. Replay collection is
  pipelined, but SGLang refit pauses generation and remains a synchronization
  barrier.
- `policy.router_replay.enabled: false`. The NeMo-Gym SGLang adapter does not
  return routed-expert traces.
- `env.nemo_gym.truncate_noncontiguous_episodes: false`. SWE1 is single-turn;
  token-prefix mismatches should fail rather than be hidden.
- A non-null SGLang `context_length` and the same
  `context_length` in the Gym adapter, so an unlimited
  Responses request consumes the remaining model context instead of SGLang's
  short default generation cap.

## Reading the Markers

`NRL_SGLANG_REFIT_GROUP_READY` means both sides completed communicator
construction. `"Connected all rings"` is an intermediate NCCL bootstrap trace,
not proof that the bootstrap all-gather or a weight broadcast completed.

`NRL_SGLANG_REFIT_SUCCESS` is emitted only after the weight transfer,
post-processing, and generation resume complete. Treat these as fatal:

- `NRL_SGLANG_REFIT_FAILURE`
- KV-cache invalidation failure
- NCCL system or remote error
- collective watchdog timeout
- `Failed to recv`
- connection reset

There is no disk-refit fallback. Diagnose and repair the failed in-memory path
before retrying.

## Troubleshooting

- **Failure before `GROUP_READY`:** inspect the trainer-rank-0 and engine-leader
  logs together. The rendezvous store can be healthy even when NCCL
  communicator finalization fails.
- **Timeout during a bucket:** the refit deadline covers lock acquisition,
  engine receive setup, Ray waits, and SGLang-side collectives. The failure
  path performs bounded communicator cleanup; another refit is refused until
  cleanup is confirmed.
- **Generation fails after a successful refit:** verify dedicated engines use
  `needs_offload: false`; otherwise released weights may not have been restored
  before generation resumes.
- **Gym token-prefix assertion:** verify the Gym SGLang adapter is active
  (`responses_api_models.sglang_model`), the context limits match, and the
  configured tool format matches the model template.

See [Weight Refit](refit.md) for the transport matrix and
[Async GRPO](async-grpo.md) for replay-age and startup-barrier semantics.
