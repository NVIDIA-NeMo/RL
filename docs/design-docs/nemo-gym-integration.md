# NeMo Gym Integration

This document describes how NeMo RL integrates with [NeMo Gym](https://docs.nvidia.com/nemo/gym/v0.2.1/index.html) for multi-step and multi-turn rollout collection. NeMo Gym rollouts are supported by GRPO and on-policy distillation.

## Overview

NeMo Gym provides HTTP-based training environments for LLMs. **NeMo Gym is CPU-only**—it runs no inference engines and holds no GPU memory. NeMo RL exposes its vLLM generation engine as an OpenAI-compatible HTTP server, which NeMo Gym calls during rollouts, enabling:

- **Decoupled architecture**: Environments don't need direct access to model internals
- **Multi-step/multi-turn support**: Agents can orchestrate complex interactions with tools
- **Refit compatibility**: NeMo RL's weight synchronization works transparently

The same NeMo Gym rollout path is used by GRPO and on-policy distillation when `env.should_use_nemo_gym` is enabled. Distillation uses the generated NeMo Gym conversations as the student on-policy samples before computing teacher logits and the distillation loss.

For on-policy distillation, NeMo Gym controls the rollout turn count from its environment and agent configuration. The standard distillation `distillation.max_rollout_turns` setting is not used by the NeMo Gym rollout path.

## Configuration

To enable NeMo Gym integration, add the following to your NeMo RL config:

```yaml
policy:
  generation:
    backend: vllm
    vllm_cfg:
      async_engine: true          # Both required for HTTP server support:
      expose_http_server: true    # async_engine enables the async worker; expose_http_server starts the server

env:
  should_use_nemo_gym: true       # Enables NeMo Gym integration
  nemo_gym:
    # NeMo Gym config paths and settings
    config_paths:
      - resources_servers/math/configs/math.yaml
      - responses_api_agents/simple_agent/configs/simple_agent.yaml

logger:
  wandb:
    # Optional debugging aid. Keep disabled for normal training because complete
    # result payloads can produce many large W&B Table artifacts.
    log_nemo_gym_full_result_tables: false
```

When `log_nemo_gym_full_result_tables` is `false`, NeMo RL does not construct
the per-agent `full_result` Tables. This prevents those payloads from entering
the async replay buffer and avoids uploading them to W&B. Numeric per-agent
rollout metrics are unaffected. Set the flag to `true` only when the complete
Gym result payloads are needed for a short debugging run.

For complete examples, see `examples/nemo_gym/run_grpo_nemo_gym.py`, `examples/nemo_gym/run_distillation_nemo_gym.py`, and their associated configs under `examples/nemo_gym/`.

### Version Requirements

NeMo Gym runs as a Ray actor within NeMo RL's Ray cluster, so the same Ray and Python versions must be used in both environments.

## Harbor and Exact-Call Training

Harbor integration uses Gym's `responses_api_agents/harbor_agent_general`.
Its `harbor_agent`, `harbor_environment`, `harbor_dataset`, and `harbor_verifier`
fields use Harbor's typed configurations. Policy and verifier harnesses can be
configured independently. Direct-provider evaluation does not require an RL
model server. For training, configure the agent's `model_server`,
`model_api_key`, and `token_id_capture: true` so policy requests traverse Gym's
capture endpoint before reaching the NeMo RL vLLM workers.

Dataset JSONL records must name the registered agent in `agent_ref`; updating
the config path alone does not migrate an existing manifest. Datasets, model
weights, provider credentials, and deployment-specific mounts remain external
inputs. See Gym's `responses_api_agents/harbor_agent_general/TRAINING.md` for
OpenSandbox and migration details.

### Representation and Ownership

Harbor's ATIF trajectory describes the interaction, but the training source of
truth is the token record captured for each inference call. A harness may
retokenize history, compact it, or invoke a subagent. The trainer must not
replace any of those exact contexts with a textually equivalent concatenation.

| Layer | Contract |
|-------|----------|
| Gym capture | Preserve prompt IDs, sampled IDs, sampled logprobs, optional expert routes, and replica/cache/weight-version metadata per call. |
| Exact-call preparation | Build a route-compatible prefix tree from exact token IDs; incompatible contexts remain separate branches. |
| Physical model input | Store unique token runs with parent/depth metadata. A node attends only to its own causal ancestry. |
| Policy loss | Maintain a separate sampled-edge stream containing targets, logprobs, masks, and advantages. Each sampled occurrence contributes once, even if its context is shared. |
| Bounded fragments | Repeat ancestor context as needed to fit a microbatch, but assign each sampled edge to exactly one fragment. |
| Reassembly | Return edge values to the original logical rollout order before policy-loss bookkeeping. |

The final generated token may lack an executed routing row because no next
token was decoded from it. If a later request executes that same context, its
known route can materialize the node. After that, another continuation with a
different known route must branch; the original missing row is not a wildcard
that permits incompatible continuations to share a hidden state. Replica,
weight-version, and cache metadata further constrain route-prefix reuse.

The pure construction algorithms live in `nemo_rl/data/exact_calls.py`.
`nemo_rl/data/packed_rollouts.py` defines layouts, ancestor-closed fragmentation,
and sampled-edge reassembly. GRPO attaches episode rewards and advantages;
these representations do not introduce a separate reward or normalization rule.

Select Gym's `token_id_capture.builder: independent_calls` for this contract.
The upstream default, `prefix_merging`, projects a verified terminal chain and
does not preserve the whole episode for training. Native external-staging
receipts currently support that single-chain contract only; combining them
with `independent_calls` is rejected before Gym servers start. Whole-episode
trees use the ordinary Gym payload with either legacy or TQ ingestion instead.

### Capture Lifetime

Freezing a capture does not delete it. The Gym actor returns a small frozen
snapshot identity alongside the token payload, without retaining a second
copy of the token or routing arrays. Successful acceptance permits retirement:

- Async TQ: after the canonical replay-buffer commit succeeds.
- Legacy async: after the replay-buffer actor accepts the complete prompt group.
- Sync TQ: after the first data-plane write succeeds.
- Sync without TQ: after the caller has assembled and validated the complete
  returned batch; this path has no replay buffer.

Retirement is conditional on the same snapshot ID and version. Unknown,
changed, failed, or capture-masked snapshots are not deleted. Cleanup failure
does not retry or discard an already accepted rollout. Set Gym's
`token_id_capture.retain_consumed: true` to keep successful captures for offline
inspection. The Gym actor closes only sources it created, never a borrowed
framework-installed source. This handoff is not a per-rollout disk checkpoint;
durability across a full job failure still depends on the training checkpoint.

### Legacy and Transfer Queue

Legacy delivery and the Transfer Queue (TQ) must preserve the same logical
rollout, exact-call metadata, sampled-edge ownership, and row order. Tree model
inputs are node-aligned; loss tensors are edge-aligned and need not have the
same sequence length. Treating both as ordinary token-aligned tensors would
silently change the objective or create unnecessarily large padded broadcasts.

For oversized TQ trees, dispatch carries fragment selections and row metadata.
The worker materializes bounded physical microbatches before model-parallel
broadcast rather than broadcasting the whole padded logical group. The legacy
path uses the same fragment planner and edge reassembly contracts.

SingleController resumes the current upstream native TQ checkpoint format.
Historical `replay_buffer.pt` and `replay_buffer/manifest.json` checkpoints are
rejected when native replay metadata is absent, rather than silently losing
buffered rollouts. Resume those using their original implementation or make
an explicit decision to restart without the old buffer; there is no automatic
format migration in this patch.

### Backend Requirements

Shared-prefix execution requires the matching Megatron-Core tree attention,
Mamba state propagation, and sampled-edge projection support, as well as
`policy.sequence_packing.enabled: true`. Physical microbatch budgets are
`policy.sequence_packing.train_mb_tokens` and `logprob_mb_tokens`; a logical
root-to-leaf path must still fit the context and fragment budget. Fragmenting a
wide tree does not truncate a long path. Multimodal tree fragmentation is not
supported, and unsupported loss/backend combinations fail explicitly.

Whole-episode independent-call training is currently a GRPO feature. PPO and
teacher/distillation consumers reject explicit independent-call requests rather
than training only the compatibility transcript. Their ordinary native-agent
and upstream prefix-merging paths retain upstream behavior. A synchronous
consumer must opt in with `allow_independent_calls=True` only after implementing
the exact-call contract; returning a token-bearing transcript is not sufficient.

CPU layout tests establish token/route alignment and single ownership. They do
not establish numerical kernel parity. Changes to backend execution also need
forward and gradient comparisons against independent paths for the configured
CP/TP topology and model family, including Mamba models.

## Architecture Overview

```mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
    subgraph RL["NeMo RL"]
        Loop["Training Loop<br/>(GRPO or Distillation)"]
        vLLM["vLLM + HTTP"]
        Bridge["NemoGym Actor"]
    end
    
    subgraph Gym["NeMo Gym"]
        Agent["Agent"]
        Model["Model (Proxy)"]
        Resources["Resources"]
    end
    
    Loop -->|refit| vLLM
    Loop -->|run_rollouts| Bridge
    Bridge -->|spawns| Gym
    Agent <--> Model
    Agent <--> Resources
    Model -->|HTTP| vLLM

    style RL fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Gym fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

**Color coding**:
- Blue = NeMo RL code (`nemo_rl/`)
- Orange = NeMo Gym code (`3rdparty/Gym-workspace/Gym/nemo_gym/`)

## The NemoGym Actor

The integration is handled by the `NemoGym` Ray actor at `nemo_rl/environments/nemo_gym.py`:

1. **Created by NeMo RL** during training setup via `NemoGym.remote(config)`
2. **Joins the existing Ray cluster** that NeMo RL already initialized
3. **Spawns NeMo Gym servers** as OS subprocesses (Head, Agent, Model, Resources)
4. **Injects vLLM base URLs** so NeMo Gym's Model Server knows where to proxy requests
5. **Exposes `run_rollouts()`** as the entry point for the training loop to call

```mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
    subgraph RL["NeMo RL"]
        Loop["Training Loop<br/>(GRPO or Distillation)"]
        Actor["NemoGym Actor"]
    end
    
    subgraph Gym["NeMo Gym"]
        RCH["RolloutCollectionHelper"]
        Agent["Agent Server"]
    end
    
    Loop --> Actor
    Actor --> Agent
    Agent --> RCH
    RCH --> Actor
    Actor --> Loop

    style RL fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Gym fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

The flow is:
1. The GRPO or distillation rollout layer starts a streaming `run_rollouts` call on the NemoGym Actor
2. Actor sends `POST /run` to the Agent Server
3. Agent Server orchestrates the rollout via RolloutCollectionHelper
4. Completed examples return to the Actor
5. Actor post-processes and streams each completed example back with its original row index
6. The rollout layer emits a prompt group after all generations for that prompt are complete; synchronous callers drain the stream and retain full-batch behavior

## vLLM HTTP Server

**NeMo Gym does not run its own vLLM engine.** The Model Server is purely an HTTP proxy:

| Aspect | NeMo RL vLLM Worker | NeMo Gym Model Server |
|--------|---------------------|----------------------|
| **Engine** | Runs actual vLLM `AsyncLLM` | No engine - HTTP proxy only |
| **GPU** | Holds model weights | No GPU required |
| **Endpoints** | `/v1/chat/completions`, `/tokenize` | `/v1/responses` |
| **Role** | Inference | API translation, forwards requests |

Data parallel vLLM workers each expose their own HTTP server. NeMo Gym's Model Server load-balances requests across them.

## Initialization Sequence

```mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
sequenceDiagram
    autonumber
    box rgb(227, 242, 253) NeMo RL
        participant RL as Training Script
        participant Ray as Ray Cluster
        participant vLLM as vLLM Workers
        participant Bridge as NemoGym Actor
    end
    box rgb(255, 243, 224) NeMo Gym
        participant Servers as NeMo Gym Servers
    end
    
    RL->>Ray: Initialize Ray cluster
    RL->>vLLM: Create vLLM workers with HTTP servers
    vLLM-->>RL: Return base URLs (one per DP rank)
    RL->>Bridge: NemoGym.remote(config, base_urls)
    Note over Bridge: Reuses existing Ray cluster
    Bridge->>Servers: Spawn subprocess servers
    Servers-->>Bridge: Health check OK
    Bridge-->>RL: Ready for rollouts
```

## Training Loop Control Flow

```mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
sequenceDiagram
    autonumber
    box rgb(227, 242, 253) NeMo RL
        participant Loop as Training Loop
        participant Policy as Policy Workers
        participant vLLM as vLLM HTTP
        participant Bridge as NemoGym Actor
    end
    box rgb(255, 243, 224) NeMo Gym
        participant Agent as Agent Server
        participant Model as Model Server
        participant Resource as Resource Server
    end
    
    Loop->>Policy: Refit (trigger weight sync)
    Policy->>vLLM: Sync weights to vLLM
    Loop->>Bridge: streaming run_rollouts(batch)
    Bridge->>Agent: POST /run
    Agent->>Model: POST /v1/responses
    Model->>vLLM: POST /v1/chat/completions
    vLLM-->>Model: Response
    Model-->>Agent: Responses API format
    Agent->>Resource: Execute tool / compute reward
    Resource-->>Agent: Tool result / reward
    Agent-->>Bridge: Completed example + reward
    Bridge-->>Loop: Stream row index, token IDs, logprobs, reward
    Note over Loop,Bridge: Async GRPO emits each complete prompt group;<br/>sync paths drain all rows before continuing
    Loop->>Policy: Compute loss and train
```

> **NeMo Gym server types** (see [Core Components](https://docs.nvidia.com/nemo/gym/v0.2.1/about/concepts/core-components/)):
> - **Agent Server**: Orchestrates the rollout loop
> - **Model Server**: HTTP proxy to vLLM; translates Responses API ↔ Chat Completions
> - **Resource Server**: Provides tools and rewards

### Key Steps

| Step | Location | Description |
|------|----------|-------------|
| **Refit** | NeMo RL | Synchronizes policy weights to vLLM workers. For async RL, refit timing may differ—see {doc}`generation` for details. |
| **Streaming `run_rollouts()`** | NeMo RL | Ray generator call from the rollout layer to the NemoGym actor; rows can arrive out of input order |
| **POST /run** | NeMo RL → NeMo Gym | HTTP request from NemoGym actor to Agent Server subprocess |
| **Rollout orchestration** | NeMo Gym | Agent calls Model Server and Resources Server via HTTP |
| **POST /v1/chat/completions** | NeMo Gym → NeMo RL | Model Server proxies to NeMo RL's vLLM HTTP endpoint |
| **Result processing** | NeMo RL | NemoGym actor extracts token IDs, logprobs, rewards |

### Async Result Processing

The NemoGym actor and NeMo RL rollout layer use an **as-completed** pattern to overlap waiting, post-processing, and downstream collection:

1. **Completed examples return out of order**: Full rollout examples complete at different times depending on conversation length and tool calls. The actor processes and streams each example as soon as it completes, tagged with its original row index.

2. **Immediate post-processing**: As each rollout completes, the actor immediately extracts token IDs and logprobs. This overlaps CPU work with network I/O from slower rollouts still in flight.

3. **Prompt-group buffering**: Async GRPO groups the streamed rows by prompt and emits a group as soon as all of that prompt's generations have arrived. A slow prompt therefore does not prevent already-complete prompt groups from entering the replay buffer. Synchronous GRPO, PPO, and distillation use the same stream but drain the complete batch before continuing.

4. **Stable ordering where required**: Each example carries a row index. Prompt groups preserve their input slices, and full-batch synchronous callers restore input order before returning.

This pattern maximizes throughput by keeping the CPU busy while waiting for network responses.

### Async GRPO Collector Invariants

The async GRPO collector uses the same prompt-group contract for Gym and native environments:

- One batch worker owns each reserved target weight until all expected prompt groups are buffered or the batch fails.
- Every yielded prompt (Gym and native) receives a monotonic `_ng_task_index` equal to its position in the dataloader stream. The counter is checkpointed, restored, and cross-checked against buffered trajectories. On a legacy resume, task identities are never reused after restart; on a frontier-aligned resume, the counter is deliberately rewound to the saved base ordinal so the covered window re-yields under its original indices (rows already trained or retained are dropped before dispatch, and per-batch uniqueness is still enforced).
- A partial Gym stream can be retried without duplicating groups that were already accepted by the replay buffer.
- Native rollouts still execute as one batch. Their per-sample metrics are aggregated separately for each prompt group before buffering, so batch-level metrics are not duplicated across groups.
- Gym rows are validated for range, uniqueness, completeness, and single-agent grouping. Results are restored to input order within a prompt group before post-processing.

## Data Format Translation

```mermaid
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
    subgraph RL1["NeMo RL Input"]
        Datum["DatumSpec"]
    end
    
    subgraph Gym["NeMo Gym"]
        Example["Example Dict"]
        ReqResp["Responses API"]
        ReqChat["Chat Completions"]
    end
    
    subgraph RL2["NeMo RL Output"]
        Result["Result"]
    end
    
    Datum --> Example
    Example --> ReqResp
    ReqResp --> ReqChat
    ReqChat --> ReqResp
    ReqResp --> Example
    Example --> Result

    style RL1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style RL2 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Gym fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

**Formats**:
- **DatumSpec** (NeMo RL): Training-focused format with `prompt`, `prompt_token_ids`, and task metadata
- **Example Dict** (NeMo Gym): Environment-focused format containing `responses_create_params` and `expected` answer
- **Responses API** (NeMo Gym): OpenAI Responses API format with `input`, `tools`, and multi-turn conversation
- **Chat Completions** (vLLM): OpenAI Chat Completions format for the actual inference call

**Data flow**: DatumSpec is converted to Example Dict, which passes through to the Responses API with generation parameters (`temperature`, `top_p`) added for on-policy sampling. The Model Server translates Responses API ↔ Chat Completions (converting message formats, extracting reasoning content, attaching token IDs). Results flow back with token IDs and logprobs extracted into the final Result.

## Tokenization and On-Policy Corrections

Token IDs are extracted at the NeMo RL vLLM layer via the `/tokenize` endpoint. This ensures:
- Tokenization matches the exact model and tokenizer used for generation
- No re-tokenization drift between generation and training

For details on on-policy token ID handling, see {doc}`../guides/environments` and the [NeMo Gym on-policy corrections documentation](https://docs.nvidia.com/nemo/gym/v0.2.1/contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction.html).
