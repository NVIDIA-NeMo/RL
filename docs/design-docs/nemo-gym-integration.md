# NeMo Gym Integration

This document describes how NeMo RL integrates with [NeMo Gym](https://docs.nvidia.com/nemo/gym/latest/index.html) for multi-step and multi-turn reinforcement learning training.

## Overview

NeMo Gym provides HTTP-based training environments for LLMs. NeMo RL exposes its vLLM generation engine as an OpenAI-compatible HTTP server, which NeMo Gym calls during rollouts, enabling:

- **Decoupled architecture**: Environments don't need direct access to model internals
- **Multi-step/multi-turn support**: Agents can orchestrate complex interactions with tools
- **Refit compatibility**: NeMo RL's weight synchronization works transparently

## Configuration

To enable NeMo Gym integration, add the following to your NeMo RL config:

```yaml
policy:
  generation:
    backend: vllm
    vllm_cfg:
      async_engine: true          # Required for HTTP server support
      expose_http_server: true    # Exposes /v1/chat/completions endpoint

env:
  should_use_nemo_gym: true       # Enables NeMo Gym integration
  nemo_gym:
    # NeMo Gym config paths and settings
    config_paths:
      - resources_servers/math/configs/math.yaml
      - responses_api_agents/simple_agent/configs/simple_agent.yaml
```

### Version Requirements

NeMo Gym runs as a Ray actor within NeMo RL's Ray cluster, so the same Ray and Python versions must be used in both environments.

## Architecture Overview

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
    subgraph RL["NeMo RL"]
        GRPO["GRPO Loop"]
        vLLM["vLLM + HTTP"]
        Bridge["NemoGym<br/>Actor"]
    end
    
    subgraph Gym["NeMo Gym"]
        Agent["Agent"]
        Model["Model<br/>(Proxy)"]
        Resources["Resources"]
    end
    
    GRPO -->|refit| vLLM
    GRPO -->|run_rollouts| Bridge
    Bridge -->|spawns| Gym
    Agent <--> Model
    Agent <--> Resources
    Model -->|HTTP| vLLM

    style RL fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Gym fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

**Color coding**: Blue = NeMo RL code (`nemo_rl/`), Orange = NeMo Gym code (`nemo_gym/`)

## The NemoGym Actor

The integration is handled by the `NemoGym` Ray actor at `nemo_rl/environments/nemo_gym.py`:

1. **Created by NeMo RL** during training setup via `NemoGym.remote(config)`
2. **Joins the existing Ray cluster** that NeMo RL already initialized
3. **Spawns NeMo Gym servers** as OS subprocesses (Head, Agent, Model, Resources)
4. **Injects vLLM base URLs** so NeMo Gym's Model Server knows where to proxy requests
5. **Exposes `run_rollouts()`** as the entry point for the training loop

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
    subgraph RL["NeMo RL"]
        GRPO["GRPO Loop<br/><i>grpo.py</i>"]
        Actor["NemoGym Actor<br/><i>nemo_rl/environments/nemo_gym.py</i>"]
    end
    
    subgraph Gym["NeMo Gym"]
        RCH["RolloutCollectionHelper<br/><i>nemo_gym/rollout_collection.py</i>"]
        Agent["Agent Server<br/><i>responses_api_agents/*/app.py</i>"]
    end
    
    GRPO -->|"1. run_rollouts.remote(batch)"| Actor
    Actor -->|"2. POST /run"| Agent
    Agent -->|"3. orchestrates rollout"| RCH
    RCH -->|"4. returns results"| Actor
    Actor -->|"5. returns to training"| GRPO

    style RL fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Gym fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

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

```{mermaid}
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

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
sequenceDiagram
    autonumber
    box rgb(227, 242, 253) NeMo RL
        participant GRPO as GRPO Loop
        participant Policy as Policy Workers
        participant vLLM as vLLM HTTP
        participant Bridge as NemoGym Actor
    end
    box rgb(255, 243, 224) NeMo Gym
        participant Agent as Agent Server
        participant Model as Model Server
    end
    
    GRPO->>vLLM: Refit (sync weights)
    GRPO->>Bridge: run_rollouts.remote(batch)
    Bridge->>Agent: POST /run
    Agent->>Model: POST /v1/responses
    Model->>vLLM: POST /v1/chat/completions
    vLLM-->>Model: Response
    Model-->>Agent: Responses API format
    Agent-->>Bridge: Results + rewards
    Bridge-->>GRPO: Token IDs, logprobs, rewards
    GRPO->>Policy: Compute loss and train
```

### Key Steps

| Step | Location | Description |
|------|----------|-------------|
| **Refit** | NeMo RL | Synchronizes policy weights to vLLM workers. For async RL, refit timing may differ—see {doc}`generation` for details. |
| **run_rollouts.remote()** | NeMo RL | Ray remote call from GRPO loop to the NemoGym actor |
| **POST /run** | NeMo RL → NeMo Gym | HTTP request from NemoGym actor to Agent Server subprocess |
| **Rollout orchestration** | NeMo Gym | Agent calls Model Server and Resources Server via HTTP |
| **POST /v1/chat/completions** | NeMo Gym → NeMo RL | Model Server proxies to NeMo RL's vLLM HTTP endpoint |
| **Result processing** | NeMo RL | NemoGym actor extracts token IDs, logprobs, rewards |

### Async Result Processing

The NemoGym actor uses an **as-completed** pattern to overlap waiting with post-processing:

1. **Results return out of order**: Rollouts complete at different times depending on conversation length and tool calls. Rather than waiting for all results, the actor processes each result as soon as it completes.

2. **Immediate post-processing**: As each rollout completes, the actor immediately extracts token IDs and logprobs. This overlaps CPU work with network I/O from slower rollouts still in flight.

3. **Reordering at the end**: Each example carries an index. After all results are collected, results are reordered to match the original batch order before returning to the training loop.

This pattern maximizes throughput by keeping the CPU busy while waiting for network responses.

## Data Format Translation

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart LR
    subgraph RL1["NeMo RL (Input)"]
        Datum["DatumSpec"]
    end
    
    subgraph Gym["NeMo Gym"]
        Example["Example Dict"]
        ReqResp["Responses API"]
        ReqChat["Chat Completions"]
    end
    
    subgraph RL2["NeMo RL (Output)"]
        Result["Result"]
    end
    
    Datum -->|"convert"| Example
    Example --> ReqResp
    ReqResp -->|"translate"| ReqChat
    ReqChat -->|"vLLM"| ReqResp
    ReqResp --> Example
    Example -->|"extract"| Result

    style RL1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style RL2 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Gym fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

### Format Differences

| Format | Owner | Contents |
|--------|-------|----------|
| **DatumSpec** | NeMo RL | Training-focused: `prompt`, `prompt_token_ids`, task metadata for loss computation |
| **Example Dict** | NeMo Gym | Environment-focused: `responses_create_params` (OpenAI format), `expected` answer for verification |
| **Responses API** | NeMo Gym | OpenAI Responses API format with `input`, `tools`, multi-turn conversation |
| **Chat Completions** | NeMo RL vLLM | OpenAI Chat Completions format, the actual inference call |

The Model Server handles Responses API ↔ Chat Completions translation, including:
- Converting message formats
- Extracting reasoning content from think tags
- Attaching token ID information for training

## Tokenization and On-Policy Corrections

Token IDs are extracted at the NeMo RL vLLM layer via the `/tokenize` endpoint. This ensures:
- Tokenization matches the exact model and tokenizer used for generation
- No re-tokenization drift between generation and training

For details on on-policy token ID handling, see {doc}`../guides/environments` and the [NeMo Gym on-policy corrections documentation](https://docs.nvidia.com/nemo/gym/latest/contribute/rl-framework-integration/openai-compatible-http-server-on-policy-correction.html).
