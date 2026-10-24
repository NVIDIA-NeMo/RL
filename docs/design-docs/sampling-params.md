# Sampling Parameters Across Generation Backends

`temperature`, `top_p` and `top_k` are configured in one place, `policy.generation`, but the
generation backends do not all read them the same way. The value that means "no top-k
restriction" differs between engines, and the HTTP servers each treat values sent on a request
differently. This page is the reference for what each path does, so the answer does not have to
be re-derived from every backend's `_build_sampling_params`.

For the surrounding generation config and the backend interface, see
[Generation](generation.md).

## Configuring sampling

All three values live under `policy.generation`. Validation has its own keys,
`val_temperature` / `val_top_p` / `val_top_k`, which normally just point at the training values:

```yaml
policy:
  generation:
    backend: vllm
    temperature: 1.0
    top_p: 1.0
    top_k: null        # null is the portable "unrestricted" value
    val_temperature: ${.temperature}
    val_top_p: ${.top_p}
    val_top_k: ${.top_k}
```

GRPO checks at setup that the validation values equal the training ones. The one exception is
vLLM with NeMo-Gym rollouts, where `val_temperature` and `val_top_p` may differ. `val_top_k` must
then be left `null`, because that path never sends a top-k (see below).

Top-p and top-k sampling were added in
[#2053](https://github.com/NVIDIA-NeMo/RL/pull/2053). They narrow the rollout distribution, so
they interact with importance sampling and off-policy correction: the tighter the truncation, the
further the sampled distribution sits from the policy the loss assumes.

## Use `null`, not `-1` or `0`

**`top_k: null` is the only value that means "unrestricted" on every backend.** Each backend
translates it to what its engine expects. An integer written in the YAML is passed through as it
is, and no single integer is accepted everywhere:

| you write | vLLM | SGLang | TRT-LLM | Megatron |
|---|---|---|---|---|
| `top_k: null` | sent as `-1`, unrestricted | key omitted, unrestricted | sent as `0`, unrestricted | sent as `0`, unrestricted |
| `top_k: -1` | unrestricted | unrestricted | **raises**, `require top_k >= 0` | unrestricted |
| `top_k: 0` | unrestricted | **rejected**, `top_k must be -1 (disable) or at least 1` | unrestricted | unrestricted |

The integer rows come from the engines themselves, at the versions pinned in `pyproject.toml`:

- vLLM's `SamplingParams` documents `0` as "consider all tokens" and also accepts `-1`.
- SGLang's `SamplingParams` reads `-1` as the whole vocabulary and rejects anything else below 1.
- TRT-LLM's `SamplingParams` documents `0` as "all logits" and rejects negative values.
- Megatron-LM's `SamplingParams` turns any `top_k <= 0` into `0`, its disabled value.

So a config carrying `top_k: -1`, which is what
[`examples/configs/evals/eval.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/evals/eval.yaml)
uses, commented `# -1 means disable`, works on the other three backends and fails at request time
on TRT-LLM. `top_k: 0` has the mirror problem on SGLang, whose server refuses the request.

## What each path does

| path | `top_k: null` becomes | config `top_k` reaches the engine | source |
|---|---|---|---|
| vLLM `generate()` / `generate_async()` | `-1` | yes | `BaseVllmGenerationWorker._build_sampling_params`, `nemo_rl/models/generation/vllm/vllm_worker.py` |
| vLLM `generate_text()` / `generate_text_async()` | `-1` | yes | `nemo_rl/models/generation/vllm/vllm_worker.py`, `vllm_worker_async.py` |
| **vLLM HTTP** (NeMo-Gym rollouts) | forced `-1` | **no, the config value is ignored** | `nemo_rl/models/generation/vllm/vllm_worker_async.py` |
| SGLang | key omitted | yes | `nemo_rl/models/generation/sglang/sglang_generation.py` |
| TRT-LLM `generate()` | `0` | yes | `nemo_rl/models/generation/trtllm/trtllm_worker_async.py` |
| TRT-LLM HTTP | `0` | yes | `nemo_rl/models/generation/trtllm/trtllm_http_server.py` |
| Megatron direct generation | `0` | yes | `_build_sampling_params`, `nemo_rl/models/generation/megatron/megatron_worker.py` |
| Megatron HTTP | `0` | as the server default | `_http_sampling_defaults`, `nemo_rl/models/generation/megatron/megatron_worker.py` |

SGLang is the one backend where "unrestricted" is not a value at all: when the resolved `top_k`
is `-1` the key is left out of the request dict.

`temperature` and `top_p` are passed to the engine as configured. Megatron is the one backend
that also accepts `top_p: null`: its direct path sends `0.0`, which Megatron-LM reads as no top-p
filtering, and its HTTP server default becomes `1.0`.

## Values sent on an HTTP request

The direct paths build their sampling params from the config alone. The OpenAI-compatible
servers, which NeMo-Gym agentic rollouts talk to, also see whatever the client puts on the
request, and each server handles that differently:

| server | `temperature` / `top_p` on the request | `top_k` on the request |
|---|---|---|
| vLLM | `top_p` must be set, and the pair must equal either the training or the validation values in the config | must be unset or `-1`, and is then pinned to `-1` |
| TRT-LLM | when present, must equal the config value | when present, must equal the config value |
| Megatron | used as sent; the config value applies when the field is omitted | used as sent; the config value applies when the field is omitted |

vLLM and TRT-LLM fail the request with an assertion when a check does not hold. Megatron has no
such guard, so a client that sends its own sampling values gets them.

## The vLLM HTTP path ignores `top_k`

The vLLM chat-completions handler asserts that the incoming request carries no top-k and then
pins it:

```python
# nemo_rl/models/generation/vllm/vllm_worker_async.py
assert request.top_k in (None, -1), (
    f"Top k sampling parameter must be unset, empty, or -1. Got `{request.top_k}`"
)
request.top_k = -1
```

The comment above it says this matches
`BaseVllmGenerationWorker::_build_sampling_params`. That holds only while `top_k` is `null`. Set
`policy.generation.top_k: 50` and the direct path samples with top-k 50 while the HTTP path
samples unrestricted, so the same config produces different rollout distributions depending on
whether the rollout went through the engine directly or over HTTP.

TRT-LLM's HTTP server used to have the same drift and no longer does; it was aligned in
[#3537](https://github.com/NVIDIA-NeMo/RL/pull/3537), whose description carries a
direct-vs-HTTP comparison worth reading. **Until the vLLM HTTP path is threaded the same way,
treat agentic rollouts on vLLM as top-k-free.** For an on-policy run this matters: the trainer
computes logprobs under the policy, and a rollout distribution the config did not ask for is
off-policy by construction.

## Greedy decoding

Greedy is an argument of the direct generation call, not a config field. Every path that takes a
`greedy` flag resolves it the same way, to `temperature=0.0` and `top_k=1`:

- `BaseVllmGenerationWorker._build_sampling_params` (vLLM)
- `generate_text` / `generate_text_async` (vLLM)
- `sglang_generation.py` (SGLang)
- `trtllm_worker_async.py` (TRT-LLM)
- `megatron_worker.py` (Megatron, which also sets `top_p` to `0.0`)

The HTTP servers take no `greedy` flag. A caller wanting greedy decoding over HTTP sets the
sampling fields on the request itself, subject to the checks above.

## Summary

- Write `top_k: null` for unrestricted. `-1` raises on TRT-LLM and `0` is rejected by SGLang.
- vLLM HTTP rollouts ignore the configured `top_k`.
- Over HTTP, vLLM and TRT-LLM check request values against the config, while Megatron only uses
  the config as its default.
- Greedy means `temperature=0.0, top_k=1` on every path that accepts the flag, and does not exist
  on the HTTP servers.
