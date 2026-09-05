# Sampling Parameters Across Generation Backends

`temperature`, `top_p` and `top_k` are configured in one place — `policy.generation` — but the
generation backends do not all read them the same way. The value that means "no top-k
restriction" differs between backends, and one path ignores `top_k` altogether. This page is the
reference for what each backend actually does, so the answer does not have to be re-derived from
six `_build_sampling_params` implementations.

For the surrounding generation config and the backend interface, see
[Generation](generation.md).

## Configuring sampling

All three values live under `policy.generation`, and validation runs share them through
`val_temperature` / `val_top_p` / `val_top_k`:

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

Top-p and top-k sampling were added in
[#2053](https://github.com/NVIDIA-NeMo/RL/pull/2053). They narrow the rollout distribution, so
they interact with importance sampling and off-policy correction: the tighter the truncation, the
further the sampled distribution sits from the policy the loss assumes. Keep the training and
validation values deliberate rather than inherited by accident.

## Use `null`, not `-1` or `0`

**`top_k: null` is the only portable way to say "unrestricted".** Each backend translates it to
whatever its engine expects. Writing the engine-level sentinel directly in YAML does not
travel:

| you write | vLLM / SGLang | TRT-LLM / Megatron |
|---|---|---|
| `top_k: null` | becomes `-1` — unrestricted | becomes `0` — unrestricted |
| `top_k: -1` | unrestricted | **raises** — TRT-LLM requires `top_k >= 0` |
| `top_k: 0` | top-k of 0 — not the disable sentinel | unrestricted |

TRT-LLM's `SamplingParams` rejects negatives outright (`require top_k >= 0`) and documents `0` as
"all logits". So a config carrying `top_k: -1` — which is what
[`examples/configs/evals/eval.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/evals/eval.yaml)
uses, commented `# -1 means disable` — works on vLLM and fails at request time on TRT-LLM.

## What each path does

Verified against `main`:

| path | "disabled" spelled as | honors `cfg["top_k"]` | source |
|---|---|---|---|
| vLLM `generate()` / `generate_async()` | `-1` | yes | `BaseVllmGenerationWorker._build_sampling_params`, `nemo_rl/models/generation/vllm/vllm_worker.py` |
| vLLM `generate_text()` | `-1` | yes | `nemo_rl/models/generation/vllm/vllm_worker.py` |
| **vLLM HTTP** (NeMo-Gym rollouts) | forced `-1` | **no — config ignored** | `nemo_rl/models/generation/vllm/vllm_worker_async.py` |
| SGLang | `-1`, key omitted entirely | yes | `nemo_rl/models/generation/sglang/sglang_generation.py` |
| TRT-LLM `generate()` | `0` | yes | `nemo_rl/models/generation/trtllm/trtllm_worker_async.py` |
| TRT-LLM HTTP | `0` | yes | `nemo_rl/models/generation/trtllm/trtllm_http_server.py` |
| Megatron | `0` | yes | `nemo_rl/models/generation/megatron/megatron_worker.py` |

SGLang goes one step further than the others: when the resolved value is `-1` it drops the
`top_k` key from the request dict rather than sending the sentinel.

`temperature` and `top_p` need no such translation — every backend passes them through as
floats.

## The vLLM HTTP path ignores `top_k`

The vLLM OpenAI-compatible server, which is what NeMo-Gym agentic rollouts talk to, asserts that
the incoming request carries no top-k and then pins it:

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
samples unrestricted — so the same config produces different rollout distributions depending on
whether the rollout went through the engine directly or over HTTP.

TRT-LLM's HTTP server used to have the same drift and no longer does; it was aligned in
[#3537](https://github.com/NVIDIA-NeMo/RL/pull/3537), whose description carries a
direct-vs-HTTP comparison worth reading. **Until the vLLM HTTP path is threaded the same way,
treat agentic rollouts on vLLM as top-k-free.** For an on-policy run this matters: the trainer
computes logprobs under the policy, and a rollout distribution the config did not ask for is off-policy
by construction.

## Greedy decoding

Greedy is a parameter of the *direct* generation call, not a config field. Every backend that
takes a `greedy` flag resolves it the same way — `temperature=0.0` and `top_k=1`:

- `BaseVllmGenerationWorker._build_sampling_params` (vLLM)
- `generate_text` / `generate_text_async` (vLLM)
- `sglang_generation.py` (SGLang)
- `trtllm_worker_async.py` (TRT-LLM)
- `megatron_worker.py` (Megatron)

The HTTP paths have no `greedy` flag at all. A caller wanting greedy decoding over HTTP sets the
sampling fields on the request itself, subject to the top-k caveat above.

## Summary

- Write `top_k: null` for unrestricted. `-1` and `0` are engine-level spellings and are not
  interchangeable.
- `-1` on TRT-LLM raises; `0` on vLLM is a literal top-k of 0.
- vLLM HTTP rollouts ignore `top_k` entirely.
- Greedy means `temperature=0.0, top_k=1` on every backend that accepts the flag, and does not
  exist on the HTTP paths.
