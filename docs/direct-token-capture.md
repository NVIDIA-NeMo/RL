# Direct vLLM token capture

The single-controller Gym rollout path can send model requests directly to NRL's
vLLM workers. Each rollout gets an ordinary OpenAI API key. NRL associates that
key with a rollout and sampling profile; agents do not send conversation IDs,
token arrays, or lineage metadata. Native engine outputs supply the training
tokens, selected-token logprobs, weight version, and optional routing decisions.

```yaml
env:
  should_use_nemo_gym: true
policy:
  generation:
    backend: vllm
    vllm_cfg:
      async_engine: true
      expose_http_server: true
data_plane:
  enabled: true
token_capture:
  enabled: true
  transport: direct   # default: gym
  direct:
    token_packing: fused  # default; legacy retains tuple-based capture for comparison
    workers: 2
    max_pending_calls: 256
    max_rollouts: 4096
    max_calls_per_rollout: 512
```

Use the TransferQueue data-plane configuration from the single-controller recipe.
The direct transport supports text/tool requests, nonstreaming Responses requests,
and both regular and streaming Chat Completions. Media capture and deferred route
assembly currently require `transport: gym`; setup rejects these combinations.
Router replay uses native tensors, with no routing arrays in model HTTP responses.
Missing or incomplete routing coverage rejects the training row.

## Gym integration

The Gym dependency adds a generic `model_endpoint` to `BaseRunRequest`:

```json
{"server_name":"policy_model","base_url":"http://worker:8000/v1","api_key":"rollout-uuid"}
```

The base agent scopes it to one run. `ServerClient` forwards it across internal
agent calls and applies it only to the named model server. Resource tools,
verifiers, and other model servers keep their existing routing. It is excluded
from serialized verifier bodies and agent results. NRL's Gym actor supplies this
field automatically, registers it with one worker, and drains capture when the
agent finishes. Requests in a rollout stay on that worker; worker recovery creates
a new attempt rather than moving an in-flight capture between workers.

Native agents derived from `SimpleResponsesAPIAgent` and using `ServerClient`
need no per-agent token-capture integration. Their Responses-to-Chat conversion
runs inside the vLLM HTTP worker, with no additional network hop. Harness adapters
can use `self.model_environment(body)` to obtain `OPENAI_BASE_URL` and
`OPENAI_API_KEY`. Harbor General merges these into each trial's `AgentConfig.env`;
MiniSWE uses the same helper. Adapters overriding the base webserver or using
their own subprocess clients must pass these settings to those clients.

For Harbor/OpenCode, use a custom OpenAI-compatible provider. For example:

```json
{
  "provider": {
    "nrl": {
      "npm": "@ai-sdk/openai-compatible",
      "env": ["OPENAI_API_KEY"],
      "options": {"baseURL": "{env:OPENAI_BASE_URL}"},
      "models": {"policy": {"name": "policy"}}
    }
  },
  "model": "nrl/policy",
  "small_model": "nrl/policy"
}
```

Set the provider model name to the served NRL model. All auxiliary model calls
must use the rollout endpoint/key to participate. Omitted temperature/top-p use
the registered training profile; explicit mismatches fail instead of sampling
off-policy. `n>1`, beam search, and legacy token-prefix overrides are unsupported.
The separate control-plane bearer credential is not passed to the agent.

## Capture and the first training view

The serving path reserves a bounded slot, retains engine output references, and
submits a staging task. The task extracts native tokens/logprobs and writes one
tensor row per call. Streaming retains delta chunks and the final routing array.
The HTTP formatter receives a shallow output copy without the routing array, so
upstream vLLM's own base64 encoder is bypassed too. There is no Gym ledger, prefix
lookup, tree update, token JSON round trip, or awaited staging operation per call.

`token_capture.direct.token_packing` selects `fused` (default) or `legacy` for
E2E comparison. Fused capture packs prompt lists directly into an owned int64
buffer and appends generated tokens; staging uses a tensor view of that buffer.
It also accepts one-dimensional contiguous native signed-int64 prompt buffers
(for example, `array("q")` or NumPy int64) and copies their bytes without a Python
token walk. Other buffer layouts fail capture rather than being reinterpreted.
Producers must keep the source stable while the captured output is retained,
until the worker finishes copying it. Subsequent producer mutation cannot change
the staged tokens. Packing and copying run in the existing background worker.

The pinned [vLLM 0.29 output API](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/outputs.py)
supplies prompt token lists, so current vLLM runs use the fused list path. Enabling
this option does not make vLLM emit packed buffers or eliminate list conversion.
Packed-input throughput requires a producer that already supplies those buffers.
Both modes produce the same staged schema, masks, logprobs, and routing alignment;
neither needs prefix or KV-cache ancestry.

Finalization closes admission for the rollout, waits for active HTTP requests and
staging jobs, and returns an idempotent receipt containing metadata only. Queue
overflow, incomplete generation, missing logprobs, or staging failures invalidate
the rollout. A release clears worker-side receipt state after receipt delivery;
abandoned attempts drain and discard staged rows. Training-row assembly deletes
all staged calls, including calls excluded from training.

The first view selects the latest admitted completed call and its longest exact
token-prefix predecessors. Equal prefix candidates choose the latest preceding
call. This is a deterministic selection rule, not semantic knowledge of which
subagent is the main agent. Compaction can therefore select a new root. Only
selected generations receive loss; context/tool tokens are masked. Metrics report
`finalize/direct_selected_calls`, `finalize/direct_excluded_calls`, and
`finalize/direct_excluded_tokens`. Train-row tags retain the selected and excluded
call IDs. The complete receipt still accounts for every captured call.

## Serialization and CPU costs

Let P be prompt tokens, G generated tokens, B request text bytes, R response text
bytes, L routed layers, K experts selected per token, C calls per rollout, and S
the selected training sequence length. Expert IDs require log2(E) bits for E
experts; the current training transport stores them as int16.

| Location | Representation/work | Size | On the model HTTP response path? |
|---|---|---|---|
| Agent → worker | Ordinary request JSON encode, HTTP, JSON decode/Pydantic validation | O(B) | Yes, also required by direct vLLM |
| Native Gym compatibility | Responses ↔ Chat model conversion/dictionary validation | O(B + R) | Yes; absent for Chat clients |
| vLLM rendering/engine transport | Existing tokenization and engine IPC | O(B + P + G + (P+G)LK) with routing | Existing backend work; unchanged |
| Capture handoff | Reference retention, lock/slot accounting, executor submission | O(1) per output chunk | Yes; no payload encoding |
| Native staging | Python integer buffers → owned binary arrays/tensor views; scalar metadata JSON | 8(P+G) + 4G bytes; O((P+G)LK) routes | Background pool; GIL contention can still delay HTTP |
| TransferQueue | Native tensor transport | O(P+G+(P+G)LK) | Background pool |
| Worker → agent | Ordinary response JSON/SSE | O(R), excluding training arrays | Yes, also required by direct vLLM |
| Rollout control | Register/finalize/release JSON | O(1), O(C), O(1) | Once per rollout; finalization waits for custody |
| Row assembly | Tensor fetch/list conversion, exact-prefix matching, masks, tensor packing | O(sum(P+G)) tokens; O(SLK) assembled routes | After rollout |

Prefix selection currently compares candidate token prefixes directly. Worst-case
CPU is O(C²(P+G)); it runs in the finalizer, never during model requests. Staging
currently stores each call's complete executed context, so storage is
O(sum(P+G+(P+G)LK)). A shared-prefix storage/learner implementation can change this
without changing model HTTP requests or capture identity. This path does not
depend on the shared-prefix training-worker PR.

GPU correctness, replay completeness under prefix caching, and GRPO convergence
still require cluster validation before production use.
