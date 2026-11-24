# TITO Inference Interface

The token-in/token-out (TITO) inference interface lets the vLLM HTTP server accept and return token IDs. With it enabled, the server also builds a multi-turn chat prompt from the tokens the parent call produced instead of re-tokenizing the conversation text. The design and its rationale are in the [TITO design document](../design-docs/tito.md).

The interface is off by default. The existing chat path is unchanged when it is off.

## Configuration

| Key | Values | Behavior |
| --- | --- | --- |
| `policy.generation.vllm_cfg.tito_interface` | `false` (default), `true` | Serve `/v1/tito/generate` and bridge chat continuations onto their parent's tokens. |
| `policy.generation.vllm_cfg.tito_reject_root_reasons` | `[]` (default), list of Gym root reasons | Answer a call with HTTP 400 instead of re-rendering its history from text when it would start a root for one of these reasons. Requires `tito_interface`. `first_call` is not allowed. |

Requirements, checked when the run starts:

| Requirement | Reason |
| --- | --- |
| `vllm_cfg.async_engine: true` and `vllm_cfg.expose_http_server: true` | The interface is part of the HTTP server. |
| `vllm_cfg.tito_model: null` | [Captured TITO](captured-tito.md) selects a different prompt construction path. |
| vLLM's Hugging Face renderer | Bridging uses the tokenizer's chat template. |
| The chat template passes qualification | See [Qualification](#qualification). |
| No routed experts, media capture, sampler top-k capture, or generation-prefix cuts | The interface does not carry them yet. |

## The token-only route

`POST /v1/tito/generate` runs one model call from token IDs.

Request:

| Field | Type | Meaning |
| --- | --- | --- |
| `token_ids` | list of int | Tokens to append after the parent, or the full prompt when the call has no parent. |
| `ng_capture` | object, optional | Gym's capture admission. A `token_in` admission names the parent call; a `text` admission starts a root. Omit it for a call that is not recorded. |
| `temperature`, `top_p` | float | Must match the training or validation sampling profile, as on the chat route. |
| `top_k` | int, optional | Must be unset or equal the profile's. |
| `max_tokens` | int, optional | Clamped to the remaining context. |
| `seed` | int, optional | Sampling seed. |

Response:

| Field | Meaning |
| --- | --- |
| `call_id` | The captured call's ID, or `null` when the call is not recorded. |
| `prompt_token_count` | Length of the engine prompt, parent included. |
| `token_ids`, `logprobs` | The sampled tokens and their log probabilities. |
| `finish_reason`, `stop_reason` | As vLLM reports them. |
| `ng_commit_coords` | Where capture staged the call, or `null`. |

The engine prompt is the parent's prompt and generated tokens followed by `token_ids`. The server does not render, truncate, or re-tokenize it.

The route answers with HTTP 400 and generates nothing when:

- `ng_capture` is sent to a worker without token capture,
- the admission is still a `candidate`, which only the chat route resolves,
- the parent's stored tokens differ in length from the admission,
- the admission restores a generation prefix,
- the prompt fills the model's context.

## Chat requests

`/v1/chat/completions` keeps its request and response format. With the interface enabled, the server prepares the engine prompt differently:

| Request | Engine prompt |
| --- | --- |
| No parent (first call, or a new root) | The chat template's rendering of the request. |
| Parent selected by Gym's capture ledger | The parent's tokens, then the rendered tokens after the parent's turn close. |
| Parent tokens sent inline (`required_prefix_token_ids`, or token fields on the last assistant message) | The same, with the inline tokens as the parent. |

The parent's turn close is found by counting turn-close tokens, so reasoning that the template drops from earlier turns does not move it. If the rendered history has fewer turn closes than the parent, the request fails instead of falling back to the rendered text.

Requests with media fail on this path.

### Refusing roots

By default a call whose parent cannot be identified starts a root and is counted. With `tito_reject_root_reasons` set, such a call is answered with HTTP 400 and the error code `tito_root_rejected` on both routes, and nothing is generated or recorded. The reasons are Gym's: `no_prefix_match`, `non_user_suffix`, `render_options_changed`, `ambiguous_candidate`, and `worker_rooted`. Use it to check that a harness never rewrites history, or in an experiment that must not contain re-rendered history; a rollout that hits it fails.

Context-measuring requests to `/tokenize` are never refused.

## Qualification

At startup the server replays fixed conversations through the chat template and requires that the bridged tokens:

- contain the appended messages in order and nothing from the history,
- do not depend on what the parent generated,
- start with the template's turn close when the parent ended without one.

A template that fails any of these stops the server from starting. The turn-close tokens are the configured `stop_token_ids`, the model's generation EOS, and the tokenizer's EOS.

## Trying it on a Gym recipe

The interface has run a two-step SWE smoke; the [design document](../design-docs/tito.md#swe-smoke) has the numbers. To run it on the SWE smoke recipe that [Captured TITO](captured-tito.md#swe-compatibility-smoke) describes, keep that guide's setup and add these overrides to the launch command:

```
policy.generation.vllm_cfg.tito_model=null policy.generation.vllm_cfg.tito_interface=true
```

Compare `finalize/token_in_rate` and the root-reason counts with a run that leaves both keys at their defaults.

## Measuring roots

These finalize metrics are reported whenever token capture is on, with or without the interface:

| Metric | Meaning |
| --- | --- |
| `finalize/root_reason_<reason>_count` | Calls that started a root, per Gym root reason. Every rollout has one `first_call`. |
| `finalize/rerooted_tokens` | Total length of the roots that are not a first call. Their history was rendered from text. |

## Limits

- vLLM backend only. The Megatron inference endpoint does not serve the route.
- Text prompts only.
- Parent matching, root reasons, and training rows are unchanged; see the [token capture ledger](../design-docs/token-capture-ledger.md).
