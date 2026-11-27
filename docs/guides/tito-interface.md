# TITO Inference Interface

The token-in/token-out (TITO) inference interface lets the vLLM HTTP server accept and return token IDs. With it enabled, the server also builds a multi-turn chat prompt from the tokens the parent call produced instead of re-tokenizing the conversation text. For a model family with an incremental tokenizer, it builds that prompt without rendering the history at all.

The interface is off by default. The existing chat path is unchanged when it is off.

## Configuration

| Key | Values | Behavior |
| --- | --- | --- |
| `policy.generation.vllm_cfg.tito_interface` | `false` (default), `true` | Serve `/v1/tito/generate` and bridge chat continuations onto their parent's tokens. |
| `policy.generation.vllm_cfg.tito_model` | `null` (default), `qwen3` | Tokenize only the appended messages of a captured chat continuation; see [Incremental tokenization](#incremental-tokenization). Requires `tito_interface`. |
| `policy.generation.vllm_cfg.tito_reject_root_reasons` | `[]` (default), list of Gym root reasons | Answer a call with HTTP 400 instead of re-rendering its history from text when it would start a root for one of these reasons. Requires `tito_interface`. `first_call` is not allowed. |

Requirements, checked when the run starts:

| Requirement | Reason |
| --- | --- |
| `vllm_cfg.async_engine: true` and `vllm_cfg.expose_http_server: true` | The interface is part of the HTTP server. |
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
| Parent selected by Gym's capture ledger, `tito_model` set, request eligible | The parent's tokens, then the tokens of the appended messages. The history is not rendered. |
| Parent tokens sent inline (`required_prefix_token_ids`, or token fields on the last assistant message) | The same, with the inline tokens as the parent. |

The parent's turn close is found by counting turn-close tokens, so reasoning that the template drops from earlier turns does not move it. If the rendered history has fewer turn closes than the parent, the request fails instead of falling back to the rendered text.

Requests with media fail on this path.

### Incremental tokenization

The bridge renders and tokenizes the whole conversation once per call to find the appended tokens. `tito_model` names a model family whose appended user and tool messages can be written directly in the chat template's format, so only they are tokenized. Both paths give the engine the same tokens; the bridge handles every request the incremental tokenizer does not.

| | Bridge | Incremental tokenization |
| --- | --- | --- |
| Chat templates | Any template that passes [qualification](#qualification) | The official Qwen3 template, matched by SHA-256 |
| History rendered per call | Once | Not at all |
| Cost grows with | History length | Appended messages only |
| What it knows about the model | The turn-close tokens | The template's format for user and tool turns, written by hand |

A continuation is bridged instead when any of these holds:

- Gym did not send a parent candidate, or the worker cannot verify the candidate's message boundary without the rendered conversation.
- A message has non-string content, or an appended message is not a user or tool message.
- The parent's generation did not end on the turn-close token.
- The request overrides the chat template, sets `reasoning_effort`, `stream`, `echo`, `n` above 1, `continue_final_message`, prompt truncation, or `add_generation_prompt: false`.

The two limits of the incremental tokenizer:

- **It is tied to a model family.** The suffix format is hand-written for one template. Another family needs its own `IncrementalTokenizer` in `nemo_rl/models/generation/tito.py`, and the server refuses to start unless that tokenizer reproduces the template's rendering on the qualification conversations.
- **It relies on one vLLM behavior.** The server hands vLLM the finished prompt through the request field `kv_transfer_params.prompt_token_ids`, which makes vLLM's chat preprocessing skip rendering and tokenization. Every incremental call checks that vLLM used those tokens and fails otherwise. The incremental tokenizer reads the messages as the client sent them, so it does not depend on how vLLM parses or renders them.

With `tito_model` set, enable DEBUG logging for `nemo_rl.models.generation.vllm.tito` to get one `Captured TITO {JSON}` event per captured chat request, with `outcome` (`incremental` or `bridge`), the `reason` a request was bridged, `reused_tokens`, `suffix_tokens`, and stage durations. Every 1000 requests each HTTP server process logs the cumulative counts at INFO as `Captured TITO outcomes over N requests: {JSON}`.

To compare prompt construction on CPU with the same tokenizer and inputs:

```bash
uv run python tools/benchmark_captured_tito.py \
  --model Qwen/Qwen3-0.6B --history-repeats 64 512 2048 \
  --iterations 30 --output /tmp/captured-tito-cpu.json
```

The benchmark checks that the splice, the bridge and the incremental tokenizer produce equal tokens before it times them. It excludes parent matching, storage reads, generation and the environment.

### Refusing roots

By default a call whose parent cannot be identified starts a root and is counted. With `tito_reject_root_reasons` set, such a call is answered with HTTP 400 and the error code `tito_root_rejected` on both routes, and nothing is generated or recorded. The reasons are Gym's: `no_prefix_match`, `non_user_suffix`, `render_options_changed`, `ambiguous_candidate`, and `worker_rooted`. Use it to check that a harness never rewrites history, or in an experiment that must not contain re-rendered history; a rollout that hits it fails.

Context-measuring requests to `/tokenize` are never refused.

## Qualification

At startup the server replays fixed conversations through the chat template and requires that the bridged tokens:

- contain the appended messages in order and nothing from the history,
- do not depend on what the parent generated,
- start with the template's turn close when the parent ended without one.

A template that fails any of these stops the server from starting. The turn-close tokens are the configured `stop_token_ids`, the model's generation EOS, and the tokenizer's EOS.

With `tito_model` set, the server also requires that:

- the chat template is the family's qualified template,
- the template takes message content as a string,
- on the same conversations, the incremental suffix equals the tail of the template's rendering, token for token, and the parent ends on the turn-close token.

## Trying it on a Gym recipe

`examples/nemo_gym/grpo_qwen3_0_6b_swe_tito.yaml` runs two short GRPO steps of Qwen/Qwen3-0.6B on multi-turn SWE-bench Verified tasks through Gym's sandboxed OpenCode agent, with one learner GPU and one vLLM GPU. It sets `tito_interface: true` and `tito_model: qwen3`. The small model checks that the protocol and the training data are compatible; it does not resolve the issues.

1. Set up the pinned Gym and the training dependencies as for any NeMo RL run, and configure the Gym OpenSandbox provider with the sandbox images and agent binaries. The recipe does not provision infrastructure.
2. In the pinned Gym checkout, prepare the official rows:

   ```bash
   uv run gym eval prepare --config benchmarks/swebench/verified/opencode.yaml
   ```

3. Select a few prepared rows and add the agent reference that `NemoGymDataset` reads:

   ```bash
   head -n 4 benchmarks/swebench/data/swebench_verified_benchmark.jsonl | \
     jq -c '.agent_ref = {type: "responses_api_agents", name: "swebench_verified_opencode_sandboxed_agent"}' \
     > /path/to/swe_tito_smoke.jsonl
   ```

4. From NeMo RL, run:

   ```bash
   NRL_LOG_LEVEL=DEBUG uv run examples/run_grpo_single_controller.py \
     --config examples/nemo_gym/grpo_qwen3_0_6b_swe_tito.yaml \
     data.train.data_path=/path/to/swe_tito_smoke.jsonl \
     data.validation.data_path=/path/to/swe_tito_smoke.jsonl
   ```

| To compare | Override |
| --- | --- |
| The bridge alone | `policy.generation.vllm_cfg.tito_model=null` |
| The existing prefix splice | `policy.generation.vllm_cfg.tito_model=null policy.generation.vllm_cfg.tito_interface=false` |

Compare `finalize/token_in_rate` and the root-reason counts across the runs, and check the worker log for `Captured TITO` events with `outcome=incremental`. A model that produces no tool calls has no second turn and exercises neither path.

## Megatron backend

The Megatron inference server's `POST /v1/completions` accepts `prompt` as a list of token IDs and returns `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs`. To continue a captured parent, send only the tokens after it and mark the request:

| Field | Value |
| --- | --- |
| `prompt` | Tokens to append after the parent. |
| `offload_params.ng_capture` | The `token_in` capture admission. |
| `offload_params.ng_tito_append` | `true`. |

The worker prepends the parent's tokens. A request with `ng_tito_append` that also carries chat splice metadata is rejected.

| Key | Values | Behavior |
| --- | --- | --- |
| `policy.generation.mcore_generation_config.tito_interface` | `false` (default), `true` | Continue a captured chat request with the shared bridge instead of the prefix splice. |

Both were checked against a Megatron inference server with Qwen3-0.6B, not on a Gym recipe.

## Measuring roots

These finalize metrics are reported whenever token capture is on, with or without the interface:

| Metric | Meaning |
| --- | --- |
| `finalize/root_reason_<reason>_count` | Calls that started a root, per Gym root reason. Every rollout has one `first_call`. |
| `finalize/rerooted_tokens` | Total length of the roots that are not a first call. Their history was rendered from text. |

## Limits

- The route, the chat adapter, and `tito_reject_root_reasons` are on the vLLM backend. On the Megatron backend, see below.
- Text prompts only.
- Parent matching, root reasons, and training rows are unchanged; see the [token capture ledger](../design-docs/token-capture-ledger.md).
