# The Life of a Prompt — Image-Zoom Multi-Turn Tool-Use GRPO

How a single training example travels from the dataset, through a multi-turn
image-zoom rollout, and back into the policy-gradient loss — with exact pointers
to where the **chat template is applied**, where things get **tokenized**, where
**prefix tokens get replaced**, and where **image placeholders expand / realign /
collapse**.

> Pointers are `path:line · function` relative to the repo root, on branch
> `matthieul/image-zoom-minimal`. Line numbers drift as the files change — treat
> them as "look here," not gospel. Companion doc: `chat-template-token-drift.md`
> (the *why*; this doc is the *what/where*).

---

## 0. The one idea to hold onto: token-in-token-out

Multi-turn RL has a fragile round trip. vLLM samples assistant tokens; the agent
decodes them to detect a tool call; a tool result (here, a cropped image) is
appended as a new user turn; the next request goes back through a chat template.
If step 4 **re-tokenizes prior assistant text differently** (a stray newline
changes BPE merges) or **expands image placeholders differently**, then at
training time Megatron would score a context that vLLM never actually generated
from. That shows up as large Token-Mult-Prob-Error (TMPE), masked samples, or
hard shape mismatches in the loss.

The whole system is built around one invariant:

> **Every assistant turn that contributes policy loss is scored on the EXACT
> `prompt_token_ids + generation_token_ids` that vLLM produced for it.**

The chat template is applied *many* times along the way (for rendering and
budgeting), but the **sampled token IDs always win** — they are carried verbatim
and spliced back over whatever the template re-rendered. Four mechanisms produce
this guarantee, and four gates enforce it (see §A and §B at the end).

---

## 1. The flow at a glance

```
DATA PREP (offline)
  prepare_image_zoom_gym_data.py  ── prepend system prompt, raw messages+image refs → jsonl

LOAD / PROCESS (per row)
  NemoGymDataset → nemo_gym_data_processor → nemo_gym_example_to_nemo_rl_datum_spec
     ├─ apply_chat_template(tokenize=True)         ← chat template + tokenize (for the FIRST turn layout + length budget)
     └─ DatumSpec{ message_log[0] = first-turn tokens+pixels,  extra_env_info = RAW gym example }

GRPO STEP (grpo.py)
  run_async_nemo_gym_rollout ──► NemoGym.run_rollouts (Ray actor)
                                    │  re-sends extra_env_info (RAW messages+images) to the Gym agent
                                    ▼
  ┌──────────────────────── MULTI-TURN ROLLOUT (per sample) ───────────────────────┐
  │ image_zoom_agent.run → _run_image_zoom_loop:                                    │
  │   loop up to max_steps:                                                         │
  │     model_body = base_input + new_outputs        (verbatim, never reformatted)  │
  │     POST /v1/responses → vLLM model server                                      │
  │        ├─ apply chat template + tokenize (vLLM /tokenize)   ← (a)+(b)            │
  │        ├─ NeMo-RL render: splice required_prefix tokens     ← (c) THE SPLICE     │
  │        │  + realign multimodal placeholders                 ← (d)               │
  │        └─ return assistant item with prompt_token_ids + generation_token_ids    │
  │            + generation_log_probs                                               │
  │     parse tool call (text, for routing only)                                    │
  │     execute crop (image.crop) → append crop as NEW user turn (input_image)      │
  └────────────────────────────────────────────────────────────────────────────────┘
                                    │  one NeMoGymResponse = full accumulated trajectory
                                    ▼
  _postprocess_nemo_gym_to_nemo_rl_result:
     ├─ STRICT exact-prefix gate (_find_prompt_delta_start)        ← enforce invariant
     ├─ rebuild per-turn message_log: token_ids = generation_token_ids (verbatim)
     ├─ reconstruct crop user-delta + expand <image> to match vLLM's baked count
     └─ EMBED-COUNT gate (reconstructed N == recorded N)           ← enforce invariant

TRAIN (Megatron policy worker)
  get_logprobs → collapse_multimodal_tokens (N→1) → model fprop → re-expand (1→N)
     ├─ drift guard: total_embeds == raw <image> count            ← enforce invariant
     └─ curr_logprobs  (the current policy, scored on the verbatim rollout tokens)

LOSS (loss_functions.py)
  log_ratios = curr_logprobs − prev_logprobs ;  ratio·advantage (clipped) ; masked_mean
```

---

## 2. Stage-by-stage

### Stage 1 — Data → `DatumSpec`

| where | what |
|---|---|
| `tools/prepare_image_zoom_gym_data.py:41-52 · _prepend_system_prompt` (from `convert_row:62`) | **Offline.** Prepends the system prompt (`examples/prompts/image_zoom_tool_system_prompt.txt`) as a `{"role":"system"}` item into `responses_create_params.input`, rewrites `agent_ref → image_zoom_simple_agent`. Output jsonl holds **raw messages + image refs**, not tokens. |
| `nemo_rl/data/datasets/response_datasets/nemogym_dataset.py:20,36` | Loads the jsonl rows verbatim into `extra_env_info` (a JSON string) + `task_name`. No parsing. |
| `nemo_rl/data/processors.py:1528-1555 · nemo_gym_data_processor` | Per-row: `json.loads(extra_env_info)`, then (VLM) `nemo_gym_example_to_nemo_rl_datum_spec`. |
| `nemo_rl/environments/nemo_gym.py:1687-1708 · nemo_gym_example_to_nemo_rl_datum_spec` | **(a)+(b)+(d)** Runs `processor.apply_chat_template(messages_with_images, tokenize=True, add_generation_prompt=True, return_dict=True)` → the **first-turn** token layout `<img><image>×N</img>` + `pixel_values`/`imgs_sizes`, packed as **one non-trainable `user` message** in `message_log[0]`. |
| `nemo_rl/environments/nemo_gym.py:1669-1682 · _compute_vllm_dynamic_prompt_length` | Computes vLLM's image budget `num_tokens_available = max_seq_length − prompt_length − 4` and feeds it to the processor so local image sizing matches what vLLM will do at rollout. |

**Takeaway:** the `DatumSpec` carries *pre-tokenized* first-turn ids only for
length/advantage bookkeeping; the **source of truth replayed at rollout is the
raw `extra_env_info`** (messages + image refs), re-sent to vLLM.

### Stage 2 — GRPO rollout entry

| where | what |
|---|---|
| `nemo_rl/algorithms/grpo.py:2511-2525` (async path `async_grpo_train:3823`) | Image-zoom takes the **async NeMo-Gym** path. Calls `run_async_nemo_gym_rollout(repeated_batch, tokenizer, task_to_env, generation_config)`; the returned batch replaces `repeated_batch`. |
| `nemo_rl/experience/rollouts.py:1328-1411 · run_async_nemo_gym_rollout` | Pulls `nemo_gym_rows = input_batch["extra_env_info"]`, clamps each row's `max_output_tokens` to remaining context, then `ray.get(nemo_gym_environment.run_rollouts.remote(rows, tokenizer, original_message_logs=input_batch["message_log"], ...))`. |
| `nemo_rl/algorithms/grpo.py:4396,3708-3737 · build_async_train_and_logprob_data` | After rollout, flattens `repeated_batch["message_log"]` with `batched_message_log_to_flat_message` into `train_data` (`generation_logprobs`, `token_mask=token_loss_mask`, `sample_mask=loss_multiplier`). |

### Stage 3 — NeMo-Gym dispatch (DatumSpec → Responses request)

| where | what |
|---|---|
| `nemo_rl/environments/nemo_gym.py:1121-1183 · NemoGym.run_rollouts` (Ray actor) | Spins the Gym servers (`_spinup:1023`), runs examples (`self.rch.run_examples`), and per result calls `_postprocess_nemo_gym_to_nemo_rl_result`. Includes a NaN-retry loop on `generation_logprobs`. |
| `nemo_rl/environments/nemo_gym.py:716 · encode_images_in_examples` | Local image paths in `responses_create_params.input[].content[]` → base64 data URLs before dispatch. |
| `3rdparty/Gym-workspace/Gym/responses_api_agents/image_zoom_agent/app.py:349-366 · ImageZoomAgent.run` | Resolves the base resource, `/seed_session`, then `_run_image_zoom_loop(body=responses_create_params)`. `responses_create_params` (system prompt + image) is the immutable **`base_input`**. |

### Stage 4 — The Gym agent multi-turn loop

| where | what |
|---|---|
| `image_zoom_agent/app.py:282-288 · _run_image_zoom_loop` | Per step: `model_body = body.model_copy(update={"input": base_input + new_outputs})`. **`new_outputs` accumulates prior token-bearing assistant items + appended crop user-turns *verbatim*** — this is the agent-side half of the invariant. |
| `image_zoom_agent/app.py:292-302` | Stop strings routed through `metadata.extra_body` (`stop`, `include_stop_str_in_output`) because the Responses param type forbids a top-level `stop`. Default `stop=["</tool_call>"]`. |
| `image_zoom_agent/app.py:304-305 · _call_model` | POST `model_body` → `/v1/responses`; `new_outputs.extend(model_response.output)` — appends the assistant item carrying `prompt_token_ids`/`generation_token_ids`/`generation_log_probs`. |
| `image_zoom_agent/app.py · _run_image_zoom_loop` → `nemo_rl/environments/image_tools_gym_tool.py · parse_image_tool_calls, process_nonterminal_turn` | Decodes assistant text **only to route**; parses the image-tool call, executes its deterministic operation, and applies duplicate/reward logic. No tool call and no malformed attempt → break (final turn). The legacy `image_zoom_gym_tool.py` module re-exports the original MR 68 names. |
| `image_zoom_agent/app.py:328-332 · _tool_user_message` | Wraps the crop as a **new user turn** `{"type":"input_image","image_url": <crop data-URL>}`, appended to `new_outputs`. Loops up to `max_steps` (default 5). |
| `image_zoom_agent/app.py:337-347` | Returns one `NeMoGymResponse` whose `output` is the **full accumulated trajectory**. |

### Stage 5 — The vLLM model server (Gym side)

| where | what |
|---|---|
| `vllm_model/app.py:607-687 · responses_to_chat_completion_create_params` | Responses items → chat messages; carries each item's `prompt_token_ids` into `state.token_information`. |
| `vllm_model/app.py:283-376 · _preprocess_chat_completion_create_params` | Sets `logprobs=True, return_tokens_as_token_ids=True, return_token_ids=True`. If `uses_reasoning_parser`, strips `<think>…</think>` out of **prior** assistant messages into `reasoning_content`. |
| `vllm_model/app.py:388-396` → `nemo_gym/openai_utils.py:543-552` | **(a)+(b)** Applies the **chat template inside vLLM `/tokenize`** and reads `prompt_token_ids = tokenize_response["tokens"]`. |
| `vllm_model/app.py:483-509` → `openai_utils.py:523-531` | **(a)+(b)** Generation via `/chat/completions`; `generation_token_ids` decoded from the `token_id:`-prefixed logprob tokens, `prompt_token_ids` from a second `/tokenize`. |
| `vllm_model/app.py:557-579 · flush_assistant` | **Attaches** `prompt_token_ids`/`generation_token_ids`/`generation_log_probs` to the assistant message. (It does **not** concatenate them — that's NeMo-RL side, Stage 6.) |

> Gotcha: `required_prefix_token_ids` / `required_prefix_message_count` are **not**
> Gym schema fields (`openai_utils.py:420-449`). They are added by NeMo-RL's
> request subclasses (Stage 6). Also, the `parse_reasoning_from_history` knob the
> design doc mentions is **not read by any code on this branch** — it's a
> permissive `++` override silently accepted by `VLLMModelConfig(extra="allow")`;
> the live behavior is the `uses_reasoning_parser` block above.

### Stage 6 — The required-prefix render (NeMo-RL) — the crux for (c) and (d)

This is where "what the template re-rendered" gets overwritten by "what vLLM
actually sampled."

| where | what |
|---|---|
| `nemo_rl/models/generation/vllm/vllm_worker_async.py:55-66 · _required_prefix_from_messages` | **(c) built.** Walks messages in reverse and returns `list(prompt_token_ids) + list(generation_token_ids)` (`:62-64`) plus `message_idx+1`. **This is the actual prompt+generation concatenation** (not Gym's `flush_assistant`). |
| `vllm_worker_async.py:453-475 · NeMoRLOpenAIChatRequestMixin.model_post_init` | **(c) threaded.** Back-fills `required_prefix_token_ids` / `required_prefix_message_count` onto the request. `required_prefix_handler` must be `"nemo_render"` (the only supported handler; config field `config.py`). |
| `vllm_worker_async.py:586-618 · build_nemo_rl_openai_serving_render_cls` (wired in `_setup_vllm_openai_api_server:369`) | Wraps vLLM's `OpenAIServingRender` with the NeMo-RL subclass; registers `/v1/chat/completions` + `/tokenize`. |
| `nemo_rl/models/generation/vllm/openai_serving_render.py:124-176 · preprocess_chat + _request_without_required_prefix` | **(a)+(b).** Reads the two prefix fields, **strips them**, then calls `super().preprocess_chat(...)` → vLLM's renderer applies the chat template + tokenizes (`3rdparty/vllm/.../renderers/hf.py:963`) and computes `mm_placeholders`. |
| `openai_serving_render.py:198-253` | Re-renders **just** `messages[:required_prefix_message_count]` (`add_generation_prompt=False`) to measure `rendered_prefix_len` = how many template tokens the prefix occupies. |
| `openai_serving_render.py:95-121 · _splice_required_prefix_tokens` | **(c) THE SPLICE.** `spliced = list(required_prefix_token_ids) + original_token_ids[rendered_prefix_len:]` (`:113-115`); sets `engine_input["prompt_token_ids"]=spliced`, drops `"prompt"`. The prefix is located **positionally** by `rendered_prefix_len`. |
| `openai_serving_render.py:43-92 · _realign_multimodal_placeholders` | **(d).** The splice shifts token offsets, so each `mm_placeholder` offset (computed against the *pre-splice* prompt) is stale; this finds each placeholder's token slice in the spliced stream and rewrites `offset`, raising if it can't be relocated. Operates on offset metadata, not pixel tensors. **This is the capability the old `_replace_prefix_tokens` lacked, and the reason it was removed in favor of the render.** |

> Gotcha: a near-identical splice exists in vendored vLLM
> (`3rdparty/vllm/.../render/serving.py:682-732`) but is **dead at runtime** —
> NeMo's override strips the NeMo-only fields before calling `super()`, so the
> live splice is the one in `openai_serving_render.py`.

### Stage 7 — The processor (image → tokens)

| where | what |
|---|---|
| `nemo_rl/models/nano_v3_vl/dynamic_resolution_processor.py:77-81` | `IMG_INPUT_TAG="<image>"` (incoming); expanded form is `<img>` + `<image>`×N + `</img>`. |
| `dynamic_resolution_processor.py:385-396 · compute_num_embeddings` | **(d) count.** `N = (num_patches // reduction_factor²)` — defines how many `<image>` tokens one image becomes. Must match vLLM's tiler. |
| `dynamic_resolution_processor.py:752-779 · _add_image_placeholders_dynamic` | **(d) expand.** Replaces each `<image>` with `<img>` + `<image>`×`N` + `</img>`. |
| `dynamic_resolution_processor.py:949-994 · _call_dynamic` | **(b)+(d).** Fits each image to the shared budget (`compute_params:458` packs patches to `num_tokens_available`), expands placeholders, tokenizes, returns `input_ids` + `pixel_values` + `imgs_sizes`. |
| `dynamic_resolution_processor.py:317-332 · preprocess_images_for_text_prompt_length` | The crop-budget entry used at postprocess: `num_tokens_available = max(1, max_model_len − text_prompt_length − 4)`. Reproduces vLLM's per-prompt image sizing. |

### Stage 8 — Gym trajectory → NeMo-RL training `message_log`

`nemo_rl/environments/nemo_gym.py:1228 · _postprocess_nemo_gym_to_nemo_rl_result`

| where | what |
|---|---|
| `:1299-1307` | Output items **without** `generation_token_ids` (user/tool/crop) are buffered; items **with** them are assistant turns. |
| `:323-370,1309-1316 · _find_prompt_delta_start` | **STRICT GATE (invariant).** Requires `seen` (prior prompt deltas + prior generations) to be an **exact prefix** of this turn's `prompt_token_ids`; **raises** on mismatch (no fuzzy matching). The new delta = `prompt_token_ids[len(seen):]`. |
| `:603-684,1337-1349 · _make_vllm_prompt_delta_message` | Reconstructs the crop user-delta. `_expanded_prompt_text_length(:485-503)` = `len(ids) − Σ(N_i+2)` over `<img><image>×N</img>` groups — recovers vLLM's `text_prompt_length` **exactly** from atomic special tokens, with **no decode→re-tokenize round trip** (this is the multi-image drift fix). |
| `:646-666 · embed-count check` | **GATE (invariant).** Reconstructed per-image embed counts must equal `_image_feature_sizes_from_expanded_token_ids(prompt_tokens)` (the `<image>` counts already baked into the rollout tokens), else `AssertionError`. |
| `:1388-1394,1510-1524` | Assistant message: `token_ids = generation_token_ids` (verbatim), `generation_logprobs = generation_log_probs`, `is_invalid_tool_call` flag (text contains `<tool_call>` patterns), `seen += prompt_token_ids + generation_token_ids`. |

### Stage 9 — Training: collapse → fprop → re-expand → policy logprobs

| where | what |
|---|---|
| `nemo_rl/algorithms/grpo.py:4472` (async) / `:2898` (sync) | `train_data["prev_logprobs"] = policy.get_logprobs(logprob_data)["logprobs"]` — a fresh policy fprop over the **rollout tokens**. |
| `nemo_rl/models/policy/workers/megatron_policy_worker.py:508-551 · get_logprobs` | Builds the microbatch iterator and runs `megatron_forward_backward(..., post_processing_fn=LogprobsPostProcessor, forward_only=True)`. |
| `nemo_rl/models/megatron/data.py:156-167` | Clones `original_input_ids` (the uncollapsed N-placeholder ids), then `collapse_multimodal_tokens`. |
| `nemo_rl/models/megatron/multimodal.py:487-594 · collapse_multimodal_tokens` | **(d) collapse.** `<img><image>×N</img>` → `<img><image></img>` (keep first placeholder); records `raw_image_token_counts` + `vision_expansion_per_sample`. |
| `nemo_rl/models/megatron/multimodal.py:146-213 · compute_vision_expansion` + `_get_num_embeddings_from_sizes:27` | **(d) re-expand.** Recomputes per-image embed count from `imgs_sizes`. The drift guard `_assert_total_embeds_match_raw_image_counts` (`:748`) raises if this ≠ the rollout's raw `<image>` count. |
| `nemo_rl/models/megatron/train.py:98-144 · model_forward` (`prepare_multimodal_data:849`) | **(b)+(d).** Feeds uncollapsed `input_ids` + `pixel_values`/`imgs_sizes` to the LLaVA model, which re-expands the single placeholder internally and CP-shards. |
| `nemo_rl/models/megatron/train.py:242-286` | Restores uncollapsed targets (`input_ids = original_input_ids`, `:284`) so logits[i] score target[i+1] at matching length. (`remap_expanded_logits_to_collapsed` is dead code.) |
| `nemo_rl/models/megatron/train.py:474-537 · LogprobsPostProcessor` | Gathers current-policy logprobs vs the restored targets; prepends a 0 for token 0. **These are `curr_logprobs`.** |

### Stage 10 — The loss

`nemo_rl/algorithms/loss_functions.py · ClippedPGLossFn.__call__`

| where | what |
|---|---|
| `:219-229` | `token_mask = data["token_mask"][:,1:]`; `mask = token_mask * sample_mask` (`sample_mask = loss_multiplier`). |
| `:274-280` | **TMPE** = `masked_mean(exp(|generation_logprobs − prev_logprobs|·mask))` — the metric that *verifies the invariant held*. |
| `:380 · the subtraction` | `log_ratios = curr_logprobs − prev_logprobs`. |
| `:390-412` | `ratios = exp(log_ratios)`, clamped; `clip_loss = max(−adv·ratio, −adv·ratio_clamped)`. |
| `:534-540 · masked_mean` | `actor_loss = masked_mean(importance_weights · clip_loss, mask)`. Prompt tokens, non-generated history, and `loss_multiplier==0` samples contribute **zero**. |

**What feeds masks & advantages:**
- `token_mask`: `grpo.py:4378-4389` (async) / `:2699-2704` (sync): `is_assistant = role=="assistant" AND "generation_logprobs" in message` ⇒ trainable; else masked. The `generation_logprobs` guard is what restricts loss to **model-generated** assistant turns.
- `advantages`: `grpo.py:4546` from `total_reward` via `calculate_baseline_and_std_per_prompt` (leave-one-out, grouped by the **initial prompt** — see the sync advantage fix).
- **invalid-tool penalty**: `grpo.py:4576-4587` (async) / `:3065-3076` (sync): assistant messages flagged `is_invalid_tool_call` get `advantages[...] = invalid_tool_call_advantage` (default −5.0). This **penalizes** malformed tool calls via negative advantage — it does *not* mask them (matches ultra; there is no `skip_policy_loss` masking on this branch).

---

## A. Cross-cutting summaries

### Where the chat template is applied (it's applied repeatedly!)
1. **Data prep / DatumSpec** — `nemo_gym.py:1687` `apply_chat_template(tokenize=True)` for the first-turn layout + length budget.
2. **vLLM model server** — `vllm_model/app.py:388` via vLLM `/tokenize` (and again for generation) each rollout turn.
3. **NeMo-RL render** — `openai_serving_render.py:124` calls vLLM's renderer (`super().preprocess_chat`) for the full request, plus `:198` re-renders just the prefix span to measure `rendered_prefix_len`.

The template output is **never trusted for the prefix** — the sampled tokens are spliced over it.

### Where tokenization happens
- DatumSpec build (`apply_chat_template(tokenize=True)`).
- vLLM `/tokenize` (model server, Stage 5).
- vLLM renderer inside the NeMo render (Stage 6e).
- The processor `_call_dynamic` for images (Stage 7).
- At training there is **no re-tokenization** — the message_log's `token_ids` are used directly.

### Where prefix tokens get replaced
**Exactly one live place:** `openai_serving_render.py:113-115 · _splice_required_prefix_tokens` —
`spliced = required_prefix_token_ids + original_token_ids[rendered_prefix_len:]`.
(`required_prefix_token_ids` is built at `vllm_worker_async.py:62-64`. The old
`_replace_prefix_tokens` in-worker swap was removed; the vendored-vLLM splice is
bypassed.)

### Where image placeholders expand / realign / collapse
- **Expand** (1 → N): processor `dynamic_resolution_processor.py:752` at prompt build.
- **Realign offsets** (after splice): `openai_serving_render.py:43`.
- **Collapse** (N → 1) for the Megatron fprop: `multimodal.py:487`.
- **Re-expand** (1 → N) inside the model + target restore: `multimodal.py:146`, `train.py:284`.

---

## B. The enforcement chain (how the invariant is guaranteed)

| # | gate | location | failure mode it catches |
|---|---|---|---|
| 1 | **Splice** | `openai_serving_render.py:113` | generation conditioned on re-rendered (drifted) prior tokens |
| 2 | **Strict exact-prefix** | `nemo_gym.py:323` (`_find_prompt_delta_start`) | the next turn's prompt isn't an exact continuation of what was sampled → raises |
| 3 | **Embed-count match** | `nemo_gym.py:646` (`_make_vllm_prompt_delta_message`) | reconstructed image-token count ≠ vLLM's baked count → raises |
| 4 | **Training drift guard** | `multimodal.py:748` (`_assert_total_embeds_match_raw_image_counts`) | Megatron re-expands a crop to a different N than the recorded tokens → raises (instead of an opaque `curr − prev` shape crash in the loss) |

Producers that *rely* on the invariant: Gym attaches token IDs per turn
(`vllm_model/app.py:557`), the agent appends them verbatim
(`image_zoom_agent/app.py:305`), and NeMo-RL concatenates + threads them
(`vllm_worker_async.py:62`).

---

## C. Non-obvious facts worth internalizing
1. **The prompt+generation concatenation lives NeMo-RL side** (`vllm_worker_async.py:62-64`), not in Gym's `flush_assistant`.
2. **Two splice implementations exist; only `openai_serving_render.py`'s runs** (the vendored-vLLM one is bypassed by field-stripping).
3. **The splice locates the prefix positionally** by `rendered_prefix_len`; it trusts that the first N rendered tokens correspond to the prefix messages — which is *why* gates 2–4 exist as the real safety net.
4. **Megatron restores uncollapsed targets up to the expanded logits** (`train.py:284`); it does not remap logits down (`remap_expanded_logits_to_collapsed` is dead code).
5. **`parse_reasoning_from_history` is a no-op on this branch**; the live history-reasoning behavior is the `uses_reasoning_parser` block in `vllm_model/app.py:336`. The real guards are the splice + the strict gate.
6. **Malformed tool calls are penalized, not masked** (negative advantage, `grpo.py:4576`), matching ultra's `nemo_gym`.
