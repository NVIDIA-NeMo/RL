# Gym Branch Review: `feature/nvidia-IF-bench-validators-integrations`

## Changes Made to the Gym Submodule

Two files modified, both backwards-compatible.

### 1. `resources_servers/turing_vif/app.py` — Dynamic Judge URL Discovery

**Problem:** When using a separate judge model (e.g., Qwen3-235B) with Turing VIF, the judge URL had to be hardcoded in the config as `judge_base_url: http://127.0.0.1:8000/v1`. However, NeMo-Gym assigns dynamic ports to all server instances at startup, so port 8000 is never actually used. The previous `vllm_model` server type with `spinup_server: true` silently accepted but never acted on those fields — no vLLM process was ever started, causing `Cannot connect to host 127.0.0.1:8000` errors at runtime.

**Fix:** Added an optional `judge_server_name` config field to `TuringVIFResourcesServerConfig`. When set, `_get_judge_client()` discovers the judge URL at runtime via `get_server_url()` from the NeMo-Gym server registry, which resolves the dynamically assigned host and port. When not set, the existing `judge_base_url` / `policy_base_url` fallback behavior is preserved.

This enables use of `local_vllm_model` (which actually spins up a vLLM instance via Ray) as the judge server type, with automatic URL resolution.

**Diff:**
```python
# New config field (TuringVIFResourcesServerConfig)
judge_server_name: Optional[str] = Field(
    default=None,
    description="NeMo Gym server instance name for the judge model. "
    "When set, the judge URL is discovered automatically from the "
    "server registry, and judge_base_url is ignored.",
)

# Updated _get_judge_client()
if self.config.judge_server_name:
    from nemo_gym.server_utils import get_server_url
    base_url = get_server_url(self.config.judge_server_name) + "/v1"
else:
    base_url = self.config.judge_base_url or getattr(
        self.config, "policy_base_url", "https://api.openai.com/v1"
    )
```

**Backwards compatibility:** Fully backwards-compatible. The new field defaults to `None`, preserving the existing behavior for all current users. The default `turing_vif.yaml` config (which uses `${policy_base_url}` interpolation) is unaffected.

### 2. `nemo_gym/profiling.py` — Lazy Import Fix

**Problem:** Top-level imports of `gprof2dot` and `pydot` cause `ModuleNotFoundError` on Ray worker nodes where these profiling dependencies are not installed. These modules are only needed when `dump()` is called, not at import time.

**Fix:** Moved the imports inside the `dump()` method so they're only loaded when actually needed.

**Backwards compatibility:** No behavioral change. The modules are imported at the same point in the execution flow — just deferred from module load time to first use.

### 3. `resources_servers/turing_vif/app.py` — Configurable Reward Aggregation

**Problem:** The reward signal was hard-coded to all-or-nothing (AND): `reward = float(all(is_following_list))`. This is the strictest aggregation — the model only receives reward 1.0 when every single check passes, and 0.0 otherwise. For tasks with many constraints, this produces a very sparse reward signal that can hinder RL training.

**Fix:** Added an `AggregationMode` enum and a configurable `aggregation_mode` field to `TuringVIFResourcesServerConfig`. The `verify()` method now converts the per-check boolean results into float scores and delegates to `_aggregate_scores()`, which supports five modes:

| Mode | Behavior | Output |
|------|----------|--------|
| `all` (default) | All checks must pass (AND) | 0.0 or 1.0 |
| `any` | At least one passes (OR) | 0.0 or 1.0 |
| `mean` | Average of binary scores | [0.0, 1.0] |
| `min` | Minimum score | 0.0 or 1.0 |
| `max` | Maximum score | 0.0 or 1.0 |

**Backwards compatibility:** Fully backwards-compatible. The default is `all`, which reproduces the original hard-coded behavior exactly. The base `turing_vif.yaml` config now declares the field explicitly. Experiment YAMLs can override it per-experiment (e.g., `aggregation_mode: mean` for multichallenge and inverse_if).

### 4. `resources_servers/turing_vif/app.py` — Thinking Trace Stripping

**Problem:** The `verify()` method extracted the response text by blindly taking `body.response.output[-1]`, which could be a `type: "reasoning"` output item instead of the actual assistant message. Inline `<think>`/`<thinking>` blocks were also passed through verbatim to both fast validators and the LLM judge. This meant word counts, keyword checks, and judge evaluations could be contaminated by the model's chain-of-thought.

**Fix:** Added two module-level helpers:
- `_extract_text_from_response()`: iterates output items in reverse to find the last `type="message"` / `role="assistant"` item (skipping `type="reasoning"` items entirely), then regex-strips `<think>`/`<thinking>` blocks.
- `_strip_thinking_traces()`: strips thinking tags from any string. Applied to judge responses in `_validate_custom_llm_judge_async`, `_validate_llm_instruction_async`, and `_get_dynamic_definition_async` before JSON parsing.

This mirrors the MultiChallenge server's approach, handling all three thinking representations: `type="reasoning"` output items, inline `<think>`/`<thinking>` tags, and `role="thinking"` messages.

**Backwards compatibility:** Fully backwards-compatible. For non-thinking models, the output list typically has a single `type="message"` item and no thinking tags, so behavior is unchanged.

### 5. `resources_servers/turing_vif/configs/turing_vif.yaml` — Base Config Update

Added `aggregation_mode: all` to the base server config so the field is explicit and discoverable.

### 6. `resources_servers/turing_vif/README.md` — Documentation Update

Updated to document the new `aggregation_mode` config field with a table of available modes and an example experiment YAML override.

### 7. `resources_servers/turing_vif/vif_validators/data_loader.py` + `app.py` — Judge Prompt Improvements

**Problem:** The `LLM_JUDGE_QUESTION_PROMPT` template had two issues causing noisy and systematically deflated reward signals for `llm_judge`-based evaluation (used by MultiChallenge):

1. **Fragile JSON output format.** The template asked the judge to return a JSON object `{"verdict": "YES" or "NO", "reasoning": "..."}`. The extraction code parsed this with `json.loads` + Pydantic validation, and any parse failure silently scored the sample as `False`. With a stochastic judge (temperature > 0), malformed JSON was common — extra text around the JSON, lowercase `"yes"`, trailing commas, missing quotes — all producing guaranteed false negatives on potentially correct evaluations.

2. **Suboptimal content ordering.** The model's response appeared *before* the conversation context in the prompt. The judge read the response to evaluate before understanding what the conversation was about or what criterion to check. This is an unnatural evaluation order that can reduce judge accuracy on context-dependent criteria (e.g., "Did the model remember the user's seafood allergy?").

**Fix (template — `data_loader.py`):** Restructured `LLM_JUDGE_QUESTION_PROMPT` to:
- Place `{question}` (which contains the conversation context and evaluation criterion) **before** `{model_response}`, so the judge reads context first.
- Replace the JSON output instruction with a simple `[[YES]]` / `[[NO]]` bracket-marker instruction: *"Analyze carefully, then respond with exactly [[YES]] or [[NO]] on the last line."*

**Fix (extraction — `app.py`):** Replaced `_validate_custom_llm_judge_async`'s JSON parsing with robust three-tier verdict extraction:
1. **Primary:** Search for the last occurrence of `[[YES]]` and `[[NO]]` in the response (last-occurrence-wins, matching the reference implementation's semantics).
2. **Fallback:** Check the last line for plain `YES` or `NO` (case-insensitive).
3. **Default:** Score as `NO` if neither marker is found.

This eliminates the entire class of JSON parse failures. The judge can produce arbitrarily long reasoning, markdown formatting, or chain-of-thought — as long as `[[YES]]` or `[[NO]]` appears anywhere, the verdict is extracted correctly.

**Backwards compatibility:** Fully backwards-compatible. The `JUDGE_SYSTEM_PROMPT` (used for `instructions`-based stylistic/linguistic validation) is unchanged and still uses JSON output. Only the `llm_judge` code path is affected. The `JudgeResponse` Pydantic model is retained for the instruction-based path.

### 8. `resources_servers/turing_vif/app.py` + configs — Configurable Judge Sampling Parameters

**Problem:** The judge's sampling parameters (`temperature`, `top_p`, `max_tokens`) were hardcoded in the `_judge_llm_api_async` method signature. The previous defaults (`temperature=1.0`, no `top_p`) produced high-variance judge verdicts, contributing to a noisy reward signal during RL training.

**Fix:** Added three new config fields to `TuringVIFResourcesServerConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `judge_temperature` | `0.7` | Sampling temperature for judge LLM calls |
| `judge_top_p` | `0.8` | Top-p (nucleus) sampling for judge LLM calls |
| `judge_max_tokens` | `10000` | Max output tokens for judge LLM calls |

The `_judge_llm_api_async` method now reads these from config, with optional per-call overrides. The `top_p` parameter is threaded through `_judge_llm_api_call_async` to the underlying `create_chat_completion` call.

Updated all config files to declare the new fields explicitly:
- `resources_servers/turing_vif/configs/turing_vif.yaml` (Gym base config)
- `experiments/turing_vif_multichallenge/grpo_turing_vif_multichallenge.yaml`
- `experiments/turing_vif_inverse_if/grpo_turing_vif_inverse_if.yaml`
- `experiments/dummy_turing_vif/grpo_dummy_turing_vif.yaml`

**Backwards compatibility:** Fully backwards-compatible. The config fields have sensible defaults and can be overridden per-experiment via YAML or CLI (`++env.nemo_gym.turing_vif.resources_servers.turing_vif.judge_temperature=0.5`).

## Experiment Configs (NeMo-RL side, not in Gym)

The following changes were made in the NeMo-RL experiment configs to work with the corrected Gym code:

- Switched judge model server type from `vllm_model` (proxy-only, `spinup_server` was a no-op) to `local_vllm_model` (actually starts vLLM via Ray)
- Replaced hardcoded `judge_base_url` with `judge_server_name: local_vllm_model` for dynamic URL discovery
- Used proper `local_vllm_model` config fields: `vllm_serve_kwargs` and `vllm_serve_env_vars`
- Switched dataset loading from inherited `OpenMathInstruct-2` to `NemoGymDataset` with `nemo_gym_data_processor`
- Fixed `cluster.num_nodes` to use policy-only node count (judge nodes managed separately by NeMo-Gym)
- Set `aggregation_mode: mean` in both `grpo_turing_vif_multichallenge.yaml` and `grpo_turing_vif_inverse_if.yaml` for denser reward signal during RL training
