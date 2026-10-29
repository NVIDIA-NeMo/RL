# Length Penalties for GRPO (NeMo-Gym)

This guide documents the length-penalty and length-bonus algorithms implemented in
`nemo_rl/utils/length_penalty.py`.

They apply to NeMo-Gym GRPO **training** rollouts on the legacy driver (`grpo_train`), the
data-plane driver (`grpo_train_sync`, `data_plane.enabled: true`) and the async
`AsyncTrajectoryCollector`. Validation rollouts never apply them, so validation metrics stay on
the raw environment reward. The SingleController path does not apply length penalties yet:
until that follow-up lands, a SingleController run with `grpo.length_penalty` set fails at
startup with `NotImplementedError` (alongside the other unsupported shapers) rather than
training on unadjusted rewards. Likewise the block requires the NeMo-Gym path: a run with
`grpo.length_penalty` set and `env.should_use_nemo_gym` false is rejected at setup.

Configure `grpo.length_penalty`. The block is typed (`LengthPenaltyConfig` in
`nemo_rl/utils/length_penalty.py`): unknown keys, misspelled `length_type` / gate values, a
`profile_band` channel with `b <= a` or `f` outside `[0, 1]` are all rejected when the config is
loaded, before the first rollout. The length adjustments mutate `full_result["reward"]` in place
during rollout postprocessing; the reward as handed to the hook is kept on the batch as
`env_reward` (see [Metrics](#metrics)).

All algorithms are resolved per prompt group. Unless otherwise stated, only rollouts with
`reward > 0` participate in length comparisons and receive length-based adjustments.

## Binary Rewards Requirement

Length adjustments are defined for **binary (0/1), single-reward environment rewards** only.
Every algorithm and the final clamp assume it, and a multi-reward verifier must keep
`reward == sum(reward_components)`, which an adjustment to the scalar alone would break. At
rollout time each prompt group's rewards are checked: a group containing any graded or negative
reward, or any `reward_components`, is skipped entirely — no adjustment, no clamp, rewards pass
through untouched — and a warning is logged once per agent. An agent disabled under
`agent_overrides` with `enabled: false` is skipped before this check, so a deliberately graded
agent (e.g. a genrm judge) produces no warning.

The one exception is the `zero_over_max` profiled cap: with
`profiled_length_allow_non_binary_rewards: true` it also caps groups with graded (non-negative)
scalar rewards instead of skipping them, and a negative reward is then an error rather than a
skip. Independently of that option, `zero_over_max` raises on `reward_components` instead of
skipping the group. See [Profiled Length Penalty](#5-profiled-length-penalty).

Consequences of binariness worth knowing:

- The adjusted reward is clamped at 0, so penalties can wipe a correct rollout's reward out but
  never flip its sign — and an all-wrong group stays all-zero (no within-group variance, hence
  no GRPO gradient, matching vanilla behavior on groups with no correctness signal).
- `top_percentile` is effectively inert: all correct rollouts tie at the top score, so every
  positive rollout is always a "top scorer".

## Common Config

```yaml
grpo:
  length_penalty:
    verbose: true
    default:
      enabled: true
      length_type: tokens
      top_percentile: 0.5
      reasoning_bonus: 0.0
      answer_bonus: 0.0
      total_bonus: 0.0
      longest_reasoning_penalty: 0.0
      longest_answer_penalty: 0.0
      longest_total_penalty: 0.0
      group_reasoning_length_penalty_coeff: 0.0
      group_answer_length_penalty_coeff: 0.0
      group_total_length_penalty_coeff: 0.0
      reasoning_zmad_threshold: 0.0
      reasoning_zmad_penalty: 0.0
      answer_zmad_threshold: 0.0
      answer_zmad_penalty: 0.0
      total_zmad_threshold: 0.0
      total_zmad_penalty: 0.0
      profiled_length_penalty: 0.0
      profiled_length_penalty_type: constant_over_n_std
      profiled_length_n_std: 1.0
      profiled_length_min_samples: 2
      profiled_reward_correct_threshold: 0.0
      profiled_length_allow_non_binary_rewards: false
      profile_band_total: false
      profile_band_reasoning: false
      profile_band_answer: false
    agent_overrides:
      math_with_judge_simple_agent:
        enabled: true
        group_total_length_penalty_coeff: 0.1
      instruction_following_simple_agent:
        enabled: false
      genrm_simple_agent:
        enabled: false
      code_gen_simple_agent:
        enabled: true
        total_zmad_threshold: 2.0
        total_zmad_penalty: 0.1
```

`length_type` may be:

- `tokens`: lengths are tokenizer token counts of the reasoning and answer text, re-tokenized
  without special tokens.
- `chars`: lengths are character counts of the reasoning and answer text.
- `all_generated_tokens`: the total length is every token the policy generated, read from the
  assistant messages of the rollout's `message_log` with no re-tokenizing. It covers all
  assistant turns, including think tags, end-of-turn and tool-call tokens, and is the length the
  reasoning pipelines' Kimi penalty uses.

With `tokens` and `chars`, lengths count the model's own output only: `reasoning` items and
`message` items with role `assistant` (or no role). Environment observations that some agents
append to the response as `user`-role messages are ignored, and so are tool-call items.
`all_generated_tokens` also excludes prompt, chat-template and environment-observation tokens,
which sit in the user messages of `message_log`.

Generated token ids carry no reasoning/answer split, so `all_generated_tokens` measures total
length only. Settings that read the reasoning or answer length on its own are rejected at config
load under it, for `default` and for each agent's merged `agent_overrides`: `reasoning_bonus`,
`answer_bonus`, `longest_reasoning_penalty`, `longest_answer_penalty`,
`group_reasoning_length_penalty_coeff`, `group_answer_length_penalty_coeff`, the reasoning and
answer zMAD penalties, `profile_band_reasoning` / `profile_band_answer` (including channels
turned on by global `profile_band.defaults`), and a `reasoning` or `answer` profile gate channel.

`agent_overrides` can override any supported parameter per agent; only the keys set under an
agent replace the `default` value, everything else is inherited. If an agent is missing from
`agent_overrides`, the implementation falls back to `default`. To disable length adjustments for
a specific environment or agent while keeping the default enabled, set `enabled: false` for that
agent:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      group_total_length_penalty_coeff: 0.1
    agent_overrides:
      instruction_following_simple_agent:
        enabled: false
      genrm_simple_agent:
        enabled: false
```

## Flag Reference

| Flag | Short description |
| --- | --- |
| `verbose` | Prints per-group length adjustment details during rollout processing. |
| `default` | Default length-adjustment config used for agents without an override. |
| `agent_overrides` | Per-agent config overrides keyed by agent name. |
| `enabled` | Enables length adjustment for this config block. |
| `length_type` | Selects length unit: `tokens` (re-tokenized text), `chars` (character counts), or `all_generated_tokens` (every generated token, total length only). |
| `top_percentile` | Fraction of positive scorers treated as top scorers for longest-penalty selection. |
| `reasoning_bonus` | Flat bonus for the shortest positive/top-scoring reasoning trace in a prompt group. |
| `answer_bonus` | Flat bonus for the shortest positive/top-scoring answer in a prompt group. |
| `total_bonus` | Flat bonus for the shortest positive/top-scoring reasoning + answer total length. |
| `longest_reasoning_penalty` | Flat penalty for the longest reasoning trace among top scorers. |
| `longest_answer_penalty` | Flat penalty for the longest answer among top scorers. |
| `longest_total_penalty` | Flat penalty for the longest reasoning + answer total length among top scorers. |
| `group_reasoning_length_penalty_coeff` | Dense group-relative coefficient for reasoning length; shorter positive rollouts get higher adjustment. |
| `group_answer_length_penalty_coeff` | Dense group-relative coefficient for answer length. |
| `group_total_length_penalty_coeff` | Dense group-relative coefficient for total reasoning + answer length. |
| `reasoning_zmad_threshold` | Modified-Z threshold for flagging long reasoning outliers. |
| `reasoning_zmad_penalty` | Flat penalty applied to reasoning lengths above the zMAD threshold. |
| `answer_zmad_threshold` | Modified-Z threshold for flagging long answer outliers. |
| `answer_zmad_penalty` | Flat penalty applied to answer lengths above the zMAD threshold. |
| `total_zmad_threshold` | Modified-Z threshold for flagging long total-length outliers. |
| `total_zmad_penalty` | Flat penalty applied to total lengths above the zMAD threshold. |
| `profiled_length_penalty` | Constant subtracted from rollouts past the `constant_over_n_std` threshold; 0 disables that type. |
| `profiled_length_penalty_type` | Profiled-length algorithm: `constant_over_n_std` (default) subtracts `profiled_length_penalty` at or past `mean + n_std * std`; `zero_over_max` sets the reward to exactly 0 past the longest passing profiled length. |
| `profiled_length_n_std` | Number of standard deviations used in `mean + n_std * std` (`constant_over_n_std` only). |
| `profiled_length_min_samples` | Minimum PASSING profiled rollouts required; below this no profiled penalty of either type is applied. |
| `profiled_reward_correct_threshold` | A profiled rollout is PASSING (correct) when its `profiled_rewards` entry is strictly greater than this; default `0.0` (i.e. `r > 0`). |
| `profiled_length_allow_non_binary_rewards` | `zero_over_max` only: also cap groups with graded (non-negative) scalar rewards instead of skipping them; a negative reward then raises. Default `false`. |
| `pass_rate_length_penalty_weight` | Weight of the pass-rate-scaled (MAI-style) dense length penalty; 0 disables. |
| `profile_band_total` | Enables per-prompt `{a,b,f}` multiplier on total length for correct rollouts. |
| `profile_band_reasoning` | Enables per-prompt `{a,b,f}` multiplier on reasoning length for correct rollouts. |
| `profile_band_answer` | Enables per-prompt `{a,b,f}` multiplier on answer length for correct rollouts. |
| `group_length_penalty_profile_gate` | Gates group-relative length coefficients using a per-prompt `profile_band` threshold. |
| `group_length_penalty_profile_gate_channel` | Selects which profile-band channel to gate on: `reasoning`, `answer`, or `total`. |
| `group_length_penalty_profile_gate_field` | Selects which field from the chosen profile-band channel to use as the gate threshold, usually `a`. |
| `group_length_penalty_profile_gate_positive_only` | If true, computes the gate mean using only `reward > 0` rollouts; if false, uses all rollouts. |

## Per-Prompt Data Format

Some algorithms depend on per-prompt metadata stored on each training-data row. NeMo-RL carries
each NeMo-Gym JSONL row as `extra_env_info`, and the rollout code reads these fields from the
row's **top level** (next to `responses_create_params` and `agent_ref`), then copies them onto
each rollout result before applying length adjustments. Do not nest them under an
`extra_env_info` key: nested fields are never read. A row whose agent has the profiled length
penalty on then fails the rollout with a `ValueError` (see below), while profile-band adjustments
are silently skipped and a profile gate without a global band stays closed.

```json
{
  "responses_create_params": {"input": [{"role": "user", "content": "Solve ..."}]},
  "agent_ref": {"name": "math_with_judge_simple_agent"},
  "expected_answer": "42",
  "profiled_rewards": [1, 1, 0, 1, 0, 1, 1, 1],
  "profiled_output_lengths": [18342, 17110, 32768, 19004, 28991, 16820, 17455, 18101],
  "profile_band": {
    "total": {"a": 18138.6667, "b": 23756.0123, "f": 0.9},
    "reasoning": {"a": 17686.3333, "b": 23111.0123, "f": 0.9},
    "answer": {"a": 452.3333, "b": 1097.3333, "f": 0.9}
  }
}
```

Field usage:

- `profiled_rewards`: used by the profiled length penalty to identify passing profiled rollouts
  (`reward > profiled_reward_correct_threshold`).
- `profiled_output_lengths`: used by the profiled length penalty to compute `mean + n_std * std`
  (`constant_over_n_std`) or the longest passing length (`zero_over_max`).
- `profile_band.total`: used by `profile_band_total` and by profile-gated group-relative
  penalties when `group_length_penalty_profile_gate_channel: total`.
- `profile_band.reasoning`: used by `profile_band_reasoning` and by profile-gated
  group-relative penalties when the gate channel is `reasoning`.
- `profile_band.answer`: used by `profile_band_answer` and by profile-gated group-relative
  penalties when the gate channel is `answer`.

The profile-band values mean:

```text
a: full reward / no penalty up to this length
b: multiplier reaches f at this length
f: multiplier at b
```

For profile-gated group-relative penalties, `a`, `b`, or `f` can be selected as the gate field,
though `a` is the normal choice:

```yaml
group_length_penalty_profile_gate: true
group_length_penalty_profile_gate_channel: total
group_length_penalty_profile_gate_field: a
```

## Implemented Algorithms

### 1. Shortest Rollout Bonus

Config keys:

- `reasoning_bonus`
- `answer_bonus`
- `total_bonus`

For each prompt group, the code finds the shortest positive rollout for the selected channel
and adds a flat bonus if that rollout is also a top scorer.

Channels:

- `reasoning_bonus`: shortest non-empty reasoning length.
- `answer_bonus`: shortest non-empty answer length.
- `total_bonus`: shortest non-empty reasoning + answer length.

This is a sparse adjustment: usually only one rollout per group gets the bonus for each enabled
channel.

Example:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      total_bonus: 0.1
```

### 2. Longest Top-Scorer Penalty

Config keys:

- `longest_reasoning_penalty`
- `longest_answer_penalty`
- `longest_total_penalty`
- `top_percentile`

For each prompt group, the code first selects positive rollouts in the top score percentile.
Among those, it subtracts a flat penalty from the longest rollout for the selected channel.

Channels:

- `longest_reasoning_penalty`
- `longest_answer_penalty`
- `longest_total_penalty`

The implementation requires at least two eligible top-scorer rollouts to compare.

Example:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      top_percentile: 0.5
      longest_total_penalty: 0.1
```

### 3. Group Relative-Length Scaling

Config keys:

- `group_reasoning_length_penalty_coeff`
- `group_answer_length_penalty_coeff`
- `group_total_length_penalty_coeff`

This is a dense group-relative adjustment over positive rollouts.

For each enabled channel:

1. Find the shortest and longest positive rollout lengths in the group.
2. Convert each length to a raw weight where shorter is larger:

   ```text
   raw_weight = 1 - (length - min_length) / (max_length - min_length)
   ```

3. Zero-center the weights by subtracting the mean raw weight.
4. Multiply by the configured coefficient.

Shorter positive rollouts receive positive adjustment; longer positive rollouts receive negative
adjustment. If all lengths are equal, the adjustment is zero.

Example:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      group_total_length_penalty_coeff: 0.1
```

### 4. zMAD Long-Outlier Penalty

Config keys:

- `reasoning_zmad_threshold`
- `reasoning_zmad_penalty`
- `answer_zmad_threshold`
- `answer_zmad_penalty`
- `total_zmad_threshold`
- `total_zmad_penalty`

This flags high-side length outliers among positive rollouts using the Iglewicz-Hoaglin modified
Z score:

```text
modified_z = 0.6745 * (length - median_length) / MAD
```

If `modified_z > threshold`, the corresponding flat penalty is subtracted.

Only long-side outliers are penalized. Short outliers are not penalized.

The implementation also has a fixed MAD floor:

```text
MAD / median >= 0.015
```

If the MAD is too small, no zMAD outliers are flagged.

Example:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      total_zmad_threshold: 2.5
      total_zmad_penalty: 0.1
```

### 5. Profiled Length Penalty

Config keys:

- `profiled_length_penalty_type`
- `profiled_length_penalty`
- `profiled_length_n_std`
- `profiled_length_min_samples`
- `profiled_reward_correct_threshold`

This uses per-prompt profiling metadata:

- `profiled_rewards`
- `profiled_output_lengths`

`profiled_length_penalty_type` picks how a rollout is penalized for running past what the
profiled model needed:

| Type | Threshold | Rollouts hit | Effect | Turned on by |
| --- | --- | --- | --- | --- |
| `constant_over_n_std` (default) | `mean + n_std * std` of passing lengths | length `>=` threshold | subtract `profiled_length_penalty` | `profiled_length_penalty > 0` |
| `zero_over_max` (Kimi-style) | `max` of passing lengths | length `>` threshold | reward set to exactly `0.0` | selecting the type |

For each prompt group:

1. Select profiled lengths from passing rollouts only: those whose `profiled_rewards` entry is
   strictly greater than `profiled_reward_correct_threshold` (default `0.0`, i.e. `> 0`). Raise
   the threshold when the profiling rewards are graded, e.g. `0.5` to count only rollouts scored
   above 0.5. The threshold also decides the count that `profiled_length_min_samples` checks.
2. If there are fewer than `profiled_length_min_samples` passing rollouts (or none at all), apply
   **no penalty** for this prompt. A profile that mostly failed carries no signal about the right
   thinking budget — the problem may simply need more than the profiled model had.
3. Otherwise compute the threshold for the selected type:

   ```text
   constant_over_n_std:  threshold = mean(profiled_lengths) + profiled_length_n_std * std(profiled_lengths)
   zero_over_max:        threshold = max(profiled_lengths)
   ```

4. `constant_over_n_std` subtracts `profiled_length_penalty` from rollouts whose total generated
   length is greater than or equal to the threshold. `zero_over_max` sets the reward to `0.0` for
   rollouts whose total generated length is strictly greater than the threshold: a rollout exactly
   as long as the longest passing profiled rollout keeps its reward.

`zero_over_max` subtracts the whole reward (an unbounded penalty that the 0-clamp floors at exactly
`0.0`), so a capped rollout ends at exactly `0.0` whatever bonus or group-relative adjustment it
received, and profile-band multipliers leave it there. Unlike `constant_over_n_std`, which is not
stacked on a rollout the group-relative channels already penalized, it always applies.

Non-binary rewards and `zero_over_max`:

| Env rewards in the group | `profiled_length_allow_non_binary_rewards: false` (default) | `true` |
| --- | --- | --- |
| binary (0/1) | over-cap rollouts set to `0.0` | over-cap rollouts set to `0.0` |
| graded, all `>= 0` (e.g. `0.5`, `2.0`) | group skipped, warning once per agent | over-cap rollouts set to `0.0`; the other length penalties still skip the group |
| any negative | group skipped, warning once per agent | `ValueError`: setting an over-cap negative reward to 0 would raise it, so negatives are not supported |
| any `reward_components` | `ValueError` | `ValueError` |

The negative-reward error fires for any negative reward in the group, not only an over-cap one, so
whether a run fails does not depend on rollout lengths. Multi-component rewards are always an error
with `zero_over_max`, because zeroing the scalar alone would break
`reward == sum(reward_components)`; turn the cap off for such agents under `agent_overrides`.
Low-effort groups are skipped before the negative-reward check, as for every other length penalty;
the `reward_components` check runs first, so it raises for them too.
`profiled_length_allow_non_binary_rewards` is rejected at config load unless the type is
`zero_over_max`, since every other length penalty requires binary rewards.

`zero_over_max` ignores `profiled_length_penalty` and `profiled_length_n_std`. A nonzero
`profiled_length_penalty` together with `profiled_length_penalty_type: zero_over_max` is rejected
when the config is loaded.

Both config checks run on `default` and on each enabled agent's settings after its
`agent_overrides` entry is merged into `default`, because those merged settings are what the agent
runs with. So `default` setting `zero_over_max` and an override setting only
`profiled_length_allow_non_binary_rewards: true` is accepted, while an override setting only
`profiled_length_penalty: 0.2` under that `default` is rejected.

While either type is on for an agent, every row of that agent must carry `profiled_rewards` and
`profiled_output_lengths` as top-level lists of equal length. A row that lacks them (forgotten,
misspelled, or nested under `extra_env_info`) or carries something else fails the rollout with a
`ValueError` naming the agent, rather than silently training without the penalty. On the legacy
and data-plane drivers this stops the run; the async collector counts it as a batch-worker failure
and aborts once failures exceed `max_generation_failures`. For mixed data where some agents have
no profiles, turn the profiled penalty off for those agents under `agent_overrides` (for example
`profiled_length_penalty_type: constant_over_n_std` and
`profiled_length_allow_non_binary_rewards: false`, with `profiled_length_penalty` left at 0; or
`enabled: false`). The `profiled_length_allow_non_binary_rewards: false` keeps the override valid
when `default` turns that option on.

`profiled_output_lengths` should be in the same unit as `length_type`. If they record the
profiled responses' generated token counts, use `length_type: all_generated_tokens`; then
`zero_over_max` measures length exactly like the reasoning pipelines' Kimi penalty. If they
record re-tokenized text, use `length_type: tokens`.

Examples:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      profiled_length_penalty_type: constant_over_n_std
      profiled_length_penalty: 0.1
      profiled_length_n_std: 1.0
      profiled_length_min_samples: 2
```

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      profiled_length_penalty_type: zero_over_max
      profiled_length_min_samples: 2
```

### 6. Profile-Band Multiplier

Config keys:

- `profile_band_total`
- `profile_band_reasoning`
- `profile_band_answer`

This uses per-row `profile_band` metadata with channel-specific `{a, b, f}` values:

```json
{
  "profile_band": {
    "total": {"a": 10000, "b": 15000, "f": 0.9},
    "reasoning": {"a": 9000, "b": 14000, "f": 0.9},
    "answer": {"a": 500, "b": 1000, "f": 0.9}
  }
}
```

For an enabled channel, the multiplier is:

```text
length <= a:     multiplier = 1
a < length < b:  multiplier = 1 - (length - a) / (b - a) * (1 - f)
length >= b:     multiplier = f
```

So the multiplier interpolates linearly from `1` at `a` down to `f` at `b`, then stays at `f` for
all lengths past `b`.

Profile-band multipliers are applied only to rollouts whose original environment reward is
positive.

Example:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      profile_band_total: true
```

#### Global Defaults (dataset without per-prompt bands)

When the dataset has no per-prompt `profile_band` metadata, global `{a, b, f}` values can be
set directly in the config under `length_penalty.profile_band`. Only the channels listed under
`defaults` are activated:

```yaml
grpo:
  length_penalty:
    profile_band:
      enabled: true
      defaults:
        total: {a: 10000, b: 20000, f: 0.5}
```

```yaml
grpo:
  length_penalty:
    profile_band:
      enabled: true
      defaults:
        reasoning: {a: 9000, b: 14000, f: 0.9}
```

```yaml
grpo:
  length_penalty:
    profile_band:
      enabled: true
      defaults:
        answer: {a: 500, b: 1000, f: 0.9}
```

The first config applies the multiplier on total length only, the second on reasoning length
only, and the last on answer length only. Multiple channels may be listed together.

Semantics:

- Channels under `defaults` are implicitly enabled — no need to also set
  `profile_band_total/reasoning/answer: true` under `length_penalty.default`. Per-agent
  `agent_overrides` can still disable a channel (e.g. `profile_band_total: false`).
- Per-prompt `profile_band` metadata, when present on a row, takes precedence over the global
  defaults on a per-channel basis (a row that only provides `total` still falls back to the
  global `reasoning`/`answer` blocks if those are configured).
- The global band also feeds profile-gated group-relative penalties
  (`group_length_penalty_profile_gate`) when rows lack metadata.
- A malformed channel block (missing `a`/`b`/`f`, `b <= a`, or `f` outside `[0, 1]`) is a
  config error and is rejected when the config is loaded.

### 7. Profile-Gated Group Relative-Length Scaling

Config keys:

- `group_length_penalty_profile_gate`
- `group_length_penalty_profile_gate_channel`
- `group_length_penalty_profile_gate_field`
- `group_length_penalty_profile_gate_positive_only`
- plus one or more group-relative coefficients:
  - `group_reasoning_length_penalty_coeff`
  - `group_answer_length_penalty_coeff`
  - `group_total_length_penalty_coeff`

This is a gate on group-relative length scaling. It does not define a separate penalty by itself.

Defaults: `group_length_penalty_profile_gate_channel: total`,
`group_length_penalty_profile_gate_field: a`,
`group_length_penalty_profile_gate_positive_only: true`.

For each prompt group:

1. Read a threshold from `profile_band[channel][field]`, for example `profile_band["total"]["a"]`.
   The band is the row's `profile_band` metadata merged over the global
   `length_penalty.profile_band.defaults` (row channels win), so the gate also works on datasets
   without per-prompt bands when global defaults are configured.
2. Compute the mean rollout length for the selected channel.
3. If `group_length_penalty_profile_gate_positive_only` is true, use only positive rollouts in
   that mean.
4. Enable group-relative length scaling only if:

   ```text
   mean_length > profile_band[channel][field]
   ```

If the gate is closed, all group-relative coefficients are set to zero for that prompt group.

The gate fails closed: when the threshold cannot be resolved — no `profile_band` available for
the channel (neither per-row nor global), an unknown channel name, or no eligible rollouts to
average (e.g. `positive_only: true` and the whole group scored 0) — group-relative scaling is
disabled for that prompt group. The verbose per-group logs record the reason
(`missing_profile_limit`, `unknown_channel`, `no_lengths`, `mean_le_limit`, `mean_gt_limit`).

`group_length_penalty_profile_gate_positive_only` only affects the gate decision. It does not
change the rollouts that receive the group-relative adjustment after the gate opens. In the
current implementation, group-relative length scaling itself still applies only to positive
rollouts.

Example:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      group_total_length_penalty_coeff: 0.1
      group_length_penalty_profile_gate: true
      group_length_penalty_profile_gate_channel: total
      group_length_penalty_profile_gate_field: a
      group_length_penalty_profile_gate_positive_only: true
```

### 8. Pass-Rate-Scaled Length Penalty (MAI-style)

Config keys:

- `pass_rate_length_penalty_weight`

Implements the MAI-paper length penalty `- w_len * R_len(y_i)` with

```text
R_len(y_i) = rho_q * |y_i| / l_max
```

where `rho_q` is the prompt group's pass rate (fraction of rollouts with `reward > 0`),
`|y_i|` is the rollout's total generated length (reasoning + answer, in `length_type` units),
and `l_max` is the longest total length among the group's CORRECT rollouts (not a hyperparameter —
the penalty is self-normalizing over the set it applies to; long wrong rollouts do not dilute it).

Properties:

- Dense pressure scaled by difficulty: easy prompts (high pass rate) get strong shortening
  pressure; hard prompts (low pass rate) get little.
- All-wrong groups (`rho_q = 0`) receive exactly zero penalty by construction, preserving the
  no-gradient-without-correctness-signal invariant.
- Applied to correct rollouts only (consistent with the rest of this module; under binary
  rewards plus the 0-clamp this matches the paper's behavior for wrong rollouts).
- The longest correct rollout in a group loses exactly `w_len * rho_q`; shorter ones lose
  proportionally less.

Example:

```yaml
grpo:
  length_penalty:
    default:
      enabled: true
      pass_rate_length_penalty_weight: 0.2
```

## Metrics

When `grpo.length_penalty` is set, the rollout postprocessor reports, next to the other reward
shapers' metrics (logged under `train/`):

| Metric | Meaning |
| --- | --- |
| `length_penalty/env_reward_mean` | Mean environment reward before length adjustments (the pass rate for binary envs). |
| `length_penalty/reward_delta_mean` | Mean `adjusted - env` reward over the rollouts. |
| `length_penalty/wiped_correct_frac` | Fraction of rollouts that were correct and whose reward the penalties clamped to 0. |
| `length_penalty/adjusted_frac` | Fraction of rollouts whose reward changed. |
| `length_penalty/skipped_non_binary_frac` | Fraction of rollouts in groups skipped for non-binary or multi-component rewards. |

All five are per-rollout means, so they read the same whether the hook sees the whole batch at
once (legacy and data-plane drivers) or one prompt group at a time (async collector, whose
per-group values are averaged).

`total_reward` and the per-agent `<agent>/reward/*` metrics carry the adjusted reward. The reward
as handed to the length hook (after effort shaping and reward penalties) is kept on the rollout
batch as `env_reward`, and the `baseline_reward/pct_0` / `pct_1` / `pct_mixed` diagnostics are
computed from it so they keep reporting the environment's all-wrong / all-correct / mixed prompt
fractions.

## Practical Notes

- Most length algorithms act only on positive rollouts (`reward > 0`).
- Additive penalties can wipe a rollout's reward out but never flip its sign: the adjusted
  reward is clamped at 0 (stacked flat penalties exceeding the reward produce 0, not a negative
  value). Non-binary rewards reach the clamp only through the `zero_over_max` cap with
  `profiled_length_allow_non_binary_rewards`, which rejects negative rewards; otherwise their
  groups are skipped wholesale (see Binary Rewards Requirement).
- `profile_band_*` multipliers also apply only to originally correct rollouts.
- `profiled_length_penalty_type: zero_over_max` is the one adjustment that overrides all
  others: a capped rollout's reward is exactly 0. Capped correct rollouts show up in
  `length_penalty/wiped_correct_frac`.
- Group-relative scaling can reduce average length aggressively because it gives dense per-group
  pressure.
- zMAD is more selective: it only hits high-side outliers.

## Final Recommendations

1. For domains where longer reasoning does not strongly correlate with higher accuracy, apply
   group-relative length scaling. This gives steady pressure toward shorter correct rollouts.

2. For domains where longer reasoning does correlate with higher accuracy, leave length
   unpenalized. Penalizing length in those domains can remove useful reasoning and hurt task
   performance.

3. If you have a target length in mind, for example when this model should be less verbose on a
   domain than another reference model, use profile-gated group-relative length scaling. The
   profile gate lets the penalty activate only when the prompt group's rollout lengths exceed a
   per-prompt target.
