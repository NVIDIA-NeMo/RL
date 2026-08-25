# Length Adjustment Algorithms

This file documents the length-penalty and length-bonus algorithms implemented in:

`nemo_rl/utils/length_adjustments.py`

The code supports two usage modes:

1. **Reward mutation mode**
   Configure `grpo.length_bonus`. The length adjustment mutates `full_result["reward"]`.

2. **GDPO feature mode**
   Configure `grpo.adv_estimator.name: gdpo` and put length feature knobs under
   `grpo.adv_estimator.reward_features`. The same calculations are recorded in
   `full_result["gdpo_reward_features"]` and consumed by GDPO without mutating the scalar
   environment reward.

All algorithms are resolved per prompt group. Unless otherwise stated, only rollouts with
`reward > 0` participate in length comparisons and receive length-based adjustments.

## Common Config

```yaml
grpo:
  length_bonus:
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
      profiled_length_n_std: 1.0
      profiled_length_min_samples: 2
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

- `tokens`: lengths are tokenizer token counts.
- anything else: lengths fall back to character counts.

`agent_overrides` can override any supported parameter per agent. If an agent is missing from
`agent_overrides`, the implementation falls back to `default`. To disable length adjustments for
a specific environment or agent while keeping the default enabled, set `enabled: false` for that
agent:

```yaml
grpo:
  length_bonus:
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
| `length_type` | Selects length unit: `tokens` uses tokenizer counts; other values use character counts. |
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
| `profiled_length_penalty` | Flat penalty for rollouts longer than a per-prompt profiled-length threshold. |
| `profiled_length_n_std` | Number of standard deviations used in `mean + n_std * std` for profiled-length thresholding. |
| `profiled_length_min_samples` | Minimum profiled length samples needed before computing the profiled threshold. |
| `profile_band_total` | Enables per-prompt `{a,b,f}` multiplier on total length for correct rollouts. |
| `profile_band_reasoning` | Enables per-prompt `{a,b,f}` multiplier on reasoning length for correct rollouts. |
| `profile_band_answer` | Enables per-prompt `{a,b,f}` multiplier on answer length for correct rollouts. |
| `group_length_penalty_profile_gate` | Gates group-relative length coefficients using a per-prompt `profile_band` threshold. |
| `group_length_penalty_profile_gate_channel` | Selects which profile-band channel to gate on: `reasoning`, `answer`, or `total`. |
| `group_length_penalty_profile_gate_field` | Selects which field from the chosen profile-band channel to use as the gate threshold, usually `a`. |
| `group_length_penalty_profile_gate_positive_only` | If true, computes the gate mean using only `reward > 0` rollouts; if false, uses all rollouts. |

## GDPO Feature Mode

In GDPO feature mode, the same feature names can be selected under `reward_features`.

```yaml
grpo:
  adv_estimator:
    name: gdpo
    reward_features:
      default:
        env_reward: 1.0
        length_adjusted_reward:
          group_total_length_penalty_coeff: 0.1
        think_count_delta: 1.0
```

Feature entries can also be weighted:

```yaml
grpo:
  adv_estimator:
    name: gdpo
    reward_features:
      default:
        env_reward: 1.0
        length_adjusted_reward:
          group_total_length_penalty_coeff: 0.1
        think_count_delta: 0.5
```

The rollout code records feature metrics to WandB using names like:

```text
train/gdpo_length_adjusted_reward/mean
train/gdpo_length_adjusted_reward/min
train/gdpo_length_adjusted_reward/max
train/gdpo_think_count_delta/mean
```

## Per-Prompt Data Format

Some algorithms depend on metadata stored on each training-data row. The rollout code copies
these fields from `extra_env_info` into each rollout result before applying length adjustments.

At minimum, a row still looks like a normal NeMo-Gym training example. The length-related fields
are extra keys:

```json
{
  "problem": "Solve ...",
  "expected_answer": "42",
  "agent_name": "math_with_judge_simple_agent",
  "extra_env_info": {
    "profiled_rewards": [1, 1, 0, 1, 0, 1, 1, 1],
    "profiled_output_lengths": [18342, 17110, 32768, 19004, 28991, 16820, 17455, 18101],
    "profile_band": {
      "total": {"a": 18138.6667, "b": 23756.0123, "f": 0.9},
      "reasoning": {"a": 17686.3333, "b": 23111.0123, "f": 0.9},
      "answer": {"a": 452.3333, "b": 1097.3333, "f": 0.9}
    }
  }
}
```

Some data files store these fields at top level instead of inside `extra_env_info`; the important
part is that by rollout time the result has:

```json
{
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

- `profiled_rewards`: used by `profiled_length_penalty` to identify passing profiled rollouts.
- `profiled_output_lengths`: used by `profiled_length_penalty` to compute
  `mean + n_std * std`.
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
  length_bonus:
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
  length_bonus:
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
  length_bonus:
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
  length_bonus:
    default:
      enabled: true
      total_zmad_threshold: 2.5
      total_zmad_penalty: 0.1
```

### 5. Profiled Length Threshold Penalty

Config keys:

- `profiled_length_penalty`
- `profiled_length_n_std`
- `profiled_length_min_samples`

This uses per-prompt profiling metadata:

- `profiled_rewards`
- `profiled_output_lengths`

For each prompt group:

1. Prefer profiled lengths from passing rollouts.
2. If there are fewer than `profiled_length_min_samples` passing rollouts, fall back to all
   profiled lengths.
3. Compute:

   ```text
   threshold = mean(profiled_lengths) + profiled_length_n_std * std(profiled_lengths)
   ```

4. Penalize rollouts whose total generated length is greater than or equal to the threshold.

Example:

```yaml
grpo:
  length_bonus:
    default:
      enabled: true
      profiled_length_penalty: 0.1
      profiled_length_n_std: 1.0
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
  length_bonus:
    default:
      enabled: true
      profile_band_total: true
```

#### Global Defaults (dataset without per-prompt bands)

When the dataset has no per-prompt `profile_band` metadata, global `{a, b, f}` values can be
set directly in the config under `length_bonus.profile_band`. Only the channels listed under
`defaults` are activated:

```yaml
grpo:
  length_bonus:
    profile_band:
      enabled: true
      defaults:
        total: {a: 10000, b: 20000, f: 0.5}
```

```yaml
grpo:
  length_bonus:
    profile_band:
      enabled: true
      defaults:
        reasoning: {a: 9000, b: 14000, f: 0.9}
```

```yaml
grpo:
  length_bonus:
    profile_band:
      enabled: true
      defaults:
        answer: {a: 500, b: 1000, f: 0.9}
```

The first config applies the multiplier on total length only, the second on reasoning length
only, and the last on answer length only. Multiple channels may be listed together.

Semantics:

- Channels under `defaults` are implicitly enabled — no need to also set
  `profile_band_total/reasoning/answer: true` under `length_bonus.default`. Per-agent
  `agent_overrides` can still disable a channel (e.g. `profile_band_total: false`).
- Per-prompt `profile_band` metadata, when present on a row, takes precedence over the global
  defaults on a per-channel basis (a row that only provides `total` still falls back to the
  global `reasoning`/`answer` blocks if those are configured).
- The global band also feeds profile-gated group-relative penalties
  (`group_length_penalty_profile_gate`) when rows lack metadata.
- A malformed channel block (missing `a`/`b`/`f`, or `b <= a`) is ignored with a warning.

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

For each prompt group:

1. Read a threshold from `profile_band[channel][field]`, for example `profile_band["total"]["a"]`.
2. Compute the mean rollout length for the selected channel.
3. If `group_length_penalty_profile_gate_positive_only` is true, use only positive rollouts in
   that mean.
4. Enable group-relative length scaling only if:

   ```text
   mean_length > profile_band[channel][field]
   ```

If the gate is closed, all group-relative coefficients are set to zero for that prompt group.

`group_length_penalty_profile_gate_positive_only` only affects the gate decision. It does not
change the rollouts that receive the group-relative adjustment after the gate opens. In the
current implementation, group-relative length scaling itself still applies only to positive
rollouts.

Example:

```yaml
grpo:
  length_bonus:
    default:
      enabled: true
      group_total_length_penalty_coeff: 0.1
      group_length_penalty_profile_gate: true
      group_length_penalty_profile_gate_channel: total
      group_length_penalty_profile_gate_field: a
      group_length_penalty_profile_gate_positive_only: true
```

## Recorded GDPO Feature Names

The implementation records these feature names in `full_result["gdpo_reward_features"]`:

- `env_reward`
- `reasoning_bonus`
- `answer_bonus`
- `total_bonus`
- `longest_reasoning_penalty`
- `longest_answer_penalty`
- `longest_total_penalty`
- `group_reasoning_length_penalty_coeff`
- `group_answer_length_penalty_coeff`
- `group_total_length_penalty_coeff`
- `reasoning_zmad_penalty`
- `answer_zmad_penalty`
- `total_zmad_penalty`
- `profiled_length_penalty`
- `profile_band_total`
- `profile_band_reasoning`
- `profile_band_answer`
- `profile_band_delta`
- `length_additive_delta`
- `length_total_delta`
- `length_adjusted_reward`

`length_adjusted_reward` is the combined length-adjusted scalar that GDPO can use as one reward
feature. `length_additive_delta`, `profile_band_delta`, and `length_total_delta` are derived
summary features.

## Formatting Feature: think_count_delta

The GDPO branch also supports a formatting feature:

```text
think_count_delta = -abs(num_close_think_tags - 1)
```

This is not a length-penalty algorithm, but it can be selected alongside length features in GDPO:

```yaml
grpo:
  adv_estimator:
    name: gdpo
    reward_features:
      default:
        env_reward: 1.0
        length_adjusted_reward:
          group_total_length_penalty_coeff: 0.1
        think_count_delta: 1.0
```

For a format-only GDPO setup, use only the environment reward and the malformed-format feature:

```yaml
grpo:
  adv_estimator:
    name: gdpo
    reward_features:
      default:
        env_reward: 1.0
        think_count_delta: 1.0
```

To make the malformed-format feature weaker than the task reward, lower its weight:

```yaml
grpo:
  adv_estimator:
    name: gdpo
    reward_features:
      default:
        env_reward: 1.0
        think_count_delta: 0.5
```

## Practical Notes

- Most length algorithms act only on positive rollouts (`reward > 0`).
- `profile_band_*` multipliers also apply only to originally correct rollouts.
- Group-relative scaling can reduce average length aggressively because it gives dense per-group
  pressure.
- zMAD is more selective: it only hits high-side outliers.
- GDPO feature mode is useful when you want length or formatting behavior to be represented as a
  separate feature instead of mixing it into the scalar environment reward.

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
