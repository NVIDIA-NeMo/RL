# Train with PivotRL

PivotRL trains a policy at an intermediate decision point from a frozen
trajectory. Each dataset row contains an immutable conversation prefix, the
expert action expected at the pivot, and the media visible to the policy at
that point. NeMo RL samples several alternative actions from the same row and
applies the standard GRPO objective to that one local decision.

The reference SpatialClaw recipe uses asynchronous GRPO with Nemotron 3 Nano
Omni. The prefix remains prompt context and is masked from policy loss; only
the newly generated assistant action is trainable. See {doc}`spatialclaw` for
the corresponding end-to-end async-GRPO baseline.

## Prerequisites

- A NeMo Gym checkout containing the `spatialclaw_pivot` resource and its
  `spatialclaw_pivot_agent`.
- A JSONL pivot dataset whose media paths are readable on every generation
  worker.
- A Nemotron 3 Nano Omni checkpoint compatible with the base video-GRPO
  recipe.

The PivotRL recipe depends on the per-turn multimodal result contract described
in {doc}`../design-docs/nemo-gym-integration`. A Gym agent should return
`prompt_multimodal_content` and `prompt_mm_processor_kwargs` for media added by
the frozen prefix or by tools.

## Dataset contract

Every row selects the PivotRL Gym agent and includes one expected local action:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "Use SpatialClaw Python actions."},
      {"role": "user", "content": "Which option is correct?"}
    ]
  },
  "expected_action": {
    "type": "final_answer",
    "answer": "B",
    "scoring_mode": "mcqa"
  },
  "pivot_id": "example:0",
  "trajectory_id": "example",
  "decision_index": 0,
  "session_kind": "main",
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "spatialclaw_pivot_agent"
  }
}
```

For a tool-action pivot, set `expected_action.type` to `python_calls` and put
the canonical expected calls in `expected_action.calls`. A production dataset
should give every row a stable, unique `pivot_id`; preserve the complete
assistant/tool history in `responses_create_params.input` instead of
reconstructing it at training time.

All generations for a row form one GRPO prompt group. Different pivot rows may
share text token IDs while carrying different images, so media identity must
come from the row and the Gym per-turn payload rather than a token-only hash.

## Run the reference recipe

Set the dataset and media roots, then launch the VLM GRPO entry point:

```bash
export SPATIALCLAW_PIVOT_DATA_PATH=/path/to/pivots.jsonl
export SPATIALCLAW_VIDEO_ROOT=/path/to/spatialclaw-videos

uv run examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-spatialclaw-pivotrl.v1.yaml
```

The recipe deliberately sets `grpo.max_rollout_turns: 1` and
`policy.generation.max_new_tokens: 4096`: PivotRL samples one local action, not
a replacement end-to-end trajectory. It retains the async video recipe's 16
generations per prompt and uses a 49,152-token context for the frozen history
and its media.

`loss_fn.reference_policy_kl_penalty` defaults to zero. Set it explicitly to a
positive experiment value when a reference-policy KL term is required.

## Data preparation checks

Before a full run, validate that:

- every selected pivot has more than one possible reward across repeated
  frozen-policy samples;
- the online verifier matches the verifier used to select or profile pivots;
- each row contains exactly one trainable decision boundary;
- frame sampling, temporal patch size, chat-template settings, and reasoning
  settings match those used when the pivot prefix was captured; and
- data rows do not contain generated rollouts from the policy being trained.

These checks prevent zero-advantage prompt groups and train/evaluation drift;
they belong in the data-generation pipeline because NeMo RL intentionally
treats the frozen prefix as opaque prompt context.
