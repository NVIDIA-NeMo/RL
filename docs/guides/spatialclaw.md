# Train SpatialClaw with Async GRPO

SpatialClaw is a multi-turn video agent that alternates a trainable main policy
with planning, reflection, auxiliary vision, and Python-tool sessions. NeMo RL
trains only generations from the main-policy session; the other sessions drive
the environment but do not contribute policy loss.

The reference recipe runs asynchronous GRPO with Nemotron 3 Nano Omni. It uses
32 initial video keyframes and preserves images introduced by later tool calls
for the exact policy turn that consumed them.

## Prerequisites

- A NeMo Gym checkout containing the `spatialclaw` resource and
  `spatialclaw_agent`.
- A SpatialClaw JSONL dataset whose video paths are readable on every
  generation worker.
- A Nemotron 3 Nano Omni checkpoint compatible with the base video-GRPO
  recipe.

The Gym agent must follow the per-turn multimodal result contract in
{doc}`../design-docs/nemo-gym-integration`: media added between policy calls is
returned as `prompt_multimodal_content` together with
`prompt_mm_processor_kwargs`. This keeps the async training forward aligned
with the pixels and image grouping used during generation.

## Dataset contract

Each row supplies a Responses API request and selects the SpatialClaw agent.
For example:

```json
{
  "responses_create_params": {
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_video", "video_url": "file:///data/example.mp4"},
          {"type": "input_text", "text": "Answer the question about this video."}
        ]
      }
    ]
  },
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "spatialclaw_agent"
  }
}
```

Keep task-specific expected-answer and resource fields required by the
SpatialClaw Gym server in the same row. Do not embed policy-generated rollout
history in an end-to-end GRPO dataset: the agent constructs that history
during each sampled trajectory.

## Run the reference recipe

Set the dataset and video roots, then launch the VLM GRPO entry point:

```bash
export SPATIALCLAW_DATA_PATH=/path/to/spatialclaw.jsonl
export SPATIALCLAW_VIDEO_ROOT=/path/to/spatialclaw-videos

uv run examples/run_vlm_grpo.py \
  --config examples/configs/recipes/vlm/vlm_grpo-nemotron-omni-30ba3b-16n8g-megatron-tp4ep4-async-gym-spatialclaw.v1.yaml
```

The recipe allocates two nodes to training and fourteen to non-colocated vLLM
generation, matching the inherited 16-node async video topology. Each optimizer
step consumes four prompt groups with sixteen generations per prompt, for a
global batch of 64 trajectories.

The model context is 16,384 tokens. SpatialClaw limits each main or auxiliary
call to 1,024 new tokens and runs at most four agent steps. Prefix-aware Gym
tokenization clamps later calls to the remaining context instead of
retokenizing or truncating an already generated assistant/tool prefix.

## Multimodal behavior

- The initial video is sampled to 32 frames with temporal patch size 2.
- `limit_mm_per_prompt.image: 512` leaves parser capacity for initial frames
  and images returned by `show()` calls; the sequence limit still bounds the
  usable history.
- Generated image delimiter tokens are blocked so malformed assistant text
  cannot become an unmatched media span when replayed as exact history.
- Multimodal payload deduplication shares immutable prompt media in driver
  memory without merging logical GRPO rows or their rewards.
- The request-ID path packs token IDs only; large repeated video tensors stay
  in their per-turn message logs until the trainer materializes them.

For a smoke run, override the data path with a small immutable subset and lower
`grpo.max_num_steps`. Keep frame sampling, chat-template, reasoning, and Gym
agent settings unchanged so the smoke run exercises the production media
contract.
