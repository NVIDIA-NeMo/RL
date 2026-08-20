# Image-Zoom Multi-Turn Tool-Use GRPO — Design

Design doc for MR !54 (`jseppanen/nemo-rl` → `super-v3-omni-vllm20-aws-dfw`): RL training
for a multimodal model that zooms into image regions via a multi-turn tool.

This is the "why + architecture + how to extend" doc. Its companion,
[`prompt-lifecycle-image-zoom.md`](prompt-lifecycle-image-zoom.md), is the deep dive on the
token-level mechanics (chat template / tokenization / the token-in-token-out splice). This doc
points there for those details rather than repeating them.

---

## 1. Motivation

Vision-language models downsample each image to a fixed token budget, so fine detail (small text,
chart values, distant objects) is lost. A model can do better if it can **zoom**: crop a
sub-region and re-encode it at full resolution. We expose that as a **multi-turn tool** — the
model emits a crop request (a bounding box), the environment crops + re-encodes the region and
feeds it back as a new image, and the model continues reasoning with the zoomed view.

This MR adds **GRPO training for that loop**: the policy learns end-to-end to (a) decide where to
zoom, (b) issue valid crop tool calls, and (c) answer correctly using the zoomed evidence. It is
built on NeMo-Gym's multi-turn Responses-API agents and NeMo-RL's GRPO.

## 2. Scope (what the MR adds)

Five layered commits:

| Commit | Adds |
|---|---|
| `8994a8b5` | Image-zoom Gym agent + NeMo-RL tool logic + sync/async launchers + data prep + tests |
| `819ee05a` | **Required-prefix token-in-token-out render** for multi-turn vLLM generation |
| `3cf5d2d8` | **Multi-image data flow** with exact token round-trip |
| `b6e7f879` | **Sync multi-turn advantage-grouping fix**; wire the multimodal processor into NeMoGym |
| `f9065d8a` | Pin the Gym submodule at the image_zoom_agent |

Baseline (non-zoom, single-turn) behavior is unchanged — see §6.

## 3. Architecture

One multi-turn rollout:

```
prompt (image + question)
  → vLLM generate (turn 1)  ──emits──>  <tool_call> crop(bbox) </tool_call>
  → NeMo-Gym image_zoom agent: parse → crop + re-encode region (tool) →
        per-crop reward (validity / IoU / dedup) → return crop image as the tool result
  → vLLM generate (turn 2, now with the crop image in context)
  → … up to max_tool_calls turns …
  → final assistant answer
  → grade answer (string-match → base reward) + sum tool rewards (aux)
  → GRPO: group-relative advantage → policy update
```

Components by layer:

- **NeMo-Gym environment** (`3rdparty/Gym-workspace/Gym/responses_api_agents/image_zoom_agent/`):
  `app.py` is the Responses-API agent that drives the loop, invokes the crop tool, and assembles
  the per-rollout reward; `configs/image_zoom_agent.yaml` holds the reward/limit knobs.
- **NeMo-RL ↔ Gym adapter** (`nemo_rl/environments/nemo_gym.py`): runs the Gym agents as resource
  servers, collects multi-turn rollouts, and **reconstructs the exact token sequence** of each
  rollout so GRPO scores precisely what vLLM produced. The deterministic tool/reward logic is in
  `nemo_rl/environments/image_tools_gym_tool.py`; `image_zoom_gym_tool.py` preserves the original
  MR 68 import names.
- **Generation / vLLM** (`nemo_rl/models/generation/vllm/`): `openai_serving_render.py` implements
  the **required-prefix render** (`required_prefix_handler=nemo_render`) — it splices the *exact*
  generated prefix token ids of each assistant turn and realigns multimodal placeholders for the
  injected crop images; wired via `vllm_worker_async.py` (+ `config.py`).
- **Multimodal data flow** (`nemo_rl/data/multimodal_utils.py`,
  `models/nano_v3_vl/dynamic_resolution_processor.py`, `models/megatron/multimodal.py`): each image
  expands to N vision embeddings; the pipeline **collapses** N→1 placeholder for transport and
  **re-expands** 1→N for training, with a guard that fails loud on any count mismatch.
- **Algorithm** (`nemo_rl/algorithms/grpo.py`): wires the multimodal processor into NeMoGym and
  fixes multi-turn advantage grouping (§4.5).

## 4. Key design decisions (and why)

### 4.1 Token-in-token-out is the central invariant
GRPO scores log-probs/advantages on token sequences. If the tokens the trainer scores differ —
even by one — from the tokens vLLM sampled, the importance ratios and the multimodal embedding
counts are wrong (here: a hard crash, not silent drift). Multi-turn tool use makes this hard:
between turns the environment injects content (the crop image + tool framing), and naively
re-tokenizing the rendered chat does **not** reproduce vLLM's exact ids. **Decision:** carry the
exact generated ids forward and *splice* them into the next turn's prompt, never re-render+re-tokenize.
Four gates enforce it (splice → strict exact-prefix check → embed-count gate → Megatron drift guard).
Details: `prompt-lifecycle-image-zoom.md`.

### 4.2 Required-prefix render replaces prefix-text-replacement
`nemo_render` wraps vLLM's `OpenAIServingRender` to splice exact prefix token ids and realign
multimodal placeholders. It replaces the older approach that round-tripped the prefix through text
(decode→re-tokenize), which drifts for multimodal multi-turn prompts.

### 4.3 Multi-image data flow with exact arithmetic
A rollout accumulates multiple images (original + crops). The expanded prompt length is computed
by **exact arithmetic over the atomic image special tokens**, not a decode→re-tokenize round trip —
the latter drifts and, under the shared image budget, corrupts crop sizing and crashes training.

### 4.4 Reward = task correctness + capped tool-shaping
`total = base_reward + aux_reward`. `base_reward` is answer correctness (binary, string-match) —
the objective we care about. `aux_reward` is per-valid-crop shaping (+0.02, capped at 0.05) with
penalties for invalid/duplicate crops — a small nudge toward competent tool use, **capped so it
can't dominate or be farmed**. Keeps the task objective primary while bootstrapping tool use.
(Tuning + a flat-reward analysis: `image-zoom-lr-reward-sweep.md`.)

### 4.5 Sync multi-turn advantage-grouping fix
GRPO groups generations per prompt for the leave-one-out baseline. The sync path grouped on the
wrong key for multi-turn rollouts; the fix groups on the *initial prompt messages*. It is a
**no-op for single-turn baselines** (one turn ⇒ grouping unchanged) — which is why the baselines
don't regress (verified by the parity runs in §6).

### 4.6 The system prompt teaches the model the tool protocol
The model doesn't know the tool exists a priori — the **system prompt**
(`examples/prompts/image_zoom_tool_system_prompt.txt`) teaches it: the task framing, *when* to use
the tool (fine-grained / blurry / text-heavy regions — and when not to), and the **exact tool-call
protocol** — a `<tools>` block declaring `image_zoom_in_tool` with `bbox_2d` (`[x1,y1,x2,y2]`,
normalized to 0–1000) and a `label`.

It is baked into the **data**, not the launcher: `tools/prepare_image_zoom_gym_data.py`
(`--system-prompt-file`, defaulting to the file above) **prepends it as the system message of every
example** at data-prep time. So changing it means **re-running data prep**, not editing a runtime
config.

**Keep three things in sync:** the tool name + parameter schema in the system prompt, the format the
Gym agent's parser (`image_zoom_agent/app.py`) expects, and the arguments the tool actually consumes.
If they disagree (e.g., the prompt says `bbox_2d` but the agent parses `box`), every call is invalid
→ the policy only ever collects the invalid-tool-call penalty and never zooms, and the reward looks
broken for a reason that has nothing to do with RL.

## 5. Adding a new multimodal Gym environment

The image-zoom env is the reference implementation. To add another multimodal multi-turn tool env:

1. **Gym agent** — under `3rdparty/Gym-workspace/Gym/responses_api_agents/<your_agent>/`, mirror
   `image_zoom_agent/`: `app.py` (parse the tool call → run the action → build the tool result +
   reward) + `configs/<your_agent>.yaml` (tool/reward knobs). If the tool feeds media back, return
   it as image/content parts (as image_zoom returns the crop).
2. **Tool logic** (if deterministic compute is needed) — model it on
   `nemo_rl/environments/image_tools_gym_tool.py` so it runs in the rollout worker.
3. **Data prep + system prompt** — adapt `tools/prepare_image_zoom_gym_data.py` to format your
   dataset (prompts, ground truth, image refs) into the NeMoGym jsonl. Write a new system-prompt
   file (`examples/prompts/<your_tool>_system_prompt.txt` — your tool's `<tools>` spec + when to use
   it) and pass it via `--system-prompt-file`; it is prepended as each example's system message
   (see §4.6). The tool name/params declared there **must match your Gym agent's parser**.
4. **Launcher** — copy `scripts/batch_nanov3_gym_grpo_image_zoom*onenv.sh` and point
   `env.nemo_gym.config_paths=[…,responses_api_agents/<your_agent>/configs/<your_agent>.yaml]` at
   your agent. Keep the multimodal-critical overrides:
   `++policy.generation.vllm_cfg.required_prefix_handler=nemo_render`,
   `++policy.generation.vllm_kwargs.limit_mm_per_prompt.image=<max images/rollout>`,
   `++loss_fn.truncated_importance_sampling_ratio=null`.
5. **If your tool injects images** you get token-in-token-out + the multi-image flow for free,
   provided (a) you use `nemo_render`, (b) `limit_mm_per_prompt` covers the max images a rollout can
   accumulate, and (c) the media uses the same atomic image special tokens. New media *types*
   (video/audio) need analogous collapse/expand + render handling.

**Gotchas that bit image-zoom:**
- Set `limit_mm_per_prompt.image` to the max a rollout can reach (original + all injected) — too low silently truncates.
- The trainer scores the *exact* generated tokens — never bypass `nemo_render` or re-tokenize the rendered chat, or you drift and crash (the gates catch it).
- Cap tool-shaping rewards so they can't be farmed.
- Baseline launchers need `++loss_fn.truncated_importance_sampling_ratio=null` (a pre-existing config gap).
- The system prompt's tool spec must match the agent's parser, and it's baked in at data-prep time — **re-run data prep to change it**, not the launcher (§4.6).
- Read `prompt-lifecycle-image-zoom.md` before touching the render or the multimodal arithmetic.

## 6. Testing & validation
- **Unit:** `tests/unit/environments/test_image_zoom_gym_tool.py` (crop/IoU/reward),
  `tests/unit/tools/test_prepare_image_zoom_gym_data.py` (data prep).
- **Invariant:** the Megatron drift guard fires **0 times** across all runs (the token round-trip
  holds at full scale).
- **Non-regression:** sync + async baselines behave identically on the base branch vs this MR
  (the advantage fix is a no-op for single-turn) — wandb parity runs
  `mr54fs-{base,mr}-{sync,async}` in `nvidia/grpo-nanov3vl`.

## 7. Pointers
- Token-level mechanics: `prompt-lifecycle-image-zoom.md`
- Full-scale config tuning + recommendation: `image-zoom-fullscale-tuning-log.md`
- LR/reward tuning + flat-reward diagnosis: `image-zoom-lr-reward-sweep.md`
- Reference code: `nemo_rl/environments/{nemo_gym.py, image_tools_gym_tool.py}`,
  `nemo_rl/models/generation/vllm/openai_serving_render.py`, `nemo_rl/models/megatron/multimodal.py`,
  `3rdparty/Gym-workspace/Gym/responses_api_agents/image_zoom_agent/`
