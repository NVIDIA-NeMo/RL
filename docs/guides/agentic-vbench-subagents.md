# Subagent RL for Agentic-VBench

## Super VL TMPE regression fix (2026-10-19)

The public Super 3.5 recipe is
`examples/configs/recipes/vlm/vlm_grpo-nemotron-3.5-super-vl-120ba12b-8n8g-megatron-agentic-vbench-sc.v1.yaml`.
The 16-node async counterpart is
`examples/configs/recipes/vlm/vlm_grpo-nemotron-3.5-super-vl-120ba12b-16n8g-megatron-agentic-vbench-async-sc.v1.yaml`.
It selects `policy.megatron_cfg.stable_policy_forward: true`, native CPU
optimizer offload, and `policy.generation.vllm_kwargs.moe_backend: triton`
with `mamba_ssm_cache_dtype: float32`. The existing launchers still own
subagent mode, dataset paths, allocation identity, and online W&B settings.
Keep the full 262144-token context and leave vLLM batch-invariant mode off:
the native Mamba2 attention guard does not support that mode.

The learner fix scopes stable BF16 TP reductions, RMSNorm statistics, and
expert forward GEMMs across the actual scheduler, including checkpoint
recomputation on autograd threads. Native TE backward GEMMs remain in use.
Ignored dense padding uses balanced dummy expert routes only for independent,
dropless routers; all real token routes remain unchanged. The optional policy
flag defaults off for other recipes. Its native TE/MCore source hashes are
checked before installation; a different runtime requires requalification.

Both native layouts completed the original failing AVB-call regression with
every original token and all 5,002 generation targets preserved:

| Layout | Generation engines | Policy replicas | Worst raw standard TMPE |
| --- | ---: | ---: | ---: |
| Sync8, TP8/EP32 learner | 8 | 4 | 1.0121357957 |
| Async16, TP8/EP64 learner | 16 | 8 | 1.0199028627 |

Each generation engine was compared using its own captured expert routes
against every policy replica. Padding residuals were at most
4.76837158203125e-7 and repeated stable forwards matched exactly. All engines'
1,024-token greedy decodes were preserved; policy comparison of those fresh
decodes covered generation DP0. Native online W&B evidence:
[sync8](https://wandb.ai/adlr/nemorl-v2-subagent-comapction-debug/runs/superfreshparitys8v42)
and
[async16](https://wandb.ai/adlr/nemorl-v2-subagent-comapction-debug/runs/superfreshparitya16v43).

These are fixed-weight numerical regressions, with zero GRPO updates. They do
not establish 200-step training stability, identical generation/learner
probabilities, or a universal low-level CUDA root cause. Standard TMPE
calculation, original thresholds, GRPO loss, rewards, advantages, and target
masks are unchanged. Existing running allocations retain their loaded backend;
the revised launch contracts take effect at the next natural allocation.
When resuming, carry the generation backend and SSM dtype from the reviewed
launch recipe into the saved config, while preserving optimizer/data-plane
state, batches, sampler semantics, checkpoint cadence, and training horizon.

The integration branch `ehsan/nemorl-v2-video-cc-subagents` builds on Rohit's
NeMo-RL v2 unified-teacher line and adds video, R3, framework-owned context
reconstruction, and subagent training modelled on `mengxiwu/nemo-rl-internal`
branch
`mengxiwu/compaction-subagent-nemorl-v2` (the `sand73` compaction/subagent
harness, read on 2026-10-09 from a local clone at
`/lustre/fsw/portfolios/nemotron/users/arushig/nemo-rl-internal-subagent`).
Arushi's `arushig/agentic-vbench-subagents` remains a comparison branch, not the
base of this integration.

## What the reference does

- One fixed `training_mode` per run: `plain`, `compaction` or `subagent`,
  validated at config time, on every published rollout and on resume. Compaction
  and subagents are never combined.
- Subagent mode: OpenCode's `task` tool is enabled for the primary agent; the
  configured `general` child runs in its own session with the parent's
  permissions, tools and turn limit, except an explicit `task` deny (depth one).
  Sibling task calls run in parallel; each is foreground and the parent waits.
- Training semantics: parent and child sessions are separate contiguous trace
  rows of one scored rollout. Each session's own generated tokens are trained;
  the copy of a child's answer inside the parent's tool result is context only.
  The reward and GRPO advantage are computed once per original rollout and
  shared across its traces; traces never add reward votes or change
  prompt-group, sibling or scheduler counts (`multi_trace.py`,
  `LogicalRolloutBatch`).
- Evidence: per-trace ownership (`session_id`, `root_session_id`,
  `parent_session_id`, `spawn_call_id`, owned `request_ids`, a
  generation-call manifest with token digests) validated by
  `session_traces.py`; checkpoint/replay recovery preserves ownership and
  refuses semantic drift.
- Mechanism: NeMo-Gym token echo (the agent returns exact prompt/generation
  tokens per session) with a Gym and OpenCode provider source overlay that
  exposes session identities and compaction boundaries.

## How it maps onto this stack

Our capture is the framework-owned ledger (PR #4200 / Gym PR #3368) with stock
OpenCode behind the agent's capture proxy, so the semantics port and the
mechanism changes:

| Reference | Here |
|---|---|
| Token echo per session, Gym/OpenCode overlay | Ledger-captured calls; the proxy identifies sessions from request shape (stock OpenCode sends no session header) and routes each session to its own capture stream |
| `traces` in the Gym result, `_postprocess_nemo_gym_traces` | Finalizer segments: a logical rollout already yields one row per chain (`{owner}_s{n}`, `segment_index`); child sessions become additional chains of the same owner, tagged with the session metadata |
| `multi_trace.py` logical-rollout collapse | Partly present (grouping prompt and `owner_mask` are shared across a rollout's rows); vendored module supplies the explicit validation of one reward vote per rollout and complete trace ownership |
| Ownership masks in the loss | Automatic: only generated tokens of each chain are targets; a child's answer copied into the parent's tool result is prompt text in the parent's chain |
| `training_mode` config + validators | `session_traces.py` vendored; agent knobs `opencode_training_mode`, `opencode_subagents_enabled`, `opencode_subagent_max_depth`, `opencode_subagent_max_active` |
| OpenCode `general` child with `task` deny | `harbor_runner.job_config`: `permission.task` allow/deny, `agent.general` (mode subagent, parent's steps, `task` deny) |

## Phases

0. **Vendored validation (done, this branch):** `nemo_rl/environments/session_traces.py`,
   `nemo_rl/algorithms/single_controller_utils/multi_trace.py` and their tests,
   unchanged from the reference, not yet wired.
1. **Harness (done on the Gym branch `cc-pr3368-unified-subagents`):** the agent
   config knobs above; `job_config` emits the OpenCode permission and `general`
   child configuration and rejects compaction/pruning in subagent mode. Default
   behaviour changes in one way for every mode: the `task` tool is now denied
   unless subagents are enabled, so single-session runs are single-session by
   construction.
2. **Session-aware capture (done):** the proxy keys sessions by the request's
   leading context (system prompt and first user turn). A child session roots a
   separate capture chain under the same rollout owner; later child calls match
   that chain by longest complete prefix. The agent returns session traces and
   `spawn_call_id` (the parent's `task` tool-call id) in its verify response;
   the finalizer binds the selected chains and tags rows with
   `session_role`/`parent_session_id`.
3. **Trainer semantics (done):** `env.nemo_gym.training_mode` plus
   `async_rl.multi_trace_enabled` wiring in config validation, one reward vote
   per owner across session rows (`logical_rollout_batch`), resume checks
   (`validate_multitrace_resume_config`), and evidence
   (`trace_training_evidence`) logged per step.
4. **Validation (in progress):** unit tests for session detection and tagging; a smoke with
   `AGENTIC_VBENCH_TRAINING_MODE=subagent` on assembly tasks checking observed
   parallel children, out-of-order returns, one reward vote per rollout, and
   zero poisoned calls; then the 20-step run.

## Capture of child sessions on this stack (phase 2 design notes)

- Gym's framework-owned admission rejects a call while another call of the same
  rollout is pending, so the proxy serializes model calls per episode (it
  already did). Parallel children therefore overlap on tool execution and
  container work, not on model calls; this is the same ledger invariant the
  reference satisfied with per-session echo.
- A child session needs no new capture rollout id: its first call has a
  different leading context (the task prompt), so the ledger roots a new chain,
  exactly as it does for a compaction rewrite; later child calls match their
  own chain by longest complete prefix.
- The explicit parent hint (`x-nemo-gym-capture-parent`) is accepted only on the
  Responses API, not on the chat route OpenCode uses, so two children spawned
  with byte-identical task prompts would be an "ambiguous source-prefix
  candidate" and that call would fail. This is a known limitation; prompts are
  not altered to disambiguate.
- The proxy tags every served call with `session_id` (digest of system prompt
  plus opening user turn), `session_role`, `parent_session_id` and
  `spawn_call_id` (the parent's `task` tool-call whose `prompt` the child's
  opening turn repeats), and the agent returns `session_traces` (response ids
  per session) in its verify response. The trainer maps those response ids to
  capture chains to tag rows for `session_traces.py` validation and the
  one-vote-per-rollout batch.

## Phase 3: trainer wiring on the ledger-based stack

Our stack already trains one scored owner as several physical rows (the
context-compaction segments, `{owner}_s{n}`, with one reward per owner in
`logical_advantage`). Subagent training reuses that machinery; the reference's
token-echo multi-trace loop is not ported.

- `env.nemo_gym.training_mode` (`plain | compaction | subagent`) is a NeMo-RL
  key consumed by `NemoGym` (never reaches Gym). It requires
  `token_capture.context_compaction=true` and is validated at actor init
  against the agent override block
  (`env.nemo_gym.agentic_vbench_agent.responses_api_agents.agentic_vbench_agent.opencode_*`)
  with the vendored `validate_session_training_config`.
- `_postprocess_captured_history` binds the agent's `session_traces` to the
  selected capture chains (`nemo_rl/environments/gym_sessions.py`): every
  selected response must be owned by exactly one session, the parent must own
  a selected response, plain/subagent modes forbid OpenCode compaction, and
  the per-session tags pass `validate_session_trace_metadata`. The tags ride
  `env_extras.session_trace_metadata` (JSON) with `num_sessions` /
  `num_child_sessions`; the `RolloutSelection` carries `session_ids` /
  `session_roles` parallel to the selected responses.
- `RolloutReassembler._plan_selected_calls` now groups selected records into
  chains by explicit `parent_call_id` even when chains interleave in capture
  order (`group_selected_chains`): a parent resumes after its children, and
  parallel children alternate. Every link is still verified; a chain cannot
  span sessions. Each `SegmentReceipt` and row tag carries `session_id` /
  `session_role`.
- Launcher: `scripts/agentic_vbench/smoke_subagents_fast_step.sh` (identity
  arm + subagents; same fast-step shaping as the compaction launcher).

Row length: without compaction an episode is one row of its whole context.
On 8 training nodes rows of 92-93k tokens trained, 114k/123k rows OOM'd the
learner (smoke 2184181), and a step holding 97k rows OOM'd with a 98k cap
(smoke 2184532). A later four-learner-node TP8/EP16 run also OOM'd on a
near-80k accepted row, so the current launcher sets
`token_capture.max_train_row_tokens=65536`
(must be passed with Hydra `+`); longer rows are rejected (`row_too_long`,
kept as the owner's rejection reason in the finalize metrics) and their owner
trains nothing that step.

Masking semantics follow from the segments: a child's final answer is
assistant text in the child's row and a tool result (prompt tokens) in the
parent's row, so it is trained once.

## Completion and overlong semantics

The parent and all child rows are one logical owner. With the standard
`overlong_filtering=true`, that owner is excluded if any of its segments ended
with `finish_reason=length`; this prevents partial parent/child trajectories
from reaching the optimizer. Do not bypass this contract by disabling
filtering, raising the row ceiling, changing masks, or altering GRPO/PPO,
rewards, advantages, clipping, the optimizer, or the TMPE calculation.

The v15 synchronous R3 run demonstrated the distinction. Its second wave had
rewards `[0 x14, 0.25, 1.0]`, but the two reward-bearing owners contained at
least one length-terminated parent or child segment. The only three physical
rows admitted to the learner belonged to reward-zero owners, so the resulting
zero gradient was correct. Raw TMPE remained healthy (maximum `1.01184`).

The harness fix gives the primary agent and the depth-one `general` child
separate completion instructions. The child is told to complete only its
bounded assignment, use tools early, return as soon as its evidence is ready,
and stop before its output limit. The parent is told to request a concise child
result and return a brief final answer immediately after verification. Token
budgets, row limits, standard overlong filtering, and all RL mathematics remain
unchanged. `opencode_subagent_prompt` is wired through the Gym config and
`harbor_runner.py`; the child still has `task` denied, preserving depth one.

## Shared decomposition prompt experiment

Use the `decomp` recipes for matched GRPO/PPO decomposition experiments. Their
shared instruction is part of the experiment configuration and must be retained
when comparing algorithms or evaluating their checkpoints. Enabling the `task`
tool alone does not select this instruction. The canonical prompt is in
[`subagent_delegation_decomposition.yaml`](../../examples/configs/subagent_delegation_decomposition.yaml):

| Algorithm and layout | Recipe |
| --- | --- |
| GRPO, synchronous, 8 nodes | [Super sync8 decomposition](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-3.5-super-vl-120ba12b-8n8g-megatron-claweval-agentic-vbench-sc-decomp.v1.yaml) |
| GRPO, asynchronous, 16 nodes | [Super async16 decomposition](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-3.5-super-vl-120ba12b-16n8g-megatron-claweval-agentic-vbench-async-sc-decomp.v1.yaml) |
| PPO, asynchronous, 16 nodes | [Super PPO async16 decomposition](../../examples/configs/recipes/vlm/vlm_ppo-nemotron-3.5-super-vl-120ba12b-16n8g-megatron-claweval-agentic-vbench-async-sc-decomp.v1.yaml) |

Gym forwards `opencode_delegation_prompt` into OpenCode's main-agent system
prompt. The parent receives the same instruction for both ClawEval and Agentic VBench.
The instruction requires at least one bounded child task and an obtained,
integrated child result before the final answer. For two or more independent
subtasks, it requires multiple child sessions. Create distinct children with
separate `task` calls that omit `task_id`, or continue an existing child with its
actual returned `task_id` and a focused follow-up. Independent calls should be
submitted together when supported; repeated calls to one session do not count
as multiple children. Both modes use OpenCode's existing tools and session capture.

This replaces the earlier optional wording, "when beneficial." It is a mandatory
system-prompt requirement, not a runtime guarantee: inspect captured Task calls,
distinct child sessions, and returned results to measure compliance. These recipes
add no delegation reward or rejection rule, so instruction violations remain
visible in the native rollouts. Record the exact prompt for each experiment;
historical checkpoints and already-running jobs retain their original prompt.

These recipes inherit their paired baseline's model, dataset, rollout settings,
context limits, admission masks, optimizer, loss, and TMPE calculation. They
change the parent instruction and use separate checkpoint/log directories and
W&B names. The PPO variant inherits the existing PPO critic configuration.
Choosing this prompt variant does not establish that a model or cluster
configuration has completed end-to-end PPO qualification.

Set `MODEL_PATH` to the same Super GA weights and `TRAIN_PATH`/`VAL_PATH` to the
same mixed prompt-only JSONL for a GRPO/PPO comparison. Use separate output roots
and W&B IDs for fresh arms; preserve an arm's ID and checkpoint root when
resuming it. `CLUSTER_NAME` controls the W&B name suffix, with
`SLURM_CLUSTER_NAME` as its fallback. `WANDB_NAME` can set the complete run name.
The existing `AGENTIC_VBENCH_DELEGATION_PROMPT` and
`CLAW_EVAL_DELEGATION_PROMPT` variables still override their respective agents.
For a matched comparison, leave both unset to use the shared default, or set
both to the same instruction. Use that same instruction when evaluating the
base GA model and trained checkpoints; changing the evaluation prompt alongside
the checkpoint would confound the comparison.

## Not ported

Score centering, the SFT export tooling, the SWE-bench harness and the
`sand71`/`sand73` validation primitives are specific to the reference's
experiments and are not part of this port.
