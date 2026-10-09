# Subagent RL for Agentic-VBench (port plan and status)

Branch `arushig/agentic-vbench-subagents` (worktree `nemo-rl-ut-avb-subagents`)
builds on the unified-teacher + context-compaction branch and adds subagent
training modelled on `mengxiwu/nemo-rl-internal` branch
`mengxiwu/compaction-subagent-nemorl-v2` (the `sand73` compaction/subagent
harness, read on 2026-10-09 from a local clone at
`/lustre/fsw/portfolios/nemotron/users/arushig/nemo-rl-internal-subagent`).

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
2. **Session-aware capture (next):** the proxy keys sessions by the request's
   leading context (system prompt and first user turn); a child session gets its
   own capture rollout id under the episode (`<rollout>-c<k>`), so Gym's
   sequential admission holds per session while siblings run in parallel. The
   agent returns the child ids and `spawn_call_id` (the parent's `task`
   tool-call id) in its verify response; the finalizer fetches all session
   receipts of an owner and tags rows with `session_role`/`parent_session_id`.
3. **Trainer semantics:** `env.nemo_gym.training_mode` plus
   `async_rl.multi_trace_enabled` wiring in config validation, one reward vote
   per owner across session rows (`logical_rollout_batch`), resume checks
   (`validate_multitrace_resume_config`), and evidence
   (`trace_training_evidence`) logged per step.
4. **Validation:** unit tests for session detection and tagging; a smoke with
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
(smoke 2184532), so the launcher sets `token_capture.max_train_row_tokens=81920`
(must be passed with Hydra `+`); longer rows are rejected (`row_too_long`,
kept as the owner's rejection reason in the finalize metrics) and their owner
trains nothing that step.

Masking semantics follow from the segments: a child's final answer is
assistant text in the child's row and a tool result (prompt tokens) in the
parent's row, so it is trained once.

## Not ported

Score centering, the SFT export tooling, the SWE-bench harness and the
`sand71`/`sand73` validation primitives are specific to the reference's
experiments and are not part of this port.
