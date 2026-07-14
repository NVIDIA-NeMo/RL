# Direct TransferQueue Rollouts with NeMo Gym

This guide explains the experimental vLLM-to-TransferQueue rollout writer and
shows how to run controlled legacy, shadow, and direct comparisons with the
eight-GPU Workplace Assistant recipe. It intentionally contains no benchmark
results.

The writer has two cursor modes. The **linear** cursor (the first half of
this guide) serves Gym's native sequential agent loop, where the driver signs
every request and turns extend one committed prefix. The **forest** cursor
(see [Black-box harness rollouts](#black-box-harness-rollouts-forest-cursor))
serves black-box harnesses — unmodified CLIs that construct their own
requests, speak text rather than token IDs, and may branch, spawn sub-agents,
or compact their context mid-rollout.

## What changes

The normal synchronous TransferQueue path returns complete token-bearing
trajectories through Gym. The rollout actor flattens them and publishes the
canonical training rows:

```text
vLLM -> Gym -> rollout actor -> TransferQueue train partition
```

With `data_plane.rollout_writer.enabled=true`, NeMo RL attaches a signed sample
identity to each Gym request. The vLLM model-owner worker reserves the next
turn, verifies the historical token prefix, and writes the new token, mask,
and sampling-log-probability delta to a staging partition. The rollout actor
then verifies and assembles the canonical row:

```text
                           +-> rollout_staging --+
Gym -> vLLM model owner ---+                      +-> finalizer -> train
                           +-> cursor registry --+
```

The cursor rejects concurrent turns, stale leases, duplicate completed
requests, prefix drift, and weight-version mismatches. The finalizer verifies
turn order, identity, hashes, tensor shapes, masks, and finite log
probabilities. A missing or corrupt trajectory becomes a masked placeholder
and is excluded from reward baselines and advantage normalization.

The main implementation is in:

- `nemo_rl/experience/rollout_writer.py`: signed context, linear and forest
  cursor state machines, staging verification, canonical assembly, and
  manifests.
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`: request reservation and
  per-turn TransferQueue writes from model-owner workers.
- `nemo_rl/experience/sync_rollout_actor.py`: finalization and canonical publish.
- `nemo_rl/experience/blackbox_finalizer.py`: receipt-reconciling finalizer for
  the forest cursor.
- `nemo_rl/experience/staged_token_source.py`: adapts verified staging rows to
  Gym's `TokenSource` protocol for trajectory assembly.
- `nemo_rl/algorithms/grpo_sync.py`: setup guards, writer lifecycle, and invalid
  trajectory masking.
- `nemo_rl/environments/nemo_gym.py`: ingress-gate installation and rollout
  register/seal orchestration for black-box mode.
- `3rdparty/Gym-workspace/Gym/nemo_gym/`: token-only response conversion needed
  to continue multi-turn tool calls without returning sampling log
  probabilities, plus the model-server ingress gate, rollout receipts, and the
  pure trajectory builder used by the black-box finalizer.

## Architecture and code flow

### Component ownership

The implementation deliberately separates untrusted request transport from the
trusted code that decides what reaches training.

| Component | Process | Responsibility |
| --- | --- | --- |
| `grpo_train_sync` | training driver | Validate the supported configuration, create the secret and cursor actor, and attach writers |
| `TQPolicy` | training driver | Bootstrap TransferQueue and register the staging and canonical partitions |
| `SyncRolloutActor` | Ray rollout actor | Mint sample identities, invoke Gym, finalize staged turns, and publish canonical rows |
| NeMo Gym | Gym Ray actor and HTTP services | Run the multi-turn agent/environment loop and forward opaque rollout metadata |
| `VllmAsyncGenerationWorkerImpl` | vLLM model-owner Ray worker | Verify request identity and prefix continuity, then write one staging delta per generated turn |
| `RolloutCursorRegistry` | dedicated Ray actor | Serialize reservations and commits for each sample |
| `assemble_staged_batch` | rollout actor | Read, verify, and assemble staging rows into trainer-facing tensors |
| GRPO/GDPO advantage estimator | training driver | Exclude rejected placeholders from baselines and normalization |

The Gym services can route successive turns to different HTTP workers. Cursor
state is therefore not stored in a vLLM worker. The dedicated registry is the
single authority for the next turn number, committed token length, prefix
hash, active lease, and terminal failure state.

### Setup lifecycle

The writer is attached once when `grpo_train_sync` starts:

1. `examples/nemo_gym/run_grpo_nemo_gym.py::_select_trainer` selects
   `grpo_train_sync` when `data_plane.enabled=true`.
2. `TQPolicy` bootstraps the configured TransferQueue backend and registers
   the normal `train` partition for canonical rows.
3. `grpo_train_sync` validates that this is synchronous NeMo Gym GRPO using
   async vLLM, a supported advantage estimator, and no router replay.
4. `TQPolicy.prepare_rollout_writer` registers `rollout_staging` with the
   per-turn tensor fields.
5. The driver creates a random 32-byte HMAC secret and a
   `RolloutCursorRegistry` Ray actor.
6. `VllmGeneration.configure_rollout_writer` sends the validated data-plane
   config, cursor handle, and secret only to vLLM model-owner workers.
7. `SyncRolloutActor` receives the same cursor handle and secret so it can mint
   identities and later request finalization manifests.

The secret is runtime state. It is never placed in YAML, Gym data, manifests,
or TransferQueue rows.

### Sample identity and trust boundary

At the start of `SyncRolloutActor.rollout_to_tq`, one `group_id` is created per
original prompt and one `sample_id` per generation:

```text
group_id  = 7d89...                         # one prompt
sample_id = 7d89..._g0                      # generation 0
sample_id = 7d89..._g1                      # generation 1
```

`_attach_rollout_contexts` adds a `RolloutContext` to the request's existing
`metadata.extra_body`. The signed fields are:

| Field | Meaning |
| --- | --- |
| `sample_id` | Stable identity of one generated trajectory |
| `group_id` | Groups generations that share the same GRPO prompt |
| `weight_version` | Training step whose policy weights produced the rollout |
| `nonce` | Random context identity |
| `issued_at`, `expires_at` | Bounded context lifetime |
| `version` | Wire-contract version |
| `signature` | HMAC-SHA256 over every preceding field |

Gym forwards this object opaquely. It does not decide a sample ID, turn number,
cursor length, or staging key. On each chat request,
`VllmAsyncGenerationWorkerImpl._prepare_rollout_request` reconstructs the
context and calls `validate_rollout_context` before accepting it.

### Per-turn write path

Each model call follows this sequence:

1. vLLM finishes chat-template rendering and obtains the exact
   `prompt_token_ids` that will be sent to the model.
2. `_prepare_rollout_request` derives an idempotency key from `sample_id` and
   the complete prompt tokens.
3. `RolloutCursorRegistry.reserve_turn` returns a `TurnReservation` containing
   `turn`, `lease`, `prev_len`, and `prev_hash`.
4. For turns after the first, Gym echoes the model-produced historical prefix
   in `required_prefix_token_ids`. The worker requires its length and hash to
   match the committed cursor and requires the rendered model prompt to start
   with those exact tokens.
5. vLLM generates one response. `_stage_rollout_response` extracts generated
   token IDs and sampling log probabilities from the non-streaming response.
6. `build_staging_delta` removes the already committed prefix. It emits only
   newly introduced prompt tokens plus newly generated tokens.
7. The worker synchronously writes the delta to
   `<sample_id>/t<turn>` in `rollout_staging`.
8. Only after the write succeeds does the worker call `commit_turn` with the
   new total length and full-prefix hash.
9. The HTTP response returns token IDs to Gym so a later tool turn can preserve
   the exact model prefix. In direct mode it removes sampling log probabilities
   from the HTTP payload because their authoritative copy is already staged.

The staging schema is intentionally small:

| Field | Dtype | Meaning |
| --- | --- | --- |
| `token_ids_delta` | `int64` | New environment/prompt tokens followed by generated tokens |
| `token_mask_delta` | `float32` | `0` for prompt/environment tokens and `1` for trainable generated tokens |
| `generation_logprobs_delta` | `float32` | `0` for prompt/environment tokens and the sampling log probability for generated tokens |

TransferQueue tags also record sample, group, turn, weight version, request
nonce, previous/new lengths, previous/new hashes, and lease state. Correctness
does not depend on trusting those tags: the finalizer uses the signed cursor
manifest and recomputes tensor-derived hashes.

### Cursor state machine

`RolloutCursorStateMachine` is single-threaded inside the Ray actor. Each
sample owns one `SampleCursor` and at most one active reservation.

```text
                    reserve_turn
          +--------------------------------+
          |                                v
no cursor/committed -----------------> reserved
                                         |   |
                              commit_turn |   | fail_turn
                                         v   v
                                     committed failed
                                         |
                                         +----> next reserve_turn
```

The important transitions are:

- A new request nonce reserves `next_turn` and captures the current
  `committed_length` and `prefix_hash`.
- Retrying the same nonce before commit returns the same reservation. After
  lease expiry it receives a new lease for the same turn.
- A different request while a turn is active fails the entire sample as a
  concurrent request.
- Committing requires the current lease and a strictly growing token length.
- Retrying an already committed request fails the sample as a duplicate.
- Once failed, a sample cannot reserve another turn.

This policy prefers rejecting one trajectory over accepting ambiguous tokens.
Other samples in the batch remain usable.

### Finalization and canonical publication

After Gym finishes the trajectories, `SyncRolloutActor` calls
`assemble_staged_batch` with the expected sample IDs, group IDs, and weight
version. For each sample, the finalizer:

1. gets an immutable `FinalizationManifest` from the cursor;
2. requires at least one committed turn and no active or failed cursor;
3. requires contiguous turn numbers beginning at zero;
4. verifies group identity and weight version on every turn;
5. verifies the previous length/hash chain;
6. reads every staging key, polling only until `finalize_timeout_s`;
7. verifies equal tensor lengths, binary token masks, and finite log
   probabilities;
8. appends each delta and recomputes the full token hash after every turn; and
9. verifies the terminal length and terminal prefix hash.

Verified rows are padded to the legacy batch width and form the canonical
training tensors: `input_ids`, `input_lengths`, `generation_logprobs`,
`token_mask`, and `sample_mask`. Reward and other driver-carry fields still
come from the trusted Gym result.

If a row fails verification, the finalizer substitutes a one-token padded row
and sets `trajectory_valid_mask=0` and `sample_mask=0`. It also zeros that
row's total and component rewards. `grpo_train_sync` passes the validity mask
into GRPO or GDPO baseline calculation and advantage estimation, then applies
it again before clipping and writeback as a fail-safe.

In shadow mode, `compare_shadow_candidate` requires every valid direct tensor
and driver-carry tensor to equal the legacy path. In direct mode, the verified
candidate is authoritative. Both modes publish only the canonical
`sample_id` to the `train` partition; staging keys never appear in trainer
metadata.

After canonical publication succeeds, the rollout actor clears the staging
keys and cursor entries. Training workers consume the canonical row through
the existing `TQPolicy` read/logprob/train pipeline.

## Two-turn rollout example

This simplified example follows one generation, `group-a_g0`, through two
model calls. Token values are illustrative, but the shapes and masks match the
implementation.

### Turn 0: initial assistant tool call

The rendered initial prompt is:

```text
prompt_token_ids    = [101, 102, 103]
generated_token_ids = [201, 202]
generated_logprobs  = [-0.10, -0.20]
```

The initial cursor has committed length zero and the well-known empty-prefix
hash:

```text
reservation = {
  turn: 0,
  prev_len: 0,
  prev_hash: EMPTY_PREFIX_HASH,
  lease: "lease-0"
}
```

`build_staging_delta` has no old prefix to remove, so the worker writes:

```text
key                         = group-a_g0/t0
token_ids_delta             = [101, 102, 103, 201, 202]
token_mask_delta            = [  0,   0,   0,   1,   1]
generation_logprobs_delta   = [0.0, 0.0, 0.0, -0.10, -0.20]
```

The worker commits:

```text
new_len  = 5
new_hash = hash_token_ids([101, 102, 103, 201, 202])
```

Gym receives the token IDs, executes the requested tool, and adds its tool
result to the next prompt. In direct mode Gym does not receive the sampling
log probabilities.

### Turn 1: answer after the tool result

Suppose chat-template rendering adds tool-result tokens `[301, 302]`, then the
model generates `[401, 402]`:

```text
required_prefix_token_ids = [101, 102, 103, 201, 202]
prompt_token_ids = [101, 102, 103, 201, 202, 301, 302]
generated_token_ids = [401, 402]
generated_logprobs = [-0.30, -0.40]
```

The cursor returns:

```text
reservation = {
  turn: 1,
  prev_len: 5,
  prev_hash: hash_token_ids([101, 102, 103, 201, 202]),
  lease: "lease-1"
}
```

The worker verifies that the required prefix is exactly the first five prompt
tokens. `build_staging_delta` removes those five committed tokens and writes
only the tool-result and new assistant tokens:

```text
key                         = group-a_g0/t1
token_ids_delta             = [301, 302, 401, 402]
token_mask_delta            = [  0,   0,   1,   1]
generation_logprobs_delta   = [0.0, 0.0, -0.30, -0.40]
```

The second commit records:

```text
new_len  = 9
new_hash = hash_token_ids(
  [101, 102, 103, 201, 202, 301, 302, 401, 402]
)
```

### Final canonical row

The finalizer reads `group-a_g0/t0` and `group-a_g0/t1`, verifies both cursor
links and hashes, and concatenates them:

```text
input_ids = [101, 102, 103, 201, 202, 301, 302, 401, 402]
token_mask = [0, 0, 0, 1, 1, 0, 0, 1, 1]
generation_logprobs = [0.0, 0.0, 0.0, -0.10, -0.20,
                       0.0, 0.0, -0.30, -0.40]
input_lengths = 9
sample_mask = 1
trajectory_valid_mask = 1
```

`kv_first_write` publishes that row under canonical key `group-a_g0` in the
`train` partition. The rollout actor then deletes `group-a_g0/t0`,
`group-a_g0/t1`, and the cursor entry. Policy logprob and training workers see
the same canonical schema they use for ordinary synchronous TransferQueue
training; they do not need to understand turns or staging.

If turn 1 had the wrong prefix, a missing staging row, or a mismatched hash,
the canonical row would instead have `sample_mask=0` and
`trajectory_valid_mask=0`, while sibling generations from `group-a` could
still train normally.

## Black-box harness rollouts (forest cursor)

Everything above assumes Gym's native agent loop: the driver signs each
request, turns are strictly sequential, and Gym echoes the exact historical
token prefix so the worker can verify continuity token-by-token. A black-box
harness — an unmodified CLI agent driven inside a Gym task — breaks all three
assumptions:

- it constructs its own request bodies, so the driver cannot attach a signed
  per-request context;
- it speaks text, so it never echoes `required_prefix_token_ids`; and
- it may issue concurrent calls, spawn sub-agents that branch from a shared
  history, or compact its context so a later prompt no longer extends any
  earlier one.

Setting `data_plane.rollout_writer.cursor=forest` (with
`accept_gateway_identity=true` and `mode=direct`) switches the write path to
a design that tolerates these behaviors while keeping the same fail-closed
guarantee: an ambiguous or broken rollout becomes a masked placeholder row,
never a failed batch.

### Identity model

| Identity | Minted by | Meaning |
| --- | --- | --- |
| `rollout_id` | driver (`mint_rollout_id`) | One GRPO generation: its Gym execution, its staged rows, its receipt, and its canonical train row. Opaque, URL-safe, 128 random bits as lowercase hex. Where a legacy field still requires a `sample_id`, it equals `rollout_id`. |
| `call_id` | Gym ingress gate | One admitted model call. The staging key is `<rollout_id>/<call_id>`. Two admitted calls with identical prompts are two nodes — prompt hashes are integrity metadata, never identity. |
| admission order | Gym ingress gate | Per-rollout monotonic order in which calls were admitted. HTTP completion order and TransferQueue write order carry no meaning. |
| `group_id` | driver | GRPO sibling grouping only. It stays in the driver's private `rollout_id -> group_id` mapping and never travels in requests, staged rows, keys, or receipts. |

In forest mode `_attach_rollout_contexts` attaches only the bare
`nemo_rl_rollout_id` to each task row's `metadata.extra_body` — deliberately
no signed context, because a signed linear context would route the worker
onto the sequential staging path. Because gateway-identified requests carry
no per-request weight version, the driver declares the step's policy weight
version to every model-owner worker up front
(`VllmGeneration.set_rollout_weight_version`).

### Trust boundary

The vLLM serving endpoint is assumed to be reachable only through Gym's
ingress gate (same host; sandbox networking does not route to the vLLM
port). Under that assumption, when `accept_gateway_identity=true`, the
worker trusts the `nemo_rl_rollout_id`/`nemo_rl_call_id` pair the gate
stamped on the internal request. The gate in turn:

- admits model calls only for registered, unsealed rollouts
  (`require_registration=true`);
- overwrites any harness-supplied identity fields;
- correlates a call to its rollout either from a `/ng-rollout/<rollout_id>`
  URL prefix (sandboxed CLIs) or from the `x-nemo-gym-rollout-id` header,
  which Gym's `ServerClient` lifts out of the trainer-stamped
  `metadata.extra_body` on every in-tree agent call; and
- strips token IDs and log probabilities from harness-facing responses — the
  staged TransferQueue row is the only token store.

The register/seal control API (`/ng-control/rollouts/...`) is guarded by a
random bearer token that `NemoGym` mints at startup; it never reaches agents
or task rows. Correctness still does not rest on the gate alone: the trusted
finalizer independently re-verifies every staged tensor (see below).

### End-to-end lifecycle

```text
driver          NemoGym env            ingress gate        vLLM worker        forest registry
  |                  |                       |                  |                    |
  | mint rollout_ids |                       |                  |                    |
  |----------------->| register each id --->|                  |                    |
  |                  | (PUT /ng-control)     |                  |                    |
  |                  | run harness --------->| admit call,      |                    |
  |                  |                       | mint call_id --->| candidates ------->|
  |                  |                       |                  | reserve_call ----->|
  |                  |                       |                  | write staging row  |
  |                  |                       |                  | commit_call ------>|
  |                  |                       |<-- token-free    |                    |
  |                  |                       |    response      |                    |
  |                  | seal rollout -------->|                  |                    |
  |                  | (POST .../seal)       |                  |                    |
  |<-- receipt ------|                       |                  |                    |
  | finalize: reconcile receipt vs forest manifest, verify rows, build chain        |
  | publish canonical row, clear staging keys and cursor entries                    |
```

1. **Setup.** `grpo_train_sync` validates the mode (forest requires
   `accept_gateway_identity=true` and `mode=direct`), registers the staging
   partition, and creates a `RolloutForestRegistry` Ray actor instead of the
   linear cursor registry. The rollout actor runs in a venv that bundles both
   the vLLM and NeMo-Gym extras (`VLLM_NEMO_GYM`) because finalization calls
   Gym's trajectory builder.
2. **Gate installation.** When the driver derives `blackbox_rollouts=true`
   from the writer config, `NemoGym` injects an observability block into the
   policy model server's config: the ingress gate, a capture directory, a
   control token, `require_registration=true`, and `return_token_ids=false`.
   Batch retries are forbidden (`rollout_max_attempts_to_avoid_lp_nan=1`)
   because registration is create-only and sealing is terminal.
3. **Registration.** Before dispatching a batch, the env reads each row's
   `nemo_rl_rollout_id` and registers it with the gate. Unregistered calls
   are rejected at admission.
4. **Per-call write path.** For each admitted call the worker: fetches the
   rollout's committed candidates from the registry (cumulative sequences,
   longest first); hashes its rendered prompt at each candidate length and
   reserves against the longest match as the storage parent; a prompt that
   extends no committed prefix reserves a `new_root` full-prompt row
   (context compaction), not a failure, and a second child of one prefix is
   a branch. It then writes one self-describing delta to
   `<rollout_id>/<call_id>` — token IDs, mask, log probabilities, plus a
   SHA-256 staging digest over all three and the identifying metadata — and
   commits the node with its new cumulative length and hash. The HTTP
   response returns neither token IDs nor log probabilities.
5. **Sealing.** After the harness (and its verifier) finishes, the env seals
   the rollout. The gate returns a token-free receipt: the call manifest
   (each admitted `call_id`, its admission index, and whether it staged) and
   a terminal status. The env attaches the verifier reward to the receipt as
   driver-carry. A seal failure yields a `seal_failed` receipt, which the
   finalizer turns into a masked placeholder. Receipts ride the batch as
   `ng_rollout_receipts` and are popped before anything is written to
   TransferQueue.

### Forest cursor state machine

`ForestCursorStateMachine` replaces the strictly sequential rules of the
linear cursor with per-node rules:

- Nodes are keyed by `call_id`. States are `reserved`, `committed`, and
  `failed`, each node with its own lease.
- Retrying an in-flight `call_id` idempotently returns the same reservation
  (a new lease after expiry); retrying a committed `call_id` is a duplicate
  and fails the node.
- A reservation's claimed parent must be a committed node whose length and
  hash match exactly. A rootless reservation must start at length zero; if
  committed nodes already exist it is flagged `is_new_root`.
- Committing requires the current lease and a strictly growing cumulative
  length, and records the staging key, prompt/generation lengths, terminal
  hash, staging digest, and weight version.
- Concurrent reservations are legal. A failed call fails only its node — the
  rollout survives, and the finalizer decides validity from reconciliation.

### Finalization: one assembler, two verifiers

`assemble_blackbox_batch` finalizes each rollout from two independent
sources: the gate's sealed receipt (what was admitted) and the forest
registry's manifest (what was committed). NeMo RL keeps transport integrity;
Gym keeps semantic assembly.

For each rollout, `finalize_blackbox_rollout`:

1. requires a `completed` receipt, a non-failed cursor, and matching rollout
   identity;
2. reconciles the receipt's staged calls one-to-one with committed nodes —
   a staged call without a committed node, a committed node absent from the
   manifest, a call that never committed, or an empty manifest all reject
   the rollout;
3. fetches every staged row and re-verifies: equal delta lengths, binary
   masks, finite log probabilities, the per-node length arithmetic, the
   expected weight version, and the recomputed staging digest (any
   storage-layer corruption or substitution is detected here);
4. rebuilds per-call `TokenEntry` records through
   `StagedSnapshotTokenSource` — the exact inverse of the worker's
   `build_staging_delta` — and checks each reconstructed cumulative sequence
   against the node's committed terminal length and hash;
5. delegates chain construction to Gym's pure `prefix_merging` builder over
   the verified snapshot and requires exactly one unambiguous `main` chain
   with no quarantined branches (the initial cardinality rule: branch and
   multi-root forests are masked until branch reward semantics are
   designed); and
6. asserts Gym's vendored NeMo-RL prefix-contiguity contract on the
   projected main-chain response.

Any failure at any step returns an invalid row whose reason feeds
`rollout_writer_manifest.jsonl`; the batch assembler substitutes a one-token
placeholder with `sample_mask=0` and `trajectory_valid_mask=0` and zeroes
every reward channel for that row. Verified rows are padded through the same
`pad_finalized_candidate_batch` tail as the linear assembler and published
under the canonical `rollout_id`. After publication the rollout actor clears
the staging keys and the forest cursor entries.

### Worked example: branch and compaction

Suppose one rollout admits four calls:

```text
c1: prompt [1, 2, 3]             generates [4, 5]   # first call
c2: prompt [1, 2, 3, 4, 5, 6]    generates [7]      # extends c1 (parent=c1)
c3: prompt [1, 2, 3, 4, 5, 8]    generates [9]      # sub-agent, also parent=c1
c4: prompt [99, 98]              generates [97]     # compacted context
```

The registry records `c2` and `c3` as two children of `c1` (a branch) and
`c4` as a `new_root` full-prompt row, and all four stage successfully — the
write path never fails a rollout for structure. At finalization, the
`prefix_merging` builder sees multiple eligible chains, so the rollout is
rejected as `ambiguous_forest` and trains as a masked placeholder. Had the
harness been purely sequential (`c1 -> c2` only), the single main chain
`[1, 2, 3, 4, 5, 6, 7]` would train with mask `[0, 0, 0, 1, 1, 0, 1]`,
exactly like the linear worked example above.

## Supported modes

| Mode | Overrides | Purpose |
| --- | --- | --- |
| Legacy TQ | `rollout_writer.enabled=false` | Control path; rollout actor publishes rows |
| Shadow | `enabled=true`, `mode=shadow` | Stage direct rows and require tensor equality with legacy rows |
| Direct | `enabled=true`, `mode=direct` | Train from verified staged tensors |
| Forest | `enabled=true`, `mode=direct`, `cursor=forest`, `accept_gateway_identity=true` | Black-box harnesses: gate-minted call identity, receipt-reconciled finalization |

The writer is opt-in and currently requires:

- synchronous NeMo Gym GRPO;
- the async vLLM HTTP backend;
- signed rollout contexts (linear cursor) or a trusted gateway identity
  (`accept_gateway_identity=true`);
- a GRPO or GDPO advantage estimator; and
- router replay and asynchronous GRPO to be disabled.

The forest cursor additionally requires `mode=direct` (finalization consumes
sealed receipts, which only the direct path produces),
`accept_gateway_identity=true` (staging rows are keyed by the gate-minted
`call_id`), and a single rollout attempt
(`rollout_max_attempts_to_avoid_lp_nan=1`).

The initial writer performs synchronous staging writes. The
`max_pending_writes_per_worker` setting is reserved for a future asynchronous
writer.

## Prerequisites

Run from the NeMo RL repository root in an interactive allocation with eight
visible GPUs.

```bash
git submodule update --init --recursive
nvidia-smi
```

Prepare the Workplace Assistant data if the recipe paths do not exist:

```bash
cd 3rdparty/Gym-workspace/Gym
printf 'hf_token: %s\n' "${HF_TOKEN:?Set HF_TOKEN first}" > env.yaml

uv run ng_prepare_data \
  "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
  +output_dirpath=data/workplace_assistant \
  +mode=train_preparation \
  +should_download=true \
  +data_source=huggingface

cd -
```

Keep `env.yaml` local. Never commit credentials.

For Nemotron Nano v2, use copies whose system prompts end in `/no_think` so
multi-turn reconstruction keeps a stable historical prefix. Set the launch
variables to those prepared files:

```bash
export TRAIN_DATA=results/runtime/workplace_assistant_no_think/train.jsonl
export VAL_DATA=results/runtime/workplace_assistant_no_think/validation.jsonl
test -s "$TRAIN_DATA"
test -s "$VAL_DATA"
```

## Launch a comparison

The helper below fixes the workload and changes only the rollout-writer mode.
Every attempt gets a new result directory.

```bash
export EXPERIMENT_ROOT=results/tq-rollout-writer-comparison
mkdir -p "$EXPERIMENT_ROOT"
set -o pipefail

run_variant() {
  mode=$1
  output_dir=$2
  test ! -e "$output_dir"

  case "$mode" in
    legacy)
      writer_overrides=(data_plane.rollout_writer.enabled=false)
      ;;
    shadow|direct)
      writer_overrides=(
        data_plane.rollout_writer.enabled=true
        "data_plane.rollout_writer.mode=$mode"
      )
      ;;
    forest)
      writer_overrides=(
        data_plane.rollout_writer.enabled=true
        data_plane.rollout_writer.mode=direct
        data_plane.rollout_writer.cursor=forest
        data_plane.rollout_writer.accept_gateway_identity=true
      )
      ;;
    *)
      echo "unsupported mode: $mode" >&2
      return 2
      ;;
  esac

  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    uv run examples/nemo_gym/run_grpo_nemo_gym.py \
      --config examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml \
      ++cluster.num_nodes=1 \
      "data.train.data_path=$TRAIN_DATA" \
      "data.validation.data_path=$VAL_DATA" \
      data.shuffle=false \
      data_plane.enabled=true \
      data_plane.backend=simple \
      data_plane.observability.enabled=true \
      "${writer_overrides[@]}" \
      grpo.seed=42 \
      grpo.max_num_steps=8 \
      grpo.num_prompts_per_step=2 \
      grpo.num_generations_per_prompt=2 \
      grpo.val_period=0 \
      grpo.val_at_start=false \
      grpo.val_at_end=false \
      policy.train_global_batch_size=4 \
      checkpointing.enabled=false \
      env.should_log_nemo_gym_responses=false \
      logger.wandb_enabled=false \
      logger.tensorboard_enabled=false \
      "logger.log_dir=$output_dir" \
      2>&1 | tee "$output_dir.console.log"
}
```

If cached worker environments are stale, set
`NRL_FORCE_REBUILD_VENVS=true` for one setup run, then unset it before a
comparison. Do not mix environment rebuild time into measured repetitions.

Run correctness gates first:

```bash
run_variant legacy "$EXPERIMENT_ROOT/gate-legacy"
run_variant shadow "$EXPERIMENT_ROOT/gate-shadow"
run_variant direct "$EXPERIMENT_ROOT/gate-direct"
```

Continue only when all jobs exit successfully, shadow mode has no tensor
mismatch, direct mode has no rejected manifest rows, an optimizer step
completes with finite loss and KL values, and all GPUs are released after each
job.

For repeated measurements, use a counterbalanced sequential order:

```bash
run_variant legacy "$EXPERIMENT_ROOT/legacy-1"
run_variant direct "$EXPERIMENT_ROOT/direct-1"
run_variant direct "$EXPERIMENT_ROOT/direct-2"
run_variant legacy "$EXPERIMENT_ROOT/legacy-2"
run_variant legacy "$EXPERIMENT_ROOT/legacy-3"
run_variant direct "$EXPERIMENT_ROOT/direct-3"
```

Discard the first step of each repetition as warm-up. Interpret timing or
transport metrics only when configuration matches, direct mode has zero
rejections, and token, turn, and HTTP-exchange distributions are comparable.

## Artifacts and diagnosis

Each run writes:

- `rollout_perf_metrics.jsonl`: per-step HTTP, Gym, TransferQueue, latency,
  CPU, and memory instrumentation;
- `rollout_writer_manifest.jsonl`: per-sample finalization or rejection status
  for shadow and direct modes; and
- the normal NeMo RL console and logger output.

Use the manifest before any performance analysis. Common rejection reasons
identify prefix mismatch, missing staging rows, invalid turn order, identity or
weight-version mismatch, hash mismatch, invalid masks, or non-finite log
probabilities. Do not treat masked rejected rows as equivalent workload.

Forest-mode runs add receipt-reconciliation reasons to the same manifest:

- `missing_receipt` / `receipt_status:*` / `missing_forest_manifest`: the
  rollout never sealed cleanly (including `seal_failed` receipts) or the
  cursor entry is gone;
- `missing_staged_node` / `orphan_staged_node` / `uncollected_call` /
  `no_staged_calls`: the sealed call manifest and the committed cursor nodes
  do not match one-to-one;
- `staging_digest_mismatch` / `delta_length_mismatch` /
  `terminal_hash_mismatch`: a fetched staging row fails integrity
  re-verification; and
- `ambiguous_forest` / `empty_forest`: the verified snapshot did not reduce
  to exactly one unambiguous main chain (branches, multiple roots, or
  quarantined nodes).

A high `ambiguous_forest` rate is expected for harnesses that branch or
compact aggressively; it indicates masked (untrained) rollouts, not
corruption.
