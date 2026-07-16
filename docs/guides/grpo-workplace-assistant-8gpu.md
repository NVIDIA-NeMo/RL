# Direct TransferQueue Rollouts with NeMo Gym

This guide explains how NeMo RL trains on NeMo Gym rollouts written directly
from vLLM workers to TransferQueue (TQ). One mechanism — gateway-minted call
identity, a forest cursor, and receipt-reconciled finalization — serves both
Gym's native agent loop and black-box harnesses (unmodified CLIs such as
Claude Code or Codex that construct their own requests, speak text rather
than token IDs, and may branch, spawn sub-agents, or compact their context).

The guide covers how the system works end to end, how to run controlled
legacy-versus-direct comparisons with the eight-GPU Workplace Assistant
recipe, and the measured results of that comparison on the synchronous GRPO
path.

Design Overview: https://terryk.gitlab-master-pages.nvidia.com/nemo-html/pthombre/tq-blackbox-rollout-lifecycle.html

## What changes

The legacy synchronous TransferQueue path returns complete token-bearing
trajectories through Gym. The rollout actor flattens them and publishes the
canonical training rows:

```text
vLLM -> Gym -> rollout actor -> TransferQueue train partition
```

With the direct writer enabled (`data_plane.rollout_writer.enabled=true`,
`mode=direct`, `cursor=forest`, `accept_gateway_identity=true`), tokens stop
riding HTTP responses back through Gym. Every model call is admitted by
Gym's ingress gate, which stamps it with a per-call identity; the vLLM
model-owner worker writes that call's token delta straight to a TQ staging
partition; and after the rollout finishes, a trusted finalizer reconciles
the gate's sealed receipt against the cursor registry, re-verifies every
staged tensor, and publishes one canonical training row:

```text
                 +----> rollout_staging ----+
harness/agent    |                          |
  -> ingress ----+-> vLLM model owner       +-> finalizer -> train
     gate        |                          |
                 +----> forest registry ----+
```

The system is fail-closed at every boundary: an unidentified call is served
without being collected, a broken or ambiguous rollout becomes a masked
placeholder row that is excluded from reward baselines and advantage
normalization, and nothing a harness does can fail the batch or silently
train on corrupt tokens.

The main implementation is in:

- `nemo_rl/experience/rollout_writer.py`: rollout identity minting, the
  forest cursor state machine, staging deltas and digests, and canonical
  batch assembly.
- `nemo_rl/models/generation/vllm/vllm_worker_async.py`: gateway-identity
  validation, prefix selection, and per-call TransferQueue writes from
  model-owner workers.
- `nemo_rl/experience/blackbox_finalizer.py`: receipt reconciliation,
  integrity re-verification, and canonical row construction.
- `nemo_rl/experience/staged_token_source.py`: adapts verified staging rows
  to Gym's `TokenSource` protocol for trajectory assembly.
- `nemo_rl/experience/sync_rollout_actor.py`: identity attachment,
  finalization, and canonical publish.
- `nemo_rl/environments/nemo_gym.py`: ingress-gate installation and rollout
  register/seal orchestration.
- `nemo_rl/algorithms/grpo_sync.py`: setup guards, writer lifecycle, and
  invalid trajectory masking.
- `3rdparty/Gym-workspace/Gym/nemo_gym/`: the model-server ingress gate,
  rollout receipts, the `ServerClient` rollout-id header lift, and the pure
  trajectory builder the finalizer delegates chain assembly to.

## Architecture and code flow

### Component ownership

The implementation deliberately separates untrusted request transport from
the trusted code that decides what reaches training.

| Component | Process | Responsibility |
| --- | --- | --- |
| `grpo_train_sync` | training driver | Validate the supported configuration and create the forest registry actor |
| `TQPolicy` | training driver | Bootstrap TransferQueue and register the staging and canonical partitions |
| `SyncRolloutActor` | Ray rollout actor | Mint rollout identities, invoke Gym, finalize sealed rollouts, and publish canonical rows |
| NeMo Gym env | Gym Ray actor and HTTP services | Install the ingress gate, register rollouts before dispatch, run the agent/harness, and seal receipts |
| Ingress gate | model-server HTTP middleware | Admit calls for registered rollouts, mint `call_id` and admission order, strip tokens from harness-facing responses |
| `VllmAsyncGenerationWorkerImpl` | vLLM model-owner Ray worker | Verify the committed prefix by hash and write one staging delta per admitted call |
| `RolloutForestRegistry` | dedicated Ray actor | Serialize node reservations and commits for each rollout |
| `assemble_blackbox_batch` | rollout actor | Reconcile receipts, verify staging rows, and assemble trainer-facing tensors |
| GRPO/GDPO advantage estimator | training driver | Exclude rejected placeholders from baselines and normalization |

Gym services can route successive calls to different HTTP workers, and a
harness can issue calls concurrently, so cursor state is never stored in a
vLLM worker. The dedicated registry is the single authority for each
rollout's committed nodes, their cumulative lengths and hashes, active
leases, and terminal failure state.

### Setup lifecycle

The writer is attached once when `grpo_train_sync` starts:

1. `examples/nemo_gym/run_grpo_nemo_gym.py::_select_trainer` selects
   `grpo_train_sync` when `data_plane.enabled=true`.
2. `TQPolicy` bootstraps the configured TransferQueue backend and registers
   the normal `train` partition for canonical rows.
3. `grpo_train_sync` validates that this is synchronous NeMo Gym GRPO using
   async vLLM, a GRPO or GDPO advantage estimator, no router replay, and a
   consistent writer config (`cursor=forest` requires
   `accept_gateway_identity=true` and `mode=direct`).
4. `TQPolicy.prepare_rollout_writer` registers `rollout_staging` with the
   per-call tensor fields.
5. The driver creates a `RolloutForestRegistry` Ray actor.
6. `VllmGeneration.configure_rollout_writer` sends the validated data-plane
   config and registry handle only to vLLM model-owner workers.
7. The driver derives `blackbox_rollouts=true` for the NeMo Gym env, which
   injects the ingress gate into the policy model server's config: a
   capture directory, a minted control token, `require_registration=true`,
   and `return_token_ids=false`. The control token guards the
   register/seal API and never reaches agents or task rows.
8. The rollout actor runs in a venv that bundles both the vLLM and NeMo-Gym
   extras (`VLLM_NEMO_GYM`) because finalization calls Gym's trajectory
   builder.

Batch-level rollout retries are forbidden
(`rollout_max_attempts_to_avoid_lp_nan=1`) because registration is
create-only and sealing is terminal: a re-run of the same `rollout_id`
would be rejected at admission.

## Identity model

| Identity | Minted by | Meaning |
| --- | --- | --- |
| `rollout_id` | driver (`mint_rollout_id`) | One GRPO generation: its Gym execution, its staged rows, its receipt, and its canonical train row. Opaque, URL-safe, 128 random bits as lowercase hex. Where a legacy field still requires a `sample_id`, it equals `rollout_id`. |
| `call_id` | ingress gate | One admitted model call. The staging key is `<rollout_id>/<call_id>`. Two admitted calls with identical prompts are two nodes — prompt hashes are integrity metadata, never identity. |
| admission order | ingress gate | Per-rollout monotonic order in which calls were admitted. HTTP completion order and TransferQueue write order carry no meaning. |
| `group_id` | driver | GRPO sibling grouping only. It stays in the driver's private `rollout_id -> group_id` mapping and never travels in requests, staged rows, keys, or receipts. |

`_attach_rollout_contexts` stamps the bare `nemo_rl_rollout_id` into each
task row's `metadata.extra_body`; per-call identity is minted later, at
admission. Because requests carry no per-request weight version, the driver
declares the step's policy weight version to every model-owner worker up
front (`VllmGeneration.set_rollout_weight_version`), and the finalizer
verifies it on every committed node.

## Trust boundary

The vLLM serving endpoint is assumed to be reachable only through Gym's
ingress gate (same host; sandbox networking does not route to the vLLM
port). Under that assumption, with `accept_gateway_identity=true`, the
worker trusts the `nemo_rl_rollout_id`/`nemo_rl_call_id` pair the gate
stamped on the internal request. The gate in turn:

- admits model calls only for registered, unsealed rollouts
  (`require_registration=true`);
- overwrites any harness-supplied identity fields;
- correlates a call to its rollout either from a `/ng-rollout/<rollout_id>`
  URL prefix (sandboxed CLIs) or from the `x-nemo-gym-rollout-id` header,
  which Gym's `ServerClient` lifts out of the trainer-stamped
  `metadata.extra_body` on every in-tree agent call; and
- strips token IDs and log probabilities from sandboxed-harness-facing
  responses — for those clients the staged TransferQueue row is the only
  token store.

The register/seal control API (`/ng-control/rollouts/...`) is guarded by a
random bearer token that `NemoGym` mints at startup; it never reaches agents
or task rows. Correctness still does not rest on the gate alone: the trusted
finalizer independently re-verifies every staged tensor (see
[Finalization](#finalization-one-assembler-two-verifiers)).

## End-to-end lifecycle

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
  |                  |                       |<-- response      |                    |
  |                  | seal rollout -------->|                  |                    |
  |                  | (POST .../seal)       |                  |                    |
  |<-- receipt ------|                       |                  |                    |
  | finalize: reconcile receipt vs forest manifest, verify rows, build chain        |
  | publish canonical row, clear staging keys and cursor entries                    |
```

1. **Registration.** Before dispatching a batch, the env reads each row's
   `nemo_rl_rollout_id` and registers it with the gate. Unregistered calls
   are rejected at admission.
2. **Admission.** The gate correlates each model call to its rollout, mints
   `call_id` and the admission index, and forwards both to vLLM inside the
   internal request.
3. **Per-call write.** The worker stages the call's token delta (see the
   next section) and commits the node to the registry.
4. **Response.** Sampling log probabilities are never returned over HTTP —
   their authoritative copy is already staged. Token IDs are returned only
   to native in-tree agents (which rebuild message logs from them);
   sandboxed harnesses receive plain text.
5. **Sealing.** After the harness (and its verifier) finishes, the env seals
   the rollout. The gate returns a token-free receipt: the call manifest
   (each admitted `call_id`, its admission index, and whether it staged)
   and a terminal status. The env attaches the verifier reward to the
   receipt as driver-carry. A seal failure yields a `seal_failed` receipt,
   which the finalizer turns into a masked placeholder. Receipts ride the
   batch as `ng_rollout_receipts` and are popped before anything is written
   to TransferQueue.
6. **Finalization and publish.** The rollout actor reconciles each receipt
   against the registry's manifest, verifies the staged rows, assembles the
   canonical row, publishes it under `rollout_id` to the `train` partition,
   and clears the staging keys and cursor entries. Training workers consume
   the canonical row through the existing `TQPolicy` read/logprob/train
   pipeline; they never see calls or staging.

## Per-call write path

Each admitted model call follows this sequence on the model-owner worker:

1. vLLM finishes chat-template rendering and obtains the exact
   `prompt_token_ids` that will be sent to the model.
2. The worker fetches the rollout's committed candidates from the registry —
   cumulative sequences, longest first — hashes its rendered prompt at each
   candidate length, and reserves against the longest match as the storage
   parent. A prompt that extends no committed prefix reserves a `new_root`
   full-prompt row (context compaction), not a failure; a second child of
   one prefix is a branch.
3. vLLM generates one response. The worker extracts generated token IDs and
   sampling log probabilities from the non-streaming response.
4. `build_staging_delta` removes the already committed parent prefix,
   emitting only newly introduced prompt tokens plus generated tokens, and
   `compute_staging_digest` hashes the delta tensors together with the
   identifying metadata.
5. The worker synchronously writes the delta to `<rollout_id>/<call_id>` in
   `rollout_staging`, then — only after the write succeeds — commits the
   node with its new cumulative length, full-sequence hash, and digest.

The staging schema is intentionally small:

| Field | Dtype | Meaning |
| --- | --- | --- |
| `token_ids_delta` | `int64` | New environment/prompt tokens followed by generated tokens |
| `token_mask_delta` | `float32` | `0` for prompt/environment tokens and `1` for trainable generated tokens |
| `generation_logprobs_delta` | `float32` | `0` for prompt/environment tokens and the sampling log probability for generated tokens |

TransferQueue tags also record the rollout, call, parent, weight version,
previous/new lengths and hashes, digest, and lease state. Correctness does
not depend on trusting those tags: the finalizer uses the registry manifest
and recomputes tensor-derived hashes and digests from the fetched rows.

## Forest cursor state machine

`ForestCursorStateMachine` is single-threaded inside the registry actor.
Each rollout owns a set of nodes keyed by `call_id`:

- Node states are `reserved`, `committed`, and `failed`, each node with its
  own lease. Concurrent reservations are legal.
- Retrying an in-flight `call_id` idempotently returns the same reservation
  (a new lease after expiry); retrying a committed `call_id` is a duplicate
  and fails the node.
- A reservation's claimed parent must be a committed node whose length and
  hash match exactly. A rootless reservation must start at length zero; if
  committed nodes already exist it is flagged `is_new_root`.
- Committing requires the current lease and a strictly growing cumulative
  length, and records the staging key, prompt/generation lengths, terminal
  hash, staging digest, and weight version.
- A failed call fails only its node — the rollout survives, and the
  finalizer decides validity from reconciliation.

This policy prefers masking one trajectory over accepting ambiguous tokens,
and it never fails a rollout for structure the harness is entitled to
produce (branches, compaction, concurrency).

## Finalization: one assembler, two verifiers

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

Verified rows are padded to the legacy batch width and form the canonical
training tensors: `input_ids`, `input_lengths`, `generation_logprobs`,
`token_mask`, and `sample_mask`. Reward and other driver-carry fields come
from the trusted Gym result.

Any failure at any step returns an invalid row whose reason feeds
`rollout_writer_manifest.jsonl`; the batch assembler substitutes a one-token
placeholder with `sample_mask=0` and `trajectory_valid_mask=0` and zeroes
every reward channel for that row. `grpo_train_sync` passes the validity
mask into GRPO or GDPO baseline calculation and advantage estimation, then
applies it again before clipping and writeback as a fail-safe. Sibling
generations in the same group train normally.

## The native agent flow on the same path

Nothing above requires the client to be a black-box CLI. Gym's native agent
loop (the Workplace Assistant's tool-calling agent, for example) runs on
exactly the same machinery and is simply the degenerate case:

- The agent forwards `responses_create_params` verbatim, so
  `ServerClient` finds the trainer-stamped `nemo_rl_rollout_id` in
  `metadata.extra_body` and lifts it into the `x-nemo-gym-rollout-id`
  header. The gate admits the call without the agent knowing the gate
  exists.
- The native loop is sequential and append-only, so every call's rendered
  prompt extends the previous committed sequence: the forest degenerates to
  a single chain, and finalization selects it as the one `main` chain.
- Prefix continuity is verified by hashing the rendered prompt against the
  committed cumulative hash — no token echo or re-tokenization round trip
  is needed on the request path.
- Native agents still receive generated token IDs in responses (they
  rebuild message logs from them); sampling log probabilities are never
  returned, because the staged copy is authoritative.

## Worked examples

### Sequential rollout (the native case)

One rollout, two admitted calls. Token values are illustrative, but the
shapes and masks match the implementation. The gate mints `c1` and `c2`;
the rollout id is abbreviated `rid`.

Call `c1` renders prompt `[101, 102, 103]` and generates `[201, 202]` with
log probabilities `[-0.10, -0.20]`. No candidates are committed yet, so the
reservation is rootless (`prev_len=0`) and the worker stages the full
sequence:

```text
key                         = rid/c1
token_ids_delta             = [101, 102, 103, 201, 202]
token_mask_delta            = [  0,   0,   0,   1,   1]
generation_logprobs_delta   = [0.0, 0.0, 0.0, -0.10, -0.20]
commit: new_len=5, new_hash=hash([101, 102, 103, 201, 202])
```

Gym executes the requested tool and the next rendered prompt appends the
tool result: `[101, 102, 103, 201, 202, 301, 302]`, generating
`[401, 402]`. The worker finds `c1`'s committed sequence as the longest
matching candidate (hash of the first five prompt tokens matches), reserves
with `parent=c1, prev_len=5`, and stages only the delta:

```text
key                         = rid/c2
token_ids_delta             = [301, 302, 401, 402]
token_mask_delta            = [  0,   0,   1,   1]
generation_logprobs_delta   = [0.0, 0.0, -0.30, -0.40]
commit: new_len=9, new_hash=hash(full 9-token sequence)
```

After sealing, the receipt lists `c1` and `c2` as staged; both reconcile
against committed nodes; the builder produces one chain; and the canonical
row is:

```text
input_ids = [101, 102, 103, 201, 202, 301, 302, 401, 402]
token_mask = [0, 0, 0, 1, 1, 0, 0, 1, 1]
generation_logprobs = [0.0, 0.0, 0.0, -0.10, -0.20,
                       0.0, 0.0, -0.30, -0.40]
input_lengths = 9
sample_mask = 1
trajectory_valid_mask = 1
```

`kv_first_write` publishes that row under canonical key `rid` in the
`train` partition, and the rollout actor deletes `rid/c1`, `rid/c2`, and
the cursor entry.

### Branch and compaction (the black-box case)

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
rejected as `ambiguous_forest` and trains as a masked placeholder while its
sibling generations train normally. Had the harness been purely sequential
(`c1 -> c2` only), the single main chain `[1, 2, 3, 4, 5, 6, 7]` would
train with mask `[0, 0, 0, 1, 1, 0, 1]`, exactly like the sequential
example above.

## Measured performance: legacy vs direct (sync GRPO)

The numbers below are from a counterbalanced eight-GPU campaign
(method: `docs/design-docs/tq-blackbox-perf-plan.md`; artifacts:
`results/tq-blackbox-e2e-perf-20260714/`, including the runner, analyzer,
per-job consoles, and a byte-regenerable `summary.json`).

| | |
| --- | --- |
| Hardware | 1 node, 8× H100, colocated Megatron TP2 policy + 8× vLLM TP1 |
| Model | nvidia/NVIDIA-Nemotron-Nano-9B-v2 |
| Workload | Workplace Assistant (native Gym `simple_agent`, tool calls), seed 42, shuffle off |
| Shape | 2 prompts × 2 generations × 8 steps/job; step 1 discarded; 21 retained steps/arm |
| Arms | legacy = `rollout_writer.enabled=false`; direct = `mode=direct cursor=forest accept_gateway_identity=true` |
| Order | legacy-1, direct-1, direct-2, legacy-2, legacy-3, direct-3 (counterbalanced) |

### Signal quality (gates the performance numbers)

Across all 96 direct-path rollouts (3 jobs × 32): **95 finalized, 1
rejected** (`ambiguous_forest`, 1.04% — under the campaign's 5% abort
threshold; the rejected rollout became a masked placeholder, fail-closed).
Chain-length distribution over finalized rollouts: 60× 1-call, 21× 2-call,
13× 3-call, 1× 4-call (mean 1.53, max 4) — multi-call chains survive
assembly with no silent forest collapse. A same-day smoke measured
generation-KL error 0.0002, confirming staged log probabilities align with
policy recomputation.

### Transport

| Metric (21 steps/arm) | Legacy | Direct (forest) | Change |
| --- | ---: | ---: | ---: |
| HTTP bytes / generated token | 1456.9 | 773.4 | **−46.9%** |
| HTTP total | 11.93 MB | 5.76 MB | **−51.7%** |
| HTTP exchanges | 280 | 124 | **−55.7%** |
| — `/tokenize` traffic | 7.01 MB | **0** | eliminated |
| — response logprob payload | 0.82 MB | **0** | eliminated |
| — response token-id echo | 0.01 MB | 2.65 MB | native echo retained |
| Terminal Gym→RL bytes / sample | 87,752 | 47,966 | **−45.3%** |

The two big wins are structural: prefix continuity is verified against the
committed cumulative hash, so the legacy exact-token echo — one full
`/tokenize` round trip per turn plus its payload — is gone entirely (that
alone is half the exchange count), and sampling log probabilities ride TQ
staging rather than the HTTP response. Token-id echo remains for the native
loop because it rebuilds message logs from it; a sandboxed harness (which
speaks text) drops that 2.65 MB too.

### TransferQueue cost

| Metric | Legacy | Direct (forest) |
| --- | ---: | ---: |
| Staging put / read | 0 / 0 | 5.42 MB / 5.42 MB |
| Canonical put | 6.97 MB | 5.42 MB |
| Write amplification vs legacy canonical | 1× | 1.56× |
| Movement amplification vs legacy canonical | 1× | 2.33× |
| Staging bytes / canonical byte | — | **1.00×** |

Because the worker stages **deltas against the longest committed prefix**
(full snapshots only on compaction roots), staging is exactly 1.00× of
canonical bytes on this workload — storage does not grow quadratically with
chain length. Per-call worker overhead (p50): reserve 2.1 ms, commit
1.7 ms, staging put 8.0 ms, put-blocked fraction 1.2%.

### Latency, throughput, and process cost

| Metric (p50, 21 steps/arm) | Legacy | Direct (forest) | Change |
| --- | ---: | ---: | ---: |
| Total step time | 16.91 s | 16.19 s | **−4.3%** |
| Rollout wall | 2.50 s | 2.01 s | −19.6% |
| Rollout start → canonical ready | 2.51 s | 2.11 s | **−15.8%** |
| Rollout return → canonical ready | 0.01 s | 0.12 s | +0.11 s (finalizer) |
| Generated tokens/s | 167.9 | 160.4 | −4.5%* |
| Rollout-actor / vLLM peak RSS | 1363 / 1310 MiB | 1366 / 1344 MiB | ~flat |

\* Sampled trajectories differ across arms (7,803 vs 8,584 generated
tokens, 124 vs 140 turns), so tokens/s ratios carry workload noise; the
per-token byte ratios and step time are the robust comparisons. Finalizer
cost is small and off the generation path: manifest fetch 3.4 ms, full
finalize (fetch + verify + Gym chain build) 96.7 ms p50 / 113.2 ms p95 per
step, assembly 0.2 ms.

### What the numbers mean

1. **Less data on the wire, structurally.** −47% HTTP bytes per generated
   token and −56% exchanges against legacy, because token integrity is
   enforced by hashes and TQ digests instead of echoing tokens over HTTP;
   −45% on the terminal Gym→RL hop because training tensors travel through
   TQ, not Ray returns.
2. **No step-time or memory tax.** Step p50 is slightly faster; RSS is
   flat; all verification (digest, hash chain, manifest reconciliation, Gym
   chain build) costs ~0.1 s per step off the generation path.
3. **Verified, not trusted, training data.** Every canonical row is
   reconciled exactly against a sealed call manifest and re-verified before
   training; the smoke's 2e-4 generation-KL error is the end-to-end proof.
4. **Fail-closed by construction.** The one broken rollout in 96 became a
   masked placeholder with an exact reason (`ambiguous_forest`).
5. **The capability unlock.** Identity is minted at the gateway and
   reconstruction is post-hoc, so an unmodified CLI that branches, spawns
   sub-agents, or compacts context trains with the same machinery — the
   native loop measured here is the degenerate single-chain case. The
   legacy path cannot drive such a client at all.

Costs to weigh: 2.33× TQ movement versus legacy's single canonical write
(~16 MB per arm over 21 steps here), +0.11 s of post-rollout finalizer per
step, roughly one masked rollout per hundred (monitor rejection reasons at
scale), the native loop's remaining token-id echo, and no batch-level
rollout retries.

## Supported configuration

| Mode | Overrides | Purpose |
| --- | --- | --- |
| Legacy TQ | `rollout_writer.enabled=false` | Control path; rollout actor publishes rows |
| Direct | `enabled=true`, `mode=direct`, `cursor=forest`, `accept_gateway_identity=true` | The supported write path for native agents and black-box harnesses |

The writer is opt-in and currently requires:

- synchronous NeMo Gym GRPO;
- the async vLLM HTTP backend;
- a trusted gateway identity (`accept_gateway_identity=true`, meaning the
  vLLM endpoint is reachable only through the gate);
- a GRPO or GDPO advantage estimator;
- router replay and asynchronous GRPO to be disabled; and
- a single rollout attempt (`rollout_max_attempts_to_avoid_lp_nan=1`).

The initial writer performs synchronous staging writes. The
`max_pending_writes_per_worker` setting is reserved for a future
asynchronous writer.

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

The helper below fixes the workload and changes only the rollout-writer
mode. Every attempt gets a new result directory.

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
    direct)
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
run_variant direct "$EXPERIMENT_ROOT/gate-direct"
```

Continue only when both jobs exit successfully, the direct gate's manifest
has no rejected rows (or only a rate well under 5%, each with an explained
reason), an optimizer step completes with finite loss and KL values, and
all GPUs are released after each job.

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
transport metrics only when configuration matches, the direct arm's
rejection rate is negligible, and token, turn, and HTTP-exchange
distributions are comparable.

## Artifacts and diagnosis

Each run writes:

- `rollout_perf_metrics.jsonl`: per-step HTTP, Gym, TransferQueue, latency,
  CPU, and memory instrumentation;
- `rollout_writer_manifest.jsonl`: per-rollout finalization or rejection
  status for the direct path; and
- the normal NeMo RL console and logger output.

Use the manifest before any performance analysis. Rejection reasons group
as follows:

- `missing_receipt` / `receipt_status:*` / `missing_forest_manifest`: the
  rollout never sealed cleanly (including `seal_failed` receipts) or the
  cursor entry is gone;
- `missing_staged_node` / `orphan_staged_node` / `uncollected_call` /
  `no_staged_calls`: the sealed call manifest and the committed cursor
  nodes do not match one-to-one;
- `missing_staging_row` / `invalid_delta_shape` / `invalid_token_mask` /
  `non_finite_generation_logprob` / `delta_length_mismatch` /
  `staging_digest_mismatch` / `terminal_hash_mismatch` /
  `weight_version_mismatch`: a fetched staging row fails integrity
  re-verification; and
- `ambiguous_forest` / `empty_forest`: the verified snapshot did not reduce
  to exactly one unambiguous main chain (branches, multiple roots, or
  quarantined nodes).

A nonzero `ambiguous_forest` rate is expected for harnesses that branch or
compact aggressively; it indicates masked (untrained) rollouts, not
corruption. Do not treat masked rejected rows as equivalent workload in
performance comparisons.
