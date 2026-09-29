# Context compaction with framework-owned training rows

This opt-in SingleController GRPO path lets a harness rewrite its semantic history while NeMo-RL owns the resulting training rows. It builds on the ordinary worker token/media capture path. It does not patch upstream vLLM.

## Ownership and request flow

The Gym integration accepts ordinary rollout output through shared capture. A harness retains its semantic message/image history, applies its own compaction policy, and sends its chosen context through the supported sequential, non-streaming Responses or Chat capture route. It does not supply tokens, segment IDs, TQ parent decisions, or a processed-image arena. The included `simple_agent_with_compaction` uses a recency policy; another harness can choose a different policy.

1. Gym's existing capture middleware assigns the attempt/call identity. The client identifies its last accepted response using `x-nemo-gym-capture-parent` (a JSON response ID, or `null` before any response is accepted). The ledger forwards that call as a **candidate**, with a versioned digest and item count covering the ordered received and exposed source history. Requests without a hint retain latest-call inference. The current request supplies per-item comparison facts transiently. It records a call intent before inference; a separate check of the observed ledger head rejects concurrent or stale admissions under the same ledger lock.
2. RL's `decide_capture_input` compares those source facts and rendering options before selecting a serving prefix. A known source rewrite renders fresh. An unchanged source prefix with appended user/tool observations can restore the candidate's exact captured tokens. Missing evidence or an unsupported converted-message boundary raises an error; a fingerprint miss is not a rewrite rule. Exposed reasoning is part of the source comparison, even if conversion later removes it.
3. The existing RL vLLM renderer processes all supplied media, resolves any serving prefix from TQ, applies the ordinary token splice, and checks/remaps every retained media occurrence. It captures the actual tokens, generated logprobs, routes, and processed pixels.
4. **A continuation stores a parent-linked delta.** RL commits the verified predecessor using the existing `token_in` record shape. The initial request and each actual context rewrite use a parentless `text` root. `candidate` is admission-only; unsupported preservation fails explicitly.
5. Gym accepts the worker-selected root or admitted predecessor, validates the returned coordinates, and records exposed output identities/content fingerprints and completion metadata centrally. The harness returns its ordinary rollout output and verifier reward; it defines no segments and need not return `LogicalCCResult`.
6. RL's shared Gym ingestion identifies accepted whole generations from that ordinary output, then combines the selection with the complete attempt receipt and verifies the selected chains. It publishes one training trace per chain. Twenty calls without compaction produce one trace; one compaction after turn 10 produces two. Twenty staged call records remain small deltas within those chains.
7. Each trace trains every selected generated span in its chain exactly once. Added observations, summaries and quoted answers have zero loss. Existing logical-owner GRPO supplies one reward/advantage per rollout, independent of segment count, and excludes execution padding.

CC finalization separately authenticates each call's occurrence metadata against its existing extras commitment. It checks media-presence/frame flags, ordered spans entirely inside the new prompt tokens, exact token IDs, per-occurrence H/W, and video frame counts before publication. One bad selected segment invalidates the whole logical rollout. The worker authenticates retained descriptors before using their offsets as well. This reuses the capture descriptors and shared integrity helpers; it does not add a new media store or authenticate pixel contents cryptographically.

The common Gym path accepts `response.output`, ordered `responses`, transition snapshots, or existing `ng_trajectory.turns[].model_calls` references. Native response/item identities constrain whole-generation content matching; synthetic exporter IDs are not treated as native capture identities. Reasoning and function calls are included. Existing trajectory references can select calls directly when no output is supplied; when both are supplied, they must agree. Missing, contradictory, duplicated, reordered, or ambiguous evidence raises an error. RL does not fall back to selecting a graph leaf for this path.

Compaction of the next model input must not erase accepted history from the returned training output. Missing calls inside a selected continuation chain are detected because the predecessor must also be selected. A completely omitted independent chain cannot always be distinguished from a rejected call using capture alone. Capture records what ran, not what the harness accepted. Thus this implementation does not promise compatibility with every exporter: it requires complete accepted history or references and a supported capture route. Stock OpenCode is not qualified by tests of a custom fork. Branching, concurrency, and multiple sessions need separate qualification.

`RolloutSelection` is internal to RL's reconstruction; it is not a new object required from Gym harnesses. The current capture protocol types remain in the shared Gym staging package; moving that package is not part of this ownership change.

The small non-Gym adapter test translates an independent episode's action IDs and score into `RolloutSelection` plus an attempt receipt, then exercises actual finalization and cleanup. It supplies no segment IDs or Gym `LogicalCCResult`. This proves the local adapter contract; it does not qualify another harness's HTTP transport or capture gateway.

## Images and TQ

Suppose call 1 sees image A, call 2 sees retained A plus new B, and a policy rewrite makes call 3 see only B. The harness sends A on call 2 because A remains in its semantic history; it does not ask vLLM to fetch raw A from TQ. Normal vLLM processing produces that call's pixels. TQ supplies historical exact tokens and media descriptors when a serving splice is requested.

The staged records contain A for call 1, only new B for continuing call 2, and B in full for rewritten call 3. Finalization publishes two traces: calls 1–2 with A+B, and call 3 with B. Training uses captured processor outputs through the foundation `PackedTensor` path. No CC-specific image preprocessing or `MediaArena` is needed.

Retained geometry includes patch dimensions, token count, placeholder coordinates and embedding spans. RL also compares the current retained processor tensors to existing TQ media bundles before omitting them from a delta. Geometry, token or pixel changes on a continuation fail explicitly before inference; they do not silently create additional segments. A genuine rewrite captures its current geometry and pixels in the new root.

Within a context, each continuation stores only its new token/media delta. A rewrite starts a new root and captures its complete current input, so content retained across a rewrite can appear in both segments. Repeated full per-call storage is not the production baseline.

## Measurement and lifecycle

Each completed call stores one `ReplaySummary`: normalization version, source-item count, ordered source digest, and render-option digest. RL hashes the corresponding prefix of the next request and compares that summary before reusing captured tokens; suffix-role and converted-assistant-boundary checks still apply. Reasoning, image references, item order, and qualified tool-argument normalization retain their existing semantics.

Candidate admission reconstructs staging ancestry from existing call parent links. The framework-owned route does not persist the full ancestor-key list on every call. Durable metadata therefore grows approximately linearly with call count. Full current-request item comparisons, manifest traversal, transient ancestor lists, and historical-media validation reads still have history-dependent cost; this change does not make total rollout processing linear. Ordinary capture keeps its existing lookup-chain representation.

`/context/{attempt}/measure` uses the same Responses conversion, candidate admission, RL preprocessing and splice decision as generation. It returns a count and an explicit feature acknowledgement, never tokens. It creates no call intent, staged record, or capture acknowledgement. Model/template/processor configuration must remain fixed during an attempt.

The existing ledger tracks exact attempted call IDs. A missing worker acknowledgement remains pending, even if the HTTP request has ended, because the worker may still have written or be writing. This initial path stops rather than replaying that generation or treating an uncertain write as cleanup permission. Evidence remains until reconciliation/run teardown; automatic ambiguous-attempt recovery is not implemented.

Completed attempt receipts retain unselected records for cleanup. Before publication, the fixed selection determines exact canonical row and padding IDs. A lost canonical-write acknowledgement preserves source staging; the controller owns the exact publication cleanup keys. Foreign scope and invalid selection fail before publication.

`select_response` may reject a definite completed response and resample within `max_response_retries` and `max_model_calls`. The client retries the same semantic request with the same last-accepted-response hint; rejected output never enters its history or selected-action result. This hint also remains the last accepted response across compaction: RL alone decides whether the rewritten request needs a new root. Measurement uses the same hint without writing an intent. Unknown, foreign, or ambiguous response IDs fail before generation. Transport/read ambiguity closes the client and is never retried automatically.

## Configuration and supported scope

Enable `token_capture.enabled: true` and `token_capture.context_compaction: true` on the SingleController GRPO configuration. For the included example, select Gym's `simple_agent_with_compaction` and configure its `context_history` policy/guards. Other harnesses use their existing ordinary output through the same ingestion path, subject to the evidence and capture constraints above. RL sets Gym's internal `framework_owned_context: true` with external `vllm_worker` staging and `rebuild_response: false`.

The first implementation supports sequential full-history Responses calls, bounded resampling of definite responses, ordinary text/images supported by the foundation processor path, token-level GRPO, and direct route assembly. It retains existing CC guards on checkpoint/resume, rollout snapshots, deferred route assembly, incompatible objectives, message-level advantage overrides, partial step batches and non-unit dataset loss multipliers. Streaming, provider-managed conversation state, assistant prefill, engine prompt truncation and concurrent model calls are rejected. Ordinary capture keeps its existing path with the option disabled.

Both repositories must use the paired capture schema v3 revision, including the `ReplaySummary` representation on this route. Earlier experimental v3 ledgers that persisted `ReplayContext.items` are rejected, as are prior v2 staged checkpoints; there is no automatic migration or CC resume. Source normalization and token/media digest versions are unchanged by the summary representation.

## Comparison with the earlier CC implementation

| Concern | Earlier CC PRs | This baseline |
|---|---|---|
| Segment boundaries/IDs | Gym client and per-segment capture scopes | RL worker selects boundaries during rollout; reassembler assigns final row IDs |
| Raw image retention | Semantic references plus `MediaArena` export | Original semantic content retained directly |
| Learner image preprocessing | CC media export and RL preprocessing/alignment path | Foundation capture of actual vLLM processor tensors |
| Serving prefix | Harness-selected parent within its segment | RL comparison of source evidence and supported conversion boundary |
| Storage | Parent-relative capture inside harness segments | Parent-relative deltas inside RL-selected segments |
| Training semantics | Logical-owner advantages, selected-token masks, direct routes | Reused logical-owner machinery and foundation media publication |
| Failure custody | Segment receipts; logical result could lose attempt receipt | One real attempt receipt, durable call intents, exact publication plan |

## Validation

The restored combined tests exercise the real semantic client/policy through HTTP custody, RL row planning, replay selection, advantages and a mocked learner for 20 turns (including one compaction after turn 10). They also cover failed owners, padding, action-specific masks, nontrivial route-tail backpatching, corruption rejection, actor replay suppression and unsupported CC restore. The CPU suites additionally exercise HTTP conversion/custody, sequential candidate admission, read-only measurement, original-image retention and recency, geometry rejection, current-pixel capture, multimodal chain publication, selected-token masks, logical advantages, execution padding and acknowledgement-loss retention. An order-sensitive recurrent-model oracle compares loss and nonzero gradients against independently recorded per-call inputs. Corruption controls change conditioning, duplicate action ownership, alter owner weighting, and use the wrong denominator.

These tests use synthetic generation and in-memory storage. They do **not** qualify a real vLLM image processor, native TransferQueue/Mooncake, Ray cancellation, distributed Megatron, or a complete optimizer step. Before release, run paired real-model rollouts and an optimizer smoke; compare actual tokens, per-occurrence geometry/pixels, behavior logprobs, action ownership, loss/gradients and cleanup. Compare against both ordinary no-compaction capture and the earlier CC implementation using identical recorded evidence. Independent fresh samples are not a strict numerical oracle.
