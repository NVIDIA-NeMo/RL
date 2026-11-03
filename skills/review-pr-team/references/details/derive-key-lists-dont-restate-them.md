# derive-key-lists-dont-restate-them

> A literal list of names that mirrors a declaration elsewhere is a review finding — work out whether it can be DERIVED (fields/enum members), else a test, else a linked restatement; never just say "these could drift"

Whenever a PR adds a literal collection of names — a `frozenset`/list/tuple/dict of field
names, config keys, or enum values — ask **what is the source of truth, and can the list be
read off it instead of retyped?** Two lists saying the same thing must be updated together
forever, and that pairing is the defect. Do not stop at "these could drift": the review's job
is to find the derivation, verify it produces the identical set, and hand it over.

Work down these tiers and name which one applies:

1. **Derivable — mirrors a class's fields.** `frozenset(f.name for f in fields(X))` for a
   dataclass, `X.__annotations__` / `get_type_hints(X)` for a TypedDict, `X.model_fields` for
   a pydantic BaseModel, `{e.value for e in X}` for an Enum. Always verify by running it and
   comparing to the literal. Check the ordering constraint too: the derived assignment must
   sit *below* the class it reads, so say so — an author who pastes it in place gets a
   `NameError` at import. Note also what the derivation quietly excludes (`dataclasses.fields`
   skips `@property`, which is usually what you want).
2. **Not derivable, but checkable — mirrors behaviour spread across functions/files.** No
   single declaration exists to read (e.g. "the keys this other module pops"). Ask for a test
   asserting the two agree, and say plainly why derivation is not possible here.
3. **Not derivable, cross-package — the source cannot be imported.** Restating is correct.
   Ask that the restatement carry a permalink to the upstream definition and one line saying
   why it is restated, **and ask for a test in a lane that CAN import the owner** that checks
   the copy against it. A comment alone does not catch drift. On PR #4052 the hand-copied vLLM
   metric names carried three `*_total` aliases that could never match. A `@pytest.mark.vllm`
   test checking `inspect.getsource(vllm.v1.metrics.loggers)` fails on exactly those three, and
   the user asked for such a test explicitly.

**Why:** On PR #3369 (`nemo_gym_shards.py`) all three tiers appeared in one 408-line file, and
only tier 1 was spotted at first. `SHARD_SPEC_KEYS` is exactly `fields(ShardSpec)` and
`SHARDING_CONFIG_KEYS` is exactly `fields(ShardPlan)` — both verified identical by running it —
while `NEMO_RL_DICT_CONFIG_KEYS` is tier 2 (its two entries are popped in two different
functions in another file) and the Gym constants are tier 3 (the driver cannot import
`nemo_gym`, which is installed only in the actor venv). The user pushed back that this pattern
recurs across many people's PRs and the review should figure out the derivation rather than
just flag the duplication.

**Evidence framing matters.** The tempting line is "you will forget to update both." Check
before writing it: in that stack the author *did* remember — #3373 adds `placement_strategy` to
`ShardPlan` and to `SHARDING_CONFIG_KEYS` in the same diff. The honest and stronger framing is
"this paired edit is a tax you are already paying" plus a real instance where it *was* missed
(the tip commit `b6bdd3303` exists only to fix a missed sync on the sibling list). Never
predict a mistake the author has demonstrably not made.

**A moved list counts as new.** If the PR moves an existing list to a new home and gives it a new
user, review it as if the PR wrote it. On PR #4326, `FORBIDDEN_RPC_KEYS` moved from
`rollout_reassembler_actor.py` into `rpc_guard.py` and became the guard for the new advantage
actor too. It was a hand-kept subset of `TOKEN_ALIGNED_FIELDS`, missing the very columns the new
actor moves (`advantages`, `prev_logprobs`, ...). The team noted only the leftover import alias
and skipped the list as "pre-existing". The user had to ask "is this derived, and is there a test?"

**Tier 0 — the producer can just say it.** When a consumer keeps name lists or name rules to
decide how to treat values another function produces (which metrics to sum/max/pool, which
aliases map to which statistic), the best fix is neither derive nor test: have the producer
emit the rule next to the value (e.g. a plain-dict `MetricSet` with `reduce`/`samples`/`views`),
and keep name rules only in one adapter for producers that don't declare yet. On PR #4068 the
design lane flagged the 9-name family list but proposed only moving it; devil-advocate killed
that because a move doesn't remove drift. The user then asked for exactly this producer-declared
shape. A scratch prototype matched the PR's reducer on all 97 keys plus V1 and mixed-replay
cases, which is what made the comment credible. When "move it" dies for "only a test would
catch drift", look for tier 0 before dropping the finding.

**How to apply:** grep the module for `frozenset({`, `= [` / `= (` of string literals, and any
set of names, including ones the diff only moved. For each, find the declaration it shadows and try the derivation in a scratch
interpreter. One comment covering every list in the file that shares the fix beats one comment
per list. Related: [collapse-enum-plus-dependent-knob](collapse-enum-plus-dependent-knob.md) (same instinct: one declaration, not
two that must agree), [typeddict-missing-key-failfast-is-convention](typeddict-missing-key-failfast-is-convention.md).
**Seen:** #3369 SHARD_SPEC_KEYS, SHARDING_CONFIG_KEYS, NEMO_RL_DICT_CONFIG_KEYS, Gym constants (team caught the two field lists; user pushed for the other two) · #4052 vLLM metric names (user) · #4326 C10 FORBIDDEN_RPC_KEYS (user) · #4068 aggregate_rollout_metrics alias table + family list + agent-name parsing (user)
