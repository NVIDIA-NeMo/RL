# capability-belongs-on-the-class

> A name-keyed list/dict/if-chain that says which classes "can do X" or "count as X" is a finding — ask for a required (abstract) property on the class, read off the built object, so a new class that forgets it fails at construction

When a PR decides something about a family of classes by looking their **name** up in a
separate collection — `FOO_CAPABLE = frozenset({"grpo", "opd"})`, a dict of name → flag,
`if name in (...)`, or an `if isinstance / if cfg.is_x` chain in the caller — ask whether the
thing being decided is really a **fact about each class's own behavior**. If it is, the fact
belongs on the class, as a property every class in the family must define. The caller reads it
off the object it already built. The name list goes away.

The shape to recommend:
- **The class states a fact about itself; the consumer draws the conclusion.** Name the property
  after the class's own behavior (`needs_cross_group_stats`), not after the consumer's feature
  (`shard_invariant`). The consumer computes its own flag from it (`shardable = not
  est.needs_cross_group_stats`). The same fact then answers other consumers' questions too.
- **Mandatory, not defaulted.** Put it on an `ABC` base as an abstract property. A class that
  forgets it raises `TypeError: Can't instantiate abstract class ...` when it is built, which is
  usually at setup, before any GPU work. A plain class attribute satisfies an abstract property;
  a `@property` handles config-dependent answers. Never `getattr(obj, "x", False)`: a default
  hides exactly the class someone forgot. If there is no base class yet, adding one is part of
  the ask.
- **Config-dependent answers become possible.** A name list can only say "this estimator, always".
  A property can say "only when `normalize_advantages` is on". A name list that has to be
  cautious for a whole class is itself a sign the fact belongs on the instance.
- **Test the property on every built instance**, not the list's contents. A test that pins
  `LIST == {"a", "b"}` guards the list, not the classes.

Scope and severity:
- Prefer code the PR introduces. A pre-existing `if is_ppo:` chain the PR only moved is a
  tracking-issue mention, not an ask.
- Low severity when the list is correct and fails safe (missing → the conservative path). Higher
  when a missing entry silently takes the unsafe path.
- Leave it to the author: this PR or a follow-up.

**Why:** On PR #4326 the new advantage-actor pool decided which estimators may be split across
actors with `SHARD_INVARIANT_ESTIMATORS = frozenset({"grpo", "opd"})` in advantage_stage.py, and
`shardable = name in` that set. The list was correct and failed safe, but:
- the fact lived in a different module from the estimator classes
- the test only pinned the set's contents
- it could not say that GAE/raw_reward are safe with `normalize_advantages` off

The user: "i don't like it when candidacy is some global dict where it could a mandated property of
some class that everyone adheres to." They then pushed the design to an abstract
`needs_cross_group_stats` on an `AdvantageEstimator(ABC)` base. The sibling
`FORBIDDEN_RPC_KEYS` in the same PR is the [derive-key-lists-dont-restate-them](derive-key-lists-dont-restate-them.md) form of the
problem: a hand-kept subset of `TOKEN_ALIGNED_FIELDS` that is missing the very columns the new
actor handles.

**Same smell, other shapes.** The class's own facts can also leak out as:
- **one class's special fields on a shared type.** `AdvantageOutcome` (an RPC type for every
  estimator) carries OPD-only metric fields.
- **a caller redoing one class's math.** The controller recomputes OPD's metrics from sums the
  estimator already reduces, kept in step only by a comment.
- **a per-name list of inputs.** Which columns each estimator reads is kept outside the estimator.

The fix has the same shape: the class owns it and exposes it through one common interface, e.g. a
mergeable partial every estimator returns. On #4326 these became C11 and follow-up #4376.

**How to apply:** For every new `frozenset`/list/dict of names and every new
`if name == / in / isinstance` chain in a caller, ask: "is this a fact about each class that the
class should state?" If yes:
1. Name the property after the class's behavior.
2. Show the ABC + abstract property sketch.
3. Give the value for every existing class, including config-dependent ones.
4. Show the one line the consumer derives from it.

Verify the abstract-property behavior in a scratch interpreter before claiming it. If the list
instead copies a *declaration* (dataclass fields, enum members, another module's constant), that
is [derive-key-lists-dont-restate-them](derive-key-lists-dont-restate-them.md). Related: [collapse-enum-plus-dependent-knob](collapse-enum-plus-dependent-knob.md).
**Seen:** #4326 C9 SHARD_INVARIANT_ESTIMATORS (user) · #4326 C11 OPD fields on AdvantageOutcome + metrics recomputed in the controller (user) · #4326 #4376 per-estimator input_fields (user)
