# SDD ledger — plan: /Users/sna/Nemo-RL_Qwen3_Roadmap/.worktrees/online-drafter-efficiency-design-20260822/docs/superpowers/plans/2026-08-22-draft-update-cadence.md

## Task 6 dependency preflight

| Producer | Consumer | Dependency | Ruling |
|---|---|---|---|
| Task 5B receipt schema | Task 6 producer selection | Receipt flags remain orthogonal to transfer selection | Do not fabricate apply receipts or science capability. |
| Reviewed Task 6 receiver | Task 6 producer | Receiver already accepts `WeightSyncSelection` and enforces draft coverage | Replay receiver commits first; producer must match the reviewed receiver contract. |
| Policy producer | IPC/collective synchronizers | Full/default sync must preserve legacy endpoint call shape | Pass `selection=` only for non-default target-only transfer; default/full calls omit the keyword. |
| Factory/preflight | Later cadence controller | Mode/capability errors must happen before worker side effects | Propagate and validate mode only; do not enable fixed/adaptive scheduling in this slice. |

### Controller rulings

1. The bounded Task 6 slice is receiver replay, producer selection, and early capability/factory validation only.
2. Target-only transfer skips draft preflight, export, and pipeline-parallel collectives and emits zero draft names and bytes.
3. A reusable policy worker must support full, then target-only, then full without stale state.
4. Direct/remote unsupported modes must reject before worker construction or other worker-side effects.
5. Controller decisions, sender/apply receipts, cadence science capability, and fixed/adaptive enablement are out of scope.

### Progress

- [x] Exact Task 5B base verified at `5495e172058d08ecf1cf027e4c356d1d44610471` with a good signature.
- [x] Replayed reviewed receiver commits through `864ff419adfc8ddb55b281076770eb1d5e0c5a46` as new signed+DCO commits: `26df6b303`, `1c1df3815`, `507ac23d9`, `252da7549`.
- [ ] Dispatch bounded producer implementation with strict RED→GREEN evidence.
- [ ] Run independent read-only review and fix loop if needed.
- [ ] Run final scoped verification and report without push or job submission.
