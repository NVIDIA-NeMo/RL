---
name: megatron-pr-reporting
description: Aggregates NeMo-RL functional test results for a Megatron-LM PR into a testname/status/comment table and upserts it as a sticky comment on the PR. Use as the final step of testing a NemoRLTest-labeled PR, or when refreshing an existing results comment.
when_to_use: Publishing functional test results back to a Megatron-LM PR; "post the results table"; refreshing a stale results comment; deciding what to write in the comment column for a failure.
---

# Reporting results back to the PR

Stage 5 of the nemo-rl-testing-agent pipeline. One sticky comment per PR,
rewritten in place on every re-run, so the PR never accumulates a wall of
near-identical bot comments.

## Never leave the PR silent

Reporting is the last *stage*, but it is not the only time the agent writes to
the PR. Because the comment is sticky, claim it up front and overwrite it as the
run progresses. A PR author must never have to guess whether the agent picked
their PR up.

| When | Command | Result |
|---|---|---|
| Immediately before submitting the first suite | `--state running` | "⏳ Running…" placeholder |
| The run died before any test executed | `--state infra --note '<what broke>'` | "⚠️ did not reach the tests", explicitly *not* a verdict on the PR |
| A suite produced results | `--state results --results …` | the table |

Posting `--state infra` is **mandatory** when a run is abandoned for
infrastructure reasons — a failed container prep, a wedged workspace sync, a
cancelled job. Silence reads to the author as "the agent never ran", and that is
the one outcome worth avoiding: it is indistinguishable from the agent being
broken. Say what broke and that it is a NeMo-RL-side problem.

## Results JSON

`parse_results.py` produces this; the agent edits `status` and `comment` before
publishing. Nothing else needs to change.

```json
{
  "prep": {"mcore_sha": "...", "mcore_subject": "..."},
  "tests": [
    {
      "name": "grpo_megatron_generation_async",
      "status": "fail",
      "rc": 1,
      "secs": 4180,
      "error_signature": "AssertionError: max(data[\"train/token_mult_prob_error\"]) < 1.05",
      "comment": ""
    }
  ]
}
```

### Status vocabulary

| Status | Meaning |
|---|---|
| `pass` | Ran clean against the PR revision. |
| `pass (suspect)` | Exited 0, but the log shows swallowed per-sample errors — treat as a failure. |
| `fail` | Genuine failure caused by the PR, not fixed. |
| `fail (pre-existing)` | Fails on `main` too; not this PR's fault. Still gets a fix attempt. |
| `fixed` | Failed, then passed with the linked draft PR applied. |
| `not run` | Never executed (branch conflicts, repeated infra failure, deliberately skipped). |

## What belongs in the comment column

Leave it **empty for `pass`**. Otherwise write for the PR author, who has not
seen any of the logs:

- **`fail`** — the error signature, the most likely root cause, and each fix
  attempt with why it did not work. Name the file and function when known.
- **`pass (suspect)`** — what the swallowed error was, and that the metric it
  reported is not trustworthy.
- **`fail (pre-existing)`** — say it reproduces on `main` at `<sha>` so nobody
  re-litigates it, **and** either link the fix PR or state where the fix attempt
  got to. "Not caused by this PR" on its own is not an acceptable cell.
- **`fixed`** — one-sentence root cause plus the review link, e.g.
  `Megatron InferenceConfig rename; needs NVIDIA-NeMo/RL#2931 (draft) merged.`
- **`not run`** — why, and what would unblock it.

When one root cause claims several tests — which is the normal shape when a
dependency breaks at import — write the explanation once and have each cell
point at it, rather than repeating a paragraph down every row. Put the shared
part in the report note under the table and keep the cells to what differs.
`post_tracking_issue.py` already groups rows by known-issue id and renders each
diagnosis once below; a table where every row carries the same eighty words is
one nobody reads, including the author it was written for.

Also flag the checkout caveat here when it applies: if the PR changed
`setup.py`, `pyproject.toml`, or native sources, the run used the image's
prebuilt dependencies and extensions (see
[megatron-pr-discovery](../megatron-pr-discovery/SKILL.md) step 4).

Keep each cell to a few sentences. Newlines become `<br>` and pipes are escaped
automatically, so write naturally.

## Publish

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/post_report.py \
  --pr 5700 \
  --results ~/.nemo-rl-testing-agent/pr-5700/l1.results.json \
  --results ~/.nemo-rl-testing-agent/pr-5700/l2.results.json \
  --integration ~/.nemo-rl-testing-agent/integration.json \
  --meta megatron-lm=<HEAD_SHA> --meta merged-with-main=<BASE_SHA> \
  --meta cluster=oci-hsg --meta image=nvcr.io/nvidian/nemo-rl:nightly
```

Results files render in the order given, so pass L1 before L2. Add `--dry-run`
to review the markdown first — always do this before the first post on a PR.

**Always pass `--integration`.** Runs carry NeMo-RL fixes that are raised but not
merged, so a green table means "green with those applied", and a reader entitled
to assume otherwise would draw the wrong conclusion. The flag renders one line
naming them; omitting it silently overstates the result.

`--note` appends free markdown under the body in every state; use it for a
finding that applies to the whole run rather than one row, such as the failing
prep step.

### Write for someone outside NVIDIA

The audience is a Megatron-LM contributor with no access to our cluster. Internal
run names and `/lustre` paths are meaningless to them and actively mislead — a
trailing `` `nrlta-pr5700-l1-a7` `` was once read as a branch the reviewer was
supposed to go look at. `post_report.py` refuses to post a body containing them,
so describe the evidence in words ("reproduces on megatron-core `main` at
`cd4afffa`") and keep run names in `ledger.md`, which is where the next agent
looks anyway.

The comment is matched by the `<!-- nemo-rl-testing-agent -->` marker: found
means PATCH, absent means POST. Never remove the marker from a rendered body.

## After posting

1. Re-read the comment (`gh pr view <N> --repo NVIDIA/Megatron-LM --comments`)
   and confirm the table rendered — an unescaped pipe or a stray newline shows
   up immediately.
2. Append the comment URL and the final table to `ledger.md`.
3. Record anything this PR taught you about running the pipeline itself with
   [nemo-rl-testing-agent-learning](../nemo-rl-testing-agent-learning/SKILL.md).
   This is the last point at which the sweep still holds the context; after the
   handoff to the next PR it is gone.
4. Report to the user: PR number, counts by status, and the links to any draft
   fix PRs that need review.
