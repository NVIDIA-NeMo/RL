---
name: cp-to-main
description: Manually cherry-pick a merged NVIDIA-NeMo/RL PR from a release branch (r0.X.0) back onto main when the auto cherry-pick bot failed. Stops at conflicts for the user to resolve, then verifies, pushes, opens a cp PR, copies CI labels and posts /ok to test.
when_to_use: >-
  'cp #N back to main', 'auto cp failed', '/cp-to-main N', cherry-picking a release-branch PR onto main;
  resuming after the user resolved conflicts ('continue').
---

# cp-to-main

Reverse of `cp-to-release`. Input: the original PR number `N`, already merged into a release branch
`rX.Y.Z`. Since r0.8.0 PRs merge into the release branch first and a label makes the bot cherry-pick
them to main as `cherry-pick-<N>-main`; this skill covers the case where that auto cp failed
(typically a conflict) and the user wants it done by hand.

The skill runs in two phases with a hand-off in the middle:

- **Phase A (start):** read PR, cherry-pick in a worktree, **stop at a conflict** and hand back.
- **Phase B (continue):** after the user resolves, finish the pick, verify, push, PR, labels,
  `/ok to test`.

A clean pick runs A and B back to back without stopping. The user invoking this skill authorizes
the commit, push, PR, label and `/ok to test` — no pause-before-commit. Never resolve a
conflict yourself, never `git add` conflicted files, never `--skip`/`--abort` unprompted. Do not
assign reviewers.

Run from the NeMo-RL checkout. `$GITHUB_TOKEN` (`public_repo` scope) must be set in every Bash call that uses it; `<user>` is the user's GitHub branch prefix.

## Phase A: start

1. **Read the original PR.**
   ```bash
   curl -sH "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/NVIDIA-NeMo/RL/pulls/<N> \
     | jq '{title, merged, merge_commit_sha, base: .base.ref, labels: [.labels[].name]}'
   ```
   Stop if `merged` is false, or if `base` is `main` (nothing to cp back). Keep `title`,
   `merge_commit_sha`, `base` (the release branch), and the labels starting with `CI:`.

2. **Check it is not already handled.**
   - `git fetch origin`; `git log origin/main --oneline --grep="(#<N>)" | head` — already on main? stop.
   - Open PR from the bot or a prior attempt:
     `curl ... "/pulls?state=open&head=NVIDIA-NeMo:cherry-pick-<N>-main"` and
     `git ls-remote origin refs/heads/<user>/cp-<N>-main`. If either exists, ask the user.

3. **Worktree + cherry-pick**, under a temp directory so the main checkout is untouched:
   ```bash
   git worktree add -b <user>/cp-<N>-main <tmpdir>/cp-<N>-main origin/main
   cd <tmpdir>/cp-<N>-main && git cherry-pick -s <merge_commit_sha>
   ```
   `-s` adds the user's `Signed-off-by` beside the author's.

4. **Conflict → stop and hand back.** If the pick exits non-zero, run
   `git status --short | grep -E '^(UU|AA|DD|AU|UA|DU|UD)'` and report:
   - the worktree path (the user resolves there, not in the main checkout),
   - each conflicting file with a one-line guess at why (e.g. depends on #M not in main, file
     renamed on main — check `git log origin/main -3 -- <file>`),
   - how to proceed: edit the files, `git add <files>` themselves, then say "continue"
     (running `git cherry-pick --continue` themselves is also fine).

   Then **end the turn**. Do not edit, stage, or commit anything.

   Clean pick (exit 0) → go straight to step 6.

## Phase B: continue

5. **Re-enter and finish the pick.** On "continue", locate the worktree
   (`git worktree list | grep cp-<N>-main`) and in it:
   ```bash
   git status --short
   git diff --name-only --diff-filter=U        # unmerged paths
   ls "$(git rev-parse --git-dir)/CHERRY_PICK_HEAD" 2>/dev/null
   ```
   - Unmerged paths remain, or leftover markers (below) → list them and stop again.
   - `CHERRY_PICK_HEAD` still present and nothing unmerged → the user staged but did not commit:
     `GIT_EDITOR=true git cherry-pick --continue` (never bare, `core.editor=vim`).
   - No `CHERRY_PICK_HEAD` → the user already continued; proceed.

6. **Verify before pushing.**
   - `git log -1` — the pick committed, and the message has the user's `Signed-off-by`. If it is
     missing (continue after a conflict can drop it): `git commit --amend -s --no-edit` (not pushed
     yet, safe).
   - `git grep -n -E '^(<<<<<<< |=======$|>>>>>>> )' -- <files>` — no leftover markers.
   - Per file, compare the `+` lines with the original and report any extra/missing lines — for a
     conflicted pick the diff is the user's resolution, so list it rather than judging it:
     `diff <(git diff HEAD~1..HEAD -- <f> | grep '^+' | grep -v '^+++' | sort) <(git show <merge_sha> -- <f> | grep '^+' | grep -v '^+++' | sort)`
   - Run `ruff check .`, `ruff check --select I .` and `ruff format --check .` only if the user
     resolved conflicts in `.py` files.

7. **Push and open a PR** targeting `main`:
   ```bash
   git push -u origin <user>/cp-<N>-main
   ```
   POST `/repos/NVIDIA-NeMo/RL/pulls` with `"base": "main"`,
   `"head": "<user>/cp-<N>-main"`.
   - Title: `cp: cherry-pick #<N> - <original title verbatim>`.
   - Body, one line:
     - Clean: `manually cp #<N> from <rel> into main since the auto cherry-pick did not land.`
     - Conflict: `manually cp #<N> from <rel> into main since the auto cherry-pick had a conflict in `<file>` (<why>).`

8. **Labels + CI.** Copy the original's `CI:*` labels (`POST /issues/<pr>/labels`). Do not copy
   release labels (`r0.8.0`, ...) or the bot's `cherry-pick` label. Then post
   `/ok to test <full head sha>` as an issue comment.

9. **Report**: PR number (bare `#N`), base, commit sha, labels, that `/ok to test` was
   posted, which files had conflicts, and cleanup (`git worktree remove <path>`; the bot's own
   `cherry-pick-<N>-main` PR, if any, can be closed as superseded — say so, don't close it).
