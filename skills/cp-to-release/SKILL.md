---
name: cp-to-release
description: Manually cherry-pick a merged NVIDIA-NeMo/RL PR onto the current release branch (r0.X.0) and open a cp PR with matching CI labels and /ok to test.
when_to_use: >-
  'cp #N to release', 'cp #N to r0.8.0', '/cp-to-release N', backporting a PR that already landed on main
  into a release branch.
---

# cp-to-release

Input: the original PR number `N` (already merged). Optional: a release branch name; otherwise use the newest `origin/rX.Y.Z`.

Release flow since r0.8.0: PRs merge into the release branch first, and a label auto cherry-picks
them back to main. A manual cp is needed when a PR was merged into main instead of the release
branch first. (Up to r0.7.0 it was the reverse: main first, auto cp to release.)

The user invoking this skill authorizes the cherry-pick commit, push, PR, label, and
`/ok to test` — no pause-before-commit for the clean path. Do not assign reviewers; the user does.

## Steps

Run from the NeMo-RL checkout. `$GITHUB_TOKEN` (`public_repo` scope) must be set in every Bash call that uses it; `<user>` is the user's GitHub branch prefix.

1. **Read the original PR.**
   ```bash
   curl -sH "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/NVIDIA-NeMo/RL/pulls/<N> \
     | jq '{title, merged, merge_commit_sha, base: .base.ref, labels: [.labels[].name]}'
   ```
   Stop if `merged` is false. Keep `title`, `merge_commit_sha`, and the labels starting with `CI:`.

2. **Pick the release branch** (unless given):
   ```bash
   git fetch origin
   git branch -r | grep -E '^ *origin/r[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -1
   ```
   Also check `git ls-remote origin refs/heads/<user>/cp-<N>` is empty; if the branch exists, ask.

3. **Branch + cherry-pick in a worktree**, so the user's main checkout (often dirty, on another
   branch) is untouched. Put it under a temp directory:
   ```bash
   git worktree add -b <user>/cp-<N> <tmpdir>/cp-<N> origin/<rel>
   cd <tmpdir>/cp-<N> && git cherry-pick -s <merge_commit_sha>
   ```
   `-s` is required: the commit must carry the user's own `Signed-off-by` alongside the author's.
   Forgetting it means amend + force-push + a second `/ok to test`.

4. **Conflicts: stop and hand back.** Show the conflicting files; do not auto-resolve. After the
   user resolves and continues, verify before pushing:
   - `git log -1` — the cherry-pick committed, and has the user's `Signed-off-by`.
   - `grep -n "^<<<<<<< \|^=======$\|^>>>>>>> " <files>` — no leftover markers.
   - Per file, compare the `+` lines with the original:
     `diff <(git diff HEAD~1..HEAD -- <f> | grep '^+' | grep -v '^+++' | sort) <(git show <merge_sha> -- <f> | grep '^+' | grep -v '^+++' | sort)`
     and report any extra/missing lines.

5. **Push and open a PR** targeting the release branch, never `main`:
   ```bash
   git push -u origin <user>/cp-<N>
   ```
   POST `/repos/NVIDIA-NeMo/RL/pulls` with `"base": "<rel>"`, `"head": "<user>/cp-<N>"`.
   - Title: `cp: cherry-pick #<N> - <original title verbatim>` (keeps semantic-pull-request CI happy).
   - Body, one line:
     - Clean: `manually cp #<N> into <rel> branch since it was merged into main instead of <rel> first.`
     - Conflict: append the conflict, e.g. `... first. There's a conflict in `<file>` (<why, e.g. depends on #M which is not in <rel>>).`

6. **Labels + CI.** Copy the original's `CI:*` labels (`POST /issues/<pr>/labels`). Do not copy
   `cherry-pick-to-main` (it makes the bot cherry-pick the PR back to main) or release labels such
   as `r0.8.0`. Then post
   `/ok to test <full head sha>` as an issue comment.

7. **Report**: PR number (bare `#N`), base, commit sha, labels, that `/ok to test` was
   posted, and the worktree path (`git worktree remove <path>` to clean up).

## Reference PRs

- #4412 (r0.8.0, clean, new flow): body `manually cp #4182 into r0.8.0 branch since it was merged into main instead of r0.8.0 first.`
- r0.7.0, old flow: #3211 (clean, empty body); conflicts: #3153 `manually cp #3147 into r0.7.0 branch since there's a conflict in `docs/guides/environments.md`.`, #3195, #3214.
