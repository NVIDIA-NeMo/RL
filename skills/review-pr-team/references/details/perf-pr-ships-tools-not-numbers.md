# perf-pr-ships-tools-not-numbers

> A perf PR that adds benchmark scripts but reports no results — ask for the results in the description and drop the scripts

When a perf PR's biggest part is benchmark tooling (scripts under `tools/`, plus tests of the tool's own counters) and the description has no numbers, the review asks for: (a) the one or two numbers that decide whether the change is worth its size, named as an already-logged metric where possible (e.g. `timing/rollout_checkpoint/total_save_seconds` before vs after), and (b) deleting the scripts, with the results table in the description and the script in a gist/comment. Check what the script actually times: on #4508 both scripts called the sink directly, so they measured TransferQueue, not the PR's code path.

**Why:** #4508 shipped 1,152 lines of benchmark scripts (one byte-identical to an older unreviewed PR) plus an 81-line test of the tool, and zero numbers. The scripts were ~40% of the PR. Repo precedent for `tools/` benchmarks is ~200 lines.
**How to apply:** Count tool + tool-test + config lines (e.g. `pyrefly.toml` includes) together. Diff the tool against any older PR by the same author. If part of the PR (e.g. a new read path) has no number behind it at all, ask to split it into its own PR that leads with its number, rather than only asking for numbers. Related: `perf-evidence`.
**Seen:** #4508 C5/C6 (team; the user's size focus)
