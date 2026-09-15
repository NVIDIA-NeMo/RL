# Regular Super RL: reviewed gold alignment

The reference is the `grpo_superv3_5_rlvr_v43_broad_falcon_r3-oci-hsg-20260905-r1`
script/YAML in pipeline commit `7e8d920115402f68d0f9047f23977246b4a88f46`.
These recipe changes require a new native smoke; they do not retroactively change
the source or configuration of a completed job. No training certification is
implied by static tests or scheduler acceptance.

## Context and policy serving

The earlier CMH adapter retained a 262144-token total/packing budget even after
reducing output to 102400. The regular recipe now uses **131072** for total
context and vLLM model length. Train/logprob packing budgets derive from that
single value and their respective microbatch sizes, avoiding stale copies.

The policy vLLM scheduler now allows **256 sequences** and **32768 batched
tokens**, matching the gold settings. This is an unbenchmarked throughput
candidate, not evidence of improved speed or memory fit. It does not change the
DeepSeek judge's separate scheduling or 8192-token response budget.

The policy and all agents retain **102400 cumulative assistant output tokens**,
reasoning enabled, and original prompts. At full output length only 28672 tokens
remain for input, template overhead and tool history. Validate real tokenized
trajectories before launch; do not silently truncate prompts, suppress reasoning,
or reduce the agreed output allowance to make them fit. Multi-turn history can
hit the context ceiling before exhausting the cumulative output allowance.

Implementation: `training_configs/super_rl/experiments/regular_s120_smoke.yaml`.
Regression coverage: `tests/unit/tools/test_regular_recipe_assets.py`.

## Output-format reward penalties

The earlier regular recipe disabled unwanted-token and malformed-think-tag
penalties. Both are now enabled, alongside the existing duplicated-reasoning
and empty-final penalties. The existing implementation and token IDs are
unchanged: unwanted `[2]`, think-open `12`, think-close `13`. Verify these IDs
against the actual tokenizer before using another model.

This deliberately changes the reward contract, not the source prompts or
reasoning mode. Check unwanted-token/malformed-think-tag rates and inspect
flagged multi-turn trajectories in the next smoke. If extraction or parsing
incorrectly flags valid output, fix that source rather than silently disabling
the requested penalty. Historical reward measurements remain unchanged and
are not measurements of the new contract.

## Explicit startup idle-reaper grace

Use the clean submission entrypoint with **`--bootstrap-grace-minutes 75`** for
this baseline. It puts the same structured `OccupiedIdleGPUsJobReaper` comment
as gold on the **outer sbatch request**, where it can be observed. Adding an
`#SBATCH` directive to a sourced `ray.sub` would not affect the outer job.

```bash
uv run --no-sync tools/super_rl/submit.py --bootstrap-grace-minutes 75 \
  -- --account=<your-account> --partition=batch --qos=short \
  --nodes=64 --gpus-per-node=4 --segment=16 --time=02:00:00 <reviewed-launcher.sbatch>
```

This command is **scheduler test-only**; actual submission additionally requires
`--submit` before `--`. Pass reviewed workload environment names via `--env` as
needed. The grace is explicit, not applied to every site's jobs by default.
It supersedes a batch-file comment; an additional first-component CLI comment
is rejected rather than silently discarding it. Later heterogeneous service
components retain their own comments and do not acquire a 75-minute or
1440-minute exemption implicitly.

The grace does not fix stalled training, extend Slurm walltime, or override site
policy. Verify the stored outer job comment after a real submission. A scheduler
test accepting the JSON does not prove that the reaper service honored it.
