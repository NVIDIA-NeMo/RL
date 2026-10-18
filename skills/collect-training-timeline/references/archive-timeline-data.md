# Extract and archive timeline data

Use this workflow to prepare a portable archive that preserves everything
needed to validate, analyze, and rebuild a training timeline without copying
rollout responses or sandbox contents.

## Contents

- [Set paths](#set-paths)
- [Validate the source traces](#validate-the-source-traces)
- [Extract useful tables](#extract-useful-tables)
- [Stage the canonical data](#stage-the-canonical-data)
- [Write the archive README](#write-the-archive-readme)
- [Create and verify the archive](#create-and-verify-the-archive)
- [Extract and analyze an archive](#extract-and-analyze-an-archive)
- [Rebuild the HTML](#rebuild-the-html)
- [Interpret the fields correctly](#interpret-the-fields-correctly)

## Set paths

Run from any directory. Set `JOB_ID`, `LOG_ROOT` (the host directory mounted
at the job's `/logs`) and `ARCHIVE_DIR` for the target run. Use a new
temporary directory so a stale file cannot enter the archive.

```bash
JOB_ID=<slurm-job-id>
LOG_ROOT=/path/to/job/nemo_logs                       # host side of /logs
ARCHIVE_DIR=/path/to/archives
SKILL_ROOT=/path/to/RL/skills/collect-training-timeline
ARCHIVE="$ARCHIVE_DIR/slurm-$JOB_ID-timeline-data.tar.gz"
STAGE_ROOT=$(mktemp -d "/tmp/slurm-$JOB_ID-timeline-data.XXXXXX")
STAGE="$STAGE_ROOT/slurm-$JOB_ID-timeline-data"
mkdir -p "$STAGE"
```

Check the required controller traces before doing a recursive rollout scan:

```bash
test -s "$LOG_ROOT/training_timeline.jsonl"
test -s "$LOG_ROOT/rollout_lifecycle.jsonl"
test -d "$LOG_ROOT/nemo_gym"
```

## Validate the source traces

Validate canonical data before packaging it. Require only metrics expected for
the target revision. A failed strict requirement is useful information; do not
silently synthesize missing fields from rollout `output.jsonl` files.

```bash
python "$SKILL_ROOT/scripts/validate_timeline.py" \
  --training-timeline "$LOG_ROOT/training_timeline.jsonl" \
  --rollout-root "$LOG_ROOT/nemo_gym" \
  --rollout-lifecycle "$LOG_ROOT/rollout_lifecycle.jsonl" \
  --require-generation-replica \
  --require-model-token-counts \
  --require-model-cache-metrics \
  --require-model-scheduling-timestamps
```

Record any known validation exception in the archive README, including exact
numerators and denominators. Do not describe partial coverage as complete.

## Extract useful tables

Canonical JSONL files are sufficient to reproduce every table below. Derived
TSV files are optional conveniences, not replacements for the JSONL source.

```bash
mkdir -p "$STAGE/derived"
```

### Rollout lifecycle

Extract generation, target, and consumption coordinates. Display steps are
one-based; weight versions are zero-based.

```bash
jq -r '
  select(.event == "rollout_consumed") |
  [
    .rollout_id,
    .prompt_idx,
    .sample_idx,
    .generation_weight_version,
    .target_display_step,
    .consumed_display_step,
    .disposition
  ] | @tsv
' "$LOG_ROOT/rollout_lifecycle.jsonl" \
  > "$STAGE/derived/rollout_consumed.tsv"
```

### Training phases

```bash
jq -r '
  select(.ph == "X") |
  [.name, .cat, .ts, .dur, (.args.status // "")] | @tsv
' "$LOG_ROOT/training_timeline.jsonl" \
  > "$STAGE/derived/training_events.tsv"
```

### Per-turn model calls

Read only per-rollout `timeline.jsonl` files. Do not scan persisted response
files. This command preserves a reported zero cache hit.

```bash
find "$LOG_ROOT/nemo_gym" -type f -name timeline.jsonl -print0 |
  xargs -0 jq -r '
    select(.name == "model_call") |
    [
      (.args.rollout_id // ""),
      .ts,
      (.ts + .dur),
      (.args.turn // ""),
      (.args.generation_replica_idx // ""),
      (.args.llm_arrival_ts_us // ""),
      (.args.queued_ts_us // ""),
      (.args.first_scheduled_ts_us // ""),
      (.args.first_token_ts_us // ""),
      (.args.last_token_ts_us // ""),
      (.args.llm_response_ready_ts_us // ""),
      (.args.gym_llm_response_received_ts_us // ""),
      (.args.gym_tokenize_duration_us // ""),
      (.args.gym_response_ready_ts_us // ""),
      (.args.openhands_response_received_ts_us // ""),
      (.args.openhands_parse_validate_duration_us // ""),
      (.args.prompt_tokens // ""),
      (.args.cached_prompt_tokens // ""),
      (.args.uncached_prompt_tokens // ""),
      (.args.decode_tokens // ""),
      (.args.status // "")
    ] | @tsv
  ' > "$STAGE/derived/model_calls.tsv"
```

The columns are:

```text
rollout_id  model_call_start_us  model_call_end_us  turn
generation_replica_idx  llm_arrival_ts_us  queued_ts_us
first_scheduled_ts_us  first_token_ts_us  last_token_ts_us
llm_response_ready_ts_us  gym_llm_response_received_ts_us
gym_tokenize_duration_us  gym_response_ready_ts_us
openhands_response_received_ts_us  openhands_parse_validate_duration_us
prompt_tokens  cached_prompt_tokens  uncached_prompt_tokens  decode_tokens  status
```

Useful examples:

```bash
# Successful model calls and complete cache misses.
awk -F '\t' '$21 == "ok" {ok++} $21 == "ok" && $18 == 0 {miss++}
  END {print "successful=" ok, "complete_misses=" miss}' \
  "$STAGE/derived/model_calls.tsv"

# Verify sticky rollout-to-replica routing. Output should normally be 1.
awk -F '\t' '$21 == "ok" && $5 != "" {seen[$1 SUBSEP $5] = 1}
  END {for (k in seen) {split(k, a, SUBSEP); n[a[1]]++}
       for (r in n) print r, n[r]}' \
  "$STAGE/derived/model_calls.tsv" | sort -k2,2nr

# Per-turn token/cache averages. Use sums for the weighted hit rate.
awk -F '\t' '$21 == "ok" && $17 != "" && $18 != "" {
    t=$4; n[t]++; p[t]+=$17; c[t]+=$18; u[t]+=$19; d[t]+=$20
  }
  END {for (t in n)
    printf "%s\t%d\t%.1f\t%.1f\t%.1f\t%.1f\t%.2f%%\n",
      t,n[t],p[t]/n[t],c[t]/n[t],u[t]/n[t],d[t]/n[t],100*c[t]/p[t]}' \
  "$STAGE/derived/model_calls.tsv" | sort -k1,1n

# Average model-call phase durations in milliseconds. Negative cross-host
# transport values indicate wall-clock skew and should be interpreted cautiously.
awk -F '\t' '$21 == "ok" && $6 != "" && $7 != "" && $8 != "" && $9 != "" &&
    $10 != "" && $11 != "" && $12 != "" && $14 != "" && $15 != "" {
    n++; client+=$6-$2; frontend+=$7-$6; queue+=$8-$7;
    prefill+=$9-$8; decode+=$10-$9; vfinal+=$11-$10;
    v2g+=$12-$11; gym+=$14-$12; g2o+=$15-$14; ohpost+=$3-$15;
    tokenize+=$13; parse+=$16
  }
  END {if (n) printf "n=%d client=%.3f frontend=%.3f queue=%.3f scheduled_to_first=%.3f decode=%.3f llm_finalize=%.3f llm_to_gym=%.3f gym_post=%.3f gym_tokenize=%.3f gym_to_openhands=%.3f openhands_post=%.3f parse_validate=%.3f ms\n",
    n,client/n/1000,frontend/n/1000,queue/n/1000,prefill/n/1000,
    decode/n/1000,vfinal/n/1000,v2g/n/1000,gym/n/1000,tokenize/n/1000,
    g2o/n/1000,ohpost/n/1000,parse/n/1000}' \
  "$STAGE/derived/model_calls.tsv"
```

Sort both inputs by `rollout_id` before using `join` to attach target and
consumption steps to model calls:

```bash
sort -t $'\t' -k1,1 "$STAGE/derived/rollout_consumed.tsv" \
  > "$STAGE/derived/rollout_consumed.sorted.tsv"
sort -t $'\t' -k1,1 "$STAGE/derived/model_calls.tsv" \
  > "$STAGE/derived/model_calls.sorted.tsv"
join -t $'\t' \
  "$STAGE/derived/rollout_consumed.sorted.tsv" \
  "$STAGE/derived/model_calls.sorted.tsv" \
  > "$STAGE/derived/consumed_model_calls.tsv"
```

## Stage the canonical data

Use a compact relative layout in the archive:

```text
slurm-<job>-timeline-data/
  README.md
  MANIFEST.sha256
  training_timeline.jsonl
  rollout_lifecycle.jsonl
  nemo_gym/**/timeline.jsonl
  derived/*.tsv
```

Copy only the two controller traces and files named exactly `timeline.jsonl`:

```bash
cp "$LOG_ROOT/training_timeline.jsonl" "$STAGE/"
cp "$LOG_ROOT/rollout_lifecycle.jsonl" "$STAGE/"
mkdir -p "$STAGE/nemo_gym"
rsync -a \
  --include='*/' \
  --include='timeline.jsonl' \
  --exclude='*' \
  "$LOG_ROOT/nemo_gym/" "$STAGE/nemo_gym/"
```

Never include these files or directories:

- rollout `output.jsonl` or `output_for_eval.jsonl`;
- prompts, responses, trajectory histories, patches, or model text;
- sandbox, container, evaluation-output, or Apptainer contents;
- environment dumps, credentials, tokens, or private datasets;
- a separately rendered HTML unless the user explicitly requests it.

## Write the archive README

Create `$STAGE/README.md`. Include:

- job ID, source log root, container/revision, and relevant configuration;
- counts of training events, lifecycle records, rollout trace files, rollout
  events, and successful model calls;
- exact coverage of prompt/decode, cache, replica, and LLM timing fields;
- event and token semantics from `references/event-schema.md`;
- known limitations and validation failures;
- the clean archive layout and HTML rebuild command;
- a statement that rollout output, prompts, responses, sandbox data, and
  credentials are intentionally excluded.

Do not claim `generation_replica_idx` is a TP-rank identifier. It selects a
complete data-parallel generation replica; all TP ranks in that replica execute
the request. A Gym/OpenHands hostname identifies the execution host, not the
generation replica.

## Create and verify the archive

Generate a content manifest before compression:

```bash
(
  cd "$STAGE"
  find . -type f ! -name MANIFEST.sha256 -print0 |
    sort -z |
    xargs -0 sha256sum > MANIFEST.sha256
)
```

Reject forbidden paths, create the archive atomically, and verify it:

```bash
if find "$STAGE" -type f \
    \( -name output.jsonl -o -name output_for_eval.jsonl \) \
    -print -quit | grep -q .; then
  echo "refusing to archive rollout response data" >&2
  exit 1
fi

ARCHIVE_TMP="$ARCHIVE.tmp"
tar -C "$STAGE_ROOT" -czf "$ARCHIVE_TMP" "$(basename "$STAGE")"
gzip -t "$ARCHIVE_TMP"
mv "$ARCHIVE_TMP" "$ARCHIVE"

tar -tzf "$ARCHIVE" |
  grep -E '(^|/)(output\.jsonl|output_for_eval\.jsonl)$' && {
    echo "forbidden response file found in archive" >&2
    exit 1
  }

tar -tzf "$ARCHIVE" | tail -n 1
stat -c '%s bytes %n' "$ARCHIVE"
```

The `grep` command is expected to find nothing. Inspect the complete member
list when the archive is unexpectedly large.

Remove only the validated temporary staging directory:

```bash
rm -rf -- "$STAGE_ROOT"
```

## Extract and analyze an archive

Extract into a new directory and verify checksums:

```bash
EXTRACT_ROOT=$(mktemp -d /tmp/timeline-data-extract.XXXXXX)
tar -C "$EXTRACT_ROOT" -xzf "slurm-$JOB_ID-timeline-data.tar.gz"
DATA_ROOT="$EXTRACT_ROOT/slurm-$JOB_ID-timeline-data"
(
  cd "$DATA_ROOT"
  sha256sum -c MANIFEST.sha256
)
```

Read `README.md` before interpreting fields. Prefer the canonical JSONL files
when a derived table and source data disagree.

## Rebuild the HTML

```bash
python "$SKILL_ROOT/scripts/plot_rl_timeline.py" \
  --training-timeline "$DATA_ROOT/training_timeline.jsonl" \
  --rollout-root "$DATA_ROOT/nemo_gym" \
  --rollout-lifecycle "$DATA_ROOT/rollout_lifecycle.jsonl" \
  --output "slurm-$JOB_ID-all-rollouts-timeline.html" \
  --zero-at-first-rollout \
  --include-all-rollouts
```

An all-rollouts render must include every discovered rollout trace. Do not use
lane sampling or time-window filtering when preparing a restorable archive.

## Interpret the fields correctly

- `prompt_tokens` is the complete input/context length for one model request.
- `cached_prompt_tokens` is request-level prefix reuse, not cache occupancy.
- `uncached_prompt_tokens` is prompt work that must be computed for that
  request; a complete miss has cached tokens equal to zero.
- `decode_tokens` is copied from the live completion usage and is not recovered
  from persisted response text.
- `generation_replica_idx` identifies the selected generation replica, not an
  individual TP rank.
- `first_scheduled_ts_us` is the engine's first scheduler-admission timestamp and a
  low-overhead proxy for GPU start; it is not a CUDA-kernel timestamp.
- `first_scheduled_ts_us - queued_ts_us` is the true engine scheduler queue
  interval. `queued_ts_us - llm_arrival_ts_us` is frontend preprocessing and
  submission, not scheduler queueing.
- `first_token_ts_us - first_scheduled_ts_us` includes prefill and the engine
  iterations needed to produce the first token. `last_token_ts_us -
  first_token_ts_us` is the subsequent decode interval.
- The response-ready/received timestamps split post-decode handling across
  the LLM server, the Gym proxy, transport, and OpenHands. Treat cross-host differences
  as approximate because wall-clock skew can make them slightly negative;
  local duration fields are monotonic and do not have that ambiguity.
- A rollout may span a refit; later turns can use a different cache salt even
  when the rollout remains sticky to the same replica.
- Use lifecycle `target_display_step` and `consumed_display_step` for human
  step labels. Do not infer display steps by adding one unless older data lacks
  the explicit fields and the limitation is documented.
