# Reference patches: NeMo Gym and OpenHands

Timeline instrumentation for the rollout side, as unified diffs against the
NeMo Gym and nested OpenHands sources baked into one container image:

| Patch | Apply from (`-p1`) | Files |
|---|---|---|
| `nemo-gym-timeline.patch` | `3rdparty/Gym-workspace/Gym` | `nemo_gym/openai_utils.py`, `responses_api_models/vllm_model/app.py`, `responses_api_agents/swe_agents/app.py` |
| `openhands-timeline.patch` | `3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swe_openhands_setup/OpenHands` | `openhands/agenthub/nemo_gym_client.py`, `openhands/runtime/base.py`, `evaluation/benchmarks/swe_bench/run_infer.py`, new `openhands/utils/nemo_gym_timeline.py` |

They write the per-rollout `timeline.jsonl` spans and the per-`model_call`
fields described in `../references/rollout-instrumentation.md`: the
`llm_*_ts_us` request milestones, `generation_replica_idx`, token and prefix
cache counts, and under PD disaggregation the `frontend_*` / `ctx_*` fields.
The NeMo-RL side (`nemo_rl/`) is not part of these patches.

## Baseline

`image-originals.sha256` lists the sha256 of the six pristine files the
patches apply to, as paths under the image's `/opt/nemo-rl`. The patches apply
cleanly to the unmodified files of a NeMo-RL image with the Gym SWE setup baked
in (Gym `17abf2dd`, OpenHands `5f018005`). The hashes, not the image name,
identify a matching baseline.

## Use

Apply directly only when every listed file matches:

```bash
IMG_ROOT=/opt/nemo-rl            # or an unpacked copy of the target image
OVERLAY=/path/to/overlay         # a writable tree to mount over the image
PATCHES=/path/to/skills/collect-training-timeline/patches
(cd "$IMG_ROOT" && sha256sum -c "$PATCHES/image-originals.sha256")
while read -r _ rel; do
  mkdir -p "$OVERLAY/$(dirname "$rel")" && cp "$IMG_ROOT/$rel" "$OVERLAY/$rel"
done < "$PATCHES/image-originals.sha256"
GYM="$OVERLAY/3rdparty/Gym-workspace/Gym"
(cd "$GYM" && patch -p1 --no-backup-if-mismatch < "$PATCHES/nemo-gym-timeline.patch")
(cd "$GYM/responses_api_agents/swe_agents/swe_openhands_setup/OpenHands" \
  && patch -p1 --no-backup-if-mismatch < "$PATCHES/openhands-timeline.patch")
```

`openhands/utils/nemo_gym_timeline.py` is created by the OpenHands patch, so it
has no entry in the checksum file. Mount the patched files over the same paths
in the image, and set `NRL_COLLECT_TIMELINE_FIELDS=1` on every node.

For any other revision (a hash mismatch), use the hunks as a semantic
reference and port them following `../references/rollout-instrumentation.md`;
never force them on with fuzz or by copying the patched files.

These patches cover timeline instrumentation only; a deployment may layer
unrelated changes on the same files.
