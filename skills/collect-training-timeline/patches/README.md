# Reference patches: NeMo Gym and OpenHands

Timeline instrumentation for the rollout side, as unified diffs against the
NeMo Gym and nested OpenHands sources baked into one container image:

| Patch | Apply from (`-p1`) | Files |
|---|---|---|
| `nemo-gym-timeline.patch` | `3rdparty/Gym-workspace/Gym` | `nemo_gym/openai_utils.py`, `responses_api_models/vllm_model/app.py`, `responses_api_agents/swe_agents/app.py` |
| `openhands-timeline.patch` | `3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/swe_openhands_setup/OpenHands` | `openhands/agenthub/nemo_gym_client.py`, `openhands/runtime/base.py`, `evaluation/benchmarks/swe_bench/run_infer.py`, new `openhands/utils/nemo_gym_timeline.py` |

They emit the event contract in `../references/event-schema.md`, including the
`llm_*_ts_us` request milestones, `generation_replica_idx`, and the PD
disaggregation `frontend_*` / `ctx_*` fields.

## Baseline

`image-originals.sha256` lists the sha256 of the six pristine files the
patches apply to, as paths under the image's `/opt/nemo-rl`. They were taken
from `nemo-rl-trtllm-disagg-trt95ea967-20261014-aarch64-swe.sqsh`; the hashes,
not the image name, are what identifies a matching baseline.

## Use

Apply directly only when every listed file matches:

```bash
IMG_ROOT=/opt/nemo-rl            # or an unpacked copy of the target image
OVERLAY=/path/to/revision-specific-overlay
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

Then run `../scripts/validate_replica_wiring.py` on the overlay and the smoke
test, as in `../SKILL.md`. For any other revision (a hash mismatch), use the
hunks as a semantic reference and port them per
`../references/porting-guide.md`; never force them on with fuzz or by copying
the patched files.

These patches cover timeline instrumentation only. A deployment may layer
unrelated changes on the same files; for example, NeMo Gym#3582's
`forward_session_id_as_conversation_id` option in
`responses_api_models/vllm_model/app.py` (rank-affine routing for a
disaggregated TRT-LLM server) is not part of `nemo-gym-timeline.patch`.
