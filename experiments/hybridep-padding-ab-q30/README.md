# Qwen3-30B-A3B HybridEP Padding A/B

This experiment compares the latest official stack with MCore PR 5008 against
the historical custom stack, and compares two DeepEP revisions on the official
stack. It does not attribute official-versus-legacy differences to PR 5008
alone because the NeMo-RL, Bridge, and MCore revisions differ. All arms run 20
packed-sequence GRPO steps on the canonical four-node,
eight-GPU-per-node H100 recipe with the same container, model cache, topology,
batch settings, and node class. The official arms use NeMo-RL base
`e496258b0` plus the one-line HybridEP PR 5008 flag mapping at `5b786edf1`,
Bridge `573e088c9`, and MCore `6513e3e23`; the legacy arm uses NeMo-RL
`d833180b9` with Bridge `a68c7c893` and MCore `f812f5b3d`.

| Arm | Dispatcher | DeepEP | Padding path |
|---|---|---|---|
| `official-alltoall` | all-to-all | unused | none |
| `official-pr5008-17cf` | HybridEP | `17cfb817` | MCore PR 5008 per-dispatch padding |
| `official-pr5008-f725` | HybridEP | `f725d296` | MCore PR 5008 per-dispatch padding |
| `legacy-prepad-17cf` | HybridEP | `17cfb817` | one NeMo pre-pad per microbatch; PR 5008 padding disabled |

The official path first takes the maximum local dispatch size across the MoE
communication group, then rounds that maximum up to MCore's 64-token HybridEP
alignment. The historical NeMo path pads the packed microbatch before the
dispatcher and is intentionally retained only as a legacy stack comparison.
`NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128` controls the DeepEP combine API chunk
size; it is not a request to add another 128-token input-padding layer.

`arm_matrix.py` is the authoritative machine-readable contract. The launcher
fails closed on exact source and recursive submodule commits, pushed branch
identity, container SHA, DeepEP wheel metadata and SHA, frozen Python
environment manifest, all eight H100 names, effective MCore configuration,
and the legacy padding contract.

The dated container is paired with a checksum-pinned frozen preflight venv
whose full manifest was validated on H100. The launcher mounts that venv at the
same Lustre path and injects its Python, Transformer Engine, cuDNN, and source
paths into every containerized `srun`. The corresponding intentional
`NRL_IGNORE_VERSION_MISMATCH=1` exception is recorded as
`validated_frozen_preflight_venv`; forced venv rebuilding remains disabled.
The Ray daemon, driver, and bootstrap use that frozen runtime. Megatron and
vLLM actors retain NeMo-RL's specialized `uv`-built actor environments, matching
the previously successful CW execution; the frozen environment intentionally
does not contain vLLM. The base-venv manifest is generated with `PYTHONPATH`
cleared so the 17cf and
f725 DeepEP overlays cannot contaminate its hash; each overlay is attested
separately by its wheel and expanded-tree hashes.

## Render and validate

Rendering is side-effect free and does not require cluster tools or artifacts.

```bash
python3 experiments/hybridep-padding-ab-q30/arm_matrix.py --list --format json
ARM=official-pr5008-17cf RENDER_ONLY=1 \
  bash experiments/hybridep-padding-ab-q30/submit-cw-qwen30-matrix.sh
```

Before every real submission, select `ACCOUNT` from a fresh FairShare check and
set the immutable container and source inputs. HybridEP arms also require the
wheel and build-generated metadata for their selected DeepEP commit.

```bash
export ACCOUNT=<fairshare-selected-account>
export SOURCE_PATH=/lustre/path/to/the-arm-specific-frozen-recursive-checkout
export DEEPEP_17CF_WHEEL=/lustre/path/to/deep_ep-17cf-x86_64.whl
export DEEPEP_17CF_METADATA=/lustre/path/to/17cf-build-metadata.json
export DEEPEP_F725_WHEEL=/lustre/path/to/deep_ep-f725-x86_64.whl
export DEEPEP_F725_METADATA=/lustre/path/to/f725-build-metadata.json
export HF_CACHE=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/hf_home

ARM=official-pr5008-17cf TEST_ONLY=1 \
  bash experiments/hybridep-padding-ab-q30/submit-cw-qwen30-matrix.sh
ARM=official-pr5008-17cf \
  bash experiments/hybridep-padding-ab-q30/submit-cw-qwen30-matrix.sh
```

The launcher submits `ray-nonexclusive.sub`, requests all eight GPUs on every
node with `--gpus-per-node=8`, does not request exclusive access, and does not
set explicit CPU or memory requests. The wrapper passes Slurm's proportional
CPU allocation and a stable job identifier to `ray.sub`. Each arm has an
independent output directory. HybridEP wheels are installed once into a
checksum-keyed immutable overlay on Lustre, mounted into every node, and
import-validated before Ray starts. The matrix records the expanded overlay
size and pins one container path, checksum, and Python
environment manifest hash for every arm,
removes inherited `SBATCH_*` allocation options, rejects non-canonical or
non-Lustre persistent paths, and redirects package/compiler caches away from
`/home`.

## Analysis contract

The primary comparison uses matched Steps 2–20. Steps 5–20 are reported as a
steady-state sensitivity window. Report arithmetic mean phase times and
arithmetic mean of the canonical logged tokens/s/GPU metrics for Policy,
LogProb, generation, and end-to-end; do not reconstruct throughput from
separately averaged counters. Also report included/missing/valid sample counts,
reward, KL, validation accuracy, and failures. Timed runs leave legacy padding
telemetry disabled. Set `COLLECT_PADDING_TELEMETRY=1` only for a separate
legacy diagnostic run so warning I/O cannot bias the performance comparison.
