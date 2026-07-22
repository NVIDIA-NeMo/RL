# Dataset / checkpoint preparation

The benchmark consumes four inputs (host paths set in
an external site data config, mounted by `../config_mounts.sh`):

| Input | Env var | Produced by |
|---|---|---|
| Qwen3.5-397B-A17B HF snapshot (policy init + tokenizer) | `HF_CKPT_PATH` | `download_base_ckpt.sh` |
| Megatron checkpoint cache (HF → mcore, converted once) | `NRL_MEGATRON_CHECKPOINT_DIR` | `convert_ckpt.sh` |
| R2E-Gym SWE instance SIF images (977 instances) | `NEMO_GYM_SWE_SIF_DIR` | `build_r2e_gym_sif.sh` |
| Train/val jsonl subsets (Benchmark-R2E-Gym-Easy-Large) | `NEMO_GYM_SWE_{TRAIN,VALIDATION}_DATA_PATH` | `create_subset.sh` |

All four are produced once and hosted on the cluster; runs mount them
read-only (except the mcore cache, which is written on first conversion).

## SIF images (`build_r2e_gym_sif.sh`)

Two-step flow driven by `dataset-processing-container/` (its own Dockerfile):

1. `run-r2e-gym-build-images.sh` — builds arm64 docker images for every
   instance id in `dataset-processing-container/r2e-gym-instances-to-build.txt`
   and pushes them to `$DOCKER_REGISTRY/r2e-gym` (needs `DOCKER_USER`/
   `DOCKER_TOKEN`).
2. `run-build-sif-images.sh` — pulls those images and converts each to an
   apptainer SIF under `$SIF_DIR`.

Both run inside the dataset-processing container (see its `entrypoint.sh`);
sizes are large (hundreds of GB total) — build on a node with docker + fast
scratch, output directly to the lustre target.

## jsonl subsets (`create_subset.sh`)

`create_r2e_gym_easy_subset_jsonl.py` streams the R2E-Gym-Subset parquet
shards and emits train/val jsonl filtered to the committed id lists
(`instances_train.txt` / `instances_val.txt`) with the container_formatter
prefix pointing at the SIF dir.

## Checkpoint (`download_base_ckpt.sh`, `convert_ckpt.sh`)

The HF snapshot is downloaded into an HF-hub cache layout (config_mounts.sh
mounts the `models--*` cache root so blob symlinks resolve). The mcore cache
is populated by nemo-rl itself on the first training run against an empty
`NRL_MEGATRON_CHECKPOINT_DIR`; `convert_ckpt.sh` runs the reference config
for one step to pre-pay that conversion so nightly runs are deterministic.
