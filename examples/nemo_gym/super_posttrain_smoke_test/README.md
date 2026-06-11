# Super Posttraining Smoke Test

This directory contains the Super-v3 NeMo-Gym smoke-test configs and launch helpers.

The large training blend JSONL files are intentionally not checked in. To prepare
data locally, run:

```bash
cd examples/nemo_gym/super_posttrain_smoke_test
bash download_prep_data.sh
```

Copy `super.env.example` to `super.env`, fill in the local container, Slurm,
model, cache, W&B, and Hugging Face values, then run the helpers from the
NeMo-RL repository root:

```bash
set -a
source examples/nemo_gym/super_posttrain_smoke_test/super.env
set +a

bash examples/nemo_gym/super_posttrain_smoke_test/prefetch_gym_venvs.sh
bash examples/nemo_gym/super_posttrain_smoke_test/prefetch_ray_venv.sh
bash examples/nemo_gym/super_posttrain_smoke_test/super_launch.sh
```

`super.env` is ignored because it may contain credentials.
