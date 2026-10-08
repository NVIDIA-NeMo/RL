# Nemotron3 Super Option B attention A/B

This experiment compares FlashInfer and Triton vLLM attention with the same
Nemotron3 Super MXFP8 training/rollout recipe. Both arms use Option B grouped
GEMM, 32 x 4 GB200 GPUs, Async-1off, GBS 256, NCCL Reshard refit, and 20 steps.
The only intended A/B variable is the attention backend. The rollout uses
TP=4 and EP=4 so the BF16 first/last expert weights have expert-dimension
destination shards supported by the current FlashInfer TRTLLM refit path.

## Lyris reproduction

Use a clean checkout of the pushed branch. The source archive is an immutable
snapshot copied to node-local storage by the launcher. The launcher uses the
Automodel, Gym, and Megatron-Bridge submodules built into the container, not
submodule checkouts from the source archive.

```bash
git clone --branch sna/mxfp8-perf https://github.com/NVIDIA-NeMo/RL.git "$HOME/RL-super-optionb"
cd "$HOME/RL-super-optionb"
export SOURCE_COMMIT="$(git rev-parse HEAD)"
export RESULT_ROOT="/lustre/fsw/coreai_dlalgo_llm/users/$USER/experiments/super-optionb-attention"
export SOURCE_ARCHIVE="$RESULT_ROOT/source-${SOURCE_COMMIT:0:10}.tar"
export CONTAINER=/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/mxfp8_backends_20261007/nemo_rl_nightly_vllm029_20261007_3265087.sqsh
mkdir -p "$RESULT_ROOT"
git archive --format=tar -o "$SOURCE_ARCHIVE" HEAD

# WANDB_API_KEY must already be exported securely; do not put it in the script.
bash experiments/super_option_b_attention_20261007/submit-lyris.sh flashinfer test-only
bash experiments/super_option_b_attention_20261007/submit-lyris.sh triton test-only
bash experiments/super_option_b_attention_20261007/submit-lyris.sh flashinfer
bash experiments/super_option_b_attention_20261007/submit-lyris.sh triton
```

The current launcher assumes the model is already cached under
`/lustre/fsw/coreai_dlalgo_llm/users/$USER/hf_home/hub/models--nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-BF16/`.
The image path above must be readable to the submitting user. `SLURM_ACCOUNT`
can override the default `coreai_dlalgo_llm` account; `RESULT_ROOT` and
`CONTAINER` must point to accessible paths. Each run writes its resolved
config, SLURM/Ray logs, and W&B metadata under `RESULT_ROOT`.

The first pair of runs (jobs `3269219` and `3269231`, commit `5db1b4563`)
failed before step 1 because rollout EP=1 produced an unsupported expert
`Shard(1)` in the BF16 TRTLLM refit path. Commit `56c8479ccb` sets rollout
EP=4 for both arms. The EP=4 20-step rerun is not yet validated, so neither
pair establishes an attention-backend speedup. Compare step 2 onward only
after both runs finish and have finite KL/reward metrics.
