# No args launches on the head node (node 0)
# Args 1-N launch on worker nodes (nodes 1 through N-1)
# Optional: set COMMAND='...' to run non-interactively instead of opening an interactive shell
WORKER_NUM=${1:-}
if [[ -z "$WORKER_NUM" ]]; then
  # Empty means we are on the head node
  if [[ -n "${COMMAND:-}" ]]; then
    srun --no-container-mount-home  -A coreai_comparch_trtllm -p gb200 --overlap --container-name=ray-head --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL --nodes=1 --ntasks=1 -w "lyris0253" --jobid 2391922 bash -c "$COMMAND"
  else
    srun --no-container-mount-home  -A coreai_comparch_trtllm -p gb200 --overlap --container-name=ray-head --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL --nodes=1 --ntasks=1 -w "lyris0253" --jobid 2391922 --pty bash
  fi
else
  # Worker numbers 1 through N-1 correspond to ray-worker-1 through ray-worker-(N-1)
  # and use nodes_array[1] through nodes_array[N-1]
  if [[ $WORKER_NUM -lt 1 || $WORKER_NUM -ge 2 ]]; then
    echo "Error: WORKER_NUM must be between 1 and 1"
    exit 1
  fi
  nodes_array=(lyris0253
lyris0254)
  if [[ -n "${COMMAND:-}" ]]; then
    srun --no-container-mount-home  -A coreai_comparch_trtllm -p gb200 --overlap --container-name=ray-worker-$WORKER_NUM --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL --nodes=1 --ntasks=1 -w "${nodes_array[$WORKER_NUM]}" --jobid 2391922 bash -c "$COMMAND"
  else
    srun --no-container-mount-home  -A coreai_comparch_trtllm -p gb200 --overlap --container-name=ray-worker-$WORKER_NUM --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL --nodes=1 --ntasks=1 -w "${nodes_array[$WORKER_NUM]}" --jobid 2391922 --pty bash
  fi
fi
