# No args launches on the head node (node 0)
# Args 1-N launch on worker nodes (nodes 1 through N-1)
# Optional: set COMMAND='...' to run non-interactively instead of opening an interactive shell
WORKER_NUM=${1:-}
if [[ -z "$WORKER_NUM" ]]; then
  if [[ -n "${COMMAND:-}" ]]; then
    srun --no-container-mount-home  -A coreai_comparch_trtllm -p gb200 --overlap --container-name=ray-head --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL --nodes=1 --ntasks=1 -w "lyris0051" --jobid 2389648 bash -c "$COMMAND"
  else
    srun --no-container-mount-home  -A coreai_comparch_trtllm -p gb200 --overlap --container-name=ray-head --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL --nodes=1 --ntasks=1 -w "lyris0051" --jobid 2389648 --pty bash
  fi
elif [[ 3 -eq 1 ]]; then
  echo "Error: Single-node mode — only the head node is available. Run without arguments."
  exit 1
else
  # All workers share the container name 'ray-worker'; target a specific one by node.
  # Log files are 0-indexed: worker K maps to ray-worker-(K-1).log
  if [[ $WORKER_NUM -lt 1 || $WORKER_NUM -ge 3 ]]; then
    echo "Error: WORKER_NUM must be between 1 and 2"
    exit 1
  fi
  nodes_array=(lyris0051
lyris0052
lyris0053)
  node="${nodes_array[$WORKER_NUM]}"
  if [[ -n "${COMMAND:-}" ]]; then
    srun --no-container-mount-home  -A coreai_comparch_trtllm -p gb200 --overlap --container-name=ray-worker --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL --nodes=1 --ntasks=1 -w "$node" --jobid 2389648 bash -c "$COMMAND"
  else
    srun --no-container-mount-home  -A coreai_comparch_trtllm -p gb200 --overlap --container-name=ray-worker --container-workdir=/lustre/fsw/coreai_comparch_trtllm/erinh/RL --nodes=1 --ntasks=1 -w "$node" --jobid 2389648 --pty bash
  fi
fi
