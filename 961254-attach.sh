# No args launches on the head node (node 0)
# Args 1-N launch on worker nodes (nodes 1 through N-1)
# Optional: set COMMAND='...' to run non-interactively instead of opening an interactive shell
WORKER_NUM=${1:-}
if [[ -z "$WORKER_NUM" ]]; then
  # Empty means we are on the head node
  if [[ -n "${COMMAND:-}" ]]; then
    srun --no-container-mount-home --gres=gpu:4 -A coreai_dlalgo_llm -p batch --overlap --container-name=ray-head --container-workdir=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl --nodes=1 --ntasks=1 -w "nvl72129-T01" --jobid 961254 bash -c "$COMMAND"
  else
    srun --no-container-mount-home --gres=gpu:4 -A coreai_dlalgo_llm -p batch --overlap --container-name=ray-head --container-workdir=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl --nodes=1 --ntasks=1 -w "nvl72129-T01" --jobid 961254 --pty bash
  fi
else
  # Worker numbers 1 through N-1 correspond to ray-worker-1 through ray-worker-(N-1)
  # and use nodes_array[1] through nodes_array[N-1]
  if [[ $WORKER_NUM -lt 1 || $WORKER_NUM -ge 16 ]]; then
    echo "Error: WORKER_NUM must be between 1 and 15"
    exit 1
  fi
  nodes_array=(nvl72129-T01
nvl72129-T02
nvl72129-T03
nvl72129-T04
nvl72129-T05
nvl72129-T06
nvl72129-T09
nvl72129-T10
nvl72129-T11
nvl72129-T12
nvl72129-T13
nvl72129-T14
nvl72129-T15
nvl72129-T16
nvl72129-T17
nvl72129-T18)
  if [[ -n "${COMMAND:-}" ]]; then
    srun --no-container-mount-home --gres=gpu:4 -A coreai_dlalgo_llm -p batch --overlap --container-name=ray-worker-$WORKER_NUM --container-workdir=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl --nodes=1 --ntasks=1 -w "${nodes_array[$WORKER_NUM]}" --jobid 961254 bash -c "$COMMAND"
  else
    srun --no-container-mount-home --gres=gpu:4 -A coreai_dlalgo_llm -p batch --overlap --container-name=ray-worker-$WORKER_NUM --container-workdir=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl --nodes=1 --ntasks=1 -w "${nodes_array[$WORKER_NUM]}" --jobid 961254 --pty bash
  fi
fi
