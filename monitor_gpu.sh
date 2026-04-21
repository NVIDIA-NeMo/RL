#!/bin/bash
# Monitor GPU utilization across all nodes in a SLURM job allocation.
# Queries each node via srun --overlap --jobid.
#
# Usage:
#   bash monitor_gpu.sh [JOB_ID]              # one-shot snapshot
#   bash monitor_gpu.sh [JOB_ID] -w 10        # refresh every 10s
#   bash monitor_gpu.sh [JOB_ID] -w 10 -v     # with per-GPU process info

set -uo pipefail

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS] [JOB_ID]

Monitor GPU utilization across all nodes in a SLURM job.

Arguments:
  JOB_ID              SLURM job ID (auto-detects most recent running job if omitted)

Options:
  -w, --watch SEC     Refresh every SEC seconds (default: one-shot)
  -v, --verbose       Show per-GPU process info
  -h, --help          Show this help message

Examples:
  $(basename "$0")                  # auto-detect job, one-shot
  $(basename "$0") 9684502          # specific job, one-shot
  $(basename "$0") -w 15            # auto-detect job, refresh every 15s
  $(basename "$0") 9684502 -w 10 -v # specific job, watch mode, verbose
EOF
  exit 0
}

JOB_ID=""
WATCH_INTERVAL=""
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--watch)   WATCH_INTERVAL="$2"; shift 2 ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help)    usage ;;
    -*)           echo "Unknown option: $1"; usage ;;
    *)            JOB_ID="$1"; shift ;;
  esac
done

# ── Auto-detect job ID ───────────────────────────────────────────────
if [[ -z "$JOB_ID" ]]; then
  JOB_ID=$(squeue --me --states=RUNNING --sort=-S --format="%i" --noheader 2>/dev/null | head -1 | tr -d ' ')
  if [[ -z "$JOB_ID" ]]; then
    echo "Error: No running SLURM jobs found. Provide a job ID explicitly."
    exit 1
  fi
  echo -e "${DIM}Auto-detected running job: ${JOB_ID}${NC}"
fi

# ── Extract job metadata ─────────────────────────────────────────────
JOB_INFO=$(scontrol show job "$JOB_ID" 2>&1)
if echo "$JOB_INFO" | grep -q "Invalid job id"; then
  echo "Error: Job $JOB_ID not found."
  exit 1
fi

JOB_STATE=$(echo "$JOB_INFO" | grep -oP 'JobState=\K\S+')
if [[ "$JOB_STATE" != "RUNNING" ]]; then
  echo "Error: Job $JOB_ID is not running (state: $JOB_STATE)."
  exit 1
fi

JOB_NAME=$(echo "$JOB_INFO" | grep -oP 'JobName=\K\S+')
NUM_NODES=$(echo "$JOB_INFO" | grep -oP 'NumNodes=\K\d+')
NODE_LIST=$(echo "$JOB_INFO" | grep -oP '(?<![A-Za-z])NodeList=\K\S+')

NODES_EXPANDED=$(scontrol show hostnames "$NODE_LIST")
NODES_ARRAY=($NODES_EXPANDED)

# ── nvidia-smi queries ───────────────────────────────────────────────
QUERY_CSV="nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits"
QUERY_PROCS="nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>/dev/null || true"

# ── Helpers ───────────────────────────────────────────────────────────
colorize_util() {
  local val=$1
  if [[ $val -lt 30 ]]; then
    printf "${GREEN}%3d%%${NC}" "$val"
  elif [[ $val -lt 70 ]]; then
    printf "${YELLOW}%3d%%${NC}" "$val"
  else
    printf "${RED}%3d%%${NC}" "$val"
  fi
}

run_on_node() {
  local idx=$1 node=$2 outfile=$3

  local cmd="$QUERY_CSV"
  if [[ $VERBOSE -eq 1 ]]; then
    cmd="${cmd}; echo '---PROCS---'; ${QUERY_PROCS}"
  fi

  srun --overlap --jobid "$JOB_ID" \
    --nodes=1 --ntasks=1 -w "$node" \
    bash -c "$cmd" \
    > "$outfile" 2>&1
}

# ── Main query + display ─────────────────────────────────────────────
query_all_nodes() {
  local tmpdir
  tmpdir=$(mktemp -d)
  local pids=()

  for ((i = 0; i < ${#NODES_ARRAY[@]}; i++)); do
    run_on_node "$i" "${NODES_ARRAY[$i]}" "$tmpdir/$i" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  # Accumulators
  local total_gpus=0 total_util=0 total_mem_used=0 total_mem_total=0 idle_gpus=0
  local timestamp
  timestamp=$(date '+%Y-%m-%d %H:%M:%S')

  echo ""
  echo -e "${BOLD}GPU Utilization — Job ${JOB_ID} (${JOB_NAME}) — ${timestamp}${NC}"
  echo -e "${DIM}Nodes: ${NUM_NODES} | Node list: ${NODE_LIST}${NC}"
  echo ""

  local hdr
  hdr=$(printf "${BOLD}%-5s  %-20s  %3s  %-22s  %5s  %-22s  %5s  %6s${NC}" \
    "Node" "Hostname" "GPU" "Model" "Util" "Memory" "Temp" "Power")
  echo -e "$hdr"
  printf '%.0s─' {1..94}; echo ""

  for ((i = 0; i < ${#NODES_ARRAY[@]}; i++)); do
    local node="${NODES_ARRAY[$i]}"
    local outfile="$tmpdir/$i"

    if [[ ! -s "$outfile" ]]; then
      printf "%-5s  %-20s  ${RED}%s${NC}\n" "$i" "$node" "[ERROR] No response"
      continue
    fi

    # Separate GPU CSV from process info (if verbose), filtering noise lines
    local gpu_csv proc_csv=""
    if [[ $VERBOSE -eq 1 ]] && grep -q -- '---PROCS---' "$outfile" 2>/dev/null; then
      gpu_csv=$(sed '/---PROCS---/,$d' "$outfile" | grep ',' || true)
      proc_csv=$(sed '1,/---PROCS---/d' "$outfile")
    else
      gpu_csv=$(grep ',' "$outfile" || true)
    fi

    # Check for error in output
    if ! echo "$gpu_csv" | grep -q ","; then
      printf "%-5s  %-20s  ${RED}%s${NC}\n" "$i" "$node" "[ERROR] $(head -1 "$outfile" | cut -c1-60)"
      continue
    fi

    local first_line=1
    while IFS=',' read -r gpu_idx gpu_name gpu_util mem_used mem_total temp power; do
      gpu_idx=$(echo "$gpu_idx" | xargs)
      gpu_name=$(echo "$gpu_name" | xargs)
      gpu_util=$(echo "$gpu_util" | xargs)
      mem_used=$(echo "$mem_used" | xargs)
      mem_total=$(echo "$mem_total" | xargs)
      temp=$(echo "$temp" | xargs)
      power=$(echo "$power" | xargs)

      if ! [[ "$gpu_util" =~ ^[0-9]+$ ]]; then
        continue
      fi

      total_gpus=$((total_gpus + 1))
      total_util=$((total_util + gpu_util))
      total_mem_used=$((total_mem_used + mem_used))
      total_mem_total=$((total_mem_total + mem_total))
      if [[ $gpu_util -eq 0 ]]; then
        idle_gpus=$((idle_gpus + 1))
      fi

      local util_colored
      util_colored=$(colorize_util "$gpu_util")

      local mem_pct=0
      if [[ $mem_total -gt 0 ]]; then
        mem_pct=$((mem_used * 100 / mem_total))
      fi
      local mem_str
      mem_str=$(printf "%d/%d MiB (%d%%)" "$mem_used" "$mem_total" "$mem_pct")

      local node_label="" hostname_label=""
      if [[ $first_line -eq 1 ]]; then
        node_label="$i"
        hostname_label="$node"
        first_line=0
      fi

      printf "%-5s  %-20s  %3s  %-22s  %b  %-22s  %4sC  %5sW\n" \
        "$node_label" "$hostname_label" "$gpu_idx" "$gpu_name" "$util_colored" "$mem_str" "$temp" "$power"
    done <<< "$gpu_csv"

    # Verbose: show processes under this node
    if [[ $VERBOSE -eq 1 ]] && [[ -n "$proc_csv" ]]; then
      while IFS=',' read -r _uuid pid pname pmem; do
        pid=$(echo "$pid" | xargs)
        pname=$(echo "$pname" | xargs)
        pmem=$(echo "$pmem" | xargs)
        if [[ -n "$pid" ]] && [[ "$pid" != "0" ]]; then
          printf "${DIM}       └─ PID %-8s  %-30s  %s MiB${NC}\n" "$pid" "$pname" "$pmem"
        fi
      done <<< "$proc_csv"
    fi

    printf "${DIM}%.0s·${NC}" {1..94}; echo ""
  done

  # ── Aggregate summary ──────────────────────────────────────────────
  echo ""
  printf '%.0s═' {1..94}; echo ""
  if [[ $total_gpus -gt 0 ]]; then
    local avg_util=$((total_util / total_gpus))
    local avg_util_colored
    avg_util_colored=$(colorize_util "$avg_util")
    local active_gpus=$((total_gpus - idle_gpus))
    local total_mem_gib=$((total_mem_total / 1024))
    local used_mem_gib=$((total_mem_used / 1024))

    echo -e "${BOLD}Summary${NC}"
    echo -e "  Total GPUs:    ${CYAN}${total_gpus}${NC}"
    echo -e "  Active GPUs:   ${CYAN}${active_gpus}${NC}  |  Idle GPUs: ${CYAN}${idle_gpus}${NC}"
    printf "  Avg Util:      %b\n" "$avg_util_colored"
    echo -e "  Memory:        ${CYAN}${used_mem_gib} / ${total_mem_gib} GiB${NC}  (${total_mem_used} / ${total_mem_total} MiB)"
  else
    echo -e "${RED}No GPU data collected from any node.${NC}"
  fi
  echo ""

  rm -rf "$tmpdir"
}

# ── Entry point ───────────────────────────────────────────────────────
if [[ -n "$WATCH_INTERVAL" ]]; then
  while true; do
    clear
    query_all_nodes
    echo -e "${DIM}Refreshing every ${WATCH_INTERVAL}s — Ctrl+C to stop${NC}"
    sleep "$WATCH_INTERVAL"
  done
else
  query_all_nodes
fi
