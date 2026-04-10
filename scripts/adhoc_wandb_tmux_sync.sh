#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-wandb-sync-ray}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"  # 15 minutes
RUN_DIRS_CSV="${RUN_DIRS_CSV:-}"

DEFAULT_RUN_DIRS=(
  "/e/project1/scifi/kryeziu1/RL/scripts/logs/348038-ray"
  "/e/project1/scifi/kryeziu1/RL/scripts/logs/348039-ray"
  "/e/project1/scifi/kryeziu1/RL/scripts/logs/348040-ray"
  "/e/project1/scifi/kryeziu1/RL/scripts/logs/348441-ray"
  "/e/project1/scifi/kryeziu1/RL/scripts/logs/348442-ray"
)

if [[ -n "$RUN_DIRS_CSV" ]]; then
  IFS=',' read -r -a RUN_DIRS <<< "$RUN_DIRS_CSV"
else
  RUN_DIRS=("${DEFAULT_RUN_DIRS[@]}")
fi

SCRIPT_PATH="$(realpath "$0")"

if command -v wandb >/dev/null 2>&1; then
  WANDB_BIN="$(command -v wandb)"
elif [[ -x "/e/project1/scifi/kryeziu1/RL/.venv/bin/wandb" ]]; then
  WANDB_BIN="/e/project1/scifi/kryeziu1/RL/.venv/bin/wandb"
else
  echo "ERROR: could not find wandb in PATH or RL/.venv/bin/wandb" >&2
  exit 1
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

normalize_wandb_root() {
  local root="$1"

  if [[ "$root" == *"/wandb/wandb/offline-run-"* || "$root" == *"/wandb/wandb/run-"* ]]; then
    root="${root%/offline-run-*}"
    root="${root%/run-*}"
    if [[ "$root" == *"/wandb/wandb" ]]; then
      root="${root%/wandb}"
    fi
  fi

  if [[ "$root" == *"/wandb/offline-run-"* || "$root" == *"/wandb/run-"* ]]; then
    root="${root%/offline-run-*}"
    root="${root%/run-*}"
  fi

  echo "$root"
}

extract_wandb_roots() {
  local run_dir driver project_root token root

  for run_dir in "${RUN_DIRS[@]}"; do
    driver="$run_dir/ray-driver.log"
    [[ -f "$driver" ]] || continue

    # ray-driver logs live at RL/scripts/logs/<id>-ray, while wandb paths are
    # printed relative to RL/. Walk up to RL to resolve logs/... correctly.
    project_root="$(dirname "$(dirname "$(dirname "$run_dir")")")"

    # Pull any wandb path token from driver log lines (relative logs/ or absolute /...).
    while IFS= read -r token; do
      [[ -n "$token" ]] || continue

      # Normalize offline-run path to the enclosing logs/.../wandb root.
      root="$(normalize_wandb_root "$token")"

      if [[ "$root" == logs/*/wandb ]]; then
        printf '%s\n' "$project_root/$root"
      elif [[ "$root" == /*/wandb ]]; then
        printf '%s\n' "$root"
      fi
    done < <(
      grep -oE '(logs/[^ ]*wandb[^ ]*|/[^ ]*wandb[^ ]*)' "$driver" | sort -u
    )
  done | sort -u
}

sync_once() {
  local wandb_root run_container run_path found_any
  local -a roots

  mapfile -t roots < <(extract_wandb_roots)

  if [[ ${#roots[@]} -eq 0 ]]; then
    log "No wandb roots found in the provided driver logs."
    return 0
  fi

  log "Found ${#roots[@]} wandb root(s). Starting sync pass."

  for wandb_root in "${roots[@]}"; do
    run_container="$wandb_root/wandb"

    if [[ ! -d "$run_container" ]]; then
      log "Skipping missing directory: $run_container"
      continue
    fi

    found_any=0
    while IFS= read -r run_path; do
      found_any=1
      log "Syncing $run_path"
      if ! "$WANDB_BIN" sync --include-offline "$run_path"; then
        log "WARN: sync failed for $run_path"
      fi
    done < <(find "$run_container" -mindepth 1 -maxdepth 1 -type d \( -name 'offline-run-*' -o -name 'run-*' \) | sort)

    if [[ $found_any -eq 0 ]]; then
      log "No run directories under $run_container"
    fi
  done

  log "Sync pass complete."
}

loop_forever() {
  while true; do
    sync_once
    log "Sleeping ${INTERVAL_SECONDS}s before next pass."
    sleep "$INTERVAL_SECONDS"
  done
}

start_session() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    log "tmux session '$SESSION_NAME' already exists."
    return 0
  fi

  tmux new-session -d -s "$SESSION_NAME" \
    "env INTERVAL_SECONDS='$INTERVAL_SECONDS' SESSION_NAME='$SESSION_NAME' RUN_DIRS_CSV='$RUN_DIRS_CSV' '$SCRIPT_PATH' __loop"
  log "Started tmux session '$SESSION_NAME' (interval=${INTERVAL_SECONDS}s)."
  log "Attach with: tmux attach -t $SESSION_NAME"
}

stop_session() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
    log "Stopped tmux session '$SESSION_NAME'."
  else
    log "tmux session '$SESSION_NAME' is not running."
  fi
}

status_session() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    log "tmux session '$SESSION_NAME' is running."
    tmux list-sessions | grep -E "^${SESSION_NAME}:" || true
  else
    log "tmux session '$SESSION_NAME' is not running."
  fi
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") <start|stop|status|once>

Commands:
  start   Start periodic sync in detached tmux session ($SESSION_NAME)
  stop    Stop tmux session
  status  Show whether tmux session is running
  once    Run one sync pass immediately (foreground)

Optional env vars:
  INTERVAL_SECONDS   Sync interval in seconds (default: 900)
  SESSION_NAME       tmux session name (default: wandb-sync-ray)
  RUN_DIRS_CSV       Comma-separated run log dirs to sync (override defaults)
USAGE
}

cmd="${1:-}"
case "$cmd" in
  start)
    start_session
    ;;
  stop)
    stop_session
    ;;
  status)
    status_session
    ;;
  once)
    sync_once
    ;;
  __loop)
    loop_forever
    ;;
  *)
    usage
    exit 1
    ;;
esac
