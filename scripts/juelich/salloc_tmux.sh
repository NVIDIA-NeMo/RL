#!/bin/bash
# Allocate an accelerated-h100 node for 2 days inside a tmux session.
# Usage: ./salloc_tmux.sh [name]   (default: gpu0)
# Hoard multiple nodes: ./salloc_tmux.sh gpu0 && ./salloc_tmux.sh gpu1
# Jump in:  tmux attach -t gpu0
# Jump out: Ctrl+b, then d
# Once attached, run:  srun --pty bash

SESSION="${1:-gpu0}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already exists. Attaching..."
    exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
    salloc --partition=develbooster \
           --account=envcomp \
           --nodes=1 \
           --gres=gpu:4 \
           --time=02:00:00 \
           --job-name=interactive_booster

echo "Started tmux session: $SESSION"
echo ""
echo "  Attach:  tmux attach -t $SESSION"
echo "  Detach:  Ctrl+b, then d"
echo "  On node: srun --pty bash"
echo "  Kill:    tmux kill-session -t $SESSION"
