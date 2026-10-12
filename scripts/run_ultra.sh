#!/bin/bash

WORK_DIR=$(pwd)
LUSTRE_DIR=/lustre
ACCOUNT=${ACCOUNT:-llmservice_nemotron_ultra}

CONTAINER_IMAGE=${CONTAINER_IMAGE:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/vllm/images/high_stripe/vllm-hsg-nightly.sqsh}
MODEL=${MODEL:-/lustre/fsw/portfolios/llmservice/users/soumyes/sft-runs/eval_and_sleep/ultra-v3-sft-bf16-hybridep-ep64-cp32-bindpcie-recompute-offload-mar13-blend-512k-filt-1e-5/iter_0001000/hf}
REASONING_PARSER=${REASONING_PARSER:-/lustre/fsw/portfolios/llmservice/users/lvega/evals/ultra_v3_reasoning_parser.py}
LOG_FILE=${WORK_DIR}/server_2nodes_$(date +%Y%m%d_%H%M%S).log
TIME=${TIME:-04:00:00}
QOS=${QOS:-normal}
RAY_PORT=6379
VLLM_PORT=8000

export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export WORK_DIR LUSTRE_DIR CONTAINER_IMAGE MODEL LOG_FILE RAY_PORT VLLM_PORT REASONING_PARSER

srun -A ${ACCOUNT} \
    -p batch \
    --time=${TIME} \
    --nodes=2 \
    --qos=${QOS} \
    --switches=1 \
    --ntasks=2 \
    --ntasks-per-node=1 \
    --gres=gpu:4 \
    --no-container-mount-home \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${WORK_DIR}:/workdir,${LUSTRE_DIR}:/lustre \
    --export=ALL \
    --mpi=pmix \
    bash -lc '
    set -euo pipefail
    cd /workdir

    HEAD_IP_FILE="/workdir/.ray_head_ip_${SLURM_JOB_ID}"

    if [ "$SLURM_PROCID" -eq 0 ]; then
        rm -f "$HEAD_IP_FILE"

        HEAD_IP=$(hostname -I | awk "{print \$1}")
        if [ -z "$HEAD_IP" ]; then
            HEAD_IP=$(getent hosts "$(hostname)" | awk "{print \$1; exit}")
        fi

        if [ -z "$HEAD_IP" ]; then
            echo "ERROR: could not determine head node IP"
            exit 1
        fi

        echo "$HEAD_IP" > "$HEAD_IP_FILE"
        echo "=== [rank0] Starting Ray head on ${HEAD_IP}:${RAY_PORT} ==="
        ray start --head --node-ip-address="${HEAD_IP}" --port="${RAY_PORT}" --disable-usage-stats

        echo "=== [rank0] Starting vLLM server (TP=8 across 2 nodes) ==="
        export FLASHINFER_WORKSPACE_BASE=/tmp

        vllm serve "${MODEL}" \
            --trust-remote-code \
            --dtype bfloat16 \
            --kv-cache-dtype fp8 \
            --tensor-parallel-size 8 \
            --max-num-seqs 256 \
            --gpu-memory-utilization 0.95 \
            --enable-prefix-caching \
            --distributed-executor-backend ray \
            --enable-auto-tool-choice \
            --tool-call-parser qwen3_coder \
            --enable-expert-parallel \
            --port "${VLLM_PORT}" \
            --served-model-name model \
            --reasoning-parser-plugin "${REASONING_PARSER}" \
            --reasoning-parser ultra_v3 \
            --compilation-config "{\"pass_config\": {\"fuse_allreduce_rms\": false}}" \
            --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 96}" > "${LOG_FILE}" 2>&1 &

        VLLM_PID=$!

        echo "=== [rank0] Waiting for server readiness (tailing ${LOG_FILE}) ==="
        tail -f "${LOG_FILE}" &
        TAIL_PID=$!

        while ! grep -q "Application startup complete" "${LOG_FILE}" 2>/dev/null; do
            if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                echo "ERROR: vLLM server process died"
                kill "$TAIL_PID" 2>/dev/null || true
                ray stop || true
                rm -f "$HEAD_IP_FILE"
                exit 1
            fi
            sleep 2
        done

        echo ""
        echo "=== Server is ready on http://${HEAD_IP}:${VLLM_PORT} ==="
        echo ""

        wait "$VLLM_PID"
        ray stop || true
        rm -f "$HEAD_IP_FILE"
    else
        for _ in $(seq 1 120); do
            [ -s "$HEAD_IP_FILE" ] && break
            sleep 1
        done

        if [ ! -s "$HEAD_IP_FILE" ]; then
            echo "ERROR: timed out waiting for Ray head IP file: $HEAD_IP_FILE"
            exit 1
        fi

        HEAD_IP=$(cat "$HEAD_IP_FILE")

        echo "=== [rank${SLURM_PROCID}] Waiting for Ray head ${HEAD_IP}:${RAY_PORT} ==="
        for _ in $(seq 1 120); do
            if ray status --address "${HEAD_IP}:${RAY_PORT}" >/dev/null 2>&1; then
                break
            fi
            sleep 2
        done

        echo "=== [rank${SLURM_PROCID}] Starting Ray worker ==="
        ray start --address "${HEAD_IP}:${RAY_PORT}" --disable-usage-stats

        # Keep worker alive while rank0 owns the vLLM process.
        tail -f /dev/null
    fi
    '
