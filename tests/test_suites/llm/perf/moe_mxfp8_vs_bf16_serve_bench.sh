#!/bin/bash
# Run INSIDE the vllm-ultra-rl container (via srun --container-image).
# Starts a vllm serve for one MODE (bf16|mxfp8), waits for health, runs a
# concurrency sweep measuring decode throughput, and records whether the
# flashinfer trtllm-gen MoE path was used.
#
# Env in: MODE, MODEL, RESULTS_DIR, TAG
set -uo pipefail

MODE=${MODE:?set MODE=bf16|mxfp8}
MODEL=${MODEL:?set MODEL=/path/to/checkpoint}
RESULTS_DIR=${RESULTS_DIR:?set RESULTS_DIR}
TAG=${TAG:-$MODE}
PORT=${PORT:-8000}
mkdir -p "$RESULTS_DIR"
SERVE_LOG="$RESULTS_DIR/serve_${TAG}.log"
RESULT_JSON="$RESULTS_DIR/bench_${TAG}.json"

# --- common engine flags (mirror NeMo-RL generation + handoff standalone serve)
COMMON_FLAGS=(
  --tensor-parallel-size 2
  --max-model-len 8192
  --trust-remote-code
  --gpu-memory-utilization 0.9
  --enable-prefix-caching
  --mamba-ssm-cache-dtype float32
  --async-scheduling
  --disable-custom-all-reduce
  --max-num-seqs 512
  --served-model-name nano-$TAG
  --port $PORT
)

export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

if [[ "$MODE" == "mxfp8" ]]; then
  # Engage the MXFP8 flashinfer trtllm-gen MoE fast path (the guarded path).
  export VLLM_USE_FLASHINFER_MOE_FP8=1
  export VLLM_FLASHINFER_MOE_BACKEND=latency
  MODE_FLAGS=( --quantization modelopt --moe-backend flashinfer_trtllm --kv-cache-dtype fp8_e4m3 )
else
  MODE_FLAGS=( --dtype bfloat16 )
fi

echo "[serve] MODE=$MODE MODEL=$MODEL" | tee "$SERVE_LOG"
vllm serve "$MODEL" "${COMMON_FLAGS[@]}" "${MODE_FLAGS[@]}" >> "$SERVE_LOG" 2>&1 &
SERVE_PID=$!

# --- wait for health (cold mxfp8 autotune can take many minutes) ---
echo "[serve] waiting for /health (pid $SERVE_PID) ..."
READY=0
for i in $(seq 1 360); do
  if ! kill -0 $SERVE_PID 2>/dev/null; then echo "[serve] process died early"; break; fi
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null)
  if [[ "$code" == "200" ]]; then READY=1; echo "[serve] healthy after ${i}0s"; break; fi
  sleep 10
done

if [[ "$READY" != "1" ]]; then
  echo "[serve] NEVER BECAME HEALTHY" | tee -a "$SERVE_LOG"
  echo "{\"mode\":\"$MODE\",\"healthy\":false}" > "$RESULT_JSON"
  kill $SERVE_PID 2>/dev/null
  exit 1
fi

# --- benchmark: concurrency sweep, decode throughput (stdlib only) ---
python3 - "$PORT" "$MODE" "$RESULT_JSON" <<'PY'
import sys, json, time, threading, urllib.request

port, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = f"http://localhost:{port}"
model = json.load(urllib.request.urlopen(f"{base}/v1/models"))["data"][0]["id"]
prompt = "word " * 1024            # ~1k-token prompt
DECODE = 512

def one(results, idx):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": DECODE,
        "min_tokens": DECODE,
        "ignore_eos": True,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"{base}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
        toks = r.get("usage", {}).get("completion_tokens", 0)
        results[idx] = (time.perf_counter() - t0, toks, True)
    except Exception as e:
        results[idx] = (time.perf_counter() - t0, 0, False)

def burst(n):
    results = [None] * n
    threads = [threading.Thread(target=one, args=(results, i)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in results if r and r[2]]
    tot = sum(r[1] for r in ok)
    return {"concurrency": n, "n_ok": len(ok), "wall_s": round(wall, 2),
            "decode_tok_s": round(tot / wall, 1) if wall else 0}

# warmup
burst(8)
out_rows = []
for c in [64, 128, 256]:
    row = burst(c)
    print(f"[bench {mode}] c={c}: {row['decode_tok_s']} tok/s ok={row['n_ok']}/{c} wall={row['wall_s']}s", flush=True)
    out_rows.append(row)
json.dump({"mode": mode, "model": model, "healthy": True, "decode_tokens": DECODE,
           "prompt_tokens": 1024, "sweep": out_rows}, open(out, "w"), indent=1)
print("BENCH DONE ->", out)
PY

# --- record whether the MXFP8 flashinfer trtllm-gen MoE path was used ---
echo "[moe-path markers in serve log]"
grep -iE "flashinfer|trtllm|moe.*backend|FusedMoE|modelopt|AUTOTUNE_FALLBACK" "$SERVE_LOG" | sort | uniq -c | sort -rn | head -30

kill $SERVE_PID 2>/dev/null
echo "[serve] done MODE=$MODE"
