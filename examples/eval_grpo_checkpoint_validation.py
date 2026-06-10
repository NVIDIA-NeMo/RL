#!/usr/bin/env python3
"""No-Ray math evaluator matching the diffuGRPO validation settings.

This script avoids NeMo-RL's Ray training stack. It reproduces the important
validation ingredients directly:

* openai/gsm8k test split or Nemo Skills AIME 2024/2025 test jsonl
* NeMo-RL math_hf_data_processor-style prompt formatting with examples/prompts/cot.txt
* tokenizer.apply_chat_template(..., add_generation_prompt=True)
* SGLang /generate with input_ids
* FastDiffuser block_size=32, max_steps=32, threshold=0.9,
  selection_policy=confidence, temperature=1.0
* NeMo-RL HFVerifyWorker-equivalent math_verify grading

The raw Megatron checkpoint must first be converted to Hugging Face format for
SGLang. The defaults point at the already-converted step_275 checkpoint.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eval_datasets import load_benchmark_samples, normalize_benchmark


DEFAULT_MODEL = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/"
    "checkpoints/diffugrpo_5k_single_node_g750_trainmb8_lpbs16_20260603_213622_step_275_hf"
)
DEFAULT_SGLANG_REPO = Path("/home/snorouzi/code/sglang-nemotron-dllm-a652eb48")
DEFAULT_SGLANG_COMMIT = "9530f475cdeb4912445aa37fba13b09dcfb9ae6a"
DEFAULT_VENV = Path(
    "/lustre/fsw/portfolios/coreai/users/snorouzi/"
    "sglang_nemotron_torch291_cu129_uvpy312_venv"
)
DEFAULT_PROMPT = Path(
    __file__
).resolve().parent / "prompts" / "cot.txt"
DEFAULT_OUTDIR = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/snorouzi/"
    "eval_results/diffugrpo_step275_gsm8k_training_style_no_ray"
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--server-model-path",
        type=Path,
        default=None,
        help="Optional model path used only to instantiate the SGLang server. Defaults to --model-path.",
    )
    parser.add_argument(
        "--refit-model-path",
        type=Path,
        default=None,
        help="Optional HF checkpoint path loaded into the running SGLang server via /update_weights_from_disk before eval.",
    )
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--benchmark",
        default="gsm8k",
        choices=("gsm8k", "aime24", "aime2024", "aime25", "aime2025"),
        help="Evaluation benchmark to load.",
    )
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--launch-server", action="store_true", default=True)
    parser.add_argument("--no-launch-server", dest="launch_server", action="store_false")
    parser.add_argument("--base-url", default="http://127.0.0.1:32000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=32000)
    parser.add_argument("--sglang-repo", type=Path, default=DEFAULT_SGLANG_REPO)
    parser.add_argument("--sglang-commit", default=DEFAULT_SGLANG_COMMIT)
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--server-random-seed", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)

    parser.add_argument("--max-new-tokens", type=int, default=750)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--concurrent", type=int, default=8)
    parser.add_argument(
        "--generation-api",
        default="generate",
        choices=("generate", "chat_completions"),
        help="SGLang request API to use.",
    )

    parser.add_argument("--dllm-algorithm", default="FastDiffuser", choices=("FastDiffuser", "LinearSpec"))
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--selection-policy", default="confidence")
    parser.add_argument("--causal-context", default="true")

    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mem-fraction-static", default="0.55")
    parser.add_argument("--max-running-requests", default="8")
    parser.add_argument("--max-total-tokens", default="20000")
    parser.add_argument("--attention-backend", default="flashinfer")
    parser.add_argument("--served-model-name", default="default")
    parser.add_argument("--val-batch-size", type=int, default=128)
    parser.add_argument("--generation-batch-size", type=int, default=0)
    parser.add_argument("--shard-dp-size", type=int, default=1)
    parser.add_argument("--shard-rank", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_prompt_template(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def select_validation_shard(
    samples: list[dict[str, str]], val_batch_size: int, dp_size: int, rank: int
) -> list[tuple[int, dict[str, str]]]:
    """Match NeMo-RL validation sharding for one SGLang DP worker.

    NeMo-RL validates in batches of grpo.val_batch_size and calls
    shard_by_batch_size(dp_size, allow_uneven_shards=True) on each batch.
    With no dynamic batching this is contiguous chunking per validation batch.
    """
    if dp_size < 1:
        raise ValueError("--shard-dp-size must be >= 1")
    if rank < 0 or rank >= dp_size:
        raise ValueError("--shard-rank must be in [0, --shard-dp-size)")
    if val_batch_size < 1:
        raise ValueError("--val-batch-size must be >= 1")

    indexed = list(enumerate(samples))
    if dp_size == 1:
        return indexed

    selected: list[tuple[int, dict[str, str]]] = []
    for batch_start in range(0, len(indexed), val_batch_size):
        batch = indexed[batch_start : batch_start + val_batch_size]
        shard_size = (len(batch) + dp_size - 1) // dp_size
        start = rank * shard_size
        end = min(start + shard_size, len(batch))
        selected.extend(batch[start:end])
    return selected


def make_prompt_ids(
    tokenizer: Any, prompt_template: str, question: str
) -> tuple[str, list[int]]:
    formatted_content = prompt_template.format(question)
    message = tokenizer.apply_chat_template(
        [{"role": "user", "content": formatted_content}],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    token_ids = tokenizer(
        message,
        return_tensors=None,
        add_special_tokens=False,
    )["input_ids"]
    return message, list(token_ids)


def check_client_dependencies() -> None:
    # Fail before launching SGLang if this client Python cannot reproduce NeMo-RL scoring.
    import requests  # noqa: F401
    import transformers  # noqa: F401
    from datasets import load_dataset  # noqa: F401
    from math_verify.errors import TimeoutException  # noqa: F401
    from math_verify.metric import math_metric  # noqa: F401
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # noqa: F401


def build_verifier():
    # No-Ray equivalent of nemo_rl.environments.math_environment.HFVerifyWorker.
    # HFVerifyWorker is a Ray actor; keep the standalone evaluator on the same
    # math_verify contract without instantiating Ray.
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

    return math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )


def score_response(verify_func: Any, response: str, gold: str) -> tuple[float, Any]:
    from math_verify.errors import TimeoutException

    try:
        gold_parsable = "\\boxed{" + gold + "}"
        score, extracted = verify_func([gold_parsable], [response])
        return float(score), extracted
    except (Exception, TimeoutException) as e:
        return 0.0, {"error": f"{type(e).__name__}: {e!r}"}


def write_dllm_config(args: argparse.Namespace) -> Path:
    args.outdir.mkdir(parents=True, exist_ok=True)
    config_path = args.outdir / "dllm_config.yaml"
    config = {
        "algorithm": args.dllm_algorithm,
        "causal_context": args.causal_context,
        "block_size": args.block_size,
        "max_steps": args.max_steps,
        "threshold": args.threshold,
        "stats_file": str(args.outdir / "stats.jsonl"),
    }
    if args.dllm_algorithm == "FastDiffuser":
        config["selection_policy"] = args.selection_policy
        config["temperature"] = args.temperature
    with open(config_path, "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    return config_path


def check_sglang_commit(repo: Path, expected: str) -> None:
    if not expected:
        return
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual != expected:
        raise RuntimeError(
            f"SGLang commit mismatch: expected {expected}, got {actual} in {repo}"
        )


def server_command(args: argparse.Namespace, dllm_config: Path) -> list[str]:
    py = args.venv / "bin" / "python"
    cmd = [
        str(py),
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(args.server_model_path or args.model_path),
        "--tokenizer-path",
        str(args.tokenizer_path or args.model_path),
        "--served-model-name",
        args.served_model_name,
        "--trust-remote-code",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tp-size",
        "1",
        "--dtype",
        args.dtype,
        "--context-length",
        str(args.context_length),
        "--allow-auto-truncate",
        "--skip-server-warmup",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(args.max_running_requests),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--attention-backend",
        args.attention_backend,
        "--json-model-override-args",
        "{}",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--dllm-algorithm",
        args.dllm_algorithm,
        "--dllm-algorithm-config",
        str(dllm_config),
    ]
    if args.server_random_seed is not None:
        cmd.extend(["--random-seed", str(args.server_random_seed)])
    return cmd


def launch_server(args: argparse.Namespace, dllm_config: Path) -> subprocess.Popen:
    check_sglang_commit(args.sglang_repo, args.sglang_commit)
    log_path = args.outdir / "server.log"
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/lustre/fsw/portfolios/coreai/users/snorouzi/hf_home")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", "1")
    env["PYTHONPATH"] = f"{args.sglang_repo}/python"
    env["DLLM_TEMPERATURE"] = str(args.temperature)
    env["SELECTION_POLICY"] = args.selection_policy
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = server_command(args, dllm_config)
    with open(args.outdir / "server_command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(json.dumps(x) for x in cmd) + "\n")
    server_log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=server_log, stderr=subprocess.STDOUT, env=env)
    return proc


def wait_for_server(base_url: str, proc: subprocess.Popen | None) -> None:
    import requests

    for i in range(600):
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {proc.returncode}")
        for path in ("/health_generate", "/health"):
            try:
                if requests.get(base_url + path, timeout=5).status_code == 200:
                    print(f"SERVER_READY after {i + 1}s", flush=True)
                    return
            except requests.RequestException:
                pass
        time.sleep(1)
    raise TimeoutError(f"SGLang server did not become ready at {base_url}")


def update_weights_from_disk(base_url: str, model_path: Path) -> None:
    import requests

    payload = {"model_path": str(model_path), "flush_cache": True}
    print(f"Refitting SGLang weights from disk: {model_path}", flush=True)
    resp = requests.post(
        base_url + "/update_weights_from_disk",
        json=payload,
        timeout=600,
    )
    try:
        body = resp.json()
    except Exception:
        body = {"raw": resp.text}
    if resp.status_code != 200 or not body.get("success", False):
        raise RuntimeError(
            f"/update_weights_from_disk failed: status={resp.status_code}, body={body}"
        )
    print(f"REFIT_READY: {body}", flush=True)


def extract_generated_token_ids(result: dict[str, Any]) -> list[int]:
    meta = result.get("meta_info", {})
    output_token_logprobs = meta.get("output_token_logprobs", [])
    if output_token_logprobs:
        token_ids = []
        for item in output_token_logprobs:
            if isinstance(item, dict):
                token_id = item.get("token_id", item.get("id"))
            else:
                token_id = item[1]
            token_ids.append(int(token_id))
        return token_ids
    return [int(x) for x in result.get("output_ids", meta.get("output_ids", []))]


def generate_one(
    base_url: str,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    context_length: int,
) -> dict[str, Any]:
    import requests

    final_max_tokens = min(max_new_tokens, max(0, context_length - len(input_ids) - 1))
    sampling_params: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": final_max_tokens,
    }
    if top_k != -1:
        sampling_params["top_k"] = top_k
    payload = {
        "input_ids": input_ids[: max(0, context_length - 1)],
        "sampling_params": sampling_params,
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    resp = requests.post(base_url + "/generate", json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()
    result["_requested_max_new_tokens"] = final_max_tokens
    return result


def generate_one_chat_completions(
    base_url: str,
    prompt_template: str,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    args: argparse.Namespace,
    benchmark_name: str,
) -> dict[str, Any]:
    import requests

    content = prompt_template.format(question)
    payload: dict[str, Any] = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "steps": args.max_steps,
        "block_length": args.block_size,
        "threshold": args.threshold,
        "benchmark_name": benchmark_name,
    }
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json=payload,
        timeout=14400,
    )
    resp.raise_for_status()
    result = resp.json()
    choice = result["choices"][0]
    message = choice.get("message") or {}
    response = message.get("content")
    if response is None:
        response = choice.get("text", "")
    return {
        "response": response,
        "raw_response": result,
        "_requested_max_new_tokens": max_new_tokens,
        "prompt": content,
        "finish_reason": choice.get("finish_reason"),
    }


def main() -> None:
    args = parse_args()
    args.benchmark = normalize_benchmark(args.benchmark)
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.base_url = args.base_url.rstrip("/")

    if not args.model_path.is_dir():
        raise FileNotFoundError(
            f"model-path must be a converted Hugging Face checkpoint: {args.model_path}"
        )

    if not args.dry_run:
        check_client_dependencies()

    dllm_config = write_dllm_config(args)
    print(f"DLLM config: {dllm_config}")

    server_proc = None
    try:
        if args.launch_server:
            cmd = server_command(args, dllm_config)
            print("Server command:")
            print(" ".join(json.dumps(x) for x in cmd))
            if args.dry_run:
                return
            server_proc = launch_server(args, dllm_config)
        elif args.dry_run:
            return
        wait_for_server(args.base_url, server_proc)
        if args.refit_model_path is not None:
            if not args.refit_model_path.is_dir():
                raise FileNotFoundError(
                    f"refit-model-path must be a converted Hugging Face checkpoint: {args.refit_model_path}"
                )
            update_weights_from_disk(args.base_url, args.refit_model_path)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(args.tokenizer_path or args.model_path),
            trust_remote_code=True,
        )
        prompt_template = load_prompt_template(args.prompt_file)
        samples = load_benchmark_samples(args.benchmark, args.num_samples, args.seed)
        indexed_samples = select_validation_shard(
            samples,
            val_batch_size=args.val_batch_size,
            dp_size=args.shard_dp_size,
            rank=args.shard_rank,
        )
        verify_func = build_verifier()

        print(f"Loaded {len(samples)} {args.benchmark} samples")
        if args.shard_dp_size != 1:
            print(
                f"Validation shard: rank={args.shard_rank}/{args.shard_dp_size}, "
                f"val_batch_size={args.val_batch_size}, samples={len(indexed_samples)}"
            )
        print(
            "Generation settings: "
            f"max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, "
            f"top_p={args.top_p}, top_k={args.top_k}, context_length={args.context_length}, "
            f"generation_api={args.generation_api}"
        )

        records: list[dict[str, Any]] = [None] * len(indexed_samples)  # type: ignore
        started = time.time()

        def work(i: int, original_idx: int, sample: dict[str, str]) -> tuple[int, dict[str, Any]]:
            prompt, input_ids = make_prompt_ids(tokenizer, prompt_template, sample["question"])
            output_ids: list[int] = []
            if args.generation_api == "chat_completions":
                result = generate_one_chat_completions(
                    args.base_url,
                    prompt_template,
                    sample["question"],
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    args,
                    args.benchmark,
                )
                prompt = result["prompt"]
                response = result["response"]
            else:
                result = generate_one(
                    args.base_url,
                    input_ids,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    args.context_length,
                )
                output_ids = extract_generated_token_ids(result)
                response = tokenizer.decode(output_ids, skip_special_tokens=True)
            return i, {
                "idx": i,
                "original_idx": original_idx,
                "source_id": sample.get("source_id"),
                "question": sample["question"],
                "gold": sample["gold"],
                "prompt": prompt,
                "prompt_len": len(input_ids),
                "requested_max_new_tokens": result.get("_requested_max_new_tokens"),
                "output_ids": output_ids,
                "nfe": result.get("nfe"),
                "finish_reason": result.get("finish_reason"),
                "response": response,
            }

        done = 0
        generation_batch_size = args.generation_batch_size or len(indexed_samples)
        if generation_batch_size < 1:
            raise ValueError("--generation-batch-size must be >= 1 when set")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as pool:
            for batch_start in range(0, len(indexed_samples), generation_batch_size):
                batch = indexed_samples[batch_start : batch_start + generation_batch_size]
                futures = [
                    pool.submit(work, batch_start + j, original_idx, sample)
                    for j, (original_idx, sample) in enumerate(batch)
                ]
                for fut in concurrent.futures.as_completed(futures):
                    i, record = fut.result()
                    score, extracted = score_response(verify_func, record["response"], record["gold"])
                    record["reward"] = score
                    record["extracted"] = repr(extracted)
                    records[i] = record
                    done += 1
                    if done % 100 == 0 or done == len(indexed_samples):
                        print(f"  [{done}/{len(indexed_samples)}] evaluated", flush=True)

        correct = sum(1 for r in records if r["reward"] == 1.0)
        total = len(records)
        avg_len = sum(len(r["output_ids"]) for r in records) / max(total, 1)
        elapsed = time.time() - started
        metrics = {
            "benchmark": args.benchmark,
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "avg_generation_tokens": avg_len,
            "elapsed_seconds": elapsed,
            "samples_per_second": total / elapsed if elapsed else 0.0,
            "settings": vars(args),
        }

        with open(args.outdir / "records.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(args.outdir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        print(
            f"{args.benchmark} Accuracy: {correct}/{total} = {100 * metrics['accuracy']:.4f}%"
        )
        print(f"Average generation length: {avg_len:.1f} tokens")
        print(f"Metrics: {args.outdir / 'metrics.json'}")
        print(f"Records: {args.outdir / 'records.jsonl'}")
    finally:
        if server_proc is not None and server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
