# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Micro-benchmark the fastokens BPE tokenizer against stock HuggingFace.

Measures encode/decode throughput of ``transformers`` before and after applying
``fastokens.patch_transformers()`` on a synthetic corpus, and reports the
speedup delta. Also verifies token-for-token equivalence so a "fast but wrong"
result is never reported as a win.

Because ``patch_transformers()`` mutates ``transformers`` globally and
irreversibly, the baseline is always measured first in the same process.

Usage:
    uv run python3 tools/benchmark_fastokens.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --num-samples 512 --approx-tokens 512 --iters 3

Requires the fastokens wheel for your platform
"""

import argparse
import random
import time
from collections.abc import Callable

from transformers import AutoTokenizer, PreTrainedTokenizerBase

# A small, mixed word bank so the synthetic corpus exercises realistic BPE merges
# (common words, punctuation, code-ish tokens, and some multi-byte unicode).
_WORD_BANK = [
    "the",
    "model",
    "reward",
    "policy",
    "gradient",
    "rollout",
    "sequence",
    "token",
    "attention",
    "transformer",
    "learning",
    "sample",
    "batch",
    "def",
    "return",
    "import",
    "self",
    "config",
    "value",
    "loss",
    "step",
    "caf\u00e9",
    "na\u00efve",
    "\u65e5\u672c\u8a9e",
    "\u2211",
    "\u222b",
    "\U0001f680",
    "https://example.com/path?q=1",
    "x_i",
    "y=mx+b",
    "O(n\u00b2)",
    "0xDEADBEEF",
    "This is a longer clause, with commas; and semicolons: plus (parentheses).",
]


def _build_corpus(num_samples: int, approx_tokens: int, seed: int) -> list[str]:
    """Build ``num_samples`` strings of roughly ``approx_tokens`` words each."""
    rng = random.Random(seed)
    # Roughly one word ~= one-to-two tokens; overshoot slightly toward the target.
    words_per_sample = max(1, int(approx_tokens * 0.9))
    return [
        " ".join(rng.choice(_WORD_BANK) for _ in range(words_per_sample))
        for _ in range(num_samples)
    ]


def _time_best(fn: Callable[[], object], iters: int) -> float:
    """Return the best (min) wall-clock seconds over ``iters`` runs."""
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def _measure(
    tokenizer: PreTrainedTokenizerBase,
    corpus: list[str],
    iters: int,
) -> dict[str, float]:
    """Measure encode (per-sample + batch) and decode timings for a tokenizer."""
    # Precompute token ids once for the decode benchmark (not part of decode timing).
    token_ids = [tokenizer(text)["input_ids"] for text in corpus]
    total_tokens = sum(len(ids) for ids in token_ids)

    def encode_loop() -> None:
        for text in corpus:
            tokenizer(text)

    def encode_batch() -> None:
        tokenizer(corpus)

    def decode_loop() -> None:
        tokenizer.batch_decode(token_ids)

    return {
        "total_tokens": float(total_tokens),
        "encode_loop_s": _time_best(encode_loop, iters),
        "encode_batch_s": _time_best(encode_batch, iters),
        "decode_loop_s": _time_best(decode_loop, iters),
    }


def _verify_equivalence(
    baseline: PreTrainedTokenizerBase,
    fast: PreTrainedTokenizerBase,
    corpus: list[str],
) -> bool:
    base_ids = [baseline(text)["input_ids"] for text in corpus]
    fast_ids = [fast(text)["input_ids"] for text in corpus]
    if base_ids != fast_ids:
        return False
    base_decode = [baseline.decode(ids) for ids in base_ids]
    fast_decode = [fast.decode(ids) for ids in fast_ids]
    return base_decode == fast_decode


def _report(label: str, base: float, fast: float, tokens: float) -> None:
    base_tps = tokens / base if base > 0 else float("inf")
    fast_tps = tokens / fast if fast > 0 else float("inf")
    speedup = base / fast if fast > 0 else float("inf")
    print(
        f"  {label:<16} "
        f"hf={base * 1e3:9.2f} ms ({base_tps:12,.0f} tok/s)  "
        f"fast={fast * 1e3:9.2f} ms ({fast_tps:12,.0f} tok/s)  "
        f"speedup={speedup:5.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--approx-tokens", type=int, default=512)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"Corpus: {args.num_samples} samples x ~{args.approx_tokens} tokens "
        f"| model={args.model} | iters={args.iters} (best-of)\n"
    )
    corpus = _build_corpus(args.num_samples, args.approx_tokens, args.seed)

    # Baseline MUST be measured before patching (patch is global + irreversible).
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf = _measure(hf_tokenizer, corpus, args.iters)

    try:
        import fastokens
    except ImportError as exc:
        raise SystemExit("fastokens is not installed for this platform; ") from exc

    fastokens.patch_transformers()
    fast_tokenizer = AutoTokenizer.from_pretrained(args.model)

    if not _verify_equivalence(hf_tokenizer, fast_tokenizer, corpus):
        raise SystemExit(
            "ERROR: fastokens output does NOT match HuggingFace; refusing to "
            "report a speedup for non-equivalent tokenization."
        )
    print("Equivalence check: PASS (token ids and decoded text match)\n")

    fast = _measure(fast_tokenizer, corpus, args.iters)
    tokens = hf["total_tokens"]
    print(f"Total tokens encoded: {tokens:,.0f}\n")
    _report("encode (loop)", hf["encode_loop_s"], fast["encode_loop_s"], tokens)
    _report("encode (batch)", hf["encode_batch_s"], fast["encode_batch_s"], tokens)
    _report("decode (batch)", hf["decode_loop_s"], fast["decode_loop_s"], tokens)


if __name__ == "__main__":
    main()
