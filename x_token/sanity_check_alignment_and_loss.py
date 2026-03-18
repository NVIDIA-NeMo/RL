#!/usr/bin/env python3
"""Sanity check: compare alignment + loss between the standalone TokenAligner
pipeline (train_distillation_ddp.py) and the NeMo RL pipeline
(off_policy_distillation.py + loss_functions.py).

Uses the actual code from both pipelines rather than re-implementing them:
  - Path A imports TokenizeAndAlignCollator from tokenalign/src/pytorch_data_loader.py
    and calls TokenAligner.compute_loss() (same as train_distillation_ddp.py)
  - Path B replicates off_policy_distillation.py's decode-reencode + align flow
    and calls CrossTokenizerDistillationLossFn (same as NeMo RL training)

Usage:
    python x_token/sanity_check_alignment_and_loss.py \
        --projection-matrix-path <path_to_projection.pt> \
        --student-model meta-llama/Llama-3.2-1B \
        --teacher-model Qwen/Qwen3-8B-Base

    # With gold loss:
    python x_token/sanity_check_alignment_and_loss.py \
        --projection-matrix-path <path_to_projection.pt> \
        --student-model meta-llama/Llama-3.2-1B \
        --teacher-model Qwen/Qwen3-8B-Base \
        --gold-loss
"""
import argparse
import os
import sys
from unittest.mock import MagicMock

# Stub out heavy dependencies that NeMo RL imports but we don't need.
# Uses a meta-path finder so *any* import under these prefixes is intercepted
# before Python's normal import machinery tries to find them on disk.
import types
import importlib
import importlib.abc
import importlib.machinery

_STUB_PREFIXES = (
    "ray", "vllm", "uvicorn", "tensorstore", "zarr", "torchdata",
    "fastapi", "starlette", "pydantic_settings", "sse_starlette",
    "mlflow", "wandb", "tensorboard",
    "nemo_rl.models.generation.vllm",
    "nemo_rl.models.policy.lm_policy",
    "nemo_rl.models.policy.hf_policy",
    "nemo_rl.distributed.virtual_cluster",
)

class _StubFinder(importlib.abc.MetaPathFinder):
    def find_module(self, fullname, path=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in _STUB_PREFIXES):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _StubModule(fullname)
        sys.modules[fullname] = mod
        return mod

class _StubModule(types.ModuleType):
    """Module stub that returns a MagicMock for any attribute access."""
    def __init__(self, name):
        super().__init__(name)
        self.__spec__ = importlib.machinery.ModuleSpec(name, None)
        self.__path__ = []
        self.__file__ = f"<stub {name}>"
        self.__package__ = name
        self.__loader__ = None
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return MagicMock()

sys.meta_path.insert(0, _StubFinder())

import torch

# Make both codebases importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
TOKENALIGN_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "tokenalign")
sys.path.insert(0, TOKENALIGN_ROOT)
sys.path.insert(0, os.path.join(TOKENALIGN_ROOT, "src"))


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def compare_aligned_pairs(pairs_a, pairs_b, label_a="Path A", label_b="Path B", max_show=10):
    """Print a summary comparison of two sets of aligned pairs."""
    for batch_idx in range(len(pairs_a)):
        n_a = len(pairs_a[batch_idx])
        n_b = len(pairs_b[batch_idx])
        print(f"\n  Batch {batch_idx}: {label_a}={n_a} pairs, {label_b}={n_b} pairs")

        if n_a == n_b:
            diffs = 0
            for i, (pa, pb) in enumerate(zip(pairs_a[batch_idx], pairs_b[batch_idx])):
                if pa[:6] != pb[:6]:
                    diffs += 1
                    if diffs <= max_show:
                        print(f"    Pair {i} differs:")
                        print(f"      {label_a}: {pa[:6]}")
                        print(f"      {label_b}: {pb[:6]}")
            if diffs == 0:
                print(f"    All {n_a} pairs are identical (first 6 fields)")
            elif diffs > max_show:
                print(f"    ... and {diffs - max_show} more differences")
        else:
            print(f"    Pair counts differ -- showing first {max_show} from each:")
            for i, p in enumerate(pairs_a[batch_idx][:max_show]):
                print(f"    {label_a}[{i}]: {p[:6]}")
            for i, p in enumerate(pairs_b[batch_idx][:max_show]):
                print(f"    {label_b}[{i}]: {p[:6]}")


def main():
    parser = argparse.ArgumentParser(description="Cross-tokenizer alignment + loss sanity check")
    parser.add_argument("--projection-matrix-path", type=str, required=True)
    parser.add_argument("--student-model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--use-sparse", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vocab-topk", type=int, default=8192)
    parser.add_argument("--exact-match-only", action="store_true", default=False)
    parser.add_argument("--reverse-kl", action="store_true", default=False)
    parser.add_argument("--gold-loss", action="store_true", default=False)
    parser.add_argument("--xtoken-loss", action="store_true", default=False)
    parser.add_argument("--text", type=str, default=None,
                        help="Sample text to use. If not provided, a default is used.")
    parser.add_argument("--max-seq-len", type=int, default=128,
                        help="Max sequence length for tokenization (ctx_length)")
    parser.add_argument("--debug-dir", type=str, default=None,
                        help="Path to debug dump dir (from CrossTokenKL DEBUG). "
                             "If provided, uses saved logits instead of random.")
    parser.add_argument("--rank", type=int, default=0, help="Rank for debug dump file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    DEFAULT_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "Artificial intelligence is transforming how we build software. "
        "Large language models learn patterns from vast amounts of text data."
    )
    text = args.text or DEFAULT_TEXT

    # ================================================================
    # Setup: Tokenizers + TokenAligner (shared by both paths)
    # ================================================================
    print_header("SETUP")
    from transformers import AutoTokenizer
    from nemo_rl.algorithms.x_token import TokenAligner

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    aligner = TokenAligner(
        teacher_tokenizer_name=args.teacher_model,
        student_tokenizer_name=args.student_model,
        init_hf_tokenizers=True,
    )
    aligner._load_logits_projection_map(
        file_path=args.projection_matrix_path,
        use_sparse_format=args.use_sparse,
        device=device,
    )
    aligner = aligner.to(device)

    print(f"  Student model: {args.student_model}")
    print(f"  Teacher model: {args.teacher_model}")
    print(f"  Projection:    {args.projection_matrix_path}")
    print(f"  Device:        {device}")
    print(f"  Temperature:   {args.temperature}")
    print(f"  Vocab top-k:   {args.vocab_topk}")
    print(f"  Gold loss:     {args.gold_loss}")
    print(f"  XToken loss:   {args.xtoken_loss}")
    print(f"  Reverse KL:    {args.reverse_kl}")

    # ================================================================
    # Load debug dump if provided (real logits from training)
    # ================================================================
    if args.debug_dir:
        print_header("LOADING DEBUG DUMP")
        debug_path = os.path.join(args.debug_dir, f"debug_rank{args.rank}.pt")
        print(f"  Loading from {debug_path}")
        dump = torch.load(debug_path, map_location="cpu", weights_only=False)
        student_logits = dump["student_logits"].to(device)
        teacher_logits = dump["teacher_logits"].to(device)
        print(f"  student_logits: {student_logits.shape}")
        print(f"  teacher_logits: {teacher_logits.shape}")
    else:
        student_logits = None
        teacher_logits = None

    # ================================================================
    # PATH A: train_distillation_ddp.py pipeline
    #
    # Uses TokenizeAndAlignCollator from tokenalign/src/pytorch_data_loader.py
    # which is what TorchDataLoaderXToken uses internally.
    # Then calls TokenAligner.compute_loss() for the loss.
    # ================================================================
    print_header("PATH A: train_distillation_ddp pipeline")
    print("  (TokenizeAndAlignCollator -> TokenAligner.compute_loss)")

    from pytorch_data_loader import TokenizeAndAlignCollator

    # Build the collator exactly as TorchDataLoaderXToken does (line 290-301)
    collator_a = TokenizeAndAlignCollator(
        tokenizer_student=student_tokenizer,
        tokenizer_teacher=teacher_tokenizer,
        token_aligner=aligner,
        ctx_length=args.max_seq_len,
        chunk_size=64,
        same_vocab=False,
        characters_per_sample=None,
        align_convert_to_tokens=True,
        text_key="text",
    )

    # Feed the text as a batch of one sample (same as dataloader would)
    input_ids_student_a, input_ids_teacher_a, aligned_pairs_a = collator_a(
        [{"text": text}]
    )
    input_ids_student_a = input_ids_student_a.to(device)
    input_ids_teacher_a = input_ids_teacher_a.to(device)
    batch_size = input_ids_student_a.shape[0]

    print(f"  Student tokens: {input_ids_student_a.shape}")
    print(f"  Teacher tokens: {input_ids_teacher_a.shape}")
    print(f"  Aligned pairs:  {sum(len(ap) for ap in aligned_pairs_a)} total")

    # Generate or reuse logits
    if student_logits is not None:
        student_logits_a = student_logits
        teacher_logits_a = teacher_logits
    else:
        s_vocab = student_tokenizer.vocab_size
        t_vocab = teacher_tokenizer.vocab_size
        torch.manual_seed(42)
        student_logits_a = torch.randn(
            batch_size, input_ids_student_a.shape[1], s_vocab,
            device=device, dtype=torch.float32,
        )
        teacher_logits_a = torch.randn(
            batch_size, input_ids_teacher_a.shape[1], t_vocab,
            device=device, dtype=torch.float32,
        )

    # Loss: TokenAligner.compute_loss (same call as train_distillation_ddp.py lines 1698-1713)
    with torch.no_grad():
        loss_a, acc_a = aligner.compute_loss(
            aligned_pairs=aligned_pairs_a,
            student_logits=student_logits_a,
            teacher_logits=teacher_logits_a,
            input_ids_student=input_ids_student_a,
            input_ids_teacher=input_ids_teacher_a,
            loss_type="KL",
            exact_token_match_only=args.exact_match_only,
            temperature=args.temperature,
            vocab_topk=args.vocab_topk,
            reverse_kl=args.reverse_kl,
            gold_loss=args.gold_loss,
            xtoken_loss=args.xtoken_loss,
        )

    print(f"\n  TokenAligner.compute_loss result:")
    print(f"    Loss:     {loss_a.item():.6f}")
    print(f"    Top1 Acc: {acc_a:.4f}")

    # ================================================================
    # PATH B: off_policy_distillation.py pipeline
    #
    # Replicates the cross-tokenizer processing from
    # off_policy_distillation.py lines 784-815:
    #   1. Decode student tokens -> text
    #   2. Re-encode with teacher tokenizer
    #   3. Align
    #   4. Call CrossTokenizerDistillationLossFn
    # ================================================================
    print_header("PATH B: off_policy_distillation pipeline")
    print("  (decode-reencode -> CrossTokenizerDistillationLossFn)")

    # --- Step 1-2: off_policy_distillation.py lines 785-806 ---
    # Use student IDs from Path A as the starting point
    # (in real training, these come from the NeMo RL data pipeline)
    student_ids = input_ids_student_a
    batch_size_ct = student_ids.shape[0]

    # off_policy_distillation.py line 788-791
    texts_b = [
        student_tokenizer.decode(student_ids[i].cpu().tolist(), skip_special_tokens=True)
        for i in range(batch_size_ct)
    ]

    # off_policy_distillation.py lines 797-806
    max_teacher_len = args.max_seq_len
    teacher_encoded = teacher_tokenizer(
        texts_b,
        max_length=max_teacher_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids_teacher_b = teacher_encoded["input_ids"].to(device)
    teacher_attention_mask = teacher_encoded["attention_mask"]
    teacher_input_lengths_ct = teacher_attention_mask.sum(dim=1)

    # off_policy_distillation.py lines 808-810
    aligned_pairs_b = aligner.align(
        student_ids, input_ids_teacher_b
    )

    input_ids_student_b = student_ids

    print(f"  Student tokens: {input_ids_student_b.shape}")
    print(f"  Teacher tokens: {input_ids_teacher_b.shape}")
    print(f"  Aligned pairs:  {sum(len(ap) for ap in aligned_pairs_b)} total")

    # Generate logits for Path B (must match teacher seq len)
    if student_logits is not None:
        student_logits_b = student_logits
        teacher_logits_b = teacher_logits
    else:
        torch.manual_seed(42)
        student_logits_b = torch.randn(
            batch_size, input_ids_student_b.shape[1], s_vocab,
            device=device, dtype=torch.float32,
        )
        teacher_logits_b = torch.randn(
            batch_size, input_ids_teacher_b.shape[1], t_vocab,
            device=device, dtype=torch.float32,
        )

    # --- Step 3: Loss via CrossTokenizerDistillationLossFn ---
    # off_policy_distillation.py lines 812-815 + 878-881
    from nemo_rl.algorithms.loss_functions import CrossTokenizerDistillationLossFn

    cfg = {
        "loss_type": "KL",
        "temperature": args.temperature,
        "vocab_topk": args.vocab_topk,
        "exact_token_match_only": args.exact_match_only,
        "reverse_kl": args.reverse_kl,
        "gold_loss": args.gold_loss,
        "xtoken_loss": args.xtoken_loss,
    }
    loss_fn = CrossTokenizerDistillationLossFn(cfg, aligner)
    loss_fn._debug_dumped = True  # skip debug dump

    # off_policy_distillation.py lines 812-815
    loss_fn.set_cross_tokenizer_data(
        teacher_input_ids=input_ids_teacher_b,
        aligned_pairs=aligned_pairs_b,
    )

    student_seq_len_b = student_logits_b.shape[1]
    nemo_data = {
        "input_ids": input_ids_student_b,
        "input_lengths": torch.tensor([student_seq_len_b] * batch_size, device=device),
        "token_mask": torch.ones(batch_size, student_seq_len_b, device=device),
        "sample_mask": torch.ones(batch_size, device=device),
    }
    global_valid_toks = torch.tensor(float(student_seq_len_b * batch_size), device=device)
    global_valid_seqs = torch.tensor(float(batch_size), device=device)

    with torch.no_grad():
        loss_b_scaled, metrics_b = loss_fn(
            next_token_logits=student_logits_b,
            data=nemo_data,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            teacher_logits=teacher_logits_b,
            mb_idx=None,
            mbs=None,
        )

    # Undo NeMo RL distributed scaling to get raw loss
    local_valid = (student_seq_len_b - 1) * batch_size
    raw_loss_b = (
        loss_b_scaled.item() * float(global_valid_toks) / local_valid
        if local_valid > 0 else 0.0
    )

    print(f"\n  CrossTokenizerDistillationLossFn result:")
    print(f"    Loss (NeMo RL scaled): {loss_b_scaled.item():.6f}")
    print(f"    Loss (raw):            {raw_loss_b:.6f}")
    print(f"    Metrics: {metrics_b}")

    # ================================================================
    # PATH B-REF: TokenAligner.compute_loss on Path B's alignment
    #   (isolates loss implementation difference from alignment difference)
    # ================================================================
    print_header("PATH B-REF: TokenAligner.compute_loss on Path B alignment")

    with torch.no_grad():
        loss_b_ref, acc_b_ref = aligner.compute_loss(
            aligned_pairs=aligned_pairs_b,
            student_logits=student_logits_b,
            teacher_logits=teacher_logits_b,
            input_ids_student=input_ids_student_b,
            input_ids_teacher=input_ids_teacher_b,
            loss_type="KL",
            exact_token_match_only=args.exact_match_only,
            temperature=args.temperature,
            vocab_topk=args.vocab_topk,
            reverse_kl=args.reverse_kl,
            gold_loss=args.gold_loss,
            xtoken_loss=args.xtoken_loss,
        )
    print(f"    Loss:     {loss_b_ref.item():.6f}")
    print(f"    Top1 Acc: {acc_b_ref:.4f}")

    # ================================================================
    # ALIGNMENT COMPARISON
    # ================================================================
    print_header("ALIGNMENT COMPARISON: Path A vs Path B")

    teacher_ids_match = torch.equal(input_ids_teacher_a, input_ids_teacher_b)
    print(f"  Teacher token IDs identical: {teacher_ids_match}")
    if not teacher_ids_match:
        for i in range(batch_size):
            mask = input_ids_teacher_a[i] != input_ids_teacher_b[i]
            n_diff = mask.sum().item()
            total = input_ids_teacher_a.shape[1]
            print(f"    Batch {i}: {n_diff}/{total} tokens differ")
            if n_diff > 0 and n_diff <= 20:
                diff_positions = torch.where(mask)[0].tolist()
                for pos in diff_positions[:10]:
                    a_tok = teacher_tokenizer.decode([input_ids_teacher_a[i, pos].item()])
                    b_tok = teacher_tokenizer.decode([input_ids_teacher_b[i, pos].item()])
                    print(f"      pos {pos}: A='{a_tok}' ({input_ids_teacher_a[i, pos].item()}) "
                          f"vs B='{b_tok}' ({input_ids_teacher_b[i, pos].item()})")

    compare_aligned_pairs(
        aligned_pairs_a, aligned_pairs_b,
        label_a="Path A (train_distillation_ddp)",
        label_b="Path B (off_policy_distillation)",
    )

    # ================================================================
    # LOSS COMPARISON
    # ================================================================
    print_header("LOSS COMPARISON")

    loss_a_val = loss_a.item()
    loss_b_ref_val = loss_b_ref.item()

    print(f"  Path A     (TokenAligner.compute_loss, train_distillation_ddp pipeline):")
    print(f"    Loss = {loss_a_val:.6f}, Acc = {acc_a:.4f}")
    print(f"  Path B     (CrossTokenizerDistillationLossFn, off_policy_distillation pipeline):")
    print(f"    Raw loss = {raw_loss_b:.6f}")
    print(f"  Path B-REF (TokenAligner.compute_loss, off_policy_distillation alignment):")
    print(f"    Loss = {loss_b_ref_val:.6f}, Acc = {acc_b_ref:.4f}")

    # --- Key comparison 1: alignment source ---
    print(f"\n  [1] Alignment difference (same loss fn, different tokenization):")
    diff_alignment = abs(loss_a_val - loss_b_ref_val)
    print(f"      |Path A - Path B-REF| = {diff_alignment:.6f}")
    if loss_a_val > 0:
        print(f"      Relative: {diff_alignment / loss_a_val * 100:.2f}%")

    # --- Key comparison 2: loss implementation ---
    print(f"\n  [2] Implementation difference (same alignment, different loss fn):")
    diff_impl = abs(loss_b_ref_val - raw_loss_b)
    print(f"      |Path B-REF - Path B| = {diff_impl:.6f}")
    if loss_b_ref_val > 0:
        print(f"      Relative: {diff_impl / loss_b_ref_val * 100:.2f}%")

    # --- Verdict ---
    print(f"\n  --- Verdict ---")
    if diff_impl < 0.01:
        print(f"  LOSS MATCH -- CrossTokenizerDistillationLossFn matches TokenAligner.compute_loss")
    elif diff_impl < 0.1:
        print(f"  ~ CLOSE -- small numerical differences between loss implementations")
    else:
        print(f"  MISMATCH -- significant loss implementation difference, investigate")

    if diff_alignment > 0.01 and not teacher_ids_match:
        print(f"\n  NOTE: Alignment differs because off_policy_distillation decodes student")
        print(f"  tokens then re-encodes with teacher tokenizer, while train_distillation_ddp")
        print(f"  tokenizes the raw text independently. This is expected.")
    elif diff_alignment < 0.001:
        print(f"\n  ALIGNMENT MATCH -- both pipelines produce same teacher tokens and alignment")


if __name__ == "__main__":
    main()
