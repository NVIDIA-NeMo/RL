#!/usr/bin/env python3
"""Sanity check: compare loss from the actual CrossTokenizerDistillationLossFn
in loss_functions.py with the original TokenAligner.compute_loss().

Usage:
    python x_token/sanity_check_loss.py \
        --debug-dir x_token/debug_dump \
        --projection-matrix-path cross_tokenizer_data/projection_map_Llama-3.2_to_Qwen3_multitoken_top_32_double_special.pt \
        --student-model meta-llama/Llama-3.2-1B \
        --teacher-model Qwen/Qwen3-8B-Base
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", type=str, default="/tmp/cross_tok_debug")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--projection-matrix-path", type=str, required=True)
    parser.add_argument("--student-model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--use-sparse", action="store_true", default=False)
    args = parser.parse_args()

    debug_path = os.path.join(args.debug_dir, f"debug_rank{args.rank}.pt")
    print(f"Loading debug tensors from {debug_path}")
    data = torch.load(debug_path, map_location="cpu", weights_only=False)

    student_logits = data["student_logits"]
    teacher_logits = data["teacher_logits"]
    input_ids_student = data["input_ids_student"]
    input_ids_teacher = data["input_ids_teacher"]
    aligned_pairs = data["aligned_pairs"]
    cfg = data["config"]

    print(f"\nShapes:")
    print(f"  student_logits: {student_logits.shape}")
    print(f"  teacher_logits: {teacher_logits.shape}")
    print(f"  input_ids_student: {input_ids_student.shape}")
    print(f"  input_ids_teacher: {input_ids_teacher.shape}")
    print(f"  aligned_pairs: {len(aligned_pairs)} batches, "
          f"{sum(len(ap) for ap in aligned_pairs)} total pairs")
    print(f"\nConfig: {cfg}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student_logits = student_logits.to(device)
    teacher_logits = teacher_logits.to(device)
    input_ids_student = input_ids_student.to(device)
    input_ids_teacher = input_ids_teacher.to(device)

    batch_size = student_logits.shape[0]
    student_seq_len = student_logits.shape[1]

    # ---- Build the TokenAligner (shared by both paths) ----
    from nemo_rl.algorithms.x_token import TokenAligner

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

    temperature = cfg.get("temperature", 1.0)
    vocab_topk = cfg.get("vocab_topk", 8192)
    exact_match = cfg.get("exact_token_match_only", False)
    reverse_kl = cfg.get("reverse_kl", False)

    print(f"\n  temperature={temperature}, vocab_topk={vocab_topk}, "
          f"exact_match={exact_match}, reverse_kl={reverse_kl}")

    # ---- Path A: Original TokenAligner.compute_loss() ----
    print("\n" + "=" * 60)
    print("  Path A: Original TokenAligner.compute_loss()")
    print("=" * 60)

    with torch.no_grad():
        orig_loss, orig_acc = aligner.compute_loss(
            aligned_pairs=aligned_pairs,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            input_ids_student=input_ids_student,
            input_ids_teacher=input_ids_teacher,
            loss_type=cfg.get("loss_type", "KL"),
            exact_token_match_only=exact_match,
            temperature=temperature,
            vocab_topk=vocab_topk,
            reverse_kl=reverse_kl,
        )
    print(f"\n  Original loss: {orig_loss.item():.6f}")
    print(f"  Original topk_acc: {orig_acc:.4f}")

    # ---- Path B: Actual CrossTokenizerDistillationLossFn from loss_functions.py ----
    print("\n" + "=" * 60)
    print("  Path B: CrossTokenizerDistillationLossFn (loss_functions.py)")
    print("=" * 60)

    from nemo_rl.algorithms.loss_functions import CrossTokenizerDistillationLossFn

    loss_fn = CrossTokenizerDistillationLossFn(cfg, aligner)
    loss_fn._debug_dumped = True  # skip re-dumping

    loss_fn.set_cross_tokenizer_data(
        teacher_input_ids=input_ids_teacher,
        aligned_pairs=aligned_pairs,
    )

    # Build the NeMo RL data dict that __call__ expects
    nemo_data = {
        "input_ids": input_ids_student,
        "input_lengths": torch.tensor([student_seq_len] * batch_size, device=device),
        "token_mask": torch.ones(batch_size, student_seq_len, device=device),
        "sample_mask": torch.ones(batch_size, device=device),
    }

    # global_valid_toks/seqs: in real training these are summed across ranks.
    # For single-rank comparison, set them equal to local counts so the
    # distributed scaling becomes: loss * local / global = loss * 1.0
    global_valid_toks = torch.tensor(float(student_seq_len * batch_size), device=device)
    global_valid_seqs = torch.tensor(float(batch_size), device=device)

    with torch.no_grad():
        our_loss, our_metrics = loss_fn(
            next_token_logits=student_logits,
            data=nemo_data,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            teacher_logits=teacher_logits,
            mb_idx=None,
            mbs=None,
        )

    # Undo the distributed scaling to get raw chunk loss:
    # loss_fn does: loss = raw_loss * local_valid_toks / global_valid_toks
    # With token_mask=all-ones: local_valid_toks = (student_seq_len - 1) * batch_size
    # (the -1 is from token_mask[:, 1:max_len+1])
    local_valid = (student_seq_len - 1) * batch_size
    raw_our_loss = our_loss.item() * float(global_valid_toks) / local_valid if local_valid > 0 else 0.0

    print(f"\n  Our loss (after NeMo RL scaling): {our_loss.item():.6f}")
    print(f"  Our loss (raw, before scaling):   {raw_our_loss:.6f}")
    print(f"  Metrics: {our_metrics}")

    # ---- Comparison ----
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    print(f"  Original TokenAligner loss (raw): {orig_loss.item():.6f}")
    print(f"  Our loss (raw, before scaling):   {raw_our_loss:.6f}")
    diff = abs(orig_loss.item() - raw_our_loss)
    print(f"  Absolute difference:              {diff:.6f}")
    if orig_loss.item() > 0:
        print(f"  Relative difference:              {diff / orig_loss.item() * 100:.2f}%")

    if diff < 0.01:
        print("\n  MATCH — losses are essentially identical")
    elif diff < 0.1:
        print("\n  ~ CLOSE — small numerical differences (likely from filtering)")
    else:
        print("\n  MISMATCH — significant difference, investigate further")


if __name__ == "__main__":
    main()
