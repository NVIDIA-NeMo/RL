"""
Test the exact dataloader pipeline used by run_off_policy_distillation_arrow.py.

Exercises the REAL code path without Ray, GPU, or model loading:
  load_response_dataset -> AllTaskProcessedDataset -> StatefulDataLoader(rl_collate_fn)

Usage (from the RL/ directory):
  python examples/test_dataloader.py                          # all arrow files, batch=4
  python examples/test_dataloader.py --max-files 1            # only 1 arrow file (fast)
  python examples/test_dataloader.py --batch-size 8 --num-batches 5
"""

import argparse
import glob
import time
from functools import partial

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log


# Inlined from run_off_policy_distillation_arrow.py to avoid import issues
def sft_preprocessor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx,
                     add_bos=True, add_eos=True, add_generation_prompt=False,
                     datum_preprocessor=None):
    """Process a datum dictionary for off-policy distillation."""
    if datum_preprocessor is not None:
        datum_dict = datum_preprocessor(datum_dict)

    message_log = get_formatted_message_log(
        datum_dict["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=add_bos,
        add_eos_token=add_eos,
        add_generation_prompt=add_generation_prompt,
        tools=datum_dict.get("tools", None),
    )

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }

ARROW_GLOB = "/lustre/fsw/portfolios/llmservice/users/sdiao/data/climb_nm5.5_phase3_400b_shuffled_text_only_global_shuffle/*.arrow"
MODEL_NAME = "Qwen/Qwen3-1.7B-Base"  # same as config


def parse_args():
    p = argparse.ArgumentParser(description="Test arrow dataloader pipeline")
    p.add_argument("--arrow-glob", type=str, default=ARROW_GLOB,
                   help="Glob pattern for arrow files")
    p.add_argument("--max-files", type=int, default=None,
                   help="Limit number of arrow files to load (None = all)")
    p.add_argument("--model", type=str, default=MODEL_NAME,
                   help="Tokenizer model name")
    p.add_argument("--max-seq-length", type=int, default=8192,
                   help="Max sequence length (matches config)")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for dataloader test")
    p.add_argument("--num-batches", type=int, default=3,
                   help="Number of batches to iterate")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ============================================================
    # 1. Discover arrow files
    # ============================================================
    all_files = sorted(glob.glob(args.arrow_glob))
    if not all_files:
        raise FileNotFoundError(f"No arrow files found at: {args.arrow_glob}")
    if args.max_files:
        all_files = all_files[: args.max_files]
    print(f"[1/5] Found {len(all_files)} arrow file(s)")

    # ============================================================
    # 2. Load tokenizer (CPU only, no model weights)
    # ============================================================
    print(f"\n[2/5] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  vocab_size={tokenizer.vocab_size}, "
          f"bos={tokenizer.bos_token_id}, eos={tokenizer.eos_token_id}")

    # ============================================================
    # 3. Load dataset via the REAL pipeline (load_response_dataset)
    # ============================================================
    # Build the same data_config that off_policy_distillation.yaml produces
    data_config = {
        "dataset_name": "arrow_text",
        "arrow_files": all_files,       # pass resolved list instead of glob
        "val_split": 0.05,
        "text_key": "text",
        "max_input_seq_length": args.max_seq_length,
        "prompt_file": None,
        "system_prompt_file": None,
        "shuffle": True,
        "add_bos": True,
        "add_eos": True,
        "add_generation_prompt": False,
    }

    print(f"\n[3/5] load_response_dataset (dataset_name='arrow_text')...")
    t0 = time.time()
    data = load_response_dataset(data_config, seed=args.seed)
    train_raw = data.formatted_ds["train"]
    val_raw = data.formatted_ds["validation"]
    task_spec = data.task_spec
    elapsed = time.time() - t0
    print(f"  Train: {len(train_raw):,} samples")
    print(f"  Val:   {len(val_raw):,} samples")
    print(f"  Loaded in {elapsed:.1f}s")

    # Quick sanity check on raw data
    sample0 = train_raw[0]
    print(f"\n  Raw sample 0 keys: {list(sample0.keys())}")
    print(f"  messages[0]['role']:    {sample0['messages'][0]['role']}")
    print(f"  messages[0]['content']: {sample0['messages'][0]['content'][:200]}...")

    # ============================================================
    # 4. Wrap with AllTaskProcessedDataset (tokenization + truncation)
    # ============================================================
    print(f"\n[4/5] AllTaskProcessedDataset + sft_preprocessor (max_seq_length={args.max_seq_length})...")
    train_dataset = AllTaskProcessedDataset(
        train_raw,
        tokenizer,
        task_spec,
        partial(
            sft_preprocessor,
            add_bos=data_config["add_bos"],
            add_eos=data_config["add_eos"],
            add_generation_prompt=data_config["add_generation_prompt"],
        ),
        max_seq_length=args.max_seq_length,
    )
    print(f"  Dataset length: {len(train_dataset):,}")

    # Test individual items
    print("\n  --- Individual sample checks ---")
    n_truncated = 0
    for i in range(min(5, len(train_dataset))):
        item = train_dataset[i]
        roles = [m["role"] for m in item["message_log"]]
        n_tokens = sum(len(m["token_ids"]) for m in item["message_log"])
        if item["loss_multiplier"] == 0.0:
            n_truncated += 1
        print(f"  [{i}] length={item['length']:>6}, tokens_after_trunc={n_tokens:>6}, "
              f"loss_mult={item['loss_multiplier']:.1f}, roles={roles}")
    if n_truncated:
        print(f"  WARNING: {n_truncated}/5 samples were truncated (loss_multiplier=0.0). "
              f"Consider increasing --max-seq-length (currently {args.max_seq_length}).")

    # ============================================================
    # 5. StatefulDataLoader + rl_collate_fn  (the real dataloader)
    # ============================================================
    print(f"\n[5/5] StatefulDataLoader (batch_size={args.batch_size}, "
          f"collate_fn=rl_collate_fn, drop_last=True)")
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=rl_collate_fn,
        drop_last=True,
    )
    print(f"  Total batches: {len(dataloader):,}")

    print(f"\n  --- Iterating {args.num_batches} batch(es) ---")
    for bi, batch in enumerate(dataloader):
        if bi >= args.num_batches:
            break

        lengths = batch["length"].tolist()
        loss_mults = batch["loss_multiplier"].tolist()
        max_len = batch["batch_max_length"][0].item()

        print(f"\n  Batch {bi}:")
        print(f"    keys:             {sorted(batch.keys())}")
        print(f"    num_samples:      {len(batch['message_log'])}")
        print(f"    lengths:          {lengths}")
        print(f"    loss_multipliers: {loss_mults}")
        print(f"    batch_max_length: {max_len}")

        # Spot-check first sample
        msg_log_0 = batch["message_log"][0]
        tok0 = msg_log_0[0]["token_ids"]
        print(f"    sample[0] roles:  {[m['role'] for m in msg_log_0]}")
        print(f"    sample[0] tok[:10]: {tok0[:10].tolist()}")
        decoded = tokenizer.decode(tok0[:30], skip_special_tokens=False)
        print(f"    sample[0] decoded[:100]: {decoded[:100]}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("DATALOADER TEST COMPLETE")
    print("=" * 60)
    print(f"  Arrow files:      {len(all_files)}")
    print(f"  Train samples:    {len(train_raw):,}")
    print(f"  Val samples:      {len(val_raw):,}")
    print(f"  Tokenizer:        {args.model}")
    print(f"  Max seq length:   {args.max_seq_length}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Batches iterated: {min(args.num_batches, len(dataloader))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
