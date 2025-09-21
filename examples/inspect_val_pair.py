#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import hf_datasets
from nemo_rl.data.datasets import AllTaskProcessedDataset, preference_collate_fn
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.utils.config import load_config, parse_hydra_overrides


def import_rm_preprocessor():
    try:
        from examples.run_rm import rm_preprocessor  # type: ignore
        return rm_preprocessor
    except Exception:
        import importlib.util, sys
        rm_path = str(Path(__file__).parent / "run_rm.py")
        spec = importlib.util.spec_from_file_location("_rm_module_inspect", rm_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_rm_module_inspect"] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, "rm_preprocessor")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "rm.yaml"))
    p.add_argument("--pair-index", type=int, required=True)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = parse_hydra_overrides(cfg, args.overrides)

    tokenizer = get_tokenizer(cfg["policy"]["tokenizer"])  # honor chat_template

    # Build processed dataset via rm_preprocessor
    ds = hf_datasets.HelpSteer3Dataset()
    task_spec: TaskDataSpec = ds.task_spec
    full_val = ds.formatted_ds["validation"]
    rec = full_val[args.pair_index]
    rm_preprocessor = import_rm_preprocessor()
    proc = rm_preprocessor(rec, task_spec, tokenizer, cfg["data"]["max_input_seq_length"], args.pair_index)

    # Collate into batch
    batch = preference_collate_fn(
        [proc],
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=cfg["policy"]["make_sequence_length_divisible_by"],
        add_loss_mask=False,
    )

    # Print diagnostics
    ids = batch["input_ids"]
    lens = batch["input_lengths"]
    print(f"input_lengths={lens.tolist()}")
    print(f"seq_len={ids.shape[1]}")
    print(f"first64_ids[0]={ids[0,:64].tolist()}")
    print(f"first64_ids[1]={ids[1,:64].tolist()}")

    # Show formatted strings snippets
    def fmt_text(messages):
        return "".join([m.get("content", "") for m in messages])[:256]
    print("chosen_text_snippet=", fmt_text(proc["message_log_chosen"]))
    print("rejected_text_snippet=", fmt_text(proc["message_log_rejected"]))


if __name__ == "__main__":
    main()


