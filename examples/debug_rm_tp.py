#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Hermetic debug script to compare DTensor v2 RM forward behavior across TP sizes.
# Extended to:
# - Accept dataset/pair-index to reconstruct exact sample from HelpSteer3
# - Compare TP=1 vs TP=4 vs pure HF
# - Process top-K worst pairs from two JSONLs in one run, reusing policies
# - Optional precision toggle and make_sequence_length_divisible_by override
# - Save per-pair reports and a summary table

import argparse
import json
import os
from pathlib import Path
from typing import Any, List, Tuple

import torch

from nemo_rl.algorithms.loss_functions import PreferenceLoss
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.data import hf_datasets
from nemo_rl.data.interfaces import DPODatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log
from nemo_rl.data.datasets import preference_collate_fn
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from transformers import AutoModelForSequenceClassification
try:
    from examples.run_rm import rm_preprocessor  # type: ignore
except Exception:
    import importlib.util
    import sys
    _rm_path = str(Path(__file__).parent / "run_rm.py")
    spec = importlib.util.spec_from_file_location("_rm_module", _rm_path)
    assert spec and spec.loader
    _rm_mod = importlib.util.module_from_spec(spec)
    sys.modules["_rm_module"] = _rm_mod
    spec.loader.exec_module(_rm_mod)
    rm_preprocessor = getattr(_rm_mod, "rm_preprocessor")



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    p.add_argument("--dataset", type=str, default="helpsteer3", choices=["helpsteer3"])  # extend as needed
    p.add_argument("--pair-index", type=int, default=None, help="Index into validation split for HelpSteer3")
    p.add_argument("--topk", type=int, default=1, help="If using before/after JSONLs, how many worst pairs to process")
    p.add_argument("--before_jsonl", type=Path, default=None, help="Optional JSONL from TP=1")
    p.add_argument("--after_jsonl", type=Path, default=None, help="Optional JSONL from TP=4")
    p.add_argument("--precision", type=str, default="bfloat16", choices=["bfloat16", "float32"], help="Policy precision for NeMo-RL path")
    p.add_argument("--divisible_by", type=int, default=0, help="Override make_sequence_length_divisible_by for both TP policies; 0 to keep defaults")
    p.add_argument("--rm_config", type=str, default=str(Path(__file__).parent / "configs" / "rm.yaml"), help="Path to rm.yaml to mirror tokenizer/chat template")
    p.add_argument("overrides", nargs=argparse.REMAINDER, help="Optional hydra-style overrides to apply to rm config")
    p.add_argument("--chat_template_mode", type=str, default="default", choices=["default", "rm_yaml", "passthrough"], help="Which chat template to use for tokenizer")
    p.add_argument("--assert_match", action="store_true", help="Assert that per-pair rewards match JSONL for given files")
    return p.parse_args()


def make_task_spec() -> TaskDataSpec:
    # Mirror run_rm: fetch task_spec from dataset helper (handles prompts if any)
    ds = hf_datasets.HelpSteer3Dataset()
    return ds.task_spec


def make_policy_cfg(model_name: str, tp_size: int, precision: str, divisible_override: int | None) -> PolicyConfig:
    cfg: PolicyConfig = {
        "model_name": model_name,
        "tokenizer": {
            "name": model_name,
        },
        "train_global_batch_size": 2,
        "train_micro_batch_size": 2,
        "precision": precision,
        "reward_model_cfg": {
            "enabled": True,
            "reward_model_type": "bradley_terry",
        },
        "dtensor_cfg": {
            "enabled": True,
            "_v2": True,
            "cpu_offload": False,
            "sequence_parallel": False,
            "activation_checkpointing": False,
            "tensor_parallel_size": tp_size,
            "context_parallel_size": 1,
            "custom_parallel_plan": None,
        },
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "make_sequence_length_divisible_by": max(1, tp_size),
        "max_total_sequence_length": 8192,
        "max_grad_norm": 1.0,
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 0.0,
                "betas": (0.9, 0.95),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "foreach": False,
                "fused": False,
            },
        },
    }
    if divisible_override and divisible_override > 0:
        cfg["make_sequence_length_divisible_by"] = divisible_override
    return cfg


def make_preference_from_helpsteer3(tokenizer, task_spec: TaskDataSpec, idx: int) -> DPODatumSpec:
    ds = hf_datasets.HelpSteer3Dataset()
    rec = ds.formatted_ds["validation"][idx]
    completions = rec["completions"]
    assert len(completions) == 2
    messages_context = rec["context"]
    messages_chosen = messages_context + completions[0]["completion"]
    messages_rejected = messages_context + completions[1]["completion"]
    ml_chosen = get_formatted_message_log(messages_chosen, tokenizer, task_spec)
    ml_rejected = get_formatted_message_log(messages_rejected, tokenizer, task_spec)
    # For debugging: capture the final formatted strings for chosen/rejected
    chosen_text = "".join([m.get("content", "") for m in ml_chosen])
    rejected_text = "".join([m.get("content", "") for m in ml_rejected])
    return DPODatumSpec(
        message_log_chosen=ml_chosen,
        message_log_rejected=ml_rejected,
        length_chosen=sum(len(m["token_ids"]) for m in ml_chosen),
        length_rejected=sum(len(m["token_ids"]) for m in ml_rejected),
        loss_multiplier=1.0,
        idx=idx,
    )


def make_preference_via_rm_preprocessor(tokenizer, task_spec: TaskDataSpec, idx: int, max_seq_length: int) -> dict[str, Any]:
    ds = hf_datasets.HelpSteer3Dataset()
    rec = ds.formatted_ds["validation"][idx]
    # Use the exact preprocessing used in run_rm
    return rm_preprocessor(rec, task_spec, tokenizer, max_seq_length, idx)


def collate_preference(item: DPODatumSpec, tokenizer, divisible_by: int) -> dict[str, Any]:
    return preference_collate_fn(
        [item], tokenizer=tokenizer, make_sequence_length_divisible_by=divisible_by, add_loss_mask=False
    )


def get_length_info(item: DPODatumSpec, batch: dict[str, Any], divisible_by: int) -> dict[str, Any]:
    input_lengths = batch.get("input_lengths")
    if isinstance(input_lengths, torch.Tensor):
        input_lengths = [int(x) for x in input_lengths.tolist()]
    return {
        "raw_message_lengths": {
            "chosen": int(item["length_chosen"]),
            "rejected": int(item["length_rejected"]),
        },
        "batch_input_lengths": input_lengths,
        "padded_seq_len": int(batch["input_ids"].shape[1]),
        "divisible_by": int(divisible_by),
    }


def _create_policy(tp_size: int, model_name: str, tokenizer, cluster: RayVirtualCluster, precision: str, divisible_override: int | None) -> Policy:
    policy_cfg = make_policy_cfg(model_name, tp_size, precision, divisible_override)
    policy = Policy(
        cluster=cluster,
        config=policy_cfg,
        tokenizer=tokenizer,
        init_optimizer=True,
        init_reference_model=False,
        name_prefix=f"lm_policy_tp{tp_size}",
    )
    policy.prepare_for_training()
    return policy


def score_policy(policy: Policy, batch: dict[str, Any]) -> dict[str, Any]:
    gbs = 2  # one pair -> 2 samples
    results = policy.train(batch, PreferenceLoss(), eval_mode=True, gbs=gbs, mbs=gbs)
    metrics = results["all_mb_metrics"]
    # Flatten helper
    def _flat(x):
        out = []
        for v in x:
            if hasattr(v, "tolist"):
                out.extend(v.tolist())
            elif isinstance(v, (list, tuple)):
                out.extend(list(v))
            else:
                out.append(v)
        return out
    per_pair_chosen = _flat(metrics.get("per_pair_reward_chosen", []))
    per_pair_rejected = _flat(metrics.get("per_pair_reward_rejected", []))
    return {
        "rewards_chosen_mean": sum(metrics.get("rewards_chosen_mean", [0.0])),
        "rewards_rejected_mean": sum(metrics.get("rewards_rejected_mean", [0.0])),
        "accuracy": sum(metrics.get("accuracy", [0.0])),
        "num_valid_samples": sum(metrics.get("num_valid_samples", [0.0])),
        "per_pair_reward_chosen": per_pair_chosen,
        "per_pair_reward_rejected": per_pair_rejected,
    }


def score_hf(model_name: str, tokenizer, item: DPODatumSpec) -> tuple[float, float]:
    device = "cuda:0"
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        num_labels=1,
        trust_remote_code=True,
    )
    def to_text(ml):
        msgs = []
        for m in ml:
            if "role" in m and "content" in m:
                msgs.append({"role": m["role"], "content": m["content"]})
        return msgs
    conv_c = to_text(item["message_log_chosen"])
    conv_r = to_text(item["message_log_rejected"])
    conv_c_fmt = tokenizer.apply_chat_template(conv_c, tokenize=False)
    conv_r_fmt = tokenizer.apply_chat_template(conv_r, tokenize=False)
    if tokenizer.bos_token is not None and conv_c_fmt.startswith(tokenizer.bos_token):
        conv_c_fmt = conv_c_fmt[len(tokenizer.bos_token) :]
    if tokenizer.bos_token is not None and conv_r_fmt.startswith(tokenizer.bos_token):
        conv_r_fmt = conv_r_fmt[len(tokenizer.bos_token) :]
    conv_c_tok = tokenizer(conv_c_fmt, return_tensors="pt").to(device)
    conv_r_tok = tokenizer(conv_r_fmt, return_tensors="pt").to(device)
    with torch.no_grad():
        score_c = rm(**conv_c_tok).logits[0][0].item()
        score_r = rm(**conv_r_tok).logits[0][0].item()
    return score_c, score_r


def _scalarize(v):
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        if len(v) == 1:
            return _scalarize(v[0])
        return tuple(_scalarize(x) for x in v)
    return v


def pick_worst_pairs(before_jsonl: Path, after_jsonl: Path, topk: int) -> List[int]:
    with open(before_jsonl) as f1, open(after_jsonl) as f2:
        before = [json.loads(l) for l in f1 if l.strip()]
        after = [json.loads(l) for l in f2 if l.strip()]
    for r in before:
        r["dataset"] = _scalarize(r.get("dataset"))
        r["pair_index"] = int(_scalarize(r.get("pair_index")))
        r["reward_delta"] = float(_scalarize(r.get("reward_delta")))
    for r in after:
        r["dataset"] = _scalarize(r.get("dataset"))
        r["pair_index"] = int(_scalarize(r.get("pair_index")))
        r["reward_delta"] = float(_scalarize(r.get("reward_delta")))
    b_map = {(r["dataset"], r["pair_index"]): r for r in before}
    a_map = {(r["dataset"], r["pair_index"]): r for r in after}
    diffs = []
    for k in set(b_map.keys()) & set(a_map.keys()):
        diffs.append((k, a_map[k]["reward_delta"] - b_map[k]["reward_delta"]))
    diffs.sort(key=lambda x: x[1])
    return [k[1] for (k, _) in diffs[:topk]]


def main():
    args = parse_args()
    out_dir = Path("logs/debug_tp_rm")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    # Mirror run_rm tokenizer config (chat_template, etc.)
    # Tokenizer/chat template selection
    if args.chat_template_mode == "rm_yaml":
        try:
            cfg = load_config(args.rm_config)
            if args.overrides:
                cfg = parse_hydra_overrides(cfg, args.overrides)
            cfg = cfg
            tok_cfg = cfg["policy"]["tokenizer"].copy()
            tok_cfg["name"] = model_name
            tokenizer = get_tokenizer(tok_cfg)
        except Exception:
            tokenizer = get_tokenizer({"name": model_name})
    elif args.chat_template_mode == "passthrough":
        tokenizer = get_tokenizer({"name": model_name, "chat_template": None})
    else:
        # default -> use tokenizer's default chat template
        tokenizer = get_tokenizer({"name": model_name})
    task_spec = make_task_spec()

    pair_indices: List[int]
    if args.pair_index is not None:
        pair_indices = [args.pair_index]
    elif args.before_jsonl and args.after_jsonl:
        pair_indices = pick_worst_pairs(args.before_jsonl, args.after_jsonl, args.topk)
        if not pair_indices:
            raise SystemExit("No overlapping pairs found in JSONLs")
    else:
        raise SystemExit("Provide --pair-index or both --before_jsonl and --after_jsonl")

    init_ray()
    cluster_tp1 = RayVirtualCluster(name="rm_debug_tp1", bundle_ct_per_node_list=[1], use_gpus=True, num_gpus_per_node=8, max_colocated_worker_groups=1)
    cluster_tp4 = RayVirtualCluster(name="rm_debug_tp4", bundle_ct_per_node_list=[4], use_gpus=True, num_gpus_per_node=8, max_colocated_worker_groups=1)

    divisible_override = args.divisible_by if args.divisible_by > 0 else None
    pol_tp1 = _create_policy(1, model_name, tokenizer, cluster_tp1, args.precision, divisible_override)
    pol_tp4 = _create_policy(4, model_name, tokenizer, cluster_tp4, args.precision, divisible_override)

    summary_rows: List[Tuple[int, float, float, float, float]] = []

    for pair_idx in pair_indices:
        # Build via rm_preprocessor to replicate run_rm exactly
        item_rm = make_preference_via_rm_preprocessor(tokenizer, task_spec, pair_idx, max_seq_length=8192)
        # Convert rm_preprocessor output to DPODatumSpec
        item = DPODatumSpec(
            message_log_chosen=item_rm["message_log_chosen"],
            message_log_rejected=item_rm["message_log_rejected"],
            length_chosen=item_rm["length_chosen"],
            length_rejected=item_rm["length_rejected"],
            loss_multiplier=item_rm["loss_multiplier"],
            idx=item_rm["idx"],
        )
        div1 = pol_tp1.cfg["make_sequence_length_divisible_by"]
        div4 = pol_tp4.cfg["make_sequence_length_divisible_by"]
        batch_tp1 = collate_preference(item, tokenizer, divisible_by=div1)
        batch_tp4 = collate_preference(item, tokenizer, divisible_by=div4)

        res_tp1 = score_policy(pol_tp1, batch_tp1)
        res_tp4 = score_policy(pol_tp4, batch_tp4)
        hf_c, hf_r = score_hf(model_name, tokenizer, item)

        report = {
            "pair_index": pair_idx,
            "precision": args.precision,
            "divisible_by": divisible_override or "default",
            "tp1": res_tp1,
            "tp4": res_tp4,
            "hf": {"chosen": hf_c, "rejected": hf_r, "delta": hf_c - hf_r},
            "tp1_lengths": get_length_info(item, batch_tp1, div1),
            "tp4_lengths": get_length_info(item, batch_tp4, div4),
            "per_pair": {
                "tp1": {
                    "reward_chosen": res_tp1.get("per_pair_reward_chosen", [None])[0] if res_tp1.get("per_pair_reward_chosen") else None,
                    "reward_rejected": res_tp1.get("per_pair_reward_rejected", [None])[0] if res_tp1.get("per_pair_reward_rejected") else None,
                    "reward_delta": (res_tp1.get("per_pair_reward_chosen", [0])[0] - res_tp1.get("per_pair_reward_rejected", [0])[0]) if (res_tp1.get("per_pair_reward_chosen") and res_tp1.get("per_pair_reward_rejected")) else None,
                },
                "tp4": {
                    "reward_chosen": res_tp4.get("per_pair_reward_chosen", [None])[0] if res_tp4.get("per_pair_reward_chosen") else None,
                    "reward_rejected": res_tp4.get("per_pair_reward_rejected", [None])[0] if res_tp4.get("per_pair_reward_rejected") else None,
                    "reward_delta": (res_tp4.get("per_pair_reward_chosen", [0])[0] - res_tp4.get("per_pair_reward_rejected", [0])[0]) if (res_tp4.get("per_pair_reward_chosen") and res_tp4.get("per_pair_reward_rejected")) else None,
                },
            },
        }
        out_fp = out_dir / f"pair_{pair_idx}_report_{args.precision}_div{divisible_override or 'default'}.json"
        with open(out_fp, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {out_fp}")

        summary_rows.append(
            (
                pair_idx,
                res_tp1["rewards_chosen_mean"] - res_tp1["rewards_rejected_mean"],
                res_tp4["rewards_chosen_mean"] - res_tp4["rewards_rejected_mean"],
                hf_c - hf_r,
                (res_tp4["rewards_chosen_mean"] - res_tp4["rewards_rejected_mean"]) - (res_tp1["rewards_chosen_mean"] - res_tp1["rewards_rejected_mean"]),
            )
        )

        # Optional: assert and print diffs against JSONLs
        if args.assert_match and args.before_jsonl and args.after_jsonl:
            import json as _json
            def _find_record(path: Path, idx: int):
                with open(path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = _json.loads(line)
                        # JSONL stores possibly wrapped fields; normalize
                        def _scalar(v):
                            if isinstance(v, list) and len(v) == 1: return v[0]
                            return v
                        if int(_scalar(rec.get("pair_index"))) == idx:
                            return float(_scalar(rec["reward_chosen"])), float(_scalar(rec["reward_rejected"]))
                return None
            b = _find_record(args.before_jsonl, pair_idx)
            a = _find_record(args.after_jsonl, pair_idx)
            if b:
                tp1_pair = (report["per_pair"]["tp1"]["reward_chosen"], report["per_pair"]["tp1"]["reward_rejected"])
                if abs(tp1_pair[0] - b[0]) > 1e-6 or abs(tp1_pair[1] - b[1]) > 1e-6:
                    print(f"ASSERT MISMATCH TP=1: expected {b} got {tp1_pair}")
            if a:
                tp4_pair = (report["per_pair"]["tp4"]["reward_chosen"], report["per_pair"]["tp4"]["reward_rejected"])
                if abs(tp4_pair[0] - a[0]) > 1e-6 or abs(tp4_pair[1] - a[1]) > 1e-6:
                    print(f"ASSERT MISMATCH TP=4: expected {a} got {tp4_pair}")

    # write summary
    summary_fp = out_dir / f"topk_probe_{args.precision}_div{divisible_override or 'default'}.txt"
    with open(summary_fp, "w") as f:
        f.write("pair_index,tp1_delta,tp4_delta,hf_delta,tp_diff\n")
        for row in summary_rows:
            f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f}\n")
    print(f"Summary saved: {summary_fp}")

    pol_tp1.shutdown(); cluster_tp1.shutdown()
    pol_tp4.shutdown(); cluster_tp4.shutdown()


if __name__ == "__main__":
    main()


