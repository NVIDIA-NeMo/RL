#!/usr/bin/env python3
import argparse
from pathlib import Path

from nemo_rl.algorithms.rm import setup as rm_setup, validate_one_dataset
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import hf_datasets
from nemo_rl.data.datasets import AllTaskProcessedDataset, preference_collate_fn
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from torchdata.stateful_dataloader import StatefulDataLoader
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(Path(__file__).parent / "configs" / "rm.yaml"))
    p.add_argument("--pair-index", type=int, required=True)
    p.add_argument("--model", type=str, default="Skywork/Skywork-Reward-V2-Qwen3-0.6B")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--chat_template_mode", type=str, default="default", choices=["default", "rm_yaml", "passthrough"], help="Tokenizer default, rm.yaml chat template, or passthrough")
    p.add_argument("--strip_eos", action="store_true", help="Remove a trailing EOS token from last message if present")
    p.add_argument("--append_eos", action="store_true", help="Append an EOS token to last message if missing")
    p.add_argument("--from_jsonl", type=str, default=None, help="Use a JSONL record (with formatted_text/token_ids fields) to build the pair instead of dataset")
    p.add_argument("--divisible_by", type=int, default=0, help="Override make_sequence_length_divisible_by; 0 uses TP size")
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    p.add_argument(
        "--pad_val_like_e2e",
        action="store_true",
        help="Expand single-pair validation batch to rm.val_global_batch_size using masked dummies to mirror e2e shapes",
    )
    p.add_argument(
        "--direct_forward",
        action="store_true",
        help="Bypass DataLoader/validate_one_dataset and call policy.train directly with flat_input_ids_* from JSONL",
    )
    p.add_argument(
        "--reconstruct_batch",
        action="store_true",
        help="Load all records with the same batch_id from the JSONL and run a single forward to mirror e2e batch context",
    )
    return p.parse_args()


def import_rm_preprocessor():
    try:
        from examples.run_rm import rm_preprocessor  # type: ignore
        return rm_preprocessor
    except Exception:
        import importlib.util, sys
        rm_path = str(Path(__file__).parent / "run_rm.py")
        spec = importlib.util.spec_from_file_location("_rm_module", rm_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_rm_module"] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, "rm_preprocessor")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = parse_hydra_overrides(cfg, args.overrides)

    # Minimize logging side-effects for a focused single-pair run
    try:
        cfg["logger"]["wandb_enabled"] = False
        cfg["logger"]["tensorboard_enabled"] = False
        cfg["logger"]["mlflow_enabled"] = False
        cfg["logger"]["monitor_gpus"] = False
        cfg["logger"]["log_dir"] = "logs/val_one_pair"
        cfg["policy"]["dtensor_cfg"]["_v2"] = True
        cfg["policy"]["dtensor_cfg"]["tensor_parallel_size"] = args.tp
        cfg["policy"]["model_name"] = args.model
        cfg["policy"]["tokenizer"]["name"] = args.model
        cfg["policy"]["make_sequence_length_divisible_by"] = (
            args.divisible_by if args.divisible_by > 0 else max(1, args.tp)
        )
        if args.tp > 1:
            cfg["cluster"]["gpus_per_node"] = max(cfg["cluster"].get("gpus_per_node", 1), args.tp)
        if args.chat_template_mode == "default":
            # Drop any custom chat_template to use the model's default
            if "chat_template" in cfg["policy"]["tokenizer"]:
                cfg["policy"]["tokenizer"].pop("chat_template")
        elif args.chat_template_mode == "passthrough":
            cfg["policy"]["tokenizer"]["chat_template"] = None
    except Exception:
        pass

    tokenizer = get_tokenizer(cfg["policy"]["tokenizer"])  # honor chat_template
    init_ray()

    # Build a single-sample validation dataset mirroring setup_data, preserving original idx
    # Build task spec placeholder
    class _HardTaskSpec:
        prompt = None
        system_prompt = None
    task_spec = _HardTaskSpec()
    rm_preprocessor = import_rm_preprocessor()
    class _SingleVal:
        def __len__(self):
            return 1
        def __getitem__(self, _):
            if args.from_jsonl:
                import json
                # load record for the requested pair index
                chosen_ids = rejected_ids = None
                flat_c = flat_r = None
                with open(args.from_jsonl) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        # normalize scalar/array
                        def _scalar(v):
                            if isinstance(v, list) and len(v) == 1:
                                return v[0]
                            return v
                        if int(_scalar(rec.get("pair_index", -1))) == args.pair_index:
                            chosen_ids = rec.get("token_ids_chosen")
                            rejected_ids = rec.get("token_ids_rejected")
                            flat_c = rec.get("flat_input_ids_chosen")
                            flat_r = rec.get("flat_input_ids_rejected")
                            break
                assert chosen_ids is not None and rejected_ids is not None, "pair not found or fields missing in JSONL; rerun validation with enhanced logging"
                # Prefer exact flat inputs used by e2e forward when available
                if isinstance(flat_c, list) and isinstance(flat_r, list) and len(flat_c) > 0 and len(flat_r) > 0:
                    chosen_ids = flat_c
                    rejected_ids = flat_r
                # Construct DPODatumSpec-like dict with single-message logs based on token ids
                out = {
                    "message_log_chosen": [{"token_ids": torch.tensor(chosen_ids, dtype=torch.long)}],
                    "message_log_rejected": [{"token_ids": torch.tensor(rejected_ids, dtype=torch.long)}],
                    "length_chosen": len(chosen_ids),
                    "length_rejected": len(rejected_ids),
                    "loss_multiplier": 1.0,
                    "idx": args.pair_index,
                }
            else:
                # Hardcode pair 1040 messages (user + two assistant completions) to avoid dataset lookup
                hard_user = "7 strange places"
                hard_chosen = "Sure, I'd be happy to help you with that! Here are seven strange and unexpected places..."
                hard_rejected = "Here are seven places..."
                rec = {
                    "context": [{"role": "user", "content": hard_user}],
                    "completions": [
                        {"rank": 0, "completion": [{"role": "assistant", "content": hard_chosen}]},
                        {"rank": 1, "completion": [{"role": "assistant", "content": hard_rejected}]},
                    ],
                }
                out = rm_preprocessor(rec, task_spec, tokenizer, cfg["data"]["max_input_seq_length"], args.pair_index)
            # Optional EOS post-processing
            def _maybe_fix_eos(msg_log):
                if not msg_log:
                    return
                last = msg_log[-1]
                if "token_ids" in last and isinstance(last["token_ids"], torch.Tensor):
                    ids = last["token_ids"]
                    eos_id = tokenizer.eos_token_id
                    if args.strip_eos and eos_id is not None and ids.numel() > 0 and ids[-1].item() == eos_id:
                        last["token_ids"] = ids[:-1]
                    if args.append_eos and eos_id is not None and (ids.numel() == 0 or ids[-1].item() != eos_id):
                        last["token_ids"] = torch.cat([ids, torch.tensor([eos_id], dtype=ids.dtype)])
            _maybe_fix_eos(out["message_log_chosen"])
            _maybe_fix_eos(out["message_log_rejected"])
            out["idx"] = args.pair_index
            return out
    val_dataset = _SingleVal()

    # Optionally expand to e2e-like batch size with masked dummies to match shapes and TP padding behavior
    if args.pad_val_like_e2e:
        class _ExpandedVal:
            def __init__(self, base_item, repeats):
                self.base_item = base_item
                self.repeats = repeats
            def __len__(self):
                return self.repeats
            def __getitem__(self, i):
                if i == 0:
                    return self.base_item
                # create a dummy masked pair with minimal token
                def _dummy_msg():
                    return [{"token_ids": torch.tensor([0], dtype=torch.long)}]
                return {
                    "message_log_chosen": _dummy_msg(),
                    "message_log_rejected": _dummy_msg(),
                    "length_chosen": 1,
                    "length_rejected": 1,
                    "loss_multiplier": 0.0,
                    "idx": -1_000_000 - i,
                }
        repeats = max(1, cfg["rm"]["val_global_batch_size"])
        # Convert the single base item to materialized dict once to avoid double work
        base_item = val_dataset[0]
        val_dataset = _ExpandedVal(base_item, repeats)

    # Build a tiny 1-sample train dataset to satisfy rm_setup contract
    # Minimal train dataset: reuse the same constructed record for simplicity
    if args.from_jsonl:
        def _proc_train(_):
            # tiny 1-sample with identical lengths
            return {
                "message_log": [{"token_ids": torch.tensor([0], dtype=torch.long)}],
                "length": 1,
                "extra_env_info": None,
                "loss_multiplier": 1.0,
                "idx": 0,
            }
        train_dataset = AllTaskProcessedDataset([{}], tokenizer, task_spec, lambda *a, **k: _proc_train({}))
    else:
        hard_rec = {
            "context": [{"role": "user", "content": hard_user}],
            "completions": [
                {"rank": 0, "completion": [{"role": "assistant", "content": hard_chosen}]},
                {"rank": 1, "completion": [{"role": "assistant", "content": hard_rejected}]},
            ],
        }
        def _proc(rec):
            return rm_preprocessor(rec, task_spec, tokenizer, cfg["data"]["max_input_seq_length"], args.pair_index)
        train_dataset = AllTaskProcessedDataset([hard_rec], tokenizer, task_spec, lambda r,ts,tok,msl,idx: _proc(r))

    (
        policy,
        cluster,
        _train_dataloader,
        _val_dls,
        loss_fn,
        logger,
        _checkpointer,
        _rm_save_state,
        master_config,
    ) = rm_setup(cfg, tokenizer, train_dataset, {"default": val_dataset})

    # Build a dataloader for our one-sample dataset
    val_bs = (cfg["rm"]["val_global_batch_size"] * 2) if args.pad_val_like_e2e else 2
    val_mbs = cfg["rm"]["val_micro_batch_size"]
    eff_divisible = cfg["policy"]["make_sequence_length_divisible_by"]
    collate = lambda batch: preference_collate_fn(  # noqa: E731
        batch,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=eff_divisible,
        add_loss_mask=False,
    )
    val_loader = StatefulDataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        collate_fn=collate,
        drop_last=False,
    )

    # If requested and JSONL provided with flat_input_ids_*, run a direct forward to mirror exact inputs
    if args.reconstruct_batch and args.from_jsonl:
        import json
        # Find target record and its batch_id
        recs = [json.loads(l) for l in open(args.from_jsonl) if l.strip()]
        target = None
        for r in recs:
            if r.get("pair_index") == args.pair_index:
                target = r; break
        assert target is not None and isinstance(target.get("batch_id"), int), "batch_id not found; regenerate JSONL with batch_id"
        bid = target["batch_id"]
        same_batch = [r for r in recs if r.get("batch_id") == bid]
        # Build flat batch with exact flat_input_ids; fallback to token_ids_* if needed
        seqs = []
        idxs = []
        for r in same_batch:
            for key in ("flat_input_ids_chosen","flat_input_ids_rejected"):
                seq = r.get(key)
                if not isinstance(seq, list):
                    seq = r.get("token_ids_chosen" if "chosen" in key else "token_ids_rejected")
                assert isinstance(seq, list) and len(seq) > 0
                seqs.append(torch.tensor(seq, dtype=torch.long))
                idxs.append(r.get("pair_index"))
        maxL = max(s.numel() for s in seqs)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(seqs), maxL), pad_id, dtype=torch.long)
        input_lengths = torch.zeros((len(seqs),), dtype=torch.long)
        for i,s in enumerate(seqs):
            input_ids[i, : s.numel()] = s
            input_lengths[i] = s.numel()
        sample_mask = torch.ones((len(seqs),), dtype=torch.float32)
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict
        batch = BatchedDataDict(
            input_ids=input_ids,
            input_lengths=input_lengths,
            sample_mask=sample_mask,
            idx=idxs,
            message_log=[{"token_ids": s} for s in seqs],
        )
        # Mirror e2e microbatching: one pair (2 samples) per microbatch
        results = policy.train(batch, loss_fn, eval_mode=True, gbs=len(seqs), mbs=2)
        # Flatten helper for nested lists across microbatches
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
        idx_list = _flat(results["all_mb_metrics"].get("per_sample_idx", []))
        rw_list = _flat(results["all_mb_metrics"].get("per_sample_rewards", []))
        isc_list = _flat(results["all_mb_metrics"].get("per_sample_is_chosen", []))
        # Collect chosen/rejected by explicit flags when available
        chosen = rejected = None
        for pos, idx_val in enumerate(idx_list):
            try:
                if int(idx_val) != args.pair_index:
                    continue
            except Exception:
                continue
            if pos < len(rw_list):
                if isc_list and pos < len(isc_list):
                    if int(isc_list[pos]) == 1:
                        chosen = float(rw_list[pos])
                    elif int(isc_list[pos]) == 0:
                        rejected = float(rw_list[pos])
                else:
                    # Fallback: first occurrence is chosen, next is rejected
                    if chosen is None:
                        chosen = float(rw_list[pos])
                    else:
                        rejected = float(rw_list[pos])
        assert chosen is not None and rejected is not None, "could not locate both chosen and rejected rewards for target pair"
        dump_dir = Path(master_config["logger"]["log_dir"]) / "validation"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_fp = dump_dir / "default_pairs_step_0.jsonl"
        out = {
            "dataset": "default",
            "pair_index": args.pair_index,
            "reward_chosen": chosen,
            "reward_rejected": rejected,
            "reward_delta": float(chosen - rejected),
            "debug_reward_source": "reconstruct_batch",
        }
        with open(dump_fp, "w") as f:
            f.write(json.dumps(out) + "\n")
        print(f"  ✓ Wrote validation dump to {dump_fp}")
    elif args.direct_forward and args.from_jsonl:
        import json
        rec = None
        with open(args.from_jsonl) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("pair_index") == args.pair_index:
                    rec = r; break
        assert rec is not None, "pair not found in JSONL"
        assert isinstance(rec.get("flat_input_ids_chosen"), list) and isinstance(rec.get("flat_input_ids_rejected"), list), "flat_input_ids_* missing; rerun e2e after recent changes"
        ids_c = torch.tensor(rec["flat_input_ids_chosen"], dtype=torch.long)
        ids_r = torch.tensor(rec["flat_input_ids_rejected"], dtype=torch.long)
        Lc, Lr = ids_c.numel(), ids_r.numel()
        maxL = max(Lc, Lr)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((2, maxL), pad_id, dtype=torch.long)
        input_ids[0, :Lc] = ids_c
        input_ids[1, :Lr] = ids_r
        input_lengths = torch.tensor([Lc, Lr], dtype=torch.long)
        sample_mask = torch.tensor([1.0, 1.0], dtype=torch.float32)
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict
        batch = BatchedDataDict(
            input_ids=input_ids,
            input_lengths=input_lengths,
            sample_mask=sample_mask,
            idx=[args.pair_index, args.pair_index],
            message_log=[{"token_ids": ids_c}, {"token_ids": ids_r}],
        )
        results = policy.train(batch, loss_fn, eval_mode=True, gbs=2, mbs=2)
        # Write a minimal JSONL aligned record
        dump_dir = Path(master_config["logger"]["log_dir"]) / "validation"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_fp = dump_dir / "default_pairs_step_0.jsonl"
        rw = results["all_mb_metrics"].get("per_sample_rewards", [[None, None]])[0]
        raw = results["all_mb_metrics"].get("per_sample_raw_logits", [[None, None]])[0]
        chosen = float(rw[0]); rejected = float(rw[1])
        out = {
            "dataset": "default",
            "pair_index": args.pair_index,
            "token_ids_chosen": rec["flat_input_ids_chosen"],
            "token_ids_rejected": rec["flat_input_ids_rejected"],
            "reward_chosen": chosen,
            "reward_rejected": rejected,
            "reward_delta": float(chosen - rejected),
            "debug_reward_source": "direct_forward",
            "per_sample_rewards": [float(r) for r in rw],
            "per_sample_raw_logits": [None if v is None else float(v) for v in raw],
        }
        with open(dump_fp, "w") as f:
            f.write(json.dumps(out) + "\n")
        print(f"  ✓ Wrote validation dump to {dump_fp}")
    else:
        # Run validation on the single-sample dataloader; logs JSONL per-pair
        validate_one_dataset(
            policy=policy,
            val_dataloader=val_loader,
            loss_fn=loss_fn,
            step=0,
            master_config=master_config,
            val_batches=1,
            val_batch_size=val_bs,
            val_mbs=val_mbs,
            dataset_name="default",
        )


if __name__ == "__main__":
    main()


