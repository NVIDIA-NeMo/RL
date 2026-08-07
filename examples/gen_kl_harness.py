# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# gen_kl_harness.py — FAITHFUL cheap replica of the NeMo-RL gen_kl determinism path.
#
# It reuses NeMo-RL's OWN code paths (setup() model construction, the real engine
# generate via run_multi_turn_rollout, the real TE-training-path get_logprobs, and
# the real calculate_kl/masked_mean) to measure gen_kl_error EXACTLY as grpo_train
# does — but WITHOUT the training loop: no policy.train(), no optimizer step, no
# refit/weight-sync, no multi-step, no val, no checkpointing.
#
# gen_kl_error is computed in grpo_train BEFORE the policy is ever updated
# (it compares the engine's generation_logprobs against the TE get_logprobs of the
# same tokens). So one {rollout -> flatten -> get_logprobs -> calculate_kl} pass is
# a complete, faithful measurement. This is the "cheap test bed" for probing
# determinism fixes/optimizations without a full GRPO job.
#
# Invoke with the SAME config + overrides as the det bed (run_grpo.py), e.g.:
#   python gen_kl_harness.py --config <yaml> <hydra overrides...>
#
# Faithfulness contract (each line here mirrors grpo_train, file:line noted):
#   preamble         -> examples/run_grpo.py:66-154
#   repeat + flatten -> grpo.py:2115-2125
#   generation       -> grpo.py:2177,2232-2243  (run_multi_turn_rollout)
#   train_data build -> grpo.py:2378-2401  (add_grpo_token_loss_masks..., flatten)
#   prev_logprobs    -> grpo.py:2449-2479  (prepare_for_lp_inference, get_logprobs)
#   gen_kl_error     -> loss/loss_functions.py:308-345 (calculate_kl + masked_mean)

import argparse
import json
import os
import pprint
import time

import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    add_grpo_token_loss_masks_and_generation_logprobs,
    refit_policy_generation,
    setup,
)
from nemo_rl.models.generation.megatron import MegatronGeneration
from nemo_rl.algorithms.utils import calculate_kl, get_tokenizer, masked_mean
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.experience.rollouts import run_multi_turn_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.megatron.router_replay import router_replay_enabled
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args():
    parser = argparse.ArgumentParser(description="gen_kl determinism harness")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="How many dataloader batches to measure (each = one gen+score+kl pass)",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=os.environ.get("NRL_GENKL_JSONL", "gen_kl_harness_results.jsonl"),
        help="Where to append per-batch results",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _report(tag, gen_lp, prev_lp, token_mask, sample_mask, kl_type):
    """Compute gen_kl EXACTLY as ClippedPGLossFn (loss_functions.py:308-345),
    plus richer diagnostics the bed floors away (max|Δ|, exact-match count)."""
    gen = gen_lp[:, 1:]
    prev = prev_lp[:, 1:]
    mask = token_mask[:, 1:] * sample_mask.unsqueeze(-1)
    global_valid_toks = mask.sum()

    # k3 (default) gen_kl — the shipped metric
    gen_kl = calculate_kl(
        logprobs=gen,
        logprobs_reference=prev,
        kl_type=kl_type,
        input_clamp_value=None,
        output_clamp_value=None,
    )
    gen_kl_mean = masked_mean(
        gen_kl, mask, global_normalization_factor=global_valid_toks
    ).item()

    # diagnostics
    lp_abs = torch.abs(gen - prev)
    m = mask.bool()
    n_valid = int(m.sum().item())
    if n_valid == 0:
        return {
            "tag": tag,
            "gen_kl_error": gen_kl_mean,
            "n_valid_toks": 0,
        }
    masked_abs = lp_abs[m]
    max_abs = masked_abs.max().item()
    mean_abs = masked_abs.mean().item()
    # bitwise-exact tokens (|Δlp| == 0.0)
    n_exact = int((masked_abs == 0.0).sum().item())
    mult_prob_error = masked_mean(
        torch.exp(lp_abs * mask), mask, global_normalization_factor=global_valid_toks
    ).item()

    return {
        "tag": tag,
        "gen_kl_error": gen_kl_mean,
        "mult_prob_error": mult_prob_error,
        "max_abs_lp_err": max_abs,
        "mean_abs_lp_err": mean_abs,
        "n_valid_toks": n_valid,
        "n_exact_toks": n_exact,
        "frac_exact": n_exact / n_valid,
    }


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if not args.config:
        raise SystemExit("--config is required (use the det bed yaml)")

    config = load_config(args.config)
    print(f"[HARNESS] Loaded configuration from: {args.config}", flush=True)
    if overrides:
        print(f"[HARNESS] Overrides: {overrides}", flush=True)
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("[HARNESS] Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"[HARNESS] log dir: {config.logger['log_dir']}", flush=True)

    init_ray()

    # ---- setup: EXACTLY run_grpo.py:99-154 (tokenizer, gen cfg, data, setup) ----
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, "generation config required"
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    megatron_cfg = config.policy.get("megatron_cfg") or {}
    trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
        trains_mtp=trains_mtp,
    )

    dataset, val_dataset, task_to_env, val_task_to_env = setup_response_data(
        tokenizer, config.data, config.env
    )

    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset, policy_factory=None)

    kl_type = master_config.loss_fn.reference_policy_kl_type
    num_gen = master_config.grpo["num_generations_per_prompt"]
    make_div_by = master_config.policy["make_sequence_length_divisible_by"]
    max_seq_len = master_config.policy["max_total_sequence_length"]
    max_rollout_turns = master_config.grpo["max_rollout_turns"]

    print(
        f"[HARNESS] kl_type={kl_type} num_gen={num_gen} "
        f"make_div_by={make_div_by} max_seq_len={max_seq_len}",
        flush=True,
    )

    # ---- first-generation weight sync: grpo.py:2014-2018, 2133-2177 ----
    # grpo_train ALWAYS refits before the first generation when gen is a separate
    # worker group (NEED_REFIT: MegatronGeneration non-colocated => True). Mirror it
    # so the gen model carries exactly the weights the scoring model has.
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]
    need_refit = not (
        isinstance(policy_generation, MegatronGeneration) and colocated_inference
    )
    print(f"[HARNESS] colocated={colocated_inference} need_refit={need_refit}", flush=True)
    refit_done = False

    results = []
    n_done = 0
    for batch in dataloader:
        if n_done >= args.num_batches:
            break
        t0 = time.time()
        # ---- repeat + flatten for generation: grpo.py:2115-2125 ----
        repeated_batch = batch.repeat_interleave(num_gen)

        # ---- generation via the REAL engine path: grpo.py:2133-2177,2232-2243 ----
        if need_refit and not refit_done:
            refit_policy_generation(policy, policy_generation, colocated_inference)
            refit_done = True
        else:
            if colocated_inference:
                policy.offload_after_refit()
            policy_generation.prepare_for_generation()
        repeated_batch, rollout_metrics = run_multi_turn_rollout(
            policy_generation=policy_generation,
            input_batch=repeated_batch,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=max_seq_len,
            max_rollout_turns=max_rollout_turns,
            greedy=False,
        )
        policy_generation.finish_generation()
        t_gen = time.time() - t0

        # ---- build train_data EXACTLY as grpo.py:2378-2401 ----
        add_grpo_token_loss_masks_and_generation_logprobs(repeated_batch["message_log"])
        flat_messages, input_lengths = batched_message_log_to_flat_message(
            repeated_batch["message_log"],
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
            make_sequence_length_divisible_by=make_div_by,
        )
        train_data = BatchedDataDict(
            {
                "input_ids": flat_messages["token_ids"],
                "input_lengths": input_lengths,
                "generation_logprobs": flat_messages["generation_logprobs"],
                "token_mask": flat_messages["token_loss_mask"],
                "sample_mask": repeated_batch["loss_multiplier"],
            }
        )
        train_data.to("cpu")

        # ---- prev_logprobs via TE training-path get_logprobs: grpo.py:2449-2479 ----
        policy.prepare_for_lp_inference()
        logprob_data = BatchedDataDict(
            {
                "input_ids": train_data["input_ids"],
                "input_lengths": train_data["input_lengths"],
                "token_mask": flat_messages["token_loss_mask"],
                "sample_mask": repeated_batch["loss_multiplier"],
            }
        )
        if (
            router_replay_enabled(master_config.policy)
            and "routed_experts" in flat_messages
        ):
            logprob_data["routed_experts"] = flat_messages["routed_experts"]
        t1 = time.time()
        prev_logprobs = policy.get_logprobs(logprob_data)["logprobs"]
        t_score = time.time() - t1

        # ---- gen_kl EXACTLY as loss_functions.py:308-345 ----
        rec = _report(
            f"batch{n_done}",
            train_data["generation_logprobs"],
            prev_logprobs,
            train_data["token_mask"],
            train_data["sample_mask"],
            kl_type,
        )
        rec["t_gen_s"] = round(t_gen, 2)
        rec["t_score_s"] = round(t_score, 2)
        rec["batch_size"] = int(train_data["input_ids"].shape[0])
        rec["seq_len"] = int(train_data["input_ids"].shape[1])

        # ---- per-sequence victim diagnostics (prod-bed tail localization) ----
        # Which sequences carry the mismatches, where does divergence start within
        # each victim, and are victims the truncated (cap-hitting) sequences?
        _gen = train_data["generation_logprobs"][:, 1:]
        _prev = prev_logprobs[:, 1:]
        _m = (train_data["token_mask"][:, 1:] * train_data["sample_mask"].unsqueeze(-1)).bool()
        _diff = (_gen != _prev) & _m
        _bad_per_seq = _diff.sum(dim=1)
        _victims = (_bad_per_seq > 0).nonzero(as_tuple=True)[0]
        _trunc = repeated_batch.get("truncated", None)
        if _trunc is not None and not torch.is_tensor(_trunc):
            _trunc = torch.tensor(_trunc, dtype=torch.bool)
        _n_trunc_total = int(_trunc.sum().item()) if _trunc is not None else -1
        vdiags = []
        for _vi in _victims.tolist():
            _row = _diff[_vi].nonzero(as_tuple=True)[0]
            _mrow = _m[_vi].nonzero(as_tuple=True)[0]
            _gen_start = int(_mrow.min().item())
            _gen_end = int(_mrow.max().item())
            vdiags.append({
                "idx": _vi,
                "n_bad": int(_bad_per_seq[_vi].item()),
                "n_valid": int(_m[_vi].sum().item()),
                "gen_span": [_gen_start, _gen_end],
                "first_bad_off": int(_row.min().item()) - _gen_start,
                "last_bad_off": int(_row.max().item()) - _gen_start,
                "input_len": int(train_data["input_lengths"][_vi].item()),
                "truncated": bool(_trunc[_vi].item()) if _trunc is not None else None,
            })
        _n_vt = sum(1 for v in vdiags if v["truncated"]) if _trunc is not None else -1
        rec["n_victims"] = len(vdiags)
        rec["n_victims_truncated"] = _n_vt
        rec["n_truncated_total"] = _n_trunc_total
        rec["victims"] = vdiags[:40]
        print(f"[VICTIMS] {rec['tag']}: {len(vdiags)} victim seqs / {rec['batch_size']} "
              f"(truncated victims: {_n_vt}/{_n_trunc_total} truncated in batch)", flush=True)
        for v in vdiags[:20]:
            _glen = v["gen_span"][1] - v["gen_span"][0] + 1
            print(f"[VICTIMS]   idx={v['idx']:5d} bad={v['n_bad']:5d}/{v['n_valid']:5d} "
                  f"first_bad@{v['first_bad_off']}/{_glen} last@{v['last_bad_off']} "
                  f"in_len={v['input_len']} trunc={v['truncated']}", flush=True)
        results.append(rec)

        print(
            f"\n[GENKL] {rec['tag']}: gen_kl_error={rec['gen_kl_error']:.3e}  "
            f"max|Δlp|={rec.get('max_abs_lp_err', float('nan')):.3e}  "
            f"exact={rec.get('n_exact_toks', 0)}/{rec.get('n_valid_toks', 0)} "
            f"({rec.get('frac_exact', 0.0) * 100:.2f}%)  "
            f"mult_prob_err={rec.get('mult_prob_error', float('nan')):.6f}  "
            f"[gen {t_gen:.1f}s score {t_score:.1f}s]\n",
            flush=True,
        )
        with open(args.jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n_done += 1

    # ---- summary ----
    if results:
        kls = [r["gen_kl_error"] for r in results]
        maxes = [r.get("max_abs_lp_err", 0.0) for r in results]
        print("\n" + "=" * 60, flush=True)
        print(f"[GENKL SUMMARY] batches={len(results)}", flush=True)
        print(
            f"[GENKL SUMMARY] gen_kl_error: min={min(kls):.3e} "
            f"max={max(kls):.3e} mean={sum(kls) / len(kls):.3e}",
            flush=True,
        )
        print(f"[GENKL SUMMARY] worst max|Δlp| over batches = {max(maxes):.3e}", flush=True)
        all_exact = all(m == 0.0 for m in maxes)
        print(
            f"[GENKL SUMMARY] VERDICT: {'BITWISE-EXACT ZERO' if all_exact else 'NON-ZERO RESIDUAL'}",
            flush=True,
        )
        print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
