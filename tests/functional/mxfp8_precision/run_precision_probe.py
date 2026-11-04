"""Fresh worker lifetimes, one canonical FP32 bank, controlled forwards before updates."""

import argparse
import copy
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from precision_probe_common import (
    compare_tensor_directory,
    compare_gradient_prefixes,
    save_json,
    tensor_error,
    tensor_hash,
)


def fixed_batch(tokenizer, rows, length):
    prompts = [
        "Calculate 7 times 8. The answer is 56.",
        "The capital of France is Paris. Explain in one sentence.",
        "Write a Python function that adds two integers.",
        "A triangle has three sides. A square has four sides.",
    ]
    data = []
    for row in range(rows):
        ids = tokenizer.encode(prompts[row % len(prompts)], add_special_tokens=False)
        # No sampled labels: all inputs and next-token targets are fixed.
        data.append((ids * (length // len(ids) + 1))[:length])
    ids = torch.tensor(data, dtype=torch.long)
    return {
        "input_ids": ids,
        "input_lengths": torch.full((rows,), length, dtype=torch.int32),
        "attention_mask": torch.ones_like(ids),
        "token_mask": torch.ones_like(ids, dtype=torch.float32),
        "sample_mask": torch.ones(rows),
        "probe_row_id": torch.arange(rows),
    }


def compare_runs(out, mode, reference_mode, case, reference_case, ranks, atol, rtol):
    results = {}
    for rank in range(ranks):
        a = out / mode / case / f"rank-{rank}"
        b = out / reference_mode / reference_case / f"rank-{rank}"
        grads = compare_tensor_directory(
            a / "gradients", b / "gradients", atol=atol, rtol=rtol
        )
        x, y = (
            torch.load(a / "logits.pt", weights_only=True),
            torch.load(b / "logits.pt", weights_only=True),
        )
        logits = {"pass": len(x) == len(y), "calls": []}
        for actual, expected in zip(x, y):
            logits["calls"].append(tensor_error(actual, expected, atol=atol, rtol=rtol))
        logits["pass"] = (
            logits["pass"]
            and bool(logits["calls"])
            and all(v["pass"] for v in logits["calls"])
        )
        results[str(rank)] = {"gradients": grads, "logits": logits}
        if (a / "gradient-prefixes.pt").exists() and (
            b / "gradient-prefixes.pt"
        ).exists():
            results[str(rank)]["gradient_prefixes"] = compare_gradient_prefixes(
                torch.load(a / "gradient-prefixes.pt", weights_only=True),
                torch.load(b / "gradient-prefixes.pt", weights_only=True),
                atol=atol,
                rtol=rtol,
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--recipe-bf16-param", required=True)
    parser.add_argument("--recipe-fp8-param", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-python", required=True)
    parser.add_argument("--refit-matrix", action="store_true")
    parser.add_argument("--fresh-process-control", action="store_true")
    parser.add_argument("--gradient-order-focus", action="store_true")
    parser.add_argument("--fp32-gradient-buffers", action="store_true")
    parser.add_argument("--nodes", type=int, default=2)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--rows-per-dp-rank", type=int, default=4)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.001)
    args = parser.parse_args()
    if args.gradient_order_focus and (args.refit_matrix or args.fresh_process_control):
        parser.error(
            "Gradient-order focus excludes refit and fresh-process comparisons"
        )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    register_omegaconf_resolvers()
    cfg = OmegaConf.to_container(load_config(args.config), resolve=True)
    policy_cfg = copy.deepcopy(cfg["policy"])
    policy_cfg["model_name"] = args.model
    policy_cfg["tokenizer"]["name"] = args.model
    if args.model.startswith("/"):
        assert Path(args.model).is_dir(), "Explicit model path does not exist"
    policy_cfg["generation"]["temperature"] = 1.0
    policy_cfg["generation"]["val_temperature"] = 1.0
    # A trainer-only probe must not inherit an unexercised native-refit flag.
    policy_cfg["megatron_cfg"]["native_mxfp8_refit"] = False
    if args.fp32_gradient_buffers:
        policy_cfg["megatron_cfg"]["distributed_data_parallel_config"][
            "grad_reduce_in_fp32"
        ] = True
    policy_cfg["megatron_cfg"]["tensor_model_parallel_size"] = 4
    policy_cfg["megatron_cfg"]["expert_model_parallel_size"] = 8
    policy_cfg["megatron_cfg"]["context_parallel_size"] = 1
    policy_cfg["train_micro_batch_size"] = policy_cfg["logprob_batch_size"] = 1
    policy_cfg["train_global_batch_size"] = args.rows_per_dp_rank * (
        args.nodes * args.gpus_per_node // 4
    )
    policy_cfg["max_total_sequence_length"] = args.seq_len
    # Identical sequence packing and activation-quantization grouping in the
    # combined and split controls: each microbatch contains one fixed sequence.
    policy_cfg["sequence_packing"]["max_sequences_per_bin"] = 1
    policy_cfg["megatron_cfg"]["checkpoint"]["async_save"] = False
    policy_cfg["megatron_cfg"]["train_iters"] = 2
    # Replay routing metadata is intentionally absent: identical fixed input and
    # unmodified router parameters define the comparison, with both modes equal.
    policy_cfg["router_replay"] = {"enabled": False}
    tokenizer = get_tokenizer(policy_cfg["tokenizer"])
    data = fixed_batch(tokenizer, args.rows_per_dp_rank, args.seq_len)
    from probe_batches import verify_probe_batch_preparation

    verify_probe_batch_preparation(data, policy_cfg)
    save_json(
        out / "batch-preparation.json",
        {"status": "PASS", "full_split_and_logprob": True},
    )
    torch.save(data, out / "fixed-batch.pt")
    save_json(
        out / "workload.json",
        {
            "input_hash": tensor_hash(data["input_ids"]),
            "sampling": False,
            "optimizer_updates": False,
            "atol": args.atol,
            "rtol": args.rtol,
            "configuration": policy_cfg,
        },
    )
    ACTOR_ENVIRONMENT_REGISTRY["precision_probe_worker.PrecisionProbeWorker"] = (
        args.worker_python
    )
    init_ray()
    masters = {}
    modes = [(False, False), (True, False)]
    if args.refit_matrix:
        from refit_probe import configure_refit

        policy_cfg = configure_refit(policy_cfg, tokenizer, args.seq_len)
        modes.append((True, True))
    if args.fresh_process_control:
        modes.append((True, False))
    labels = []
    for mode_index, (fp8, native) in enumerate(modes):
        fresh_control = args.fresh_process_control and mode_index == len(modes) - 1
        mode = "fp8_param_true" if fp8 else "fp8_param_false"
        if native:
            mode += "_native_refit"
        if fresh_control:
            mode += "_fresh_control"
        labels.append(mode)
        config = copy.deepcopy(policy_cfg)
        config["megatron_cfg"]["native_mxfp8_refit"] = native
        config["megatron_cfg"]["fp8_cfg"]["fp8_param"] = fp8
        config["megatron_cfg"]["te_precision_config_file"] = (
            args.recipe_fp8_param if fp8 else args.recipe_bf16_param
        )
        save_json(out / (mode + "-configuration.json"), config)
        cluster = RayVirtualCluster(
            name="precision-" + mode,
            bundle_ct_per_node_list=[args.gpus_per_node] * args.nodes,
            num_gpus_per_node=args.gpus_per_node,
            use_gpus=True,
            max_colocated_worker_groups=1,
        )
        policy = None
        try:
            policy = Policy(
                cluster=cluster,
                config=config,
                tokenizer=tokenizer,
                init_reference_model=False,
                worker_extension_cls_fqn="precision_probe_worker.PrecisionProbeWorker",
            )

            def call(method, **kw):
                return ray.get(
                    policy.worker_group.run_all_workers_single_data(method, **kw)
                )

            masters[mode] = call(
                "canonical_masters",
                folder=str(out / "fp32-master-bank"),
                export=not fp8,
            )
            save_json(out / (mode + "-masters.json"), masters[mode])
            cases = [
                ("combined", False, False),
                ("repeat", False, False),
                ("chunked", True, False),
                ("streamed", True, True),
            ]
            if args.gradient_order_focus:
                cases.extend(
                    [
                        ("combined_packer_order", False, False),
                        ("chunked_packer_order", True, False),
                    ]
                )
            for case, streaming, detour in cases:
                if fresh_control and case != "combined":
                    continue
                call(
                    "teacher_forced_probe",
                    data=data,
                    folder=str(out / mode / case),
                    streaming=streaming,
                    logprob_detour=detour,
                    canonical_order=case
                    not in ["combined_packer_order", "chunked_packer_order"],
                    atol=args.atol,
                    rtol=args.rtol,
                )
            if not fresh_control and not args.gradient_order_focus:
                call("test_stale_storage", folder=str(out / mode / "stale"), data=data)
            if args.refit_matrix and not fresh_control:
                from refit_probe import run_refit_probe

                run_refit_probe(policy, cluster, config, data, out / mode / "refit")
        finally:
            if policy is not None:
                policy.shutdown()
            cluster.shutdown()
    assert all(value == masters[labels[0]] for value in masters.values()), (
        "Canonical FP32 masters differ across storage modes"
    )
    ranks = args.nodes * args.gpus_per_node
    comparisons = {}
    for mode in labels[:-1] if args.fresh_process_control else labels:
        comparisons[mode + "_repeat"] = compare_runs(
            out, mode, mode, "repeat", "combined", ranks, args.atol, args.rtol
        )
        comparisons[mode + "_streamed"] = compare_runs(
            out, mode, mode, "streamed", "combined", ranks, args.atol, args.rtol
        )
        comparisons[mode + "_chunking_only"] = compare_runs(
            out, mode, mode, "chunked", "combined", ranks, args.atol, args.rtol
        )
        comparisons[mode + "_logprob_detour_only"] = compare_runs(
            out, mode, mode, "streamed", "chunked", ranks, args.atol, args.rtol
        )
    order_sensitivity = {}
    if args.gradient_order_focus:
        for mode in labels:
            order_sensitivity[mode] = compare_runs(
                out,
                mode,
                mode,
                "chunked_packer_order",
                "combined_packer_order",
                ranks,
                args.atol,
                args.rtol,
            )
        save_json(out / "order-sensitivity.json", order_sensitivity)
    else:
        comparisons["cross_storage"] = compare_runs(
            out,
            "fp8_param_true",
            "fp8_param_false",
            "combined",
            "combined",
            ranks,
            args.atol,
            args.rtol,
        )
    if args.fresh_process_control:
        comparisons["fresh_process_repeat"] = compare_runs(
            out,
            labels[-1],
            "fp8_param_true",
            "combined",
            "combined",
            ranks,
            args.atol,
            args.rtol,
        )
    if args.refit_matrix:
        from refit_probe import compare_refit_outputs

        comparisons["native_flag_training"] = compare_runs(
            out,
            labels[2],
            labels[1],
            "combined",
            "combined",
            ranks,
            args.atol,
            args.rtol,
        )
        refit_results = compare_refit_outputs(out, labels[:3], args.atol, args.rtol)
        save_json(out / "refit-matrix.json", refit_results)
    passed = all(
        v["gradients"]["pass"] and (v["logits"] is None or v["logits"]["pass"])
        for comparison in comparisons.values()
        for v in comparison.values()
    )
    if args.refit_matrix:
        passed = passed and all(v["pass"] for v in refit_results.values())
    save_json(
        out / "result.json",
        {
            "status": "PASS" if passed else "FAIL",
            "identical_fp32_masters": True,
            "refit_matrix_exercised": args.refit_matrix,
            "gradient_order_focus": args.gradient_order_focus,
            "fp32_gradient_buffers": args.fp32_gradient_buffers,
            "order_sensitivity_is_diagnostic_only": args.gradient_order_focus,
            "comparisons": comparisons,
        },
    )
    raise SystemExit(not passed)


if __name__ == "__main__":
    main()
