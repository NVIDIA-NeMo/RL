"""Nano storage A/B at initialization, shared buffers, reduction, resume and refit."""

import argparse
import copy
import json
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from precision_probe_common import save_json, compare_tensor_directory
from refit_probe import configure_refit, run_refit_probe
from run_precision_probe import fixed_batch, compare_runs
from boundary_probe_common import compare_master_snapshots
from boundary_refit_compare import compare_refit_boundaries
from probe_batches import verify_probe_batch_preparation
from precision_probe_common import tensor_error


def main():
    parser = argparse.ArgumentParser()
    for field in ["config", "model", "recipe-bf16-param", "recipe-fp8-param", "worker-python", "output"]:
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--train-nodes", type=int, default=4)
    parser.add_argument("--context-parallel", type=int, default=4)
    parser.add_argument("--expert-parallel", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    register_omegaconf_resolvers()
    cfg = OmegaConf.to_container(load_config(args.config), resolve=True)["policy"]
    cfg["model_name"] = cfg["tokenizer"]["name"] = args.model
    cfg["generation"]["temperature"] = cfg["generation"]["val_temperature"] = 1.0
    mc = cfg["megatron_cfg"]
    mc["native_mxfp8_refit"] = False
    mc["tensor_model_parallel_size"] = 4
    mc["context_parallel_size"] = args.context_parallel
    mc["expert_model_parallel_size"] = args.expert_parallel
    mc["distributed_data_parallel_config"]["overlap_grad_reduce"] = False
    mc["checkpoint"]["async_save"] = False
    mc["train_iters"] = 3
    cfg["train_micro_batch_size"] = cfg["logprob_batch_size"] = 1
    cfg["train_global_batch_size"] = 2 * (args.train_nodes * 4 // (4 * args.context_parallel))
    cfg["max_total_sequence_length"] = args.seq_len
    cfg["sequence_packing"]["max_sequences_per_bin"] = 1
    cfg["router_replay"] = {"enabled": False}
    tokenizer = get_tokenizer(cfg["tokenizer"])
    cfg = configure_refit(cfg, tokenizer, args.seq_len)
    batch = fixed_batch(tokenizer, 2, args.seq_len)
    verify_probe_batch_preparation(batch, cfg)
    save_json(out / "scope.json", {"trainer_nodes": args.train_nodes, "tp": 4,
        "ep": args.expert_parallel, "cp": args.context_parallel,
        "fixed_sequences_per_dp_rank": 2, "objective": "NLL",
        "native_mxfp8_refit": False, "private_seed_optimizer_updates_per_mode": 1,
        "audited_optimizer_updates": 0, "configuration": cfg})
    ACTOR_ENVIRONMENT_REGISTRY["boundary_probe_worker.BoundaryProbeWorker"] = args.worker_python
    init_ray()
    bank = out / "fp32-master-bank"
    signatures = {}
    initial = {}
    for resumed in [False, True]:
        for fp8 in [False, True]:
            label = f"fp8_{str(fp8).lower()}_" + ("resume" if resumed else "fresh")
            config = copy.deepcopy(cfg)
            config["megatron_cfg"]["fp8_cfg"]["fp8_param"] = fp8
            config["megatron_cfg"]["te_precision_config_file"] = args.recipe_fp8_param if fp8 else args.recipe_bf16_param
            config["generation"]["vllm_cfg"]["env_vars"] = {
                **config["generation"]["vllm_cfg"].get("env_vars", {}),
                "NANO_BOUNDARY_ROOT": "/audit/boundaries", "NANO_BOUNDARY_MODE": label,
                "VLLM_RAY_EXTRA_ENV_VARS": "NANO_BOUNDARY_ROOT,NANO_BOUNDARY_MODE,PYTHONPATH",
            }
            cluster = RayVirtualCluster(name="boundary-" + label,
                bundle_ct_per_node_list=[4] * args.train_nodes, num_gpus_per_node=4,
                use_gpus=True, max_colocated_worker_groups=1)
            policy = None
            checkpoint = out / f"checkpoint-fp8-{str(fp8).lower()}"
            try:
                policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer,
                    init_reference_model=False,
                    weights_path=str(checkpoint) if resumed else None,
                    optimizer_path=str(checkpoint) if resumed else None,
                    worker_extension_cls_fqn="boundary_probe_worker.BoundaryProbeWorker")

                def call(method, **kwargs):
                    result = ray.get(policy.worker_group.run_all_workers_single_data(method, **kwargs))
                    assert len(result) == args.train_nodes * 4, (method, "Missing worker coverage")
                    if all(isinstance(v, dict) and "rank" in v for v in result):
                        assert {v["rank"] for v in result} == set(range(args.train_nodes * 4)), (method, "Missing/duplicate ranks")
                    return result

                loaded = call("loaded_snapshot", folder=str(out / label / "as-loaded"))
                save_json(out / label / "as-loaded.json", loaded)
                if not resumed:
                    initial[label] = loaded
                if not resumed:
                    call("initialize_private_optimizer_state", data=batch)
                    call("canonical_masters", folder=str(bank), export=not fp8)
                    ready = call("loaded_snapshot", folder=str(out / label / "canonical"))
                else:
                    ready = loaded
                    call("load_probe_controls", folder=str(bank))
                signatures[label] = [{"rank": r["rank"], "masters": r["masters"]} for r in ready]
                save_json(out / label / "masters.json", signatures[label])
                equality = compare_master_snapshots(signatures[label], signatures["fp8_false_fresh"])
                save_json(out / label / "master-equality.json", equality)
                assert equality["equal"], (label, "FP32 master mismatch", equality)
                call("arm_boundary_observers", folder=str(out / label / "buffers"))
                if not resumed:
                    call("verify_and_publish_current_masters")
                    synthetic = call("synthetic_reduction_probe", folder=str(out / label / "synthetic-reduction"))
                    save_json(out / label / "synthetic-source.json", synthetic)
                cases = [("combined", False, False)] if resumed else [
                    ("combined", False, False), ("repeat", False, False), ("streamed", True, True)]
                for case, streaming, detour in cases:
                    call("teacher_forced_probe", data=batch, folder=str(out / label / case),
                        streaming=streaming, logprob_detour=detour, canonical_order=True,
                        atol=0.001, rtol=0.001)
                run_refit_probe(policy, cluster, config, batch, out / label / "refit")
                if not resumed:
                    call("stale_boundary_probe", folder=str(out / label / "stale"), data=batch)
                    policy.save_checkpoint(weights_path=str(checkpoint), optimizer_path=str(checkpoint), is_final_checkpoint=False)
                save_json(out / "progress.json", {"completed_modes": list(signatures), "last_mode": label})
            finally:
                if policy is not None:
                    policy.shutdown()
                cluster.shutdown()
    ranks = args.train_nodes * 4
    assert all(compare_master_snapshots(s, signatures["fp8_false_fresh"])["equal"] for s in signatures.values()), "Loaded FP32 masters differ across canonical/resumed lifetimes"
    reports = {}
    refit_results = {}
    for a, b in [("fp8_true_fresh", "fp8_false_fresh"),
                 ("fp8_false_resume", "fp8_false_fresh"),
                 ("fp8_true_resume", "fp8_true_fresh")]:
        result = compare_refit_boundaries(out.parent / "boundaries", a, b)
        x, y = [torch.load(out / label / "refit" / "first.pt", weights_only=True) for label in [a, b]]
        same = torch.equal(x["output_ids"], y["output_ids"]) and torch.equal(x["generation_lengths"], y["generation_lengths"])
        result["rollout"] = {"same_tokens_and_lengths": same, "logprobs": tensor_error(x["logprobs"], y["logprobs"], atol=0, rtol=0) if same else None}
        refit_results[a + "_vs_" + b] = result
    save_json(out / "refit-boundaries.json", refit_results)
    sources = [json.loads((out / label / "synthetic-source.json").read_text())
               for label in ["fp8_false_fresh", "fp8_true_fresh"]]
    assert {v["rank"]: v["source"] for v in sources[0]} == {v["rank"]: v["source"] for v in sources[1]}, "Synthetic local gradient inputs differ across storage modes"
    reports["identical_synthetic_local_gradients"] = {
        str(rank): compare_tensor_directory(
            out / "fp8_true_fresh" / "synthetic-reduction" / f"rank-{rank}" / "gradients",
            out / "fp8_false_fresh" / "synthetic-reduction" / f"rank-{rank}" / "gradients",
            atol=0, rtol=0) for rank in range(ranks)
    }
    pairs = [("fp8_true_fresh", "fp8_false_fresh", "combined", "combined"),
             ("fp8_false_resume", "fp8_false_fresh", "combined", "combined"),
             ("fp8_true_resume", "fp8_true_fresh", "combined", "combined")]
    for label in ["fp8_false_fresh", "fp8_true_fresh"]:
        pairs.extend([(label, label, "repeat", "combined"), (label, label, "streamed", "combined")])
    for a, b, ca, cb in pairs:
        key = f"{a}/{ca}_vs_{b}/{cb}"
        reports[key] = compare_runs(out, a, b, ca, cb, ranks, 0.001, 0.001)
        for rank in range(ranks):
            reports[key][str(rank)]["reduced_gradients"] = compare_tensor_directory(
                out / a / ca / f"rank-{rank}" / "reduced-gradients",
                out / b / cb / f"rank-{rank}" / "reduced-gradients", atol=0.001, rtol=0.001)
    save_json(out / "result.json", {"completed": True, "identical_fp32_masters": True,
        "as_loaded_initial_masters": compare_master_snapshots(initial["fp8_true_fresh"], initial["fp8_false_fresh"]),
        "comparisons": reports, "refit": refit_results})


if __name__ == "__main__":
    main()
