"""Resolve the two smoke configs and print exactly what setup() will compute.

Runs without Ray or GPUs: config inheritance + _override_ handling, then the
same normalize_weights/compute_quota calls grpo.setup makes.
"""
import sys

from nemo_rl.data.weights import TaskWeightSpec, compute_quota, normalize_weights
from nemo_rl.utils.config import load_config

for path in sys.argv[1:]:
    cfg = load_config(path)
    data, grpo = cfg["data"], cfg["grpo"]
    n = grpo["num_prompts_per_step"]

    print(f"\n=== {path} ===")
    print(f"  use_multiple_dataloader = {data['use_multiple_dataloader']}")
    print(f"  custom_dataloader       = {data['custom_dataloader']}")
    print(f"  num_prompts_per_step    = {n}")
    print(f"  async_grpo.enabled      = {grpo['async_grpo']['enabled']}")
    print(f"  use_dynamic_sampling    = {grpo['use_dynamic_sampling']}")
    print(f"  cluster.gpus_per_node   = {cfg['cluster']['gpus_per_node']}")
    colo = cfg["policy"]["generation"]["colocated"]
    print(f"  generation.colocated    = {colo['enabled']} (resources={colo['resources']})")
    print(f"  train_global_batch_size = {cfg['policy']['train_global_batch_size']}")
    print(f"  gbs == n_prompts*n_gens : "
          f"{cfg['policy']['train_global_batch_size'] == n * grpo['num_generations_per_prompt']}")

    specs = [
        TaskWeightSpec(
            task_name=e["dataset_name"],
            weight=e.get("weight"),
            evaluation_only=bool(e.get("evaluation_only")),
        )
        for e in data["train"]
    ]
    weights = normalize_weights(specs)
    quota = compute_quota(n, weights)
    print(f"  normalized weights      = {dict(weights)}")
    print(f"  >>> task quota          = {quota}  (sum={sum(quota.values())})")
    assert sum(quota.values()) == n, "quota does not sum to num_prompts_per_step"
    assert all(v >= 1 for v in quota.values()), "a task is starved"

print("\nALL CONFIG CHECKS PASSED")
