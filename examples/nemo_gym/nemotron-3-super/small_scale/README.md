# H100 Convergence Configs

These configs inherit from the Nemotron 3 Super prod configs one directory up and reduce the train/generation-DP footprint for smaller H100 convergence runs. They keep the full training batch shapes unless a stage-specific note says otherwise.

When launching with `super_launch.sh`, set `SBATCH_NUM_NODES` to the total GPU node count below when it is larger than `cluster.num_nodes`.

| Config | Train DP | NeMo-RL `cluster.num_nodes` | Extra NeMo-Gym GPU nodes | Total GPU nodes needed |
|---|---:|---:|---:|---:|
| `stage1_rlvr_convergence_27node_h100.yaml` | 2 | 20 | 7 | 27 |
| `stage2_swe1_convergence_20node_h100.yaml` | 2 | 20 | 0 | 20 |
| `stage2_swe2_convergence_20node_h100.yaml` | 2 | 20 | 0 | 20 |
| `stage3_rlhf_convergence_28node_h100.yaml` | 8 | 24 | 4 | 28 |

For stages with extra NeMo-Gym GPU nodes, `cluster.num_nodes` only covers the NeMo-RL training and policy generation Ray workers. The larger `SBATCH_NUM_NODES` allocation reserves the remaining nodes for Gym judge/environment servers.
