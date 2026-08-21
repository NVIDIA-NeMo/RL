# Shared Nemotron 3 Nano v3 (30B-A3B) overrides for smoke-testing the Ultra
# pipeline on GB200 NVL72 nodes (4 GPUs/node). Include this AFTER the Ultra
# stage config so it overrides the Ultra-scale mesh and sizing, e.g.:
#
#   defaults:
#     - swe_teacher.yaml            # Ultra stage config (rewards, data, optimizer)
#     - _nano_smoke_gb200.yaml.inc  # this file (shrink to a Nano smoke shape)
#
# Stage-specific sequence lengths, rewards, data routing, and optimizer settings
# are inherited from the corresponding Ultra config. This file ONLY changes what
# is common to every Nano smoke run: the GPU mesh, the smoke sizing, and the
# Nano-vs-Ultra architecture toggles (no MTP). Per-stage judge shrinking lives in
# the nano_<stage>.yaml files, because each stage serves a different set of Gym
# judge models.

# =============================================================================
# Cluster — GB200 NVL72, 4 GPUs/node. num_nodes is overridden by the launcher
# (NUM_ACTOR_NODES).
#
# segment_size MUST be disabled (null). It is an NVL72 topology filter: at Ultra
# scale (segment_size=16) the virtual cluster keeps only nodes that form a
# complete 16-node NVLink segment and DISCARDS the rest — per
# virtual_cluster._filter_bundles_by_segment:
#     usable_nodes = (domain_node_count // segment_size) * segment_size
# On a ~5-node smoke allocation that is (5 // 16) * 16 = 0 usable nodes, so every
# placement bundle is discarded and nothing can schedule. null takes the early
# return in build_topology_aware_ordering and skips topology filtering entirely.
# (Distinct from the launcher's SEGMENT_SIZE env var, which is only the sbatch
# divisibility check.)
# =============================================================================
cluster:
  gpus_per_node: 4
  num_nodes: 5          # train(4) + gen(1); overridden by ultra_launch.sh
  segment_size: null    # disable NVLink-domain segment filtering for small allocs

# =============================================================================
# GRPO — shrink the step so a smoke run completes in minutes.
# =============================================================================
grpo:
  num_prompts_per_step: 16
  num_generations_per_prompt: 4    # GBS = 16 * 4 = 64 (see policy below)
  val_num_generations_per_prompt: 2
  val_period: 100000               # validation effectively off for the smoke run
  val_at_start: false
  val_at_end: false
  overlong_filtering: true         # drop prompts longer than the (small) context

# =============================================================================
# Checkpointing — save less often so the 30B save doesn't dominate a short run.
# =============================================================================
checkpointing:
  save_period: 100
  checkpoint_must_save_by: null    # let walltime end the job naturally

# =============================================================================
# Policy — Nano v3 30B-A3B, small batch + short context.
# model_name / tokenizer are overridden by the launcher (MODEL_PATH).
# =============================================================================
policy:
  train_global_batch_size: 64      # = num_prompts_per_step * num_generations_per_prompt
  max_total_sequence_length: 16384 # down from the Ultra stage lengths

  megatron_cfg:
    # --- GPU mesh grid (the core smoke-test change) -------------------------
    # world = TP*PP*CP*DP = 2*1*4*2 = 16 (4 GB200 nodes); experts sharded over
    # EP=8 (EDP=2). Nano has far fewer experts than Ultra, so EP shrinks too.
    tensor_model_parallel_size: 2     # was 8
    expert_tensor_parallel_size: 1
    expert_model_parallel_size: 8     # was 32/64
    context_parallel_size: 4          # was 8/32
    pipeline_model_parallel_size: 1   # unchanged (mirror Ultra)

    # --- MTP off for the Nano base checkpoint -------------------------------
    # The Nano base checkpoint is not guaranteed to ship MTP heads.
    mtp_num_layers: 0
    mtp_loss_scaling_factor: 0.0

  generation:
    max_new_tokens: 4096             # cap generation length for a fast smoke run
    vllm_cfg:
      tensor_parallel_size: 4        # was 8 -> one GB200 node
      expert_parallel_size: 1        # was 8 (EP=TP not needed at this scale)
      pipeline_parallel_size: 1
      max_model_len: 16384

    colocated:
      enabled: false
      resources:
        gpus_per_node: 4
        num_nodes: 1                 # overridden by launcher (NUM_GEN_NODES)
