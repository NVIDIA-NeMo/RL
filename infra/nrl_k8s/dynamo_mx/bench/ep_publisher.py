"""EP-configurable Megatron publisher for the EP->TP2 first-party refit benchmark.

Loads the real Qwen3-30B-A3B MoE in Megatron-Core with expert parallelism EP=N
(one rank per GPU), then publishes each rank's native shards via MxV2TrainingPublisher
so a TP2 receiver can plan+pull+reshard+load through the real MX v2 path.

Launch (torchrun; EP == world size):
  EP=4 single node (4 GPUs):
    MODEL_EXPRESS_URL=... torchrun --nproc_per_node=4 ep_publisher.py
  EP=8 two nodes (4 GPUs each), rank0 node:
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
      --master_addr=<r0-ip> --master_port=29500 ep_publisher.py
  (node1: --node_rank=1)

Runs in the trainer image (jwillthomson/nemo-rl-mx:*) which ships megatron-core,
nemo_rl, and megatron.bridge. See §8b of the JulyAlignment session doc.
"""
from __future__ import annotations
import os, socket, time
from collections import Counter
import torch
import torch.distributed as dist

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
EP = WORLD  # expert-parallel over the whole world (TP=PP=1)
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# Device packing: when nproc_per_node > visible GPUs (e.g. EP8 rank-packed onto a
# 3-GPU node because the cluster has no 4-free node), map ranks round-robin onto the
# visible devices. NCCL among the ranks stays on-node (NVLink). One GPU can host
# several EP ranks; each still owns a distinct expert shard + its own NIXL agent.
_NVIS = torch.cuda.device_count()
DEV_ID = LOCAL_RANK % _NVIS
torch.cuda.set_device(DEV_ID)

dist.init_process_group(backend="nccl", world_size=WORLD, rank=RANK)
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
    expert_model_parallel_size=EP,
)
ep_rank = parallel_state.get_expert_model_parallel_rank()
print(f"[pub r{RANK}] EP={EP} ep_rank={ep_rank} host={socket.gethostname()} "
      f"gpu={DEV_ID} (local_rank={LOCAL_RANK})", flush=True)

from megatron.bridge import AutoBridge
t0 = time.perf_counter()
bridge = AutoBridge.from_hf_pretrained(MODEL_ID, trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=True)
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.expert_model_parallel_size = EP
provider.expert_tensor_parallel_size = 1
provider.bf16 = True
provider.gradient_accumulation_fusion = False
provider.sequence_parallel = False
provider.finalize()
model_list = provider.provide_distributed_model(wrap_with_ddp=False)
model = model_list[0] if isinstance(model_list, list) else model_list
print(f"[pub r{RANK}] model loaded EP={EP} in {time.perf_counter()-t0:.1f}s", flush=True)

from nemo_rl.distributed.mx_megatron_helpers import collect_megatron_publish_set
from modelexpress import MxV2TrainingPublisher, TrainerWorldLayout

pub = MxV2TrainingPublisher(
    agent_name=f"{socket.gethostname()}-ep{EP}-pub-r{ep_rank}",
    device_id=DEV_ID,
    mx_server_url=os.environ.get("MODEL_EXPRESS_URL",
                                 "modelexpress-server.kavin.svc.cluster.local:8001"),
    worker_rank=ep_rank,
    world_layout=TrainerWorldLayout(fsdp_world_size=1, tp_world_size=1,
                                    pp_world_size=1, ep_world_size=EP),
)
pub.initialize(model_name=MODEL_ID, dtype="bfloat16")
pub.set_megatron_mesh_position(tp_rank=0, pp_rank=0, ep_rank=ep_rank)

# sidecar (rank 0 derives the global name map via Bridge; every rank sets it —
# the receiver only needs one, but setting on all is harmless).
tasks = bridge.get_conversion_tasks([model])
name_map_entries = []
for task in tasks:
    m_name = task.global_param_name or task.param_name
    hf_attr = getattr(task.mapping, "hf_param", None)
    if isinstance(hf_attr, str):
        hf_names = [hf_attr]
    elif isinstance(hf_attr, dict):
        hf_names = ([hf_attr["q"], hf_attr["k"], hf_attr["v"]]
                    if set(hf_attr.keys()) == {"q", "k", "v"} else list(hf_attr.values()))
    else:
        continue
    name_map_entries.append((m_name, list(hf_names)))
tcfg = getattr(bridge, "transformer_config", None) or provider
num_heads = getattr(tcfg, "num_attention_heads", None)
kv_groups = getattr(tcfg, "num_query_groups", None) or num_heads
hidden = getattr(tcfg, "hidden_size", None)
kv_channels = getattr(tcfg, "kv_channels", None) or (hidden // num_heads if num_heads else None)
num_experts_total = (getattr(tcfg, "num_moe_experts", None)
                     or getattr(tcfg, "num_experts", None))
num_local_experts = (int(num_experts_total) // EP) if num_experts_total else None
print(f"[pub r{RANK}] num_experts_total={num_experts_total} ep={EP} "
      f"num_local_experts={num_local_experts} (global expert_id = ep_rank*num_local + local)",
      flush=True)
pub.set_megatron_sidecar({
    "megatron_transformer_config": {"num_attention_heads": num_heads,
        "num_query_groups": kv_groups, "kv_channels": kv_channels, "hidden_size": hidden},
    "megatron_hf_name_map": name_map_entries,
})

added = 0
roles: Counter = Counter()
for name, local, spec, extras in collect_megatron_publish_set(
    model, tp_size=1, pp_size=1, pp_rank=0, ep_size=EP, ep_rank=ep_rank, tp_rank=0,
    num_local_experts=num_local_experts,
    num_attention_heads=num_heads, num_kv_heads=kv_groups, head_dim=kv_channels,
    target_dtype=torch.bfloat16,
):
    pub.add_tensor(name=name,
        tensor=local.to(f"cuda:{DEV_ID}").contiguous() if not local.is_cuda else local.contiguous(),
        is_expert=spec.is_expert, expert_axis=spec.expert_axis,
        owned_expert_ids=spec.owned_expert_ids,
        megatron_role=spec.role, megatron_extras=spec.descriptor_extras)
    roles[spec.role] += 1
    added += 1
print(f"[pub r{RANK}] ep_rank={ep_rank} added {added} tensors; roles={dict(roles)}", flush=True)

sid = pub.publish(version=1)
pub.mark_ready()
print(f"[pub r{RANK}] published ep_rank={ep_rank} sid={sid} READY", flush=True)
# No post-publish barrier: each rank publishes independently and the receiver
# discovers all EP ranks from the MX server, so a collective here only risks
# tearing ranks down (NCCL comm can be perturbed by NIXL CUDA registration under
# rank-packing). Ranks must stay alive holding their NIXL agents for the pull.
time.sleep(int(os.environ.get("HOLD_S", "2400")))
pub.shutdown()
