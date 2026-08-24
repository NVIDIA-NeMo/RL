from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DFLASH_REVISION = "9b41424b7109f9c5413454f481b09a82b85333f4"
DSPARK_REVISION = "03326e5043815da1f81b109078b2889737c26017"
USER_ROOT = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna"
TARGET_SNAPSHOT = (
    f"{USER_ROOT}/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/{TARGET_REVISION}"
)
DFLASH_SNAPSHOT = (
    f"{USER_ROOT}/hf_home/hub/models--z-lab--Qwen3-8B-DFlash-b16/"
    f"snapshots/{DFLASH_REVISION}"
)
DSPARK_SNAPSHOT = (
    f"{USER_ROOT}/hf_home/hub/models--deepseek-ai--dspark_qwen3_8b_block7/"
    f"snapshots/{DSPARK_REVISION}"
)
CONTAINER = f"{USER_ROOT}/containers/nemo_rl_nightly_20260818_20260818_6296116.sqsh"
CONTAINER_SHA256 = "6940409542de6669f77e91c7ce7aac0ef7e91bd56839772e1ae7efc371718d44"
CHECKPOINT_STEPS = (50, 100, 150, 200)
WINDOW = (21, 200)

ADAPTIVE_SCHEDULE: dict[str, object] = {
    "mode": "adaptive",
    "action": "sparse_update",
    "min_interval": 5,
    "max_interval": 20,
    "ewma_alpha": 0.1,
    "degradation_threshold": 0.02,
    "recovery_threshold": 0.01,
    "min_observations": 20,
    "max_burst_updates": 10,
}

Drafter = Literal["none", "dflash", "dspark"]
Cadence = Literal[
    "baseline", "static", "always", "fixed-5", "fixed-10", "fixed-20", "adaptive"
]


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    drafter: Drafter
    cadence: Cadence
    schedule: dict[str, object] | None
    config_path: str
    target_revision: str = TARGET_REVISION
    drafter_revision: str | None = None
    target_snapshot: str = TARGET_SNAPSHOT
    drafter_snapshot: str | None = None
    max_steps: int = 200
    seed: int = 42
    output_sequence_length: int = 1024
    global_batch_size: int = 8
    prompts_per_step: int = 2
    generations_per_prompt: int = 4
    nodes: int = 1
    gpus_per_node: int = 4
    tensor_parallel_size: int = 2
    context_parallel_size: int = 1
    dataset: str = "DAPOMath17K"
    k: int = 5
    wandb_entity: str = "nvidia"
    wandb_project: str = "sna-specdec"
    wandb_group: str = "qwen3-8b-dflash-dspark-cadence-200step-v1"

    @property
    def wandb_name(self) -> str:
        if self.drafter == "none":
            return "q8-cadence-200-baseline-nospec-seed42"
        return f"q8-cadence-200-{self.name}-k5-seed42"

    def deterministic_update_steps(self) -> tuple[int, ...]:
        if self.cadence == "baseline" or self.cadence == "static":
            return ()
        if self.cadence == "always":
            return tuple(range(1, self.max_steps + 1))
        if self.cadence.startswith("fixed-"):
            interval = int(self.cadence.split("-", 1)[1])
            return tuple(range(interval, self.max_steps + 1, interval))
        raise ValueError("adaptive update steps are data-dependent")

    def validate_product_source(self, source_root: Path) -> None:
        runtime_source = (
            source_root / "nemo_rl/algorithms/draft_cadence_runtime.py"
        ).read_text()
        sync_source = (source_root / "nemo_rl/algorithms/grpo_sync.py").read_text()
        interface_source = (
            source_root / "nemo_rl/weight_sync/interfaces.py"
        ).read_text()
        tq_source = (source_root / "nemo_rl/models/policy/tq_policy.py").read_text()
        failures = []
        if (
            "adaptive draft cadence requires selected-rollout acceptance provenance"
            in runtime_source
        ):
            failures.append("adaptive runtime remains explicitly rejected")
        if "prepare_sync_draft_decision(" not in sync_source:
            failures.append("sync loop does not prepare count-weighted decisions")
        if sync_source.count("apply_scheduled_refit(") < 2:
            failures.append("sync loop does not call the scheduled refit finalizer")
        if "draft_apply_receipt" not in interface_source:
            failures.append("weight-sync interface does not declare apply receipts")
        if "supports_draft_apply_receipts" not in tq_source:
            failures.append("TQ policy does not advertise apply receipts")
        if failures:
            raise RuntimeError(
                "production cadence integration is incomplete: " + "; ".join(failures)
            )


def _online_arm(drafter: Literal["dflash", "dspark"], cadence: Cadence) -> Arm:
    if cadence == "static":
        schedule: dict[str, object] = {
            "mode": "fixed",
            "action": "sparse_update",
            "fixed_interval": 201,
        }
    elif cadence == "always":
        schedule = {"mode": "always"}
    elif cadence.startswith("fixed-"):
        schedule = {
            "mode": "fixed",
            "action": "sparse_update",
            "fixed_interval": int(cadence.split("-", 1)[1]),
        }
    elif cadence == "adaptive":
        schedule = dict(ADAPTIVE_SCHEDULE)
    else:
        raise ValueError(f"unsupported cadence: {cadence}")
    if drafter == "dflash":
        revision = DFLASH_REVISION
        snapshot = DFLASH_SNAPSHOT
        recipe = "examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dflash.yaml"
    else:
        revision = DSPARK_REVISION
        snapshot = DSPARK_SNAPSHOT
        recipe = "examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dspark.yaml"
    return Arm(
        name=f"{drafter}-{cadence}",
        drafter=drafter,
        cadence=cadence,
        schedule=schedule,
        config_path=recipe,
        drafter_revision=revision,
        drafter_snapshot=snapshot,
    )


def build_arms() -> tuple[Arm, ...]:
    baseline = Arm(
        name="baseline",
        drafter="none",
        cadence="baseline",
        schedule=None,
        config_path="examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dflash.yaml",
    )
    cadence_order: tuple[Cadence, ...] = (
        "static",
        "always",
        "fixed-5",
        "fixed-10",
        "fixed-20",
        "adaptive",
    )
    return (
        baseline,
        *(_online_arm("dflash", cadence) for cadence in cadence_order),
        *(_online_arm("dspark", cadence) for cadence in cadence_order),
    )


def _value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ",".join(_value(item) for item in value) + "]"
    return str(value)


def render_hydra_overrides(arm: Arm, *, result_dir: str) -> tuple[str, ...]:
    overrides = [
        f"grpo.max_num_steps={arm.max_steps}",
        f"grpo.seed={arm.seed}",
        f"grpo.num_prompts_per_step={arm.prompts_per_step}",
        f"grpo.num_generations_per_prompt={arm.generations_per_prompt}",
        "grpo.val_period=1000000",
        "grpo.val_at_start=false",
        "grpo.val_at_end=false",
        "grpo.async_grpo.enabled=false",
        "policy.max_total_sequence_length=4096",
        f"policy.train_global_batch_size={arm.global_batch_size}",
        "policy.train_micro_batch_size=1",
        "policy.logprob_batch_size=1",
        f"policy.megatron_cfg.tensor_model_parallel_size={arm.tensor_parallel_size}",
        "policy.megatron_cfg.pipeline_model_parallel_size=1",
        f"policy.megatron_cfg.context_parallel_size={arm.context_parallel_size}",
        "policy.megatron_cfg.sequence_parallel=false",
        "policy.sequence_packing.enabled=false",
        f"policy.generation.max_new_tokens={arm.output_sequence_length}",
        "policy.generation.vllm_cfg.max_model_len=4096",
        "policy.generation.vllm_cfg.gpu_memory_utilization=0.7",
        "policy.generation.vllm_kwargs.max_num_seqs=8",
        "policy.generation.vllm_kwargs.compilation_config.backend=eager",
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE",
        "policy.generation.vllm_kwargs.compilation_config.cudagraph_capture_sizes=[1,2,4,6,8,10,12,16,18,20,24,28,30,32,36,40,42,48,50,56,60,64]",
        "data.max_input_seq_length=2048",
        "data.shuffle=true",
        f"data.train.dataset_name={arm.dataset}",
        f"data.train.seed={arm.seed}",
        "data_plane.enabled=true",
        "checkpointing.enabled=true",
        "checkpointing.save_period=50",
        "checkpointing.keep_top_k=null",
        "checkpointing.metric_name=null",
        "checkpointing.save_optimizer=true",
        f"checkpointing.checkpoint_dir={result_dir}/checkpoints",
        "cadence_runtime.enabled=true",
        f"cadence_runtime.result_dir={result_dir}",
        "cadence_runtime.required_checkpoint_steps=[50,100,150,200]",
        "logger.wandb_enabled=true",
        f"logger.log_dir={result_dir}/logs",
        f"logger.wandb.project={arm.wandb_project}",
        f"logger.wandb.entity={arm.wandb_entity}",
        f"logger.wandb.group={arm.wandb_group}",
        f"logger.wandb.name={arm.wandb_name}",
        "cluster.num_nodes=1",
        f"cluster.gpus_per_node={arm.gpus_per_node}",
        f"policy.model_name={arm.target_snapshot}",
        f"policy.tokenizer.name={arm.target_snapshot}",
    ]
    if arm.drafter == "none":
        overrides.extend(
            (
                "policy.draft.enabled=false",
                "policy.draft.optimizer=null",
                "policy.generation.vllm_kwargs.speculative_config=null",
            )
        )
    else:
        assert arm.drafter_snapshot is not None
        overrides.extend(
            (
                "policy.draft.enabled=true",
                f"policy.draft.model_name={arm.drafter_snapshot}",
                f"policy.draft.model_revision={arm.drafter_revision}",
                "policy.draft.optimizer.lr=5e-06",
                "policy.draft.optimizer.min_lr=5e-07",
                "policy.draft.optimizer.weight_decay=0.01",
                f"policy.generation.vllm_kwargs.speculative_config.model={arm.drafter_snapshot}",
                f"policy.generation.vllm_kwargs.speculative_config.revision={arm.drafter_revision}",
                f"policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens={arm.k}",
            )
        )
        assert arm.schedule is not None
        overrides.extend(
            f"policy.draft.update_schedule.{key}={_value(value)}"
            for key, value in arm.schedule.items()
        )
    return tuple(
        override if override.startswith(("+", "~")) else f"++{override}"
        for override in overrides
    )
