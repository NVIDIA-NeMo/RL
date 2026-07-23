#!/usr/bin/env python3
"""Multi-LoRA SFT runner — NeMo-RL native (no external packages).

Trains N LoRA adapters in ONE job on a shared frozen base with per-microbatch
adapter routing, using the multi-LoRA modules vendored at
``nemo_rl.models.multi_lora`` (originally developed in nousnet).

Usage (mirrors examples/run_sft.py):
    uv run python examples/run_sft_multi_lora.py --config <full_nemo_rl_config.yaml> [hydra.overrides ...]

The config is a standard NeMo-RL SFT config plus a ``multi_lora:`` block:
    multi_lora:
      enabled: true
      batch_size_per_adapter: 16
      adapters:
        - name: adapter_a
          lora_cfg: {...}
          data: {train: {data_path: ...}, validation: {data_path: ...}}
        - ...

Single-LoRA configs (no multi_lora block / enabled: false) fall through to the
stock NLLLoss + stock dataloader — same code path as examples/run_sft.py.

Env knobs (kept verbatim from the equivalence campaign so artifacts stay
comparable): NOUSNET_DIAG_ENABLED, NOUSNET_DIAG_LOSS_TRACE,
NOUSNET_DETERMINISTIC_SEED, NOUSNET_INIT_IMPORT_DIR, NOUSNET_INIT_SLOT,
NOUSNET_PER_ADAPTER_GRAD_CLIP.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to full NeMo-RL SFT YAML config")
    args, overrides = ap.parse_known_args()
    return args, overrides


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, overrides = parse_args()

    # Bit-equivalence determinism (workers re-call this from their own entry hook).
    if os.environ.get("NOUSNET_DIAG_ENABLED", "0") == "1":
        from nemo_rl.models.multi_lora.diag import enable_absolute_determinism

        status = enable_absolute_determinism(
            seed=int(os.environ.get("NOUSNET_DETERMINISTIC_SEED", "42"))
        )
        logger.info("multi_lora diag determinism: %s", status)

    from omegaconf import OmegaConf

    from nemo_rl.algorithms.sft import setup, sft_train
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.utils.config import load_config, parse_hydra_overrides
    from nemo_rl.utils.logger import get_next_experiment_dir

    master_config = load_config(args.config)
    if overrides:
        master_config = parse_hydra_overrides(master_config, overrides)
    master_config = OmegaConf.to_container(master_config, resolve=True)

    master_config["logger"]["log_dir"] = get_next_experiment_dir(
        master_config["logger"]["log_dir"]
    )
    logger.info("Log directory: %s", master_config["logger"]["log_dir"])

    tokenizer = get_tokenizer(master_config["policy"]["tokenizer"])
    init_ray()

    # --- data setup (same path as examples/run_sft.py) ---
    from nemo_rl.data.datasets import AllTaskProcessedDataset, load_response_dataset
    from nemo_rl.data.datasets.utils import update_single_dataset_config
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.processors import sft_processor

    data_config = master_config["data"]
    default_task_spec = TaskDataSpec(
        task_name="sft_default",
        prompt_file=data_config.get("default", {}).get("prompt_file"),
        system_prompt_file=data_config.get("default", {}).get("system_prompt_file"),
    )
    task_data_processors = defaultdict(lambda: (default_task_spec, sft_processor))

    # Merge data.default into train/validation so `processor:` keys propagate;
    # without this set_processor() falls back to math_hf_data_processor which
    # drops the assistant turn and zeros the loss mask -> Loss=0.0000.
    default_cfg = data_config.get("default") or {}
    if default_cfg:
        update_single_dataset_config(data_config["train"], default_cfg)

    train_data = load_response_dataset(data_config["train"])
    task_data_processors[train_data.task_name] = (train_data.task_spec, train_data.processor)
    train_dataset = AllTaskProcessedDataset(
        train_data.dataset,
        tokenizer,
        default_task_spec,
        task_data_processors,
        max_seq_length=data_config.get("max_input_seq_length"),
    )

    val_dataset = None
    if data_config.get("validation"):
        if default_cfg:
            update_single_dataset_config(data_config["validation"], default_cfg)
        val_data = load_response_dataset(data_config["validation"])
        task_data_processors[val_data.task_name] = (val_data.task_spec, val_data.processor)
        val_dataset = AllTaskProcessedDataset(
            val_data.dataset,
            tokenizer,
            default_task_spec,
            task_data_processors,
            max_seq_length=data_config.get("max_input_seq_length"),
        )

    # --- multi-LoRA pluggable wiring (vendored modules) ---
    ml_block = master_config.get("multi_lora") or {}
    loss_fn_factory = None
    train_dataloader_override = None
    if ml_block.get("enabled"):
        from nemo_rl.models.multi_lora.config import MultiLoRAConfig
        from nemo_rl.models.multi_lora.data import MultiAdapterDataLoader
        from nemo_rl.models.multi_lora.loss import MultiAdapterLoss

        ml_cfg = MultiLoRAConfig.from_dict(ml_block)
        ml_cfg.validate()
        logger.info(
            "multi_lora: enabled with %d adapters (%s)",
            len(ml_cfg.adapters),
            [a.name for a in ml_cfg.adapters],
        )
        # n_adapters>1 triggers the MultiLinearLoRA dispatch in Automodel's
        # apply_lora_to_linear_modules (see patches/automodel/). Worker reads
        # lora_cfg from policy.dtensor_cfg.lora_cfg.
        lora_cfg = (
            master_config.setdefault("policy", {})
            .setdefault("dtensor_cfg", {})
            .setdefault("lora_cfg", {})
        )
        lora_cfg["n_adapters"] = len(ml_cfg.adapters)
        lora_cfg["adapter_names"] = [a.name for a in ml_cfg.adapters]
        loss_fn_factory = lambda _mc: MultiAdapterLoss()  # noqa: E731
        train_dataloader_override = MultiAdapterDataLoader.from_config(
            ml_cfg=ml_cfg,
            base_data_cfg=master_config["data"],
            tokenizer=tokenizer,
            max_seq_length=master_config["data"].get("max_input_seq_length"),
        )

    logger.info("Setting up NeMo-RL SFT...")
    (
        policy, cluster, dataloader, val_dataloader,
        loss_fn, nemo_logger, checkpointer, sft_state, resolved_config,
    ) = setup(
        master_config, tokenizer, train_dataset, val_dataset,
        loss_fn_factory=loss_fn_factory,
        train_dataloader_override=train_dataloader_override,
    )

    logger.info("Starting SFT training...")
    sft_train(
        policy, dataloader, val_dataloader,
        tokenizer, loss_fn,
        resolved_config, nemo_logger, checkpointer, sft_state,
    )
    logger.info("SFT training complete.")


if __name__ == "__main__":
    main()
