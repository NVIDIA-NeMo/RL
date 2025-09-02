#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import os
import sys
from typing import Any, Dict

import yaml


OLD_TOP_TARGET = "nemo.tron.config.ConfigContainer"
NEW_TOP_TARGET = "megatron.bridge.training.config.ConfigContainer"


TOP_KEY_RENAMES = {
    "checkpoint_config": "checkpoint",
    "dataset_config": "dataset",
    "ddp_config": "ddp",
    "dist_config": "dist",
    "ft_config": "ft",
    "logger_config": "logger",
    "model_config": "model",
    "optimizer_config": "optimizer",
    "profiling_config": "profiling",
    "rerun_state_machine_config": "rerun_state_machine",
    "rng_config": "rng",
    "scheduler_config": "scheduler",
    "straggler_config": "straggler",
    "tokenizer_config": "tokenizer",
    "train_config": "train",
}


MODEL_TARGET_MAP = {
    # Qwen2
    "nemo.collections.llm.gpt.model.qwen2.Qwen2Config": "megatron.bridge.models.qwen.qwen_provider.Qwen2ModelProvider",
    # DeepSeek V3
    "nemo.collections.llm.gpt.model.deepseek.DeepSeekV3Config": "megatron.bridge.models.deepseek.deepseek_provider.DeepSeekV3Provider",
}


LOGGER_TARGET_OLD = "nemo.tron.config.LoggerConfig"
LOGGER_TARGET_NEW = "megatron.bridge.training.config.LoggerConfig"

CKPT_TARGET_OLD = "nemo.tron.config.CheckpointConfig"
CKPT_TARGET_NEW = "megatron.bridge.training.config.CheckpointConfig"

RERUN_SM_TARGET_OLD = "nemo.tron.config.RerunStateMachineConfig"
RERUN_SM_TARGET_NEW = "megatron.bridge.training.config.RerunStateMachineConfig"

RNG_TARGET_OLD = "nemo.tron.config.RNGConfig"
RNG_TARGET_NEW = "megatron.bridge.training.config.RNGConfig"


def _pop_if_present(d: Dict[str, Any], key: str) -> None:
    if isinstance(d, dict) and key in d:
        d.pop(key, None)


def convert_config_dict(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)  # shallow copy

    # Top-level _target_
    if cfg.get("_target_") == OLD_TOP_TARGET:
        cfg["_target_"] = NEW_TOP_TARGET

    # Rename top-level keys *_config -> new names
    for old_key, new_key in TOP_KEY_RENAMES.items():
        if old_key in cfg and new_key not in cfg:
            cfg[new_key] = cfg.pop(old_key)

    # checkpoint subtree
    if isinstance(cfg.get("checkpoint"), dict):
        ckpt = cfg["checkpoint"]
        if ckpt.get("_target_") == CKPT_TARGET_OLD:
            ckpt["_target_"] = CKPT_TARGET_NEW
        # New default toggles per examples
        ckpt.setdefault("load_main_params_from_ckpt", False)
        # In examples we flip save_rng to false
        if "save_rng" in ckpt:
            ckpt["save_rng"] = False
        ckpt.setdefault("use_persistent_ckpt_worker", True)
        # Remove legacy keys present in old
        _pop_if_present(ckpt, "auto_detect_ckpt_format")
        _pop_if_present(ckpt, "ckpt_convert_update_legacy_dist_opt_format")

    # logger subtree
    if isinstance(cfg.get("logger"), dict):
        logger = cfg["logger"]
        if logger.get("_target_") == LOGGER_TARGET_OLD:
            logger["_target_"] = LOGGER_TARGET_NEW
        logger.setdefault("log_energy", False)
        # Set logging level if null per example
        if logger.get("logging_level") in (None, "null") or "logging_level" not in logger:
            logger["logging_level"] = 20

    # model subtree: switch _target_ to provider classes, set some defaults per examples
    if isinstance(cfg.get("model"), dict):
        model = cfg["model"]
        t = model.get("_target_")
        if isinstance(t, str) and t in MODEL_TARGET_MAP:
            model["_target_"] = MODEL_TARGET_MAP[t]

        # Example toggles (conservative; only touch present keys)
        if "apply_rope_fusion" in model:
            model["apply_rope_fusion"] = True
        if "bias_dropout_fusion" in model:
            model["bias_dropout_fusion"] = True
        if "masked_softmax_fusion" in model:
            model["masked_softmax_fusion"] = False
        # Add mtp_enabled default if not present
        model.setdefault("mtp_enabled", False)
        # Perform initialization to false in new schema per examples
        if "perform_initialization" in model:
            model["perform_initialization"] = False
        # Example defaults
        if model.get("pipeline_dtype") in (None, "null"):
            model["pipeline_dtype"] = "bfloat16"

        # Transformers adapter fields
        if isinstance(model.get("hf_adapter"), dict):
            hf = model["hf_adapter"]
            # bump transformers version and trust_remote_code
            hf["transformers_version"] = "4.53.3"
            hf["trust_remote_code"] = True

    # Optional new top-level fields present in examples
    cfg.setdefault("mixed_precision", None)
    cfg.setdefault("nvrx_straggler", None)
    cfg.setdefault("comm_overlap", None)
    cfg.setdefault("peft", None)
    cfg.setdefault("profiling", None)

    # rerun_state_machine subtree
    if isinstance(cfg.get("rerun_state_machine"), dict):
        rsm = cfg["rerun_state_machine"]
        if rsm.get("_target_") == RERUN_SM_TARGET_OLD:
            rsm["_target_"] = RERUN_SM_TARGET_NEW

    # rng subtree
    if isinstance(cfg.get("rng"), dict):
        rng = cfg["rng"]
        if rng.get("_target_") == RNG_TARGET_OLD:
            rng["_target_"] = RNG_TARGET_NEW

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert old Tron run_config.yaml to Megatron Bridge schema")
    parser.add_argument("input", help="Path to old run_config.yaml")
    parser.add_argument("--output", "-o", help="Output path for converted YAML. If omitted, overwrite input.")
    parser.add_argument("--dry-run", action="store_true", help="Print converted YAML to stdout and do not write")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        print("Invalid YAML format: expected a mapping at top level", file=sys.stderr)
        sys.exit(2)

    new_cfg = convert_config_dict(cfg)

    if args.dry_run:
        yaml.safe_dump(new_cfg, sys.stdout, default_flow_style=False)
        return

    out_path = args.output or args.input
    with open(out_path, "w") as f:
        yaml.safe_dump(new_cfg, f, default_flow_style=False)

    print(f"Converted config written to: {out_path}")


if __name__ == "__main__":
    main()


