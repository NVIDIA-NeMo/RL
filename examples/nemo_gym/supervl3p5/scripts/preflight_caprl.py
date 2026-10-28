# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the CapRL smoke without starting Ray or loading model weights."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]


def load_config(path: Path, overrides: list[str]) -> dict[str, Any]:
    """Use the production loader without importing Ray on the submit host."""
    spec = importlib.util.spec_from_file_location(
        "caprl_config_loader", ROOT / "nemo_rl/utils/config.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the production config loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.register_omegaconf_resolvers()
    config = module.parse_hydra_overrides(module.load_config(path), overrides)
    return cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))


def validate_config(config: dict[str, Any]) -> None:
    """Validate the prefix smoke or its explicit no-rollout-snapshot control."""
    policy, grpo = config["policy"], config["grpo"]
    generation = policy["generation"]
    if not policy["is_vlm"] or not policy["megatron_cfg"]["enabled"]:
        raise ValueError("CapRL capture requires the Omni VLM and Megatron learner")
    if generation["backend"] != "vllm" or not generation["vllm_cfg"]["async_engine"]:
        raise ValueError("Prefix cuts require asynchronous vLLM generation")
    if grpo["deduplicate_multimodal_data"]:
        raise ValueError("Multimodal token capture requires media deduplication OFF")
    if not config["token_capture"]["enabled"]:
        raise ValueError("Token capture must be enabled")
    if not str(config["token_capture"]["capture_dir"]).startswith("/tmp/"):
        raise ValueError("The smoke uses node-local /tmp capture ledgers")
    recovery_target = config["rollout_recovery"].get("target_level")
    interval = config["rollout_checkpointing"]["snapshot_attempt_interval_s"]
    if not (
        (recovery_target == "prefix" and interval is not None and interval > 0)
        or (recovery_target == "sibling" and interval is None)
    ):
        raise ValueError(
            "Use prefix recovery with a positive snapshot interval, or sibling "
            "recovery with snapshot_attempt_interval_s=null for the baseline"
        )
    if "default_granularity" in config["rollout_recovery"]:
        raise ValueError("Remove the obsolete default_granularity setting")
    checkpointing = config["checkpointing"]
    if not checkpointing["enabled"] or not checkpointing["save_data_plane"]:
        raise ValueError("Trainer and data-plane checkpointing must be enabled")
    if checkpointing["save_period"] != 1:
        raise ValueError("Save a matching trainer anchor every smoke step")
    if (
        policy["router_replay"]["enabled"]
        or generation["vllm_kwargs"]["enable_return_routed_experts"]
    ):
        raise ValueError("The CapRL reference and initial smoke keep router replay OFF")
    if generation["vllm_kwargs"]["video_pruning_rate"]:
        raise ValueError("Media capture does not support video token pruning")
    if (
        config["cluster"]["num_nodes"]
        <= generation["colocated"]["resources"]["num_nodes"]
    ):
        raise ValueError("The allocation must leave GPU nodes for training")
    if (
        config["cluster"]["gpus_per_node"]
        != generation["colocated"]["resources"]["gpus_per_node"]
    ):
        raise ValueError("Trainer and generation GPU counts must match the allocation")


def validate_assets(config: dict[str, Any]) -> dict[str, Any]:
    """Check model shards, dataset routing, and the first sample's cached frames."""
    model = Path(config["policy"]["model_name"])
    for name in [
        "config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "processor_config.json",
    ]:
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    for shard in shards:
        if not (model / shard).is_file():
            raise FileNotFoundError(model / shard)
    cache = os.environ.get("CAPRL_CONVERSION_CACHE")
    if cache:
        # Match the production conversion-cache key, including HF overrides.
        overrides = config["policy"]["hf_config_overrides"]
        canonical = json.dumps(
            overrides, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        suffix = hashlib.sha256(canonical.encode()).hexdigest()[:12]
        key = f"model_{str(model).replace('/', '_')}__hfovr_{suffix}"
        if not (Path(cache) / key / "iter_0000000/run_config.yaml").is_file():
            raise FileNotFoundError(
                f"No matching completed conversion in {cache}: {key}"
            )
    dataset = Path(config["data"]["train"][0]["data_path"])
    with dataset.open() as stream:
        row = json.loads(next(stream))
    if row["agent_ref"]["name"] != "mcqa_simple_agent":
        raise ValueError("The CapRL smoke dataset must route to mcqa_simple_agent")
    frames = [
        part["image_url"]
        for message in row["responses_create_params"]["input"]
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("_is_video_frame")
    ]
    expected = config["data"]["default"]["num_frames"]
    if len(frames) != expected:
        raise ValueError(
            f"Dataset has {len(frames)} cached frames; expected {expected}"
        )
    for frame in frames:
        if not os.access(frame, os.R_OK):
            raise FileNotFoundError(f"Unreadable cached frame: {frame}")
    return {
        "model_shards": len(shards),
        "first_sample_frames": len(frames),
        "dataset": str(dataset),
    }


def validate_video_sample(config: dict[str, Any]) -> dict[str, Any]:
    """Process the first real row through the same path as the dataloader."""
    # Optional training dependencies: only the container preflight needs them.
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.processors import nemo_gym_data_processor
    from nemo_rl.environments.nemotron_utils import NEMOTRON_VIDEO_PROCESSOR_NAMES

    processor = get_tokenizer(config["policy"]["tokenizer"], get_processor=True)
    processor_name = type(processor).__name__
    if processor_name not in NEMOTRON_VIDEO_PROCESSOR_NAMES:
        raise ValueError(
            f"CapRL requires native Nemotron video routing: {processor_name}"
        )
    task = {**config["data"]["default"], **config["data"]["train"][0]}
    with Path(task["data_path"]).open() as stream:
        row = json.loads(next(stream))
    task_spec = TaskDataSpec(
        **{
            field.name: task[field.name]
            for field in fields(TaskDataSpec)
            if field.name in task
        }
    )
    datum = nemo_gym_data_processor(
        {"extra_env_info": json.dumps(row), "task_name": "nemo_gym"},
        task_spec,
        processor,
        config["data"]["max_input_seq_length"],
        0,
    )
    if datum["loss_multiplier"] == 0 or datum["length"] <= 0:
        raise ValueError(
            "The CapRL preflight sample was filtered or produced no prompt"
        )
    outbound = datum["extra_env_info"]["responses_create_params"]["input"]
    videos = [
        part
        for message in outbound
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "input_video"
    ]
    if len(videos) != 1:
        raise ValueError("Cached frames must become one native-video manifest")
    message = datum["message_log"][0]
    frame_counts = message["num_frames"].as_tensor().reshape(-1).tolist()
    if frame_counts != [task_spec.num_frames]:
        raise ValueError(f"Processed video frame count changed: {frame_counts}")
    return {
        "processor_class": processor_name,
        "prompt_tokens": datum["length"],
        "frames": frame_counts[0],
        "native_video_parts": len(videos),
    }


def validate_runtime(config: dict[str, Any]) -> None:
    """Check actual serving dependencies without initializing an engine or Ray."""
    # Optional, heavy dependencies: only the container runtime preflight uses them.
    from vllm import SamplingParams
    from vllm.model_executor.models.nano_nemotron_vl import NanoNemotronVLProcessingInfo
    from vllm.tokenizers import get_tokenizer

    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        _generation_cut_blockers,
    )
    from nemo_rl.utils.outdated_config_checks import check_outdated_config

    check_outdated_config(config)
    MasterConfig(**config)
    # vLLM's wrapper supplies max_token_id, required by its bad-word resolver.
    tokenizer = get_tokenizer(config["policy"]["model_name"], trust_remote_code=True)
    generation = config["policy"]["generation"]
    params = SamplingParams(bad_words=generation["bad_words"])
    params.update_from_tokenizer(tokenizer)
    blockers = _generation_cut_blockers(params)
    if blockers:
        raise ValueError(
            f"Reference sampling constraints remain uncuttable: {blockers}"
        )
    if not callable(NanoNemotronVLProcessingInfo.get_video_pruning_rate):
        raise RuntimeError("Container lacks the expected Omni video processor API")
    video_sample = validate_video_sample(config)
    print(
        json.dumps(
            {
                "runtime": "PASS",
                "bad_word_token_ids": params.bad_words_token_ids,
                "video_sample": video_sample,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--allocation", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    args, overrides = parser.parse_known_args()
    config = load_config(args.config, overrides)
    validate_config(config)
    assets = validate_assets(config)
    if args.allocation:
        print(config["cluster"]["num_nodes"], config["cluster"]["gpus_per_node"])
    else:
        print(json.dumps({"config": "PASS", **assets}))
    if args.runtime:
        validate_runtime(config)


if __name__ == "__main__":
    main()
