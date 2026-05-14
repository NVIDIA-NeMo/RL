# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Increase the W&B single object size warning threshold.
import wandb.util

wandb.util.VALUE_BYTES_LIMIT = 10_000_000

import ray
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.distillation import MasterConfig, check_vocab_equality
from nemo_rl.algorithms.distillation_mixed_generation import (
    build_prompt_identities_from_batch,
    dataset_namespace_from_config,
    serialize_final_batch_sample,
)
from nemo_rl.algorithms.grpo import _should_use_nemo_gym
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.nemo_gym import NemoGymConfig, setup_nemo_gym_config
from nemo_rl.environments.utils import create_env
from nemo_rl.experience.rollouts import run_async_nemo_gym_rollout
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Generate NeMo-Gym teacher rollout JSONL records for mixed OPD distillation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output JSONL file. Defaults to <log_dir>/mixed_teacher_rollouts.jsonl.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "validation"),
        default="train",
        help="Dataset split to generate from.",
    )
    parser.add_argument(
        "--num-generations-per-prompt",
        type=int,
        default=None,
        help=(
            "Number of teacher generations to write per prompt. "
            "Defaults to distillation.mixed_generation.teacher_generations_per_prompt "
            "when set, else distillation.num_generations_per_prompt."
        ),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit on the number of prompt batches to generate.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="Optional limit on the total number of prompts to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an incomplete JSONL by appending missing records. "
            "The final partial line is repaired before appending."
        ),
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _default_config_path() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "configs",
        "distillation_math.yaml",
    )


def _disable_validation(config: MasterConfig) -> None:
    config["distillation"]["val_period"] = 0
    config["distillation"]["val_at_start"] = False
    config["distillation"]["val_at_end"] = False
    config["distillation"]["max_val_samples"] = 0


def _resolve_generation_count(
    config: MasterConfig, requested_count: int | None
) -> int:
    if requested_count is not None:
        generation_count = requested_count
    else:
        mixed_generation = config["distillation"].get("mixed_generation")
        generation_count = 0
        if isinstance(mixed_generation, Mapping):
            generation_count = int(
                mixed_generation.get("teacher_generations_per_prompt") or 0
            )
        if generation_count <= 0:
            generation_count = int(config["distillation"]["num_generations_per_prompt"])

    if generation_count <= 0:
        raise ValueError("num_generations_per_prompt must be positive")
    return generation_count


def _resolve_output_path(
    config: MasterConfig,
    requested_path: str | None,
    *,
    overwrite: bool,
    resume: bool,
) -> Path:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if resume and requested_path is None:
        raise ValueError("--resume requires an explicit --output path")

    if requested_path is None:
        requested_path = os.path.join(
            config["logger"]["log_dir"],
            "mixed_teacher_rollouts.jsonl",
        )
    output_path = Path(requested_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Pass --resume to continue or --overwrite to replace it."
        )
    return output_path


def _done_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.done")


def _inprogress_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.inprogress")


def _record_key(record: Mapping[str, Any]) -> tuple[str, int]:
    return (str(record["prompt_uid"]), int(record["teacher_generation_id"]))


@dataclass(frozen=True)
class ResumeState:
    existing_keys: set[tuple[str, int]]
    existing_record_count: int
    already_complete: bool = False


def _prepare_output_for_write(
    output_path: Path,
    *,
    overwrite: bool,
    resume: bool,
) -> ResumeState:
    done_path = _done_path(output_path)
    inprogress_path = _inprogress_path(output_path)
    if overwrite:
        output_path.unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)
        inprogress_path.unlink(missing_ok=True)
        return ResumeState(existing_keys=set(), existing_record_count=0)

    if done_path.exists():
        if not output_path.exists():
            raise FileNotFoundError(
                f"Completion sentinel exists but output file is missing: {done_path}"
            )
        if resume:
            existing_keys = _load_existing_keys(
                output_path,
                repair_final_line=False,
            )
            return ResumeState(
                existing_keys=existing_keys,
                existing_record_count=len(existing_keys),
                already_complete=True,
            )
        raise FileExistsError(
            f"Output is already marked complete: {done_path}. "
            "Pass --overwrite to regenerate it."
        )

    if not output_path.exists():
        return ResumeState(existing_keys=set(), existing_record_count=0)

    if not resume:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Pass --resume to continue or --overwrite to replace it."
        )

    existing_keys = _load_existing_keys(output_path, repair_final_line=True)
    return ResumeState(
        existing_keys=existing_keys,
        existing_record_count=len(existing_keys),
    )


def _write_inprogress_marker(output_path: Path) -> None:
    marker = {
        "output": str(output_path),
        "pid": os.getpid(),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    _inprogress_path(output_path).write_text(
        json.dumps(marker, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_existing_records_for_resume(
    output_path: Path,
    *,
    sampling_config: Mapping[str, Any],
    model_name: str | None,
    tokenizer_name: str | None,
) -> None:
    if not output_path.exists():
        return
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_sampling = record.get("sampling")
            if not isinstance(record_sampling, Mapping):
                raise ValueError(
                    f"Existing rollout line {line_number} missing sampling metadata"
                )
            for key in ("temperature", "top_p", "top_k"):
                if key not in sampling_config:
                    continue
                if record_sampling.get(key) != sampling_config.get(key):
                    raise ValueError(
                        f"Existing rollout line {line_number} sampling mismatch for "
                        f"{key}: expected {sampling_config.get(key)!r}, "
                        f"got {record_sampling.get(key)!r}"
                    )
            model_metadata = record.get("model")
            if not isinstance(model_metadata, Mapping):
                raise ValueError(
                    f"Existing rollout line {line_number} missing model metadata"
                )
            if model_name is not None and _canonical_tokenizer_or_model_identity(
                model_metadata.get("name_or_path")
            ) != _canonical_tokenizer_or_model_identity(model_name):
                raise ValueError(
                    f"Existing rollout line {line_number} model mismatch: "
                    f"expected {model_name!r}, got {model_metadata.get('name_or_path')!r}"
                )
            if tokenizer_name is not None and _canonical_tokenizer_or_model_identity(
                model_metadata.get("tokenizer")
            ) != _canonical_tokenizer_or_model_identity(tokenizer_name):
                raise ValueError(
                    f"Existing rollout line {line_number} tokenizer mismatch: "
                    f"expected {tokenizer_name!r}, got {model_metadata.get('tokenizer')!r}"
                )


def _load_existing_keys(
    output_path: Path,
    *,
    repair_final_line: bool,
) -> set[tuple[str, int]]:
    if not output_path.exists():
        return set()

    existing_keys: set[tuple[str, int]] = set()
    valid_end_offset = 0
    file_size = output_path.stat().st_size
    truncate_offset: int | None = None

    with output_path.open("rb") as handle:
        line_number = 0
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            is_final_line = handle.tell() == file_size
            if not line.strip():
                valid_end_offset = handle.tell()
                continue
            if not line.endswith(b"\n"):
                if repair_final_line:
                    truncate_offset = line_start
                    break
                raise ValueError(
                    f"Output JSONL has an unterminated final line at {line_number}: "
                    f"{output_path}"
                )
            try:
                record = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                if repair_final_line and is_final_line:
                    truncate_offset = line_start
                    break
                raise ValueError(
                    f"Output JSONL has invalid JSON at line {line_number}: {output_path}"
                ) from exc
            key = _record_key(record)
            if key in existing_keys:
                raise ValueError(f"Output JSONL contains duplicate record key: {key}")
            existing_keys.add(key)
            valid_end_offset = handle.tell()

    if truncate_offset is not None:
        with output_path.open("r+b") as handle:
            handle.truncate(truncate_offset)
        print(
            f"Repaired partial JSONL tail in {output_path}: "
            f"truncated to byte offset {truncate_offset}",
            flush=True,
        )
    elif repair_final_line and valid_end_offset < file_size:
        with output_path.open("r+b") as handle:
            handle.truncate(valid_end_offset)
    return existing_keys


def _canonical_tokenizer_or_model_identity(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    path = Path(text).expanduser()
    if path.is_absolute() or text.startswith((".", "~")) or path.exists():
        return str(path.resolve())
    return text


def _tokenizer_name_from_config(config: MasterConfig, tokenizer) -> str | None:
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    if isinstance(tokenizer_name, str) and tokenizer_name:
        return _canonical_tokenizer_or_model_identity(tokenizer_name)
    tokenizer_config = config["policy"].get("tokenizer")
    if isinstance(tokenizer_config, Mapping):
        if tokenizer_config.get("name") is not None:
            return _canonical_tokenizer_or_model_identity(tokenizer_config["name"])
        if tokenizer_config.get("path") is not None:
            return _canonical_tokenizer_or_model_identity(tokenizer_config["path"])
    return None


def _teacher_generation_config(config: MasterConfig) -> VllmConfig:
    generation_config = config["policy"]["generation"]
    if generation_config is None:
        raise ValueError(
            "A vLLM generation config is required for NeMo-Gym teacher rollout generation"
        )
    if generation_config.get("backend") != "vllm":
        raise ValueError("Teacher rollout generation requires policy.generation.backend=vllm")

    teacher_model_name = config["teacher"]["model_name"]
    generation_config["model_name"] = _canonical_tokenizer_or_model_identity(
        teacher_model_name
    )
    generation_config["colocated"]["enabled"] = False
    generation_config.pop("quant_cfg", None)
    if "vllm_kwargs" in generation_config:
        generation_config["vllm_kwargs"]["hf_overrides"] = config["teacher"].get(
            "hf_config_overrides",
            {},
        )
    return generation_config


def _create_teacher_generation(
    config: MasterConfig,
    generation_config: VllmConfig,
) -> VllmGeneration:
    cluster_config = config["cluster"]
    generation_cluster = RayVirtualCluster(
        name="teacher_rollout_generation_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    return VllmGeneration(
        cluster=generation_cluster,
        config=generation_config,
        name_prefix="teacher_rollout_vllm",
    )


def _build_row_metadata(
    prompt_batch: Mapping[str, Any],
    *,
    num_generations_per_prompt: int,
    dataset_namespace: str,
    sampling_config: Mapping[str, Any],
    model_name: str,
    tokenizer_name: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_uids = build_prompt_identities_from_batch(
        prompt_batch,
        dataset_namespace=dataset_namespace,
    )
    prompt_indices = prompt_batch["idx"]
    prompt_extra_env_info = prompt_batch.get(
        "extra_env_info", [{} for _ in range(len(prompt_indices))]
    )

    metadata_rows: list[dict[str, Any]] = []
    extra_env_info_rows: list[dict[str, Any]] = []
    for prompt_uid, dataset_index, row_extra in zip(
        prompt_uids, prompt_indices, prompt_extra_env_info, strict=True
    ):
        base_extra = dict(row_extra) if isinstance(row_extra, Mapping) else {}
        base_extra["source_prompt_uid"] = prompt_uid
        for generation_id in range(num_generations_per_prompt):
            metadata_rows.append(
                {
                    "prompt_uid": str(prompt_uid),
                    "dataset_index": int(dataset_index),
                    "teacher_generation_id": generation_id,
                    "sampling": {
                        "temperature": sampling_config.get("temperature"),
                        "top_p": sampling_config.get("top_p"),
                        "top_k": sampling_config.get("top_k"),
                    },
                    "model": {
                        "name_or_path": model_name,
                        "tokenizer": tokenizer_name,
                    },
                }
            )
            extra_env_info_rows.append(dict(base_extra))
    return metadata_rows, extra_env_info_rows


def _missing_row_indices(
    metadata_rows: Sequence[Mapping[str, Any]],
    existing_keys: set[tuple[str, int]],
) -> list[int]:
    missing_indices = []
    for row_index, metadata in enumerate(metadata_rows):
        key = _record_key(metadata)
        if key not in existing_keys:
            missing_indices.append(row_index)
    return missing_indices


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = _default_config_path()

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    assert isinstance(config, dict)
    print("Applied CLI overrides")

    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    _disable_validation(config)

    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
        check_vocab_equality(
            tokenizer,
            config["policy"]["model_name"],
            config["teacher"]["model_name"],
        )
    if config["policy"]["generation"] is None:
        raise ValueError(
            "A vLLM generation config is required for NeMo-Gym teacher rollout generation"
        )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    setup_nemo_gym_config(config, tokenizer)
    generation_config = _teacher_generation_config(config)

    try:
        use_nemo_gym = _should_use_nemo_gym(config)
    except AssertionError as exc:
        raise ValueError(
            "NeMo-Gym rollout generation requires env.should_use_nemo_gym=true, "
            "policy.generation.backend=vllm, "
            "policy.generation.vllm_cfg.async_engine=true, and "
            "policy.generation.vllm_cfg.expose_http_server=true."
        ) from exc
    if not use_nemo_gym:
        raise ValueError("NeMo-Gym rollout generation requires env.should_use_nemo_gym=true.")

    train_dataset, val_dataset = setup_response_data(
        tokenizer, config["data"], env_configs=None
    )
    if isinstance(train_dataset, dict) or isinstance(val_dataset, dict):
        raise ValueError(
            "generate_nemo_gym_rollouts.py does not support data.use_multiple_dataloader=true"
        )

    if args.split == "validation":
        if val_dataset is None:
            raise ValueError("No validation dataset is configured for --split=validation")
        selected_dataset = val_dataset
    else:
        selected_dataset = train_dataset

    if not hasattr(selected_dataset, "__len__"):
        raise TypeError("Selected dataset does not support __len__")

    num_generations_per_prompt = _resolve_generation_count(
        config, args.num_generations_per_prompt
    )
    dataset_namespace = dataset_namespace_from_config(
        config["data"],
        split=args.split,
    )
    sampling_config = config["policy"]["generation"]
    tokenizer_name = _tokenizer_name_from_config(config, tokenizer)
    output_path = _resolve_output_path(
        config,
        args.output,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    resume_state = _prepare_output_for_write(
        output_path,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    _validate_existing_records_for_resume(
        output_path,
        sampling_config=sampling_config,
        model_name=generation_config["model_name"],
        tokenizer_name=tokenizer_name,
    )
    if resume_state.already_complete:
        print(
            f"Output is already complete, nothing to do: {output_path} "
            f"({resume_state.existing_record_count} records)"
        )
        return
    _write_inprogress_marker(output_path)

    print(
        f"Generating {num_generations_per_prompt} rollout(s) per prompt from the "
        f"{args.split} split into {output_path}"
    )
    if resume_state.existing_record_count:
        print(
            f"Resuming with {resume_state.existing_record_count} existing records",
            flush=True,
        )

    init_ray()
    teacher_generation = _create_teacher_generation(config, generation_config)

    nemo_gym_config = NemoGymConfig(
        model_name=teacher_generation.cfg["model_name"],
        base_urls=teacher_generation.dp_openai_server_base_urls,
        initial_global_config_dict=config["env"]["nemo_gym"],
    )
    nemo_gym = None
    nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)
    ray.get(nemo_gym.health_check.remote())
    task_to_env = {"nemo_gym": nemo_gym}

    prompt_dataloader = StatefulDataLoader(
        selected_dataset,
        batch_size=config["distillation"]["num_prompts_per_step"],
        shuffle=False,
        collate_fn=rl_collate_fn,
        drop_last=False,
    )

    prompts_seen = 0
    prompts_written = 0
    records_written = resume_state.existing_record_count
    new_records_written = 0
    batches_written = 0

    try:
        teacher_generation.prepare_for_generation()
        output_mode = "a" if args.resume and output_path.exists() else "w"
        with output_path.open(output_mode, encoding="utf-8") as handle:
            for batch_index, prompt_batch in enumerate(prompt_dataloader):
                if args.max_batches is not None and batch_index >= args.max_batches:
                    break
                if args.max_prompts is not None and prompts_seen >= args.max_prompts:
                    break

                if args.max_prompts is not None:
                    remaining_prompts = args.max_prompts - prompts_seen
                    if remaining_prompts <= 0:
                        break
                    if len(prompt_batch["idx"]) > remaining_prompts:
                        prompt_batch = prompt_batch.slice(0, remaining_prompts)
                batch_prompt_count = len(prompt_batch["idx"])
                prompts_seen += batch_prompt_count

                metadata_rows, extra_env_info_rows = _build_row_metadata(
                    prompt_batch,
                    num_generations_per_prompt=num_generations_per_prompt,
                    dataset_namespace=dataset_namespace,
                    sampling_config=sampling_config,
                    model_name=teacher_generation.cfg["model_name"],
                    tokenizer_name=tokenizer_name,
                )
                missing_indices = _missing_row_indices(
                    metadata_rows,
                    resume_state.existing_keys,
                )
                if not missing_indices:
                    batches_written += 1
                    print(
                        f"Skipped batch {batches_written}: prompts={batch_prompt_count}, "
                        "all records already exist",
                        flush=True,
                    )
                    continue

                repeated_batch = prompt_batch.repeat_interleave(num_generations_per_prompt)
                rollout_input_batch = repeated_batch.select_indices(missing_indices)
                missing_metadata_rows = [
                    metadata_rows[row_index] for row_index in missing_indices
                ]
                missing_extra_env_info_rows = [
                    extra_env_info_rows[row_index] for row_index in missing_indices
                ]
                rollout_result = run_async_nemo_gym_rollout(
                    policy_generation=teacher_generation,
                    input_batch=rollout_input_batch,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    generation_config=sampling_config,
                    max_seq_len=None,
                    max_rollout_turns=None,
                    greedy=False,
                )

                final_batch = rollout_result.final_batch
                final_batch["extra_env_info"] = missing_extra_env_info_rows

                for row_index, metadata in enumerate(missing_metadata_rows):
                    record = serialize_final_batch_sample(
                        final_batch,
                        index=row_index,
                        metadata=metadata,
                    )
                    handle.write(json.dumps(record) + "\n")
                    resume_state.existing_keys.add(_record_key(metadata))
                    new_records_written += 1
                    records_written += 1
                handle.flush()
                os.fsync(handle.fileno())

                prompts_written += batch_prompt_count
                batches_written += 1
                print(
                    f"Wrote batch {batches_written}: prompts={batch_prompt_count}, "
                    f"new_records={len(missing_metadata_rows)}, "
                    f"total_records={records_written}",
                    flush=True,
                )
    finally:
        if teacher_generation is not None:
            teacher_generation.finish_generation()
            teacher_generation.shutdown()
        if nemo_gym is not None:
            ray.get(nemo_gym.shutdown.remote())

    done_payload = {
        "output": str(output_path),
        "split": args.split,
        "num_generations_per_prompt": num_generations_per_prompt,
        "prompts_seen": prompts_seen,
        "prompts_written": prompts_written,
        "records": records_written,
        "new_records": new_records_written,
    }
    _done_path(output_path).write_text(
        json.dumps(done_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _inprogress_path(output_path).unlink(missing_ok=True)
    print(
        f"Completed rollout generation: prompts_seen={prompts_seen}, "
        f"prompts_written={prompts_written}, records={records_written}, "
        f"new_records={new_records_written}, output={output_path}"
    )


if __name__ == "__main__":
    main()
