#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Stage official DAPO data and validate the actual Flint GRPO driver config.

Run with python-driver, network access, the same HF cache as training, and both
reviewed runtime overlays already installed. No Ray workers are constructed.
The built-in data setup is exercised with env_configs=None. Identical formatted
question/answer inputs share a real processor invocation, while row counts,
ordered hashes, and masking statistics include every row of both datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import sys
import tempfile
import traceback
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

SOURCE_CONFIG_SHA256 = "5fa9d05d3ef34c7b9306f7c839f9c232b122db5c85c783e79694556131830669"  # pragma: allowlist secret - SHA256 integrity checksum
VALIDATED_SETUP_SHA256 = "b9d81b8a75ee4e67cb7bef737798274593ae57dc695536350d6452a965c5beef"  # pragma: allowlist secret - SHA256 integrity checksum
RUNTIME_PINS = {
    "/opt/flint/Automodel/nemo_automodel/components/models/flint/layers.py": "e57d036664e889d4ac639d1aeec198c5e127d2926de00c0656ede4b233ed4a4c",  # pragma: allowlist secret - SHA256 integrity checksum
    "/opt/flint/envs/sglang/lib/python3.13/site-packages/sglang/srt/mem_cache/kv_cache_configurator.py": "c67d197b06f407964b5b209f140535bb08665d2fe9c502c89e3f2b68fc1ad28d",  # pragma: allowlist secret - SHA256 integrity checksum
    "/opt/flint/envs/sglang/lib/python3.13/site-packages/sglang/srt/mem_cache/swa_memory_pool.py": "75a2f5d385c1c1ec886f0f553e19178b8bed498ce688fa99c88651231a8e8f45",  # pragma: allowlist secret - SHA256 integrity checksum
    "/opt/flint/envs/sglang/lib/python3.13/site-packages/sglang_router/router_args.py": "09cd8984a1f693cc2efe40b963befd6d5bc578e972272e994a046f183807f3e8",  # pragma: allowlist secret - SHA256 integrity checksum
}
DATASETS = {
    "train": (
        "BytedTsinghua-SIA/DAPO-Math-17k",  # pragma: allowlist secret - public dataset identifier
        "DAPOMath17K",
    ),
    "validation": ("BytedTsinghua-SIA/AIME-2024", "DAPOMathAIME2024"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        default=json_default,
    ).encode("utf-8")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                value,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=json_default,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def mark(report, stage):
    report["stage"] = stage
    print(json.dumps({"preflight_stage": stage}, sort_keys=True), flush=True)


def plain(value):
    """Normalize the mixed Pydantic/TypedDict config tree before indexing it."""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def check_configuration(raw, cfg, report):
    policy, grpo, loss = plain(cfg.policy), plain(cfg.grpo), plain(cfg.loss_fn)
    data, cluster, env = plain(cfg.data), plain(cfg.cluster), plain(cfg.env)
    auto, generation = policy["automodel_cfg"], policy["generation"]
    serving = generation["sglang_cfg"]
    server = serving["sglang_server_config"]
    require(
        cluster["num_nodes"] == 8 and cluster["gpus_per_node"] == 4,
        "Expected the user-selected eight 4-GPU nodes",
    )
    require(
        grpo["num_prompts_per_step"] == 32 and grpo["num_generations_per_prompt"] == 8,
        "Expected 32 prompts by eight responses",
    )
    require(grpo["max_num_steps"] == 50, "Expected 50 GRPO steps")
    require(
        policy["train_global_batch_size"] == 128
        and policy["train_micro_batch_size"] == 1,
        "Expected global batch 128 and microbatch one",
    )
    require(
        policy["train_global_batch_size"] % 32 == 0,
        "Training batch must divide 32 ranks",
    )
    require(policy["logprob_batch_size"] == 1, "Expected logprob microbatch one")
    require(
        policy["max_total_sequence_length"] == 5120
        and generation["max_new_tokens"] == 4096,
        "Expected requested 5120 total and 4096 generation token limits",
    )
    require(data["max_input_seq_length"] == 1024, "Expected 1024-token input budget")
    require(
        generation["temperature"] == 1.0
        and generation["top_p"] == 1.0
        and generation["top_k"] is None,
        "Expected unfiltered temperature-one sampling",
    )
    require(
        generation["backend"] == "sglang" and not generation["use_async_rollouts"],
        "Expected synchronous SGLang generation",
    )
    require(generation["colocated"]["enabled"], "Expected colocated rollout/training")
    require(
        auto["enabled"] and not policy["megatron_cfg"]["enabled"],
        "Expected native Automodel training",
    )
    for key in (
        "tensor_parallel_size",
        "context_parallel_size",
        "expert_parallel_size",
    ):
        require(auto[key] == 1, f"Expected Automodel {key}=1")
    require(
        auto["activation_checkpointing"] and not auto["sequence_parallel"],
        "Expected activation checkpointing and no sequence parallelism",
    )
    require(not auto["lora_cfg"]["enabled"], "Expected full-parameter training")
    require(
        not policy["sequence_packing"]["enabled"], "Flint requires unpacked sequences"
    )
    require(policy["dynamic_batching"]["enabled"], "Expected dynamic batching")
    require(policy["precision"] == "bfloat16", "Expected BF16 bulk compute")
    require(
        policy["model_name"] == "/flint-model",
        "Expected the original full checkpoint mount",
    )
    require(
        policy["tokenizer"]["name"] == "/flint-model",
        "Tokenizer must come from original model",
    )
    overrides = policy["hf_config_overrides"]
    require(
        overrides["router_aux_loss_coef"] == 0
        and overrides["residual_dtype"] == "fp32"
        and not overrides["use_cache"],
        "Expected FP32 residuals and disabled auxiliary loss/cache",
    )
    native = auto["automodel_kwargs"]
    require(
        native["force_hf"] is False and native["enable_hf_state_dict_adapter"] is True,
        "Expected native Flint with HF state dict adapter",
    )
    for key in ("attn", "linear", "rms_norm", "experts", "dispatcher"):
        expected = "sdpa" if key == "attn" else "torch"
        require(native["backend"][key] == expected, f"Unexpected native backend {key}")
    optimizer = policy["optimizer"]
    opt = optimizer["kwargs"]
    require(optimizer["name"] == "torch.optim.AdamW", "Expected native torch AdamW")
    require(
        opt["lr"] == 1.0e-6
        and opt["weight_decay"] == 0.1
        and list(opt["betas"]) == [0.9, 0.95]
        and opt["eps"] == 1.0e-8,
        "Expected Qwen learning rate, decay, betas and epsilon",
    )
    require(
        opt["foreach"] is False and opt["fused"] is False,
        "Expected bounded-memory non-foreach, non-fused AdamW",
    )
    require(
        not opt.get("master_weights", False),
        "Torch AdamW uses the FP32 model parameters as masters",
    )
    scheduler = policy["scheduler"]
    expected_scheduler = [
        {
            "name": "torch.optim.lr_scheduler.LinearLR",
            "kwargs": {"start_factor": 1.0e-8, "end_factor": 1.0, "total_iters": 10},
        },
        {
            "name": "torch.optim.lr_scheduler.ConstantLR",
            "kwargs": {"factor": 1.0, "total_iters": 10000000000},
        },
        {"milestones": [10]},
    ]
    require(
        scheduler == expected_scheduler,
        "Expected Qwen ten-step warmup followed by constant LR",
    )
    require(policy["max_grad_norm"] == 1.0, "Expected gradient clipping at one")
    require(
        policy["refit_buffer_size_gb"] == 1 and policy["offload_optimizer_for_logprob"],
        "Expected one-GiB refit buffer and optimizer offload",
    )
    for key in ("tp_size", "dp_size", "pp_size", "ep_size"):
        require(serving[key] == 1, f"Expected SGLang {key}=1")
    require(
        server["num_gpus"] == 32 and server["num_gpus_per_engine"] == 1,
        "Expected 32 TP1 SGLang replicas across eight nodes",
    )
    require(
        server["needs_offload"] and server["cpu_weight_backup"],
        "Expected cache/weight memory saver and CPU weight backup",
    )
    require(
        serving["context_length"] == 5120 and serving["max_total_tokens"] == 40960,
        "Expected 40960 KV tokens per serving replica",
    )
    require(
        serving["max_running_requests"] == 8
        and server["sglang_server_concurrency"] == 8,
        "Expected at most eight concurrent serving requests",
    )
    require(
        serving["sglang_router_config"].get("router_policy") == "round_robin",
        "Expected even request distribution across all 32 serving replicas",
    )
    require(
        serving["attention_backend"] == "fa4" and serving["enable_fp32_lm_head"],
        "Expected FA4 serving and protected FP32 output head",
    )
    require(
        serving["moe_runner_backend"] == "flashinfer_trtllm_routed",
        "Expected the user-requested unchanged FlashInfer TRT-LLM expert runner",
    )
    for key in (
        "disable_cuda_graph",
        "disable_radix_cache",
        "disable_overlap_schedule",
    ):
        require(serving[key] is True, f"Expected reviewed serving {key}")
    require(
        loss["reference_policy_kl_penalty"] == 0.0
        and loss["use_importance_sampling_correction"]
        and loss["truncated_importance_sampling_type"] == "tis"
        and loss["truncated_importance_sampling_ratio"] == 2
        and loss["ratio_clip_min"] == 0.2
        and loss["ratio_clip_max"] == 0.28
        and loss["ratio_clip_c"] == 10,
        "Expected Qwen DAPO clipped loss/TIS settings",
    )
    require(
        not grpo["use_leave_one_out_baseline"]
        and not grpo["use_dynamic_sampling"]
        and grpo["skip_reference_policy_logprobs_calculation"],
        "Unexpected GRPO baseline/reference settings",
    )
    shaping = grpo["reward_shaping"]
    require(
        not shaping["enabled"] and grpo["overlong_filtering"],
        "Expected reward shaping disabled and overlong filtering enabled",
    )
    logger = plain(cfg.logger)
    require(
        logger["wandb_enabled"]
        and logger["wandb"].get("id")
        and logger["wandb"].get("resume") == "allow",
        "Expected persistent online W&B run",
    )
    require(
        grpo["reward_scaling"]["enabled"]
        and grpo["reward_scaling"]["target_min"] == -1.0,
        "Expected rewards scaled to [-1, 1]",
    )
    require(
        grpo["val_at_start"] and grpo["val_at_end"] and grpo["val_period"] == 5,
        "Expected AIME validation initially, every five steps, and at completion",
    )
    require(
        grpo["max_val_samples"] == 960 and grpo["val_num_generations_per_prompt"] == 1,
        "Expected all 960 AIME rows, one generated response each",
    )
    require(
        grpo["val_batch_size"] == 960,
        "Expected one full AIME queue to avoid serial validation batch barriers",
    )
    require(
        env["math"]["math_verify_impl"] == "dapo_math_verify", "Expected DAPO verifier"
    )
    require(
        data["train"]["dataset_name"] == "DAPOMath17K"
        and data["validation"]["dataset_name"] == "DAPOMathAIME2024",
        "Expected both official built-in DAPO datasets",
    )
    forbidden = {"data_path", "input_key", "output_key", "split_validation_size"}
    for split in ("train", "validation"):
        require(
            not (forbidden & set(data[split])),
            f"Inherited local arithmetic data keys in {split}",
        )
    require(
        data["default"]["processor"] == "math_hf_data_processor"
        and data["default"]["env_name"] == "math",
        "Expected the real math processor/environment",
    )
    require(
        data["default"]["prompt_file"] is None
        and data["default"]["system_prompt_file"] is None,
        "Unexpected prompt override",
    )
    checkpoint = plain(cfg.checkpointing)
    require(
        checkpoint["enabled"]
        and checkpoint["save_optimizer"]
        and checkpoint["save_period"] == 5,
        "Expected resumable model/optimizer checkpoints every five steps",
    )
    require(
        checkpoint["checkpoint_dir"] == "/flint-run/results/dapo50",
        "Checkpoint output must be isolated",
    )
    require(
        checkpoint["checkpoint_must_save_by"] == "00:03:30:00",
        "Expected a three-hour-thirty-minute checkpoint deadline within each four-hour allocation",
    )
    require(
        auto["checkpoint"]["model_save_format"] == "safetensors"
        and auto["checkpoint"]["save_consolidated"] == "false",
        "Raw safetensors must preserve FP32 masters, head and sinks without BF16 consolidation",
    )
    report["configuration"] = {
        "status": "passed",
        "nodes": 8,
        "gpus_per_node": 4,
        "training_ranks": 32,
        "serving_tp1_replicas": 32,
        "grpo_steps": 50,
        "prompts_per_step": 32,
        "responses_per_prompt": 8,
        "responses_per_step": 256,
        "train_global_batch_size": 128,
        "adam_updates_per_grpo_step": 2,
        "max_input_tokens": 1024,
        "max_new_tokens": 4096,
        "max_total_tokens": 5120,
        "serving_kv_tokens_per_replica": 40960,
        "router_policy": serving["sglang_router_config"]["router_policy"],
        "serving_moe_runner_backend": serving["moe_runner_backend"],
        "validation_batch_size": grpo["val_batch_size"],
        "validation_batches": grpo["max_val_samples"] // grpo["val_batch_size"],
        "optimizer": optimizer,
        "scheduler": scheduler,
        "checkpoint": auto["checkpoint"],
        "config_payload_sha256": digest(canonical(raw)),
        "checkpoint_deadline_from_grpo_loop_entry": checkpoint[
            "checkpoint_must_save_by"
        ],
        "allocation_window_hours": 4,
        "target_may_span_resumed_allocations": True,
    }


def iter_rows(dataset):
    # Arrow batches avoid one Python/Arrow lookup for every repeated DAPO row.
    for batch in dataset.iter(batch_size=1024):
        keys = list(batch)
        for values in zip(*(batch[key] for key in keys)):
            yield dict(zip(keys, values))


def raw_pair(row):
    return [row["prompt"][0]["content"], row["reward_model"]["ground_truth"]]


def formatted_pair(row):
    return [row["messages"][0]["content"], row["messages"][1]["content"]]


def cache_description(dataset):
    return {
        "fingerprint": dataset._fingerprint,
        "download_checksums_manifest_sha256": digest(
            canonical(dataset.info.download_checksums)
        ),
        "cache_files": [
            {
                "filename": item["filename"],
                "size_bytes": Path(item["filename"]).stat().st_size,
            }
            for item in dataset.cache_files
        ],
    }


def audit_raw(dataset):
    payload_hash, pair_hash = hashlib.sha256(), hashlib.sha256()
    pairs = Counter()
    for row in iter_rows(dataset):
        pair = canonical(raw_pair(row))
        payload_hash.update(canonical(row) + b"\n")
        pair_hash.update(pair + b"\n")
        pairs[digest(pair)] += 1
    return {
        "rows": len(dataset),
        "ordered_payload_sha256": payload_hash.hexdigest(),
        "ordered_question_answer_sha256": pair_hash.hexdigest(),
        "unique_question_answer_pairs": len(pairs),
        "pair_multiplicity_histogram": dict(sorted(Counter(pairs.values()).items())),
        **cache_description(dataset),
    }


def audit_processed(wrapped, expected_name, generation_limit, total_limit):
    from nemo_rl.data.processors import math_hf_data_processor

    require(not wrapped.task_data_preprocessors, "Unexpected dataset preprocessor")
    require(
        set(wrapped.task_data_processors) == {expected_name},
        "Unexpected processed task names",
    )
    _, processor = wrapped.task_data_processors[expected_name]
    require(
        processor is math_hf_data_processor,
        "Actual dataset does not use the expected processor",
    )
    source = inspect.getsource(processor)
    # The actual processor's only idx use is an output annotation. Memoization
    # below is exact for identical messages/task/spec/tokenizer and token budget.
    require(
        source.count("idx") == 3 and '"idx": idx' in source,
        "Processor idx usage changed; repeated-row memoization needs review",
    )
    cached, pair_hash, observation_hash = {}, hashlib.sha256(), hashlib.sha256()
    lengths, unmasked_lengths, token_counts = [], [], []
    masked, bad_budget, rows = 0, 0, 0
    for index, row in enumerate(iter_rows(wrapped.dataset)):
        require(row["task_name"] == expected_name, "Unexpected row task name")
        pair = canonical(formatted_pair(row))
        pair_hash.update(pair + b"\n")
        key = digest(canonical(row))
        if key not in cached:
            actual = wrapped[index]
            original_length = int(actual["length"])
            input_tokens = sum(
                len(message["token_ids"]) for message in actual["message_log"]
            )
            multiplier = float(actual["loss_multiplier"])
            require(
                math.isfinite(multiplier) and multiplier in (0.0, 1.0),
                "Invalid processor loss multiplier",
            )
            require(
                actual["extra_env_info"]["ground_truth"] == formatted_pair(row)[1],
                "Processor changed the ground-truth answer",
            )
            cached[key] = (original_length, input_tokens, multiplier)
        original_length, input_tokens, multiplier = cached[key]
        lengths.append(original_length)
        token_counts.append(input_tokens)
        if multiplier == 0:
            masked += 1
        else:
            unmasked_lengths.append(original_length)
            if (
                input_tokens + generation_limit > total_limit
                or original_length >= wrapped.max_seq_length
            ):
                bad_budget += 1
            require(
                input_tokens == original_length,
                "Unmasked processor input length changed",
            )
        observation_hash.update(
            canonical([index, key, original_length, input_tokens, multiplier]) + b"\n"
        )
        rows += 1
        if rows % 100000 == 0:
            print(
                json.dumps(
                    {
                        "auditing_dataset": expected_name,
                        "rows": rows,
                        "unique_processor_invocations": len(cached),
                    }
                ),
                flush=True,
            )
    require(rows > 0, "Empty processed dataset")
    return {
        "rows": rows,
        "unique_processor_invocations": len(cached),
        "unique_unmasked_formatted_inputs": sum(
            value[2] != 0 for value in cached.values()
        ),
        "memoization": "Actual wrapped[index] processor once per identical full formatted row; all rows counted and hashed",
        "masked_rows": masked,
        "unmasked_rows": rows - masked,
        "raw_input_tokens_min": min(lengths),
        "raw_input_tokens_max": max(lengths),
        "processed_input_tokens_min": min(token_counts),
        "processed_input_tokens_max": max(token_counts),
        "unmasked_input_tokens_min": min(unmasked_lengths)
        if unmasked_lengths
        else None,
        "unmasked_input_tokens_max": max(unmasked_lengths)
        if unmasked_lengths
        else None,
        "unmasked_context_budget_violations": bad_budget,
        "ordered_question_answer_sha256": pair_hash.hexdigest(),
        "ordered_processor_observation_sha256": observation_hash.hexdigest(),
        "processor_source_sha256": digest(source.encode()),
        **cache_description(wrapped.dataset),
    }


def run(args, report):
    from datasets import config as datasets_config
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from huggingface_hub import constants as hub_constants
    from nemo_automodel._transformers import registry
    from omegaconf import OmegaConf
    from transformers import AutoConfig

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.environments.dapo_math_verifier import compute_score
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    from nemo_rl.utils.outdated_config_checks import check_outdated_config

    mark(report, "configuration")
    register_omegaconf_resolvers()
    raw = OmegaConf.to_container(load_config(args.config), resolve=True)
    require(isinstance(raw, dict), "Expected a mapping config")
    check_outdated_config(raw)
    cfg = MasterConfig(**deepcopy(raw))
    resolved = args.output.with_name(args.output.stem + ".resolved-config.json")
    atomic_json(resolved, raw)
    report["resolved_config"] = str(resolved)
    check_configuration(raw, cfg, report)

    mark(report, "runtime_and_full_model_identity")
    overlay_checks = []
    for filename, expected in RUNTIME_PINS.items():
        path = Path(filename)
        require(
            path.is_file() and not path.is_symlink() and path.resolve() == path,
            f"Missing or redirected runtime overlay: {filename}",
        )
        observed = digest(path.read_bytes())
        require(
            observed == expected,
            f"Runtime overlay differs from reviewed candidate: {filename}",
        )
        overlay_checks.append({"path": filename, "sha256": observed})
    report["runtime_overlays"] = {"status": "passed", "files": overlay_checks}
    setup_path = Path("/flint-work/RL/nemo_rl/models/automodel/setup.py")
    setup_hash = digest(setup_path.read_bytes())
    require(
        setup_hash == VALIDATED_SETUP_SHA256,
        "Native optimizer/load-dtype setup differs from the validated FP32-master path",
    )
    model_root = Path(cfg.policy["model_name"])
    observed = digest((model_root / "config.json").read_bytes())
    require(
        observed == SOURCE_CONFIG_SHA256, "Original full-model config identity changed"
    )
    require(
        not (model_root / ".PROXY_BUILD_INCOMPLETE").exists(),
        "Incomplete proxy model marker exists",
    )
    require(
        "FlintForCausalLM" in registry.MODEL_ARCH_MAPPING,
        "Native Flint architecture is not registered",
    )
    hf = AutoConfig.from_pretrained(
        str(model_root), **cfg.policy["hf_config_overrides"]
    )
    require(
        hf.num_hidden_layers == 49 and len(hf.layer_types) == 49,
        "Expected all 49 original model layers",
    )
    require(
        hf.model_type == "flint" and hf.router_aux_loss_coef == 0,
        "Unexpected native architecture or auxiliary loss",
    )
    require(
        hf.residual_dtype == "fp32",
        "Native residual precision override was not applied",
    )
    report["model"] = {
        "status": "passed",
        "model_path": str(model_root),
        "config_sha256": observed,
        "native_registered": True,
        "config_class": type(hf).__module__ + "." + type(hf).__qualname__,
        "num_hidden_layers": hf.num_hidden_layers,
        "layer_types": hf.layer_types,
        "fp32_master_strategy": "Native Torch AdamW uses FP32 loaded model parameters; BF16 mixed compute",
        "validated_optimizer_setup_sha256": setup_hash,
        "checkpoint_safeguard": "Raw safetensors, save_consolidated='false'; FP32 head/sinks preserved by the validated native path",
        "scope": "Configuration, architecture registration, and pinned source identities; no GPU model instantiated",
    }
    tokenizer = get_tokenizer(cfg.policy["tokenizer"])
    report["tokenizer"] = {
        "class": type(tokenizer).__name__,
        "name_or_path": tokenizer.name_or_path,
        "chat_template_sha256": digest(canonical(tokenizer.chat_template)),
        "vocab_size": len(tokenizer),
    }

    mark(report, "download_and_cache_official_datasets")
    api, revisions, loaded = HfApi(), {}, {}
    report["hf_cache"] = {
        "HF_HOME": str(hub_constants.HF_HOME),
        "HF_HUB_CACHE": str(hub_constants.HF_HUB_CACHE),
        "HF_DATASETS_CACHE": str(datasets_config.HF_DATASETS_CACHE),
    }
    report["datasets"] = {}
    for split, (repo, _) in DATASETS.items():
        info = api.dataset_info(repo_id=repo)
        require(bool(info.sha), f"Missing dataset revision: {repo}")
        revisions[split] = info.sha
        # The built-in wrappers use the default revision. Downloading that exact
        # path populates the same refs/cache which training will use offline.
        loaded[split] = load_dataset(repo, split="train")
        report["datasets"][split] = {
            "repo_id": repo,
            "repo_revision_sha": info.sha,
            "split": "train",
        }

    mark(report, "actual_builtin_dataset_setup_without_ray_environments")
    train, validation = setup_response_data(
        tokenizer, deepcopy(cfg.data), env_configs=None
    )
    require(
        not isinstance(train, dict) and validation is not None,
        "Expected one train and one validation dataset",
    )
    require(
        len(train) == len(loaded["train"])
        and len(validation) == len(loaded["validation"]),
        "Built-in dataset setup changed row counts",
    )
    for split, wrapped in (("train", train), ("validation", validation)):
        mark(report, "audit_entire_" + split + "_dataset")
        entry = report["datasets"][split]
        entry["raw"] = audit_raw(loaded[split])
        entry["processed"] = audit_processed(wrapped, DATASETS[split][1], 4096, 5120)
        require(
            entry["raw"]["ordered_question_answer_sha256"]
            == entry["processed"]["ordered_question_answer_sha256"],
            f"Built-in {split} formatting changed question/answer order or content",
        )
        require(
            entry["processed"]["unmasked_context_budget_violations"] == 0,
            f"Unmasked {split} input exceeds the real context budget",
        )
        after = api.dataset_info(repo_id=DATASETS[split][0])
        require(
            after.sha == revisions[split],
            f"Dataset revision changed while staging {split}",
        )
        entry["revision_rechecked_after_staging"] = after.sha
        entry["status"] = "passed"
    training, val = report["datasets"]["train"], report["datasets"]["validation"]
    require(
        training["raw"]["rows"] >= 16000, "DAPO train dataset has fewer than 16000 rows"
    )
    require(
        training["processed"]["unmasked_rows"] >= 1600,
        "Insufficient unmasked train prompts for 50 times 32 requested prompts",
    )
    require(
        val["raw"]["rows"] == 960
        and val["raw"]["unique_question_answer_pairs"] == 30
        and val["raw"]["pair_multiplicity_histogram"] == {32: 30},
        "Expected unaltered AIME-2024 30 question/answer pairs, each repeated 32 times",
    )
    require(
        val["processed"]["masked_rows"] == 0
        and val["processed"]["unmasked_rows"] == 960
        and val["processed"]["unmasked_input_tokens_max"] <= 1024,
        "All 960 AIME rows must actually remain unmasked within the input budget",
    )

    mark(report, "dapo_verifier_smoke")
    correct, incorrect = (
        compute_score("Answer:42", "42"),
        compute_score("Answer:41", "42"),
    )
    require(
        isinstance(correct, dict) and isinstance(incorrect, dict),
        "Verifier returned unexpected schema",
    )
    require(
        correct["score"] == 1.0
        and correct["acc"] is True
        and incorrect["score"] == 0.0
        and incorrect["acc"] is False,
        "Actual DAPO verifier did not distinguish correct and incorrect answers",
    )
    report["verifier"] = {
        "status": "passed",
        "implementation": "dapo_math_verify",
        "correct": correct,
        "incorrect": incorrect,
        "scope": "Two direct verifier calls; no training reward or model quality claim",
    }
    report["package_versions"] = {}
    for name in (
        "nemo-rl",
        "nemo-automodel",
        "torch",
        "transformers",
        "datasets",
        "huggingface-hub",
    ):
        try:
            report["package_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["package_versions"][name] = None
    report["status"] = "passed"
    report["stage"] = "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    report = {
        "schema_version": 1,
        "status": "failed",
        "stage": "imports",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "scope": "Driver configuration, pinned runtime overlays, original 49-layer model config, official dataset staging, and real CPU data processors/verifier only",
        "gpu_capacity_proven": False,
        "training_completed": False,
        "full_logits_parity_proven": False,
        "dataset_content_in_receipt": "No question or answer payloads; ordered hashes and aggregate counts only",
    }
    result = 0
    try:
        run(args, report)
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        traceback.print_exc()
        result = 1
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "stage": report["stage"],
                    "receipt": str(args.output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return result


if __name__ == "__main__":
    sys.exit(main())
