# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Native CUA integration smoke; optional short desktop task, not a benchmark."""

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import ray
import requests
import torch
from logra.actor_environments import register_actor_environments
from logra.cua import TaskBaseline, make_cua_loss
from logra.osworld import (
    CuaSegment,
    DesktopRollouts,
    Episode,
    TurnRecorder,
    make_training_batch,
    read_tasks,
)
from logra.policy import make_policy_factory
from omegaconf import OmegaConf
from PIL import Image
from run_cua import ResearchConfig

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.multimodal_utils import image_to_data_url
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def check_expert_weights(manifests, checkpoint):
    """Compare native EP shards with original HF checkpoint entries, exactly."""
    import re
    from contextlib import ExitStack

    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    files = {}
    checked = 0
    with ExitStack() as stack:
        for manifest in manifests:
            for entry in manifest:
                if "sample_values" not in entry:
                    continue
                name = re.sub(
                    r"\._checkpoint_wrapped_module|\._fsdp_wrapped_module|\._orig_mod",
                    "",
                    entry["name"],
                )
                prefix, kind = name.rsplit(".", 1)
                prefix = prefix.replace(
                    "language_model.model.", "language_model.backbone."
                )
                projection = "up_proj" if kind == "gate_and_up_projs" else "down_proj"
                for row, values in zip(
                    entry["sample_rows"], entry["sample_values"], strict=True
                ):
                    expert, native_row = divmod(row, entry["shape"][1])
                    key = f"{prefix}.{expert}.{projection}.weight"
                    shard = index[key]
                    if shard not in files:
                        files[shard] = stack.enter_context(
                            safe_open(checkpoint / shard, framework="pt", device="cpu")
                        )
                    weights = files[shard].get_slice(key)
                    for column, actual in zip(
                        entry["sample_columns"], values, strict=True
                    ):
                        expected = (
                            weights[column : column + 1, native_row : native_row + 1]
                            .float()
                            .item()
                        )
                        assert actual == expected, (
                            key,
                            column,
                            native_row,
                            actual,
                            expected,
                        )
                        checked += 1
    print("CUA_HF_EXPERT_WEIGHT_SAMPLES_OK", checked, flush=True)


def check_rollout_expert_weights(wrapper, checkpoint):
    """Compare refitted TP expert shards inside each GPU worker; return no weights."""
    from contextlib import ExitStack

    from safetensors import safe_open
    from vllm.distributed import get_tensor_model_parallel_rank

    worker = wrapper.worker
    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tp_rank = get_tensor_model_parallel_rank()
    files, mismatched = {}, []
    checked = 0
    with ExitStack() as stack, torch.no_grad():
        for name, parameter in worker.model_runner.model.named_parameters():
            if not name.endswith((".w13_weight", ".w2_weight")):
                continue
            up = name.endswith(".w13_weight")
            prefix = name.rsplit(".routed_experts.", 1)[0].replace(
                "language_model.model.", "language_model.backbone."
            )
            for expert in (0, parameter.shape[0] // 2, parameter.shape[0] - 1):
                key = f"{prefix}.{expert}.{'up_proj' if up else 'down_proj'}.weight"
                shard = index[key]
                if shard not in files:
                    files[shard] = stack.enter_context(
                        safe_open(checkpoint / shard, framework="pt", device="cpu")
                    )
                weights = files[shard].get_slice(key)
                for fraction in (0, 0.5, 1):
                    row, column = (
                        min(int(size * fraction), size - 1)
                        for size in parameter.shape[1:]
                    )
                    hf_row = row + (tp_rank * parameter.shape[1] if up else 0)
                    hf_column = column + (0 if up else tp_rank * parameter.shape[2])
                    expected = weights[hf_row : hf_row + 1, hf_column : hf_column + 1]
                    if (
                        parameter[expert, row, column].item()
                        != expected.to(parameter.dtype).item()
                    ):
                        mismatched.append(name)
                    checked += 1
    return {"checked": checked, "mismatched": sorted(set(mismatched))}


def capture_turns(generation, processor, model_name, messages):
    """Exercise native HTTP tokens, R3 and continuation; no desktop/reward claim."""
    endpoint = generation.dp_openai_server_base_urls[0]
    assert endpoint is not None
    recorder = TurnRecorder(processor)
    payload = {
        "model": model_name,
        "messages": copy.deepcopy(messages),
        "max_tokens": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
    }
    generated = 0
    with requests.Session() as session:
        for turn in range(3):
            response = session.post(
                endpoint.rstrip("/") + "/chat/completions",
                json=recorder.request_body(payload),
                timeout=120,
            )
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            generated += len(message["generation_token_ids"])
            reply = recorder.record(payload, message)
            if turn == 0:
                payload["messages"].extend(
                    [
                        {"role": "assistant", "content": reply["content"] or ""},
                        {"role": "user", "content": "Name another visible feature."},
                    ]
                )
            else:
                # Removing the previous turns emulates screenshot compaction.
                payload["messages"] = copy.deepcopy(messages)
                payload["messages"][0]["content"][0]["text"] = (
                    "Describe this new screenshot."
                )
    assert generated > 0
    assert sum(int(s.token_mask.sum()) for s in recorder.segments) == generated
    assert len(recorder.segments) >= 2
    print(
        "CUA_NATIVE_HTTP_CAPTURE_OK",
        {"tokens": generated, "segments": len(recorder.segments)},
        flush=True,
    )
    return recorder.segments


def check_http_template(
    generation: VllmGeneration,
    processor: Any,
    model_name: str,
    messages: list[dict[str, Any]],
    picture: Image.Image,
) -> None:
    """Require native HTTP renders to match the original client-side template."""
    history = [
        {"role": "assistant", "content": "First line.\nSecond line."},
        {"role": "user", "content": "Continue with the next observation."},
    ]
    cases = [
        ([{"role": "user", "content": "First instruction"}, *history], []),
        (copy.deepcopy(messages) + history + copy.deepcopy(history), [picture]),
    ]
    with requests.Session() as session:
        for chat, images in cases:
            rendered = processor.apply_chat_template(
                TurnRecorder._text_messages(chat),
                tokenize=False,
                add_generation_prompt=True,
            )
            expected = (
                processor(text=rendered, images=images, return_tensors="pt")[
                    "input_ids"
                ][0].tolist()
                if images
                else processor.tokenizer.encode(rendered, add_special_tokens=False)
            )
            native_messages = TurnRecorder(processor).request_body({"messages": chat})[
                "messages"
            ]
            for endpoint in generation.dp_openai_server_base_urls:
                response = session.post(
                    endpoint.rstrip("/").removesuffix("/v1") + "/tokenize",
                    json={
                        "model": model_name,
                        "messages": native_messages,
                        "add_generation_prompt": True,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                actual = response.json()["tokens"]
                assert actual == expected, {
                    "expected_length": len(expected),
                    "actual_length": len(actual),
                    "first_difference": next(
                        (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                        None,
                    ),
                }
    print(
        "CUA_HTTP_TEMPLATE_MATCH_OK",
        {"engines": len(generation.dp_openai_server_base_urls), "cases": len(cases)},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--with-generation",
        action="store_true",
        help="Split one four-GPU node into EP2 training and TP2 rollout workers",
    )
    parser.add_argument(
        "--with-desktop",
        action="store_true",
        help="Run the first held-out desktop task using its real verifier",
    )
    parser.add_argument(
        "--full-allocation",
        action="store_true",
        help="Use the recipe's complete train/rollout allocation and context length",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="Synthetic long-context backward stress test; skips checkpoint/refit",
    )
    parser.add_argument(
        "--screenshots",
        nargs=2,
        type=Path,
        metavar=("BEFORE", "AFTER"),
        help="Train on two real screenshots of different sizes; skip checkpointing",
    )
    parser.add_argument("--desktop-steps", type=int, default=3)
    args, overrides = parser.parse_known_args()
    if args.desktop_steps < 1:
        parser.error("--desktop-steps must be positive")
    if args.sequence_length and args.with_generation:
        parser.error("--sequence-length is a training-only synthetic stress test")
    if args.with_desktop and not args.with_generation:
        parser.error("--with-desktop requires --with-generation")
    if args.screenshots and args.with_generation:
        parser.error("--screenshots is a training-only input regression")
    register_omegaconf_resolvers()
    raw = parse_hydra_overrides(load_config(args.config), overrides)
    cfg = ResearchConfig.model_validate(OmegaConf.to_container(raw, resolve=True))
    tasks = read_tasks(cfg.osworld.train_data, cfg.osworld.train_task_base)
    validation = read_tasks(
        cfg.osworld.validation_data, cfg.osworld.validation_task_base
    )
    processor = get_tokenizer(cfg.policy["tokenizer"], get_processor=True)
    tokenizer = processor.tokenizer
    # Exercise the real processor and model with a tiny test image; no OSWorld
    # success rate or rollout/train agreement is inferred from synthetic actions.
    pictures = (
        [Image.open(path).convert("RGB") for path in args.screenshots]
        if args.screenshots
        else [Image.new("RGB", (256, 256), (100, 120, 140))]
    )
    if args.screenshots and len({image.size for image in pictures}) != 2:
        parser.error("--screenshots requires two different image sizes")
    picture = pictures[0]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the screenshot."},
                *[
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image)},
                    }
                    for image in pictures
                ],
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Describe the screenshot." + "<image>" * len(pictures),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = processor(
        text=rendered,
        images=pictures,
        return_tensors=None if args.screenshots else "pt",
    )
    prompt = torch.as_tensor(encoded["input_ids"], dtype=torch.long)[0]
    answer = torch.tensor(
        tokenizer.encode("The screen is blue.", add_special_tokens=False)
    )
    if args.sequence_length:
        remaining = args.sequence_length - len(prompt)
        if (
            remaining < 1
            or args.sequence_length > cfg.policy["max_total_sequence_length"]
        ):
            parser.error(
                "--sequence-length must contain the prompt and fit the configured context"
            )
        answer = answer.repeat((remaining + len(answer) - 1) // len(answer))[:remaining]
    ids = torch.cat((prompt, answer))
    mask = torch.cat((torch.zeros_like(prompt), torch.ones_like(answer))).float()
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        cfg.policy["model_name"], trust_remote_code=True
    )
    text_config = model_config.llm_config
    routes = torch.full(
        (len(ids), text_config.num_hidden_layers, text_config.num_experts_per_tok),
        -1,
        dtype=torch.int16,
    )
    segment = CuaSegment(ids, mask, torch.full_like(mask, -3.0), routes, messages)
    ep = Episode(tasks[0], "test", 1.0, True, None, [segment])
    node_gpus = cfg.cluster.gpus_per_node
    train_layout = [node_gpus // (2 if args.with_generation else 1)]
    rollout_layout = train_layout
    if args.full_allocation:
        rollout_nodes = (
            cfg.policy["generation"]["colocated"]["resources"]["num_nodes"]
            if args.with_generation
            else 0
        )
        train_layout = [node_gpus] * (cfg.cluster.num_nodes - rollout_nodes)
        rollout_layout = [node_gpus] * rollout_nodes
    train_gpus, rollout_gpus = sum(train_layout), sum(rollout_layout)
    batch = make_training_batch([ep], TaskBaseline(cfg.cua), processor, train_gpus)
    print(
        json.dumps(
            {
                "train_rows": len(tasks),
                "eval_tasks": len(validation),
                "processor": type(processor).__name__,
                "tokens": len(ids),
                "batch_keys": sorted(batch),
            }
        ),
        flush=True,
    )
    if args.preflight_only:
        print("CUA_DATA_PROCESSOR_PREFLIGHT_OK", flush=True)
        return
    register_actor_environments(serial_refit=True)
    init_ray()
    cluster = RayVirtualCluster(
        name="logra_cua_model_smoke",
        bundle_ct_per_node_list=train_layout,
        use_gpus=True,
        num_gpus_per_node=train_layout[0],
        max_colocated_worker_groups=1,
    )
    cfg.policy["automodel_cfg"]["expert_parallel_size"] = train_gpus
    cfg.policy["generation"] = configure_generation_config(
        cfg.policy["generation"], tokenizer
    )
    test_context = (
        cfg.policy["max_total_sequence_length"]
        if args.full_allocation
        else (16384 if args.with_desktop else 2048)
    )
    cfg.policy["max_total_sequence_length"] = (
        test_context if args.with_generation else len(ids)
    )
    policy = generation = generation_cluster = None
    try:
        if args.with_generation:
            generation_cluster = RayVirtualCluster(
                name="logra_cua_http_smoke",
                bundle_ct_per_node_list=rollout_layout,
                use_gpus=True,
                num_gpus_per_node=rollout_layout[0],
                max_colocated_worker_groups=1,
            )
            generation_config = cfg.policy["generation"]
            generation_config["model_name"] = cfg.policy["model_name"]
            generation_config["vllm_cfg"]["max_model_len"] = test_context
            generation_config["vllm_cfg"]["expose_http_server"] = True
            generation = VllmGeneration(
                cluster=generation_cluster, config=generation_config
            )
            generation.finish_generation()
        policy = make_policy_factory(cfg.logra, cfg.cua)(
            cluster=cluster,
            config=cfg.policy,
            tokenizer=tokenizer,
            processor=processor,
            init_reference_model=False,
        )
        manifests = policy.run_all_workers_single_data(
            "logra_manifest", sample_weights=True
        )
        output = Path(cfg.logger.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "projection-targets.json").write_text(json.dumps(manifests, indent=2))
        assert all(len(rows) == 162 for rows in manifests), [
            len(rows) for rows in manifests
        ]
        if generation is not None:
            host, port = cluster.get_master_address_and_port()
            futures = policy.init_collective(
                host, port, train_gpus + rollout_gpus, train_world_size=train_gpus
            )
            futures += generation.init_collective(
                host, port, train_gpus + rollout_gpus, train_world_size=train_gpus
            )
            ray.get(futures)
            info = policy.prepare_refit_info()
            generation.prepare_refit_info(info or {})
            refit_policy_generation(policy, generation, colocated_inference=False)
            audits = ray.get(
                generation.worker_group.run_all_workers_single_data(
                    "checkpoint_engine_rpc_async",
                    checkpoint_method=check_rollout_expert_weights,
                    method_args=(cfg.policy["model_name"],),
                    run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
                )
            )
            for engine in audits:
                for audit in engine:
                    assert audit["checked"] > 0 and not audit["mismatched"], audit
            print("CUA_VLLM_REFIT_EXPERT_SAMPLES_OK", audits, flush=True)
            check_http_template(
                generation, processor, cfg.policy["model_name"], messages, picture
            )
            generation.prepare_for_generation()
            if args.with_desktop:
                cfg.osworld.max_steps = args.desktop_steps
                cfg.osworld.max_tokens = 2048
                cfg.osworld.concurrency = 1
                rollouts = DesktopRollouts(
                    cfg.osworld,
                    processor,
                    cfg.policy["model_name"],
                    generation.dp_openai_server_base_urls,
                    output / "episodes",
                )
                episodes = rollouts.run(
                    validation[:1], step=0, seed=42, evaluation=True
                )
                summary = [
                    {
                        "task": x.task.key,
                        "score": x.score,
                        "valid": x.valid,
                        "error": x.error,
                        "segments": len(x.segments),
                        "action_tokens": sum(
                            int(s.token_mask.sum()) for s in x.segments
                        ),
                    }
                    for x in episodes
                ]
                (output / "desktop-smoke.json").write_text(
                    json.dumps(summary, indent=2)
                )
                print("CUA_REAL_DESKTOP_RESULT", summary, flush=True)
                assert all(x.valid and x.segments for x in episodes), summary
            else:
                ep.segments = capture_turns(
                    generation, processor, cfg.policy["model_name"], messages
                )
                episodes = [ep]
            generation.finish_generation()
            batch = make_training_batch(
                episodes, TaskBaseline(cfg.cua), processor, train_gpus
            )
            current = policy.get_logprobs(batch)["logprobs"]
            action_mask = batch["token_mask"] * batch["sample_mask"].reshape(-1, 1)
            gap = (
                (current - batch["generation_logprobs"]).abs() * action_mask
            ).sum() / action_mask.sum()
            assert torch.isfinite(gap)
            print("CUA_ROLLOUT_POLICY_LOGPROB_MAE", float(gap), flush=True)
        check_expert_weights(manifests, cfg.policy["model_name"])
        policy.prepare_for_training()
        result = policy.train(
            batch,
            make_cua_loss(cfg.cua, cfg.loss_fn),
            gbs=len(batch["input_lengths"]),
            mbs=1,
        )
        assert math.isfinite(float(result["loss"]))
        assert math.isfinite(float(result["grad_norm"]))
        print("CUA_NATIVE_FULL_MODEL_UPDATE_OK", result, flush=True)
        if args.screenshots:
            assert float(result["grad_norm"]) > 0
            print(
                "CUA_MIXED_RESOLUTION_FULL_MODEL_UPDATE_OK",
                {
                    "screenshots": [str(path) for path in args.screenshots],
                    "sizes": [image.size for image in pictures],
                },
                flush=True,
            )
            return
        if args.sequence_length:
            assert float(result["grad_norm"]) > 0
            print("CUA_LONG_CONTEXT_BACKWARD_OK", args.sequence_length, flush=True)
            return
        if args.with_desktop:
            # A failed task has zero advantage at initialization. Do not substitute
            # an artificial reward just to force a nonzero smoke-test update.
            print("CUA_REAL_DESKTOP_TRAINING_PASS", flush=True)
            return
        if generation is not None:
            refit_policy_generation(policy, generation, colocated_inference=False)
            generation.prepare_for_generation()
            capture_turns(generation, processor, cfg.policy["model_name"], messages)
            generation.finish_generation()
            print("CUA_NATIVE_UPDATED_POLICY_REFIT_OK", flush=True)
        destination = Path(cfg.logger.log_dir) / "smoke_checkpoint"
        policy.save_checkpoint(
            str(destination / "weights"),
            str(destination / "optimizer"),
            str(destination / "tokenizer"),
            is_final_checkpoint=True,
        )
        policy.finalize_async_save()
        print("CUA_NATIVE_FULL_MODEL_CHECKPOINT_OK", flush=True)
        saved = policy.run_all_workers_single_data(
            "logra_manifest", sample_weights=True
        )
        # Change weights before reloading so a no-op loader cannot pass this test.
        policy.prepare_for_training()
        policy.train(
            batch,
            make_cua_loss(cfg.cua, cfg.loss_fn),
            gbs=len(batch["input_lengths"]),
            mbs=1,
        )
        changed = policy.run_all_workers_single_data(
            "logra_manifest", sample_weights=True
        )
        assert changed != saved, "Second update did not change sampled expert weights"
        policy.run_all_workers_single_data(
            "load_checkpoint",
            weights_path=str(destination / "weights"),
            optimizer_path=str(destination / "optimizer"),
        )
        restored = policy.run_all_workers_single_data(
            "logra_manifest", sample_weights=True
        )
        assert saved == restored, "Checkpoint reload changed selected expert weights"
        print("CUA_NATIVE_FULL_MODEL_RELOAD_OK", flush=True)
    finally:
        if generation is not None:
            generation.shutdown()
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()
        if generation_cluster is not None:
            generation_cluster.shutdown()


if __name__ == "__main__":
    main()
