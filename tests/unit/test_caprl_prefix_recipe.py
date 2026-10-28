# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU recipe checks; no Ray, GPU, model download, or Slurm needed."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "examples/nemo_gym/supervl3p5/scripts"
CONFIG = (
    ROOT
    / "examples/configs/recipes/vlm/vlm_grpo-supervl3p5-caprl-video-32n4g-megatron-prefix-smoke.yaml"
)
spec = importlib.util.spec_from_file_location(
    "caprl_preflight", SCRIPT_DIR / "preflight_caprl.py"
)
assert spec is not None and spec.loader is not None
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class TestCaprlPrefixRecipe(unittest.TestCase):
    def setUp(self) -> None:
        self.config = preflight.load_config(CONFIG, [])

    def test_reference_training_and_video_settings(self) -> None:
        config = self.config
        preflight.validate_config(config)
        self.assertEqual(config["cluster"]["num_nodes"], 32)
        self.assertEqual(config["cluster"]["gpus_per_node"], 4)
        self.assertEqual(config["cluster"]["segment_size"], 8)
        policy = config["policy"]
        self.assertEqual(
            policy["generation"]["colocated"]["resources"]["num_nodes"], 16
        )
        self.assertEqual(config["grpo"]["num_prompts_per_step"], 128)
        self.assertEqual(config["grpo"]["num_generations_per_prompt"], 16)
        self.assertEqual(policy["train_global_batch_size"], 2048)
        self.assertEqual(policy["megatron_cfg"]["optimizer"]["lr"], 3e-6)
        self.assertEqual(policy["megatron_cfg"]["scheduler"]["lr_decay_iters"], 60)
        self.assertEqual(policy["megatron_cfg"]["mtp_num_layers"], 0)
        self.assertTrue(policy["megatron_cfg"]["freeze_vision_model"])
        self.assertTrue(policy["megatron_cfg"]["freeze_vision_projection"])
        self.assertEqual(config["data"]["default"]["num_frames"], 64)
        self.assertEqual(config["data"]["default"]["video_temporal_patch_size"], 2)
        self.assertEqual(
            config["async_rl"]["sampler"],
            {"name": "ready_first", "max_staleness_versions": 1},
        )

    def test_checkpoint_smoke_settings(self) -> None:
        config = self.config
        self.assertEqual(config["grpo"]["max_num_steps"], 5)
        self.assertEqual(config["checkpointing"]["keep_top_k"], 3)
        self.assertIsNone(config["checkpointing"]["checkpoint_must_save_by"])
        self.assertEqual(
            config["rollout_checkpointing"]["snapshot_attempt_interval_s"], 120
        )
        self.assertEqual(config["rollout_checkpointing"]["keep_latest_k"], 3)
        self.assertTrue(
            config["rollout_checkpointing"]["fallback_to_previous_trainer_checkpoint"]
        )
        self.assertTrue(config["checkpointing"]["load_replay_buffer"])
        self.assertIsNone(config["grpo"]["async_grpo"])
        self.assertNotIn(
            "offload_policy_before_refit",
            config["policy"]["generation"]["mcore_generation_config"],
        )

    def test_only_mcqa_services_and_preserved_bad_words(self) -> None:
        gym = self.config["env"]["nemo_gym"]
        self.assertEqual(
            gym["config_paths"],
            [
                "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
                "resources_servers/mcqa/configs/mcqa.yaml",
            ],
        )
        words = self.config["policy"]["generation"]["bad_words"]
        self.assertEqual(
            words,
            ["<image>", "<img>", "</img>", "<so_embedding>", "<so_start>", "<so_end>"],
        )
        self.assertEqual(
            gym["policy_model"]["responses_api_models"]["vllm_model"]["extra_body"][
                "bad_words"
            ],
            words,
        )
        self.assertNotIn("control_auth_token", self.config["token_capture"])

    def test_rejects_unsafe_overrides(self) -> None:
        for override in [
            "grpo.deduplicate_multimodal_data=true",
            "token_capture.enabled=false",
            "rollout_recovery.target_level=sibling",
            "checkpointing.save_data_plane=false",
            "policy.generation.vllm_kwargs.video_pruning_rate=0.7",
        ]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                preflight.validate_config(preflight.load_config(CONFIG, [override]))

    def test_no_snapshot_baseline_preserves_training_and_capture(self) -> None:
        config = preflight.load_config(
            CONFIG,
            [
                "rollout_recovery.target_level=sibling",
                "rollout_checkpointing.snapshot_attempt_interval_s=null",
                "++rollout_checkpointing.telemetry_interval_s=30",
                "++async_rl.batch_attribution=true",
            ],
        )
        preflight.validate_config(config)
        self.assertTrue(config["async_rl"]["batch_attribution"])
        self.assertEqual(config["rollout_checkpointing"]["telemetry_interval_s"], 30)
        for key in ["policy", "grpo", "data", "token_capture", "checkpointing"]:
            self.assertEqual(config[key], self.config[key])

    def test_rejects_prefix_without_snapshot_interval(self) -> None:
        with self.assertRaises(ValueError):
            preflight.validate_config(
                preflight.load_config(
                    CONFIG,
                    ["rollout_checkpointing.snapshot_attempt_interval_s=null"],
                )
            )

    def test_shell_syntax(self) -> None:
        for script in [
            "launch_caprl_prefix_smoke.sh",
            "launch_caprl_snapshot_ablation.sh",
            "caprl_runtime.sh",
        ]:
            subprocess.run(["bash", "-n", str(SCRIPT_DIR / script)], check=True)

    def test_runtime_preflight_uses_vllm_tokenizer_wrapper(self) -> None:
        raw_tokenizer = SimpleNamespace()
        wrapped_tokenizer = SimpleNamespace(max_token_id=29)
        get_tokenizer = MagicMock(return_value=wrapped_tokenizer)
        hf_loader = MagicMock(return_value=raw_tokenizer)
        params = SimpleNamespace(bad_words_token_ids=None)

        def resolve_bad_words(tokenizer: SimpleNamespace) -> None:
            # Reproduce the field access that failed with raw AutoTokenizer.
            params.bad_words_token_ids = [[tokenizer.max_token_id]]

        params.update_from_tokenizer = MagicMock(side_effect=resolve_bad_words)
        sampling_params = MagicMock(return_value=params)
        blockers = MagicMock(return_value=[])
        modules = {
            name: ModuleType(name)
            for name in [
                "transformers",
                "vllm",
                "vllm.tokenizers",
                "vllm.model_executor.models.nano_nemotron_vl",
                "nemo_rl.algorithms.single_controller_utils.config",
                "nemo_rl.models.generation.vllm.vllm_worker_async",
                "nemo_rl.utils.outdated_config_checks",
            ]
        }
        modules["transformers"].AutoTokenizer = SimpleNamespace(
            from_pretrained=hf_loader
        )
        modules["vllm"].SamplingParams = sampling_params
        modules["vllm.tokenizers"].get_tokenizer = get_tokenizer
        modules[
            "vllm.model_executor.models.nano_nemotron_vl"
        ].NanoNemotronVLProcessingInfo = SimpleNamespace(
            get_video_pruning_rate=lambda: 0.0
        )
        modules[
            "nemo_rl.algorithms.single_controller_utils.config"
        ].MasterConfig = MagicMock()
        modules[
            "nemo_rl.models.generation.vllm.vllm_worker_async"
        ]._generation_cut_blockers = blockers
        modules[
            "nemo_rl.utils.outdated_config_checks"
        ].check_outdated_config = MagicMock()

        with (
            patch.dict(sys.modules, modules),
            patch.object(
                preflight, "validate_video_sample", return_value={"frames": 64}
            ) as video_check,
            patch("builtins.print") as report,
        ):
            preflight.validate_runtime(self.config)

        get_tokenizer.assert_called_once_with(
            self.config["policy"]["model_name"], trust_remote_code=True
        )
        hf_loader.assert_not_called()
        sampling_params.assert_called_once_with(
            bad_words=self.config["policy"]["generation"]["bad_words"]
        )
        params.update_from_tokenizer.assert_called_once_with(wrapped_tokenizer)
        blockers.assert_called_once_with(params)
        video_check.assert_called_once_with(self.config)
        self.assertEqual(json.loads(report.call_args.args[0])["runtime"], "PASS")

    def test_caprl_processor_is_in_native_video_allowlist(self) -> None:
        # Read the production constant without importing torch on the submit host.
        tree = ast.parse((ROOT / "nemo_rl/environments/nemotron_utils.py").read_text())
        names = next(
            node.value.args[0]
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "NEMOTRON_VIDEO_PROCESSOR_NAMES"
                for target in node.targets
            )
        )
        self.assertIn("NemotronH_Omni_Reasoning_V3Processor", ast.literal_eval(names))

    def test_video_preflight_exercises_dataset_processor(self) -> None:
        @dataclass
        class TaskSpec:
            num_frames: int | None = None
            video_temporal_patch_size: int | None = None

        processor = type("NemotronH_Omni_Reasoning_V3Processor", (), {})()
        loader = MagicMock(return_value=processor)
        frames = MagicMock()
        frames.as_tensor.return_value.reshape.return_value.tolist.return_value = [64]
        datum = {
            "length": 9264,
            "loss_multiplier": 1.0,
            "message_log": [{"num_frames": frames}],
            "extra_env_info": {
                "responses_create_params": {
                    "input": [
                        {
                            "content": [
                                {
                                    "type": "input_video",
                                    "video_url": {"url": "data:video/test;base64,AA=="},
                                }
                            ]
                        }
                    ]
                }
            },
        }
        process_row = MagicMock(return_value=datum)
        modules = {
            name: ModuleType(name)
            for name in [
                "nemo_rl.algorithms.utils",
                "nemo_rl.data.interfaces",
                "nemo_rl.data.processors",
                "nemo_rl.environments.nemotron_utils",
            ]
        }
        modules["nemo_rl.algorithms.utils"].get_tokenizer = loader
        modules["nemo_rl.data.interfaces"].TaskDataSpec = TaskSpec
        modules["nemo_rl.data.processors"].nemo_gym_data_processor = process_row
        modules[
            "nemo_rl.environments.nemotron_utils"
        ].NEMOTRON_VIDEO_PROCESSOR_NAMES = frozenset({type(processor).__name__})
        with tempfile.TemporaryDirectory(prefix="caprl-video-") as directory:
            dataset = Path(directory) / "sample.jsonl"
            row = {
                "agent_ref": {"name": "mcqa_simple_agent"},
                "responses_create_params": {"input": []},
            }
            dataset.write_text(json.dumps(row) + "\n")
            self.config["data"]["train"][0]["data_path"] = str(dataset)
            with patch.dict(sys.modules, modules):
                result = preflight.validate_video_sample(self.config)
                self.assertEqual(
                    result,
                    {
                        "processor_class": type(processor).__name__,
                        "prompt_tokens": 9264,
                        "frames": 64,
                        "native_video_parts": 1,
                    },
                )
                args = process_row.call_args.args
                self.assertEqual(json.loads(args[0]["extra_env_info"]), row)
                self.assertEqual(args[1].num_frames, 64)
                self.assertEqual(args[1].video_temporal_patch_size, 2)
                self.assertIs(args[2], processor)
                self.assertEqual(args[3], 65536)
                loader.assert_called_once_with(
                    self.config["policy"]["tokenizer"], get_processor=True
                )

                datum["message_log"][0][
                    "num_frames"
                ].as_tensor.return_value.reshape.return_value.tolist.return_value = [32]
                with self.assertRaisesRegex(ValueError, "frame count changed"):
                    preflight.validate_video_sample(self.config)
                datum["message_log"][0][
                    "num_frames"
                ].as_tensor.return_value.reshape.return_value.tolist.return_value = [64]
                datum["extra_env_info"]["responses_create_params"]["input"] = []
                with self.assertRaisesRegex(ValueError, "native-video manifest"):
                    preflight.validate_video_sample(self.config)

    def test_assets_and_dry_run_without_cluster(self) -> None:
        with tempfile.TemporaryDirectory(prefix="caprl-recipe-") as directory:
            temp = Path(directory)
            model = temp / "model"
            model.mkdir()
            for name in [
                "config.json",
                "tokenizer.json",
                "chat_template.jinja",
                "preprocessor_config.json",
                "processor_config.json",
                "model-00001.safetensors",
            ]:
                (model / name).touch()
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "model-00001.safetensors"}})
            )
            frame = temp / "frame.png"
            frame.touch()
            data = temp / "train.jsonl"
            data.write_text(
                json.dumps(
                    {
                        "agent_ref": {"name": "mcqa_simple_agent"},
                        "responses_create_params": {
                            "input": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_image",
                                            "image_url": str(frame),
                                            "_is_video_frame": True,
                                        }
                                        for _ in range(64)
                                    ],
                                }
                            ]
                        },
                    }
                )
                + "\n"
            )
            container = temp / "image.sqsh"
            container.touch()
            cache = temp / "conversion"
            overrides = self.config["policy"]["hf_config_overrides"]
            digest = hashlib.sha256(
                json.dumps(overrides, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:12]
            entry = (
                cache
                / f"model_{str(model).replace('/', '_')}__hfovr_{digest}"
                / "iter_0000000"
            )
            entry.mkdir(parents=True)
            (entry / "run_config.yaml").touch()
            env = {
                **os.environ,
                "CAPRL_MODEL_PATH": str(model),
                "CAPRL_TRAIN_PATH": str(data),
                "CAPRL_CONVERSION_CACHE": str(cache),
                "CONTAINER": str(container),
                "RESULTS_ROOT": str(temp / "results"),
                "EXP_NAME": "fixture",
                "DRY_RUN": "1",
                "RESUME": "0",
                "HOST_PYTHON": sys.executable,
                "SLURM_DEPENDENCY": "afterany:12345",
            }
            with patch.dict(os.environ, env):
                result = preflight.validate_assets(preflight.load_config(CONFIG, []))
            self.assertEqual(result["first_sample_frames"], 64)
            run = subprocess.run(
                ["bash", str(SCRIPT_DIR / "launch_caprl_prefix_smoke.sh")],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("nodes=32; GPUs/node=4", run.stdout)
            self.assertIn("--time=1:00:00", run.stdout)
            self.assertIn("--dependency=afterany:12345", run.stdout)
            self.assertIn("DRY RUN:", run.stdout)
            self.assertFalse((temp / "results").exists())

    def test_invalid_media_capture_combination(self) -> None:
        config = copy.deepcopy(self.config)
        config["policy"]["is_vlm"] = False
        with self.assertRaises(ValueError):
            preflight.validate_config(config)


if __name__ == "__main__":
    unittest.main()
