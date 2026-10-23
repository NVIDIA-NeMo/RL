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

"""CPU-only recipe and launcher checks; no Slurm submission or Ray startup."""

import importlib.util
import os
import shlex
import shutil
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

G_ROOT = Path(__file__).resolve().parents[2]
# Load the actual production composer without importing the training package.
G_SPEC = importlib.util.spec_from_file_location(
    "nano35_recipe_config_loader", G_ROOT / "nemo_rl/utils/config.py"
)
assert G_SPEC is not None and G_SPEC.loader is not None
G_CONFIG_UTILS = importlib.util.module_from_spec(G_SPEC)
G_SPEC.loader.exec_module(G_CONFIG_UTILS)

G_RECIPE_DIR = G_ROOT / "examples/nemo_gym/nemotron-3.5-nano"
G_LAUNCHER = G_RECIPE_DIR / "scripts/launch_nano35_bf16_009_prefix_checkpoint.sh"


class PrefixRecipeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        G_CONFIG_UTILS.register_omegaconf_resolvers()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "recipe-test"}):
            cls.cfg = G_CONFIG_UTILS.load_config(
                G_RECIPE_DIR / "configs/rlvr_sc_bf16_009_prefix_checkpoint.yaml"
            )
            OmegaConf.resolve(cls.cfg)

    def test_reference_training_settings(self) -> None:
        cfg = self.cfg
        self.assertEqual(cfg.policy.precision, "bfloat16")
        self.assertEqual(cfg.policy.megatron_cfg.mtp_num_layers, 5)
        self.assertTrue(cfg.policy.megatron_cfg.mtp_use_repeated_layer)
        self.assertTrue(cfg.policy.megatron_cfg.mtp_detach_heads)
        self.assertEqual(cfg.policy.megatron_cfg.mtp_loss_scaling_factor, 0.3)
        self.assertEqual(dict(cfg.policy.hf_config_overrides), {})
        self.assertFalse(cfg.policy.megatron_cfg.fp32_lm_head)
        self.assertFalse(cfg.policy.generation.vllm_cfg.fp32_lm_head)
        self.assertFalse(cfg.policy.router_replay.enabled)
        self.assertEqual(cfg.policy.megatron_cfg.optimizer.lr, 4e-6)
        self.assertEqual(cfg.policy.megatron_cfg.scheduler.lr_warmup_iters, 10)

    def test_reference_load_and_gym_bundle(self) -> None:
        cfg = self.cfg
        self.assertEqual(cfg.grpo.max_num_steps, 100)
        self.assertEqual(cfg.grpo.seed, 42)
        self.assertEqual(cfg.grpo.num_prompts_per_step, 512)
        self.assertEqual(cfg.grpo.num_generations_per_prompt, 16)
        self.assertEqual(cfg.policy.train_global_batch_size, 512 * 16)
        self.assertEqual(cfg.async_rl.sampler.name, "ready_first")
        self.assertEqual(cfg.async_rl.sampler.max_staleness_versions, 1)
        self.assertEqual(cfg.async_rl.max_inflight_prompts, 992)
        self.assertEqual(cfg.async_rl.max_buffered_rollouts, 1024)
        self.assertEqual(cfg.async_rl.rollout_failure.on_dropped_prompt, "replace")
        self.assertEqual(cfg.policy.generation.vllm_kwargs.max_num_seqs, 768)
        self.assertEqual(cfg.env.nemo_gym.num_gpu_nodes, 2)
        self.assertEqual(cfg.env.nemo_gym.global_aiohttp_connector_limit, 16384)
        paths = cfg.env.nemo_gym.config_paths
        self.assertEqual(len(paths), 33)
        gym = G_ROOT / "3rdparty/Gym-workspace/Gym"
        for path in paths:
            self.assertTrue((gym / path).is_file(), path)

    def test_explicit_objective_differences(self) -> None:
        cfg = self.cfg
        self.assertTrue(cfg.reward_penalties.penalize_duplicated_reasoning)
        self.assertTrue(cfg.reward_penalties.penalize_empty_final_answer)
        self.assertTrue(cfg.reward_penalties.penalize_unwanted_tokens)
        self.assertFalse(cfg.reward_penalties.penalize_malformed_think_tag)
        self.assertEqual(list(cfg.reward_penalties.token_ids.unwanted), [2])
        self.assertIsNone(cfg.grpo.invalid_tool_call_advantage)
        self.assertIsNone(cfg.grpo.malformed_thinking_advantage)
        self.assertFalse(cfg.loss_fn.force_on_policy_ratio)
        self.assertTrue(cfg.loss_fn.use_importance_sampling_correction)

    def test_prefix_checkpoint_contract(self) -> None:
        cfg = self.cfg
        self.assertTrue(cfg.checkpointing.enabled)
        self.assertTrue(cfg.checkpointing.save_optimizer)
        self.assertTrue(cfg.checkpointing.save_data_plane)
        self.assertEqual(cfg.checkpointing.save_period, 10)
        self.assertEqual(cfg.checkpointing.ft_save_period, 1)
        self.assertEqual(cfg.checkpointing.checkpoint_must_save_by, "00:01:15:00")
        self.assertTrue(cfg.token_capture.enabled)
        self.assertEqual(
            cfg.token_capture.capture_dir, "/tmp/nemo_rl/token_capture/recipe-test"
        )
        self.assertEqual(cfg.rollout_recovery.target_level, "prefix")
        self.assertEqual(cfg.rollout_checkpointing.snapshot_attempt_interval_s, 120)
        self.assertEqual(cfg.rollout_checkpointing.restore_mode, "latest")
        self.assertEqual(cfg.rollout_checkpointing.keep_latest_k, 2)
        self.assertEqual(cfg.token_capture.control_timeout_s, 120)
        self.assertEqual(cfg.policy.generation.generation_cut_control.max_workers, 2)
        self.assertEqual(cfg.rollout_recovery.gym_prepare_timeout_s, 300)
        self.assertGreater(
            cfg.async_rl.stall_watchdog.stall_timeout_s,
            cfg.rollout_recovery.gym_prepare_timeout_s
            + 2 * cfg.token_capture.control_timeout_s,
        )
        self.assertEqual(cfg.data_plane.backend, "simple")

    def test_snapshot_disabled_comparison_override(self) -> None:
        cfg = OmegaConf.merge(
            self.cfg,
            OmegaConf.from_dotlist(
                ["rollout_checkpointing.snapshot_attempt_interval_s=null"]
            ),
        )
        self.assertIsNone(cfg.rollout_checkpointing.snapshot_attempt_interval_s)
        self.assertTrue(cfg.token_capture.enabled)
        self.assertTrue(cfg.checkpointing.save_data_plane)

    def _launcher_environment(self, overrides: dict[str, str]) -> str:
        # The stub stops at the common-launcher boundary: no site credentials,
        # container inspection, shared-cache writes or scheduler calls.
        with tempfile.TemporaryDirectory() as directory:
            scripts = (
                Path(directory)
                / "standalone-rl/examples/nemo_gym/nemotron-3.5-nano/scripts"
            )
            scripts.mkdir(parents=True)
            launcher = scripts / G_LAUNCHER.name
            launcher.write_text(G_LAUNCHER.read_text())
            (scripts / "nano35_rlvr_common.sh").write_text(
                'printf "SETTINGS=%s,%s,%s,%s,%s,%s,%s,%s\\n" '
                '"$NUM_TRAIN_NODES" "$NUM_GEN_NODES" "$NUM_GYM_NODES" '
                '"$GENRM_REPLICAS" "$NL2BASH_REPLICAS" "$NRL_MAX_STEPS" '
                '"$WALLTIME" "$NRL_CHECKPOINT_SUBSET"\n'
                'printf "OVERRIDES=%s\\n" "$*"\n'
            )
            env = {"PATH": os.environ["PATH"], "EXP_NAME": "recipe-test"}
            env.update(overrides)
            return subprocess.run(
                [
                    "bash",
                    str(launcher),
                    "rollout_checkpointing.snapshot_attempt_interval_s=null",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            ).stdout

    def test_launcher_defaults_and_argument_forwarding(self) -> None:
        subprocess.run(["bash", "-n", str(G_LAUNCHER)], check=True)
        output = self._launcher_environment({})
        self.assertIn("SETTINGS=32,32,2,8,4,100,1:30:00,0", output)
        self.assertIn(
            "OVERRIDES=rollout_checkpointing.snapshot_attempt_interval_s=null", output
        )

    def test_launcher_explicit_overrides(self) -> None:
        output = self._launcher_environment(
            {"WALLTIME": "2:00:00", "NRL_MAX_STEPS": "101"}
        )
        self.assertIn("SETTINGS=32,32,2,8,4,101,2:00:00,0", output)

    def test_full_standalone_dry_run(self) -> None:
        """No parent pipeline, real assets, Slurm or container runtime required."""
        with tempfile.TemporaryDirectory() as directory:
            shared = Path(directory)
            root = shared / "standalone-rl"
            recipe = root / "examples/nemo_gym/nemotron-3.5-nano"
            shutil.copytree(G_RECIPE_DIR, recipe)
            shutil.copytree(
                G_ROOT / "tools/external_gym_vllm",
                root / "tools/external_gym_vllm",
            )
            # Only host-side existence checks run in dry-run mode.
            for name in (
                "ray.sub",
                "examples/run_grpo_single_controller.py",
                "3rdparty/Gym-workspace/Gym/responses_api_models/"
                "local_vllm_model/local_vllm_model_actor.py",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            assets = shared / "assets"
            assets.mkdir()
            variables = {
                "EXTERNAL_VLLM_SHARED_ROOT": str(shared),
                "RESULTS_ROOT": str(root / "results"),
                "PERSISTENT_CACHE": str(root / "cache"),
                "X11_LIB_DIR": str(assets),
                "SLURM_ACCOUNT": "fixture",
                "SLURM_PARTITION": "fixture",
                "SLURM_QOS": "normal",
            }
            for variable in (
                "CONTAINER",
                "SANDBOX_CONTAINER",
                "MODEL_PATH",
                "TRAIN_PATH",
                "GENRM_MODEL",
                "GENRM_REASONING_PARSER",
                "NL2BASH_JUDGE_MODEL",
                "SAFETY_JUDGE_MODEL",
            ):
                path = assets / variable.lower()
                path.touch()
                variables[variable] = str(path)
            for name in (
                "libX11.so.6",
                "libXau.so.6",
                "libXdmcp.so.6",
                "libXext.so.6",
                "libXrender.so.1",
                "libxcb.so.1",
            ):
                (assets / name).touch()
            overlay = shared / "site.sh"
            overlay.write_text(
                "".join(
                    f"export {name}={shlex.quote(value)}\n"
                    for name, value in variables.items()
                )
            )
            # A scheduler call would be a test failure, even with real Slurm on PATH.
            bin_dir = shared / "bin"
            bin_dir.mkdir()
            sentinel = shared / "sbatch-called"
            sbatch = bin_dir / "sbatch"
            sbatch.write_text(
                f"#!/bin/bash\ntouch {shlex.quote(str(sentinel))}\nexit 99\n"
            )
            sbatch.chmod(0o755)
            env = {
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "EXP_NAME": "native-recipe-test",
                "DRY_RUN": "1",
                # Verify a relative overlay survives the launcher's cwd change.
                "SITE_OVERLAY": overlay.name,
            }
            result = subprocess.run(
                [
                    "bash",
                    str(recipe / "scripts" / G_LAUNCHER.name),
                    "rollout_checkpointing.snapshot_attempt_interval_s=null",
                ],
                cwd=shared,
                env=env,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(sentinel.exists())
            self.assertIn("nothing submitted", result.stdout)
            self.assertIn("/opt/nano35-config/", result.stdout)
            self.assertIn(
                "rollout_checkpointing.snapshot_attempt_interval_s=null", result.stdout
            )
            provenance = list((root / "results").glob("*/runs/*/provenance.txt"))
            self.assertTrue(provenance)
            metadata = provenance[0].read_text()
            self.assertIn("nodes_total: 86", metadata)
            self.assertIn(f"recipe_root: {recipe}", metadata)
            self.assertIn(f"config_host: {recipe}/configs/", metadata)
            self.assertIn(f"{root}/nemo_rl:/opt/nemo-rl/nemo_rl", metadata)
            self.assertNotIn("pipeline_commit:", metadata)
            self.assertNotIn("pipeline_branch:", metadata)


if __name__ == "__main__":
    unittest.main()
