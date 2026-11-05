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

"""Two-GPU V1/full-batch SC regression, paired with MCore's split parity suite."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

pytest.importorskip("nemo_automodel")
pytestmark = pytest.mark.automodel


@pytest.fixture
def tiny_dense_model_path(tmp_path: Path) -> Path:
    """Create deterministic local assets without gated tokenizer downloads."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocabulary = {f"token_{index}": index for index in range(1024)}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="token_0")),
        unk_token="token_0",
        pad_token="token_1",
        bos_token="token_2",
        eos_token="token_3",
    )
    torch.manual_seed(731)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
    )
    path = tmp_path / "tiny_dense"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.fixture
def tiny_moe_model_path(tiny_dense_model_path: Path, tmp_path: Path) -> Path:
    """Native MoE with real expert dispatch; reuse the existing test tokenizer."""
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(tiny_dense_model_path)
    config = Qwen3MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.0,
        tie_word_embeddings=False,
    )
    model = Qwen3MoeForCausalLM(config)
    path = tmp_path / "tiny_moe"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


@pytest.mark.timeout(660)
@pytest.mark.parametrize(
    "expert_parallel_size,dispatch_sizes,precision,activation_checkpointing,masked_rank,recipe_optimizer",
    [
        (1, "2", "bfloat16", False, None, False),
        (2, "2", "bfloat16", False, None, False),
        (1, "3,3", "float32", False, None, False),
        (1, "1,2,3", "float32", False, None, False),
        (1, "3,3", "bfloat16", False, None, False),
        (1, "1,2,3", "bfloat16", False, None, False),
        (2, "3,3", "bfloat16", False, None, False),
        (2, "1,2,3", "bfloat16", False, None, False),
        (2, "1,2,3", "bfloat16", True, None, False),
        (1, "1,2,3", "float32", False, 1, False),
        (2, "1,2,3", "bfloat16", True, None, True),
    ],
)
def test_full_batch_split_parity(
    request: pytest.FixtureRequest,
    expert_parallel_size: int,
    dispatch_sizes: str,
    precision: str,
    activation_checkpointing: bool,
    masked_rank: int | None,
    recipe_optimizer: bool,
    tmp_path: Path,
) -> None:
    """Exercise real FSDP gradients, updates, metadata fetch, and abort on two ranks.

    Reuse the standalone numerical harness so CI and larger model qualification
    execute the same assertions. Two updates per loss normalization exercise
    optimizer/scheduler progression, with two microbatches per rank and a masked
    sample. Network TQ, Ray dispatch, and live refits are covered by L1.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    root = Path(__file__).resolve().parents[4]
    model_path = request.getfixturevalue(
        "tiny_dense_model_path" if expert_parallel_size == 1 else "tiny_moe_model_path"
    )
    register_omegaconf_resolvers()
    config = load_config("examples/configs/grpo_math_1B.yaml")
    config.policy.automodel_cfg.expert_parallel_size = expert_parallel_size
    config.policy.automodel_cfg.activation_checkpointing = activation_checkpointing
    if recipe_optimizer:
        large_recipe = load_config(
            "examples/configs/recipes/llm/grpo-qwen3.5-35ba3b-base-3n8g-automodel-ep16-single-controller-full-batch.yaml"
        )
        config.policy.optimizer = large_recipe.policy.optimizer
        config.policy.scheduler = large_recipe.policy.scheduler
    if expert_parallel_size > 1:
        config.policy.automodel_cfg.automodel_kwargs.backend = {
            "_target_": "nemo_automodel.components.models.common.utils.BackendConfig",
            "attn": "te",
            "linear": "te",
            "rms_norm": "te",
            "experts": "gmm",
            "dispatcher": "hybridep",
            "rope_fusion": False,
            "enable_hf_state_dict_adapter": True,
        }
    recipe = tmp_path / "recipe.yaml"
    OmegaConf.save(config, recipe)
    env = os.environ.copy()
    env.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(root / "tests/functional/automodel_sc_full_batch_parity.py"),
        "--model",
        str(model_path),
        "--recipe",
        str(recipe),
        "--strict-determinism",
        "--rtol",
        "0" if dispatch_sizes == "2" else "1e-5",
        "--atol",
        "0" if dispatch_sizes == "2" else "1e-7",
        "--local-batch-size",
        str(sum(int(size) for size in dispatch_sizes.split(","))),
        "--dispatch-sizes",
        dispatch_sizes,
        "--steps",
        "2" if dispatch_sizes == "2" else "3",
        "--precision",
        precision,
        "--output",
        str(tmp_path),
        "--source-sha",
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
    ]
    if expert_parallel_size == 1:
        command.append("--force-hf")
    if precision == "bfloat16" and dispatch_sizes != "2":
        command.append("--bf16-contract")
        command.append("--optimizer-oracle")
    if masked_rank is not None:
        command.extend(["--mask-rank", str(masked_rank)])
    if recipe_optimizer:
        command.append("--recipe-optimizer")
    # Kill the entire torchrun group on timeout, including its CUDA workers.
    log_path = tmp_path / "torchrun.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=600)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    assert returncode == 0, log_path.read_text()
    report = json.loads((tmp_path / "parity.json").read_text())
    assert report["passed"]
    assert {(row["token_level"], row["step"]) for row in report["records"]} == {
        (token_level, step)
        for token_level in (True, False)
        for step in range(2 if dispatch_sizes == "2" else 3)
    }
