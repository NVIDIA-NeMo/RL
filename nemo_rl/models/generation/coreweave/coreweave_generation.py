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

"""Exact-token rollouts from CoreWeave Dedicated Inference."""

import math
from pathlib import Path
from typing import Any

from cwtrain.integrations.nemo_rl.generation import (
    CoreWeaveGeneration as CookbookGeneration,
)
from cwtrain.integrations.nvext import SERVED_IDENTITY_FILE, as_cache_salt
from cwtrain.integrations.rollout import Completion, CompletionSampler, RolloutError

from nemo_rl.models.generation.coreweave.config import CoreWeaveConfig
from nemo_rl.models.policy.draft_config import draft_refit_enabled


class CoreWeaveGeneration(CookbookGeneration):
    """Use the cookbook transport while requiring exact token IDs and probabilities."""

    @classmethod
    def validate_settings(cls, master_config: Any) -> None:
        """Reject unsupported training modes before allocating workers."""
        generation = master_config.policy["generation"]
        config = CoreWeaveConfig(**generation["coreweave_cfg"])
        config.validate_training()
        if generation["colocated"]["enabled"]:
            raise ValueError("CoreWeave generation requires colocated.enabled=false")
        if master_config.grpo.async_grpo.enabled:
            raise ValueError("CoreWeave currently supports synchronous GRPO only")
        policy = master_config.policy
        if policy.get("is_vlm") or (policy.get("megatron_cfg") or {}).get("enabled"):
            raise ValueError(
                "CoreWeave currently supports text-only Automodel policies"
            )
        if policy.get("quant_cfg") or (
            (policy.get("automodel_cfg") or {}).get("lora_cfg") or {}
        ).get("enabled"):
            raise ValueError(
                "CoreWeave checkpoint export does not support quantization or LoRA"
            )
        if draft_refit_enabled(policy.get("draft")) or (
            policy.get("router_replay") or {}
        ).get("enabled"):
            raise ValueError(
                "CoreWeave does not support draft training or router replay"
            )
        if not (policy.get("automodel_cfg") or {}).get("enabled"):
            raise ValueError("CoreWeave requires policy.automodel_cfg.enabled=true")
        if master_config.env.get("should_use_nemo_gym"):
            gym = master_config.env["nemo_gym"]
            model_config = "responses_api_models/external_model/configs/external_model_for_training.yaml"
            if model_config not in gym["config_paths"]:
                raise ValueError(
                    "CoreWeave Gym requires the external_model training config"
                )
            expected = {
                "policy_tokenizer_path": policy["tokenizer"]["name"],
                "policy_max_model_len": config.max_model_len,
                "policy_served_identity_file": str(
                    Path(config.state_dir).resolve() / SERVED_IDENTITY_FILE
                ),
                "policy_api_key_file": config.api_key_file,
            }
            for key, value in expected.items():
                if gym[key] != value:
                    raise ValueError(
                        f"CoreWeave Gym {key} must match the policy configuration"
                    )
        if master_config.data_plane is not None and master_config.data_plane["enabled"]:
            raise ValueError("CoreWeave currently requires data_plane.enabled=false")
        if policy.get("worker_extension_cls_fqn") is not None:
            raise ValueError("CoreWeave currently requires the stock Automodel worker")
        if policy["offload_policy_before_refit"]:
            raise ValueError("CoreWeave requires offload_policy_before_refit=false")
        if (
            generation.get("ignore_eos")
            or generation.get("bad_words")
            or generation.get("top_logprobs")
        ):
            raise ValueError(
                "CoreWeave does not support ignore_eos, bad_words, or top_logprobs"
            )
        if generation.get("use_async_rollouts"):
            raise ValueError("CoreWeave currently supports synchronous rollouts only")
        if generation.get("refit_transport") is not None:
            raise ValueError(
                "CoreWeave owns checkpoint hot-load; set refit_transport=null"
            )
        if (
            generation["temperature"] != 1.0
            or generation["top_p"] != 1.0
            or generation["top_k"] is not None
        ):
            raise ValueError(
                "CoreWeave requires temperature=1, top_p=1, top_k=null for logprob consistency"
            )
        resources = generation["colocated"]["resources"]
        if resources["num_nodes"] not in (None, 0) or resources[
            "gpus_per_node"
        ] not in (None, 0):
            raise ValueError("CoreWeave requires zero local inference resources")

    def __init__(self, config: dict[str, Any], tokenizer: Any) -> None:
        settings = CoreWeaveConfig(**config["coreweave_cfg"])
        settings.validate_training()
        config = {**config, "coreweave_cfg": settings.model_dump()}
        super().__init__(config, tokenizer)

    async def _sample_group(
        self,
        sampler: CompletionSampler,
        prompt_ids: list[int],
        n: int,
        stop: Any,
        greedy: bool,
    ) -> list[Completion]:
        """Submit token IDs without falling back to text prompts."""
        assert self.max_model_len is not None
        remaining = self.max_model_len - len(prompt_ids)
        if remaining <= 0:
            raise RolloutError("Prompt reaches the CoreWeave context limit")
        row_stops = [stop] if isinstance(stop, str) else list(stop or [])
        return await sampler.sample(
            prompt_ids,
            n,
            max_tokens=min(self.cfg["max_new_tokens"], remaining),
            temperature=0.0 if greedy else self.cfg["temperature"],
            top_p=self.cfg["top_p"],
            top_k=self.cfg["top_k"],
            stop=list(dict.fromkeys([*self.stop_strings, *row_stops])) or None,
            stop_token_ids=self.stop_token_ids or None,
            include_stop_str_in_output=True,
            cache_salt=self._cache_salt(),
        )

    def _cache_salt(self) -> str:
        """Refuse generation until a checkpoint update has been verified."""
        synchronizer = self.weight_synchronizer
        if (
            synchronizer is None
            or synchronizer.is_stale
            or synchronizer.served_identity is None
        ):
            raise RolloutError("CoreWeave sampling requires a verified checkpoint")
        return as_cache_salt(synchronizer.served_identity)

    def _decode(self, completion: Completion) -> tuple[list[int], list[float], bool]:
        """Validate sampled IDs and logprobs before they enter a training batch."""
        ids, logprobs = completion.token_ids, completion.token_logprobs
        if ids is None or logprobs is None or len(ids) != len(logprobs):
            raise RolloutError(
                "CoreWeave must return exact token IDs with aligned logprobs"
            )
        if any(not math.isfinite(value) for value in logprobs):
            raise RolloutError("CoreWeave returned non-finite sampled logprobs")
        if completion.finish_reason not in ("stop", "length"):
            raise RolloutError(
                f"Unsupported CoreWeave finish reason: {completion.finish_reason}"
            )
        return list(ids), list(logprobs), completion.finish_reason == "length"
