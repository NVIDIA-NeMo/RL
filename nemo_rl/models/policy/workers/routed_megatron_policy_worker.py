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

import math
import os
from typing import Any, Optional

import ray
import torch
from torch import nn

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
)
from nemo_rl.models.policy.routing import SequenceRouter
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)


class RoutedMegatronPolicyWorkerImpl(MegatronPolicyWorkerImpl):
    """Megatron worker extension that trains only a layer-skip router.

    This MVP intentionally uses slow, no-KV top-k autoregressive generation so
    that route masks can alter the actual model forward pass without teaching
    vLLM or Megatron dynamic inference caches about skipped layers.
    """

    def __init__(
        self,
        config,
        tokenizer,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_optimizer=init_optimizer,
            init_reference_model=init_reference_model,
            **kwargs,
        )

        routing_cfg = self.cfg.get("routing", {})
        if not routing_cfg.get("enabled", False):
            raise ValueError(
                "RoutedMegatronPolicyWorker requires policy.routing.enabled=true"
            )

        # Keep base parameters requiring grad because Megatron DDP forward
        # pre-hooks assert this during model-parallel gathers. The base model
        # is still unchanged: routed training only calls train_router() and
        # only steps self.router_optimizer.

        self.routing_cfg = routing_cfg
        self.gate_every_n_layers = int(routing_cfg.get("gate_every_n_layers", 4))
        if self.gate_every_n_layers <= 0:
            raise ValueError("policy.routing.gate_every_n_layers must be positive")

        self.num_model_layers = int(getattr(self.model.config, "num_layers"))
        self.num_routes = math.ceil(self.num_model_layers / self.gate_every_n_layers)
        self.layer_counts = torch.tensor(
            [
                max(
                    0,
                    min((idx + 1) * self.gate_every_n_layers, self.num_model_layers)
                    - idx * self.gate_every_n_layers,
                )
                for idx in range(self.num_routes)
            ],
            dtype=torch.float32,
        )
        self.total_routable_layers = float(self.layer_counts.sum().item())

        protect_first = int(routing_cfg.get("protect_first_n_groups", 1))
        protect_last = int(routing_cfg.get("protect_last_n_groups", 1))
        trainable = torch.ones(self.num_routes, dtype=torch.bool)
        if protect_first > 0:
            trainable[:protect_first] = False
        if protect_last > 0:
            trainable[-protect_last:] = False
        self.trainable_route_mask = trainable

        vocab_size = int(
            getattr(self, "final_padded_vocab_size", None)
            or getattr(self.model.config, "padded_vocab_size", 0)
            or getattr(self.model.config, "vocab_size")
        )
        self.router = SequenceRouter(
            vocab_size=vocab_size,
            num_routes=self.num_routes,
            hidden_size=int(routing_cfg.get("router_hidden_size", 128)),
            init_keep_bias=float(routing_cfg.get("init_keep_bias", 3.0)),
        ).cuda()
        self.router_optimizer = torch.optim.AdamW(
            self.router.parameters(),
            lr=float(routing_cfg.get("router_lr", 1.0e-4)),
            weight_decay=float(routing_cfg.get("router_weight_decay", 0.0)),
        )
        self.route_entropy_coef = float(routing_cfg.get("entropy_coef", 0.01))
        self.route_ratio_clip_min = float(routing_cfg.get("ratio_clip_min", 0.2))
        self.route_ratio_clip_max = float(routing_cfg.get("ratio_clip_max", 0.2))
        self.generation_top_k = int(routing_cfg.get("generation_top_k", 50))
        self._active_route_mask: Optional[torch.Tensor] = None
        self._install_layer_skip_hooks()

        self._try_load_router_checkpoint(weights_path, optimizer_path)

    def is_routed_worker(self) -> bool:
        return True

    def _try_load_router_checkpoint(
        self, weights_path: Optional[str], optimizer_path: Optional[str]
    ) -> None:
        if weights_path is not None:
            router_path = os.path.join(weights_path, "router.pt")
            if os.path.exists(router_path):
                self.router.load_state_dict(torch.load(router_path, map_location="cuda"))
                print(f"Loaded router weights from {router_path}")
        if optimizer_path is not None:
            optimizer_file = os.path.join(optimizer_path, "router_optimizer.pt")
            if os.path.exists(optimizer_file):
                self.router_optimizer.load_state_dict(
                    torch.load(optimizer_file, map_location="cuda")
                )
                print(f"Loaded router optimizer from {optimizer_file}")

    def _install_layer_skip_hooks(self) -> None:
        layers = self._find_transformer_layers()
        if not layers:
            raise ValueError(
                "Routed worker could not find Megatron transformer layers. "
                f"Inspected model structure: {self._describe_model_roots()}"
            )

        for layer in layers:
            group_idx = (int(layer.layer_number) - 1) // self.gate_every_n_layers
            original_forward = layer.forward

            def routed_forward(*args, _group_idx=group_idx, _orig=original_forward, **kwargs):
                if self._should_skip_group(_group_idx):
                    if "hidden_states" in kwargs:
                        hidden_states = kwargs["hidden_states"]
                    elif args:
                        hidden_states = args[0]
                    else:
                        raise ValueError("Could not find hidden_states for routed skip")
                    return hidden_states, kwargs.get("context", None)
                return _orig(*args, **kwargs)

            layer.forward = routed_forward
        print(f"Installed routed layer-skip hooks on {len(layers)} transformer layers")

    def _iter_model_roots(self) -> list[Any]:
        roots = (
            list(self.model)
            if isinstance(self.model, (list, tuple, nn.ModuleList))
            else [self.model]
        )
        discovered: list[Any] = []
        seen: set[int] = set()
        stack = list(roots)
        while stack:
            module = stack.pop(0)
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            discovered.append(module)
            for attr in ("module", "model"):
                child = getattr(module, attr, None)
                if child is not None and child is not module:
                    stack.append(child)
        return discovered

    def _get_module_path(self, root: Any, path: str) -> Any:
        current = root
        for attr in path.split("."):
            if not hasattr(current, attr):
                return None
            current = getattr(current, attr)
        return current

    def _looks_like_transformer_layer(self, module: Any) -> bool:
        return (
            hasattr(module, "layer_number")
            and hasattr(module, "forward")
            and (hasattr(module, "self_attention") or hasattr(module, "mlp"))
        )

    def _find_transformer_layers(self) -> list[Any]:
        candidate_paths = (
            "decoder.layers",
            "transformer.layers",
            "language_model.encoder.layers",
            "model.layers",
        )

        for root in self._iter_model_roots():
            for path in candidate_paths:
                layers = self._get_module_path(root, path)
                if layers is None:
                    continue
                layers_list = list(layers)
                if layers_list and any(
                    self._looks_like_transformer_layer(layer) for layer in layers_list
                ):
                    return [
                        layer
                        for layer in layers_list
                        if self._looks_like_transformer_layer(layer)
                    ]

        best_layers: list[Any] = []
        for root in self._iter_model_roots():
            if not hasattr(root, "named_modules"):
                continue
            for _name, module in root.named_modules():
                if not isinstance(module, nn.ModuleList):
                    continue
                layers_list = list(module)
                matching = [
                    layer
                    for layer in layers_list
                    if self._looks_like_transformer_layer(layer)
                ]
                if len(matching) > len(best_layers):
                    best_layers = matching
        return best_layers

    def _describe_model_roots(self) -> str:
        descriptions = []
        for root in self._iter_model_roots()[:8]:
            child_names = []
            if hasattr(root, "named_children"):
                child_names = [name for name, _child in root.named_children()][:8]
            descriptions.append(
                f"{type(root).__name__}(children={child_names or '[]'})"
            )
        return "; ".join(descriptions)

    def _should_skip_group(self, group_idx: int) -> bool:
        if self._active_route_mask is None:
            return False
        if group_idx < 0 or group_idx >= self._active_route_mask.numel():
            return False
        if not bool(self.trainable_route_mask[group_idx].item()):
            return False
        return float(self._active_route_mask[group_idx].item()) < 0.5

    def _sample_route(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        greedy: bool = False,
    ) -> dict[str, torch.Tensor]:
        self.router.eval()
        with torch.no_grad():
            logits = self.router(input_ids.cuda(), input_lengths.cuda())
            route_mask = torch.ones_like(logits)
            logprob = torch.zeros(logits.shape[0], device=logits.device)
            entropy = torch.zeros(logits.shape[0], device=logits.device)

            trainable = self.trainable_route_mask.to(logits.device)
            if trainable.any():
                trainable_logits = logits[:, trainable]
                dist = torch.distributions.Bernoulli(logits=trainable_logits)
                if greedy:
                    sampled = (torch.sigmoid(trainable_logits) >= 0.5).to(logits.dtype)
                else:
                    sampled = dist.sample()
                route_mask[:, trainable] = sampled
                logprob = dist.log_prob(sampled).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)

            layer_counts = self.layer_counts.to(logits.device)
            compute_fraction = (route_mask * layer_counts).sum(dim=-1) / max(
                self.total_routable_layers, 1.0
            )
            keep_fraction = route_mask.mean(dim=-1)
            return {
                "route_masks": route_mask.detach().cpu(),
                "route_logprobs": logprob.detach().cpu(),
                "route_entropies": entropy.detach().cpu(),
                "route_compute_fractions": compute_fraction.detach().cpu(),
                "route_keep_fractions": keep_fraction.detach().cpu(),
            }

    def _sample_next_token(
        self,
        token_ids: torch.Tensor,
        greedy: bool,
    ) -> tuple[int, float]:
        input_ids = token_ids.unsqueeze(0)
        input_lengths = torch.tensor([token_ids.numel()], dtype=torch.long)
        data = BatchedDataDict(
            {
                "input_ids": input_ids.cpu(),
                "input_lengths": input_lengths.cpu(),
            }
        )
        sequence_packing_enabled = self.cfg["sequence_packing"]["enabled"]
        dynamic_batching_enabled = self.cfg["dynamic_batching"]["enabled"]
        self.cfg["sequence_packing"]["enabled"] = False
        self.cfg["dynamic_batching"]["enabled"] = False
        try:
            topk = self.get_topk_logits(
                data=data,
                k=self.generation_top_k,
                micro_batch_size=1,
            )
        finally:
            self.cfg["sequence_packing"]["enabled"] = sequence_packing_enabled
            self.cfg["dynamic_batching"]["enabled"] = dynamic_batching_enabled
        logits = topk["topk_logits"][0, token_ids.numel() - 1].to(torch.float32)
        indices = topk["topk_indices"][0, token_ids.numel() - 1].to(torch.long)

        if greedy:
            next_idx = int(torch.argmax(logits).item())
            logprob = torch.log_softmax(logits, dim=-1)[next_idx].item()
        else:
            probs = torch.softmax(logits, dim=-1)
            next_idx = int(torch.multinomial(probs, num_samples=1).item())
            logprob = torch.log(probs[next_idx].clamp_min(1.0e-12)).item()
        return int(indices[next_idx].item()), float(logprob)

    @torch.no_grad()
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        if data.size == 0:
            raise ValueError("Cannot generate an empty routed batch")

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_size = input_ids.shape[0]
        pad_id = self.tokenizer.pad_token_id
        eos_ids = set(self.cfg["generation"].get("stop_token_ids") or [])
        max_total_len = int(self.cfg["max_total_sequence_length"])
        max_new_tokens_cfg = int(self.cfg["generation"]["max_new_tokens"])

        output_rows: list[torch.Tensor] = []
        logprob_rows: list[torch.Tensor] = []
        generation_lengths: list[int] = []
        unpadded_lengths: list[int] = []
        route_outputs: dict[str, list[torch.Tensor]] = {
            "route_masks": [],
            "route_logprobs": [],
            "route_entropies": [],
            "route_compute_fractions": [],
            "route_keep_fractions": [],
        }

        self.model.eval()
        for row_idx in range(batch_size):
            prompt_len = int(input_lengths[row_idx].item())
            tokens = input_ids[row_idx, :prompt_len].clone().cuda()
            route = self._sample_route(
                tokens.unsqueeze(0).cpu(),
                torch.tensor([prompt_len], dtype=torch.long),
                greedy=greedy,
            )
            for key, value in route.items():
                route_outputs[key].append(value[0])

            self._active_route_mask = route["route_masks"][0].cuda()
            token_logprobs = [0.0] * prompt_len
            budget = max(0, min(max_new_tokens_cfg, max_total_len - prompt_len))
            try:
                for _ in range(budget):
                    next_token, logprob = self._sample_next_token(tokens, greedy)
                    next_tensor = torch.tensor([next_token], device=tokens.device)
                    tokens = torch.cat([tokens, next_tensor], dim=0)
                    token_logprobs.append(logprob)
                    if next_token in eos_ids:
                        break
            finally:
                self._active_route_mask = None

            output_rows.append(tokens.cpu())
            logprob_rows.append(torch.tensor(token_logprobs, dtype=torch.float32))
            unpadded_lengths.append(int(tokens.numel()))
            generation_lengths.append(int(tokens.numel() - prompt_len))

        max_len = max(row.numel() for row in output_rows)
        output_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        logprobs = torch.zeros((batch_size, max_len), dtype=torch.float32)
        for idx, (tokens, row_logprobs) in enumerate(zip(output_rows, logprob_rows)):
            output_ids[idx, : tokens.numel()] = tokens
            logprobs[idx, : row_logprobs.numel()] = row_logprobs

        out_dict: dict[str, torch.Tensor] = {
            "output_ids": output_ids,
            "logprobs": logprobs,
            "generation_lengths": torch.tensor(generation_lengths, dtype=torch.long),
            "unpadded_sequence_lengths": torch.tensor(
                unpadded_lengths, dtype=torch.long
            ),
        }
        for key, values in route_outputs.items():
            out_dict[key] = torch.stack(values)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    def train_router(self, *, data: BatchedDataDict[Any]) -> dict[str, Any]:
        self.router.train()
        data.to("cuda")

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        route_masks = data["route_masks"].to(torch.float32)
        old_logprobs = data["old_route_logprobs"].to(torch.float32)
        advantages = data["advantages"].to(torch.float32)
        sample_mask = data.get("sample_mask", torch.ones_like(advantages)).to(
            torch.float32
        )
        sample_mask = sample_mask.to(advantages.device)

        logits = self.router(input_ids, input_lengths)
        trainable = self.trainable_route_mask.to(logits.device)
        if trainable.any():
            dist = torch.distributions.Bernoulli(logits=logits[:, trainable])
            selected_masks = route_masks[:, trainable]
            new_logprobs = dist.log_prob(selected_masks).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            new_logprobs = torch.zeros_like(old_logprobs)
            entropy = torch.zeros_like(old_logprobs)

        ratios = torch.exp(new_logprobs - old_logprobs)
        clipped_ratios = ratios.clamp(
            1.0 - self.route_ratio_clip_min, 1.0 + self.route_ratio_clip_max
        )
        objective = torch.minimum(ratios * advantages, clipped_ratios * advantages)
        denom = sample_mask.sum().clamp_min(1.0)
        policy_loss = -(objective * sample_mask).sum() / denom
        entropy_bonus = (entropy * sample_mask).sum() / denom
        loss = policy_loss - self.route_entropy_coef * entropy_bonus

        self.router_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
        self.router_optimizer.step()

        with torch.no_grad():
            keep_probs = torch.sigmoid(logits)
            layer_counts = self.layer_counts.to(logits.device)
            expected_compute = (keep_probs * layer_counts).sum(dim=-1) / max(
                self.total_routable_layers, 1.0
            )

        return {
            "router_loss": loss.detach().cpu(),
            "router_policy_loss": policy_loss.detach().cpu(),
            "router_entropy": entropy_bonus.detach().cpu(),
            "router_grad_norm": torch.as_tensor(float(grad_norm)).cpu(),
            "router_ratio_mean": ratios.detach().mean().cpu(),
            "router_expected_compute_fraction": expected_compute.detach().mean().cpu(),
        }

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        super().save_checkpoint(weights_path, optimizer_path, **kwargs)
        if torch.distributed.get_rank() == 0:
            os.makedirs(weights_path, exist_ok=True)
            torch.save(self.router.state_dict(), os.path.join(weights_path, "router.pt"))
            if optimizer_path is not None:
                os.makedirs(optimizer_path, exist_ok=True)
                torch.save(
                    self.router_optimizer.state_dict(),
                    os.path.join(optimizer_path, "router_optimizer.pt"),
                )


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("routed_megatron_policy_worker")
)  # pragma: no cover
class RoutedMegatronPolicyWorker(RoutedMegatronPolicyWorkerImpl):
    pass
