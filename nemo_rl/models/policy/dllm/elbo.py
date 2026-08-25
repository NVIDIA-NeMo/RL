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
r"""Semi-deterministic Monte Carlo (SDMC) ELBO estimation for masked diffusion LMs.

Masked diffusion LMs admit no exact sequence likelihood, so GDPO
(https://arxiv.org/abs/2510.08554) substitutes the ELBO

.. math::
    \mathcal{L}(y|q) = \int_0^1 \mathbb{E}_{y_t}\left[\frac{1}{t}
        \sum_i \mathbf{1}[y_t^i = M] \log \pi_\theta(y^i | y_t, q)\right] dt

The paper's contribution is *how* this integral is approximated: sampling ``t``
at random dominates the estimator variance, so the time integral is instead
evaluated with a deterministic quadrature rule and only the inner masking
expectation is left to Monte Carlo -- hence "semi-deterministic". Two or three
quadrature points match a 100+ sample double-Monte-Carlo estimator.

This module is deliberately free of any model or parallelism concern. It emits
the masked inputs to score (:meth:`SdmcElboEstimator.mask_points`) and folds the
resulting log probabilities back into a per-position tensor
(:meth:`SdmcElboEstimator.accumulate`); the caller owns the forward passes. The
per-position form is what lets the rest of NeMo RL stay unchanged -- it occupies
the same ``[batch, seq_len]`` slot as autoregressive token log probabilities, and
its masked sum is the scalar ELBO the GDPO objective needs.
"""

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import torch

from nemo_rl.models.policy.dllm.config import DllmConfig

# Gauss-Legendre nodes and weights on [-1, 1]. Mapped onto the unit interval by
# t = (x + 1) / 2, w = w_x / 2 in get_quadrature().
_GAUSS_LEGENDRE: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "gauss-1": ((0.0,), (2.0,)),
    "gauss-2": ((-(1 / 3**0.5), 1 / 3**0.5), (1.0, 1.0)),
    "gauss-3": ((-((3 / 5) ** 0.5), 0.0, (3 / 5) ** 0.5), (5 / 9, 8 / 9, 5 / 9)),
    "gauss-4": (
        (
            -((3 / 7 + 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
            -((3 / 7 - 2 / 7 * (6 / 5) ** 0.5) ** 0.5),
            (3 / 7 - 2 / 7 * (6 / 5) ** 0.5) ** 0.5,
            (3 / 7 + 2 / 7 * (6 / 5) ** 0.5) ** 0.5,
        ),
        (
            (18 - 30**0.5) / 36,
            (18 + 30**0.5) / 36,
            (18 + 30**0.5) / 36,
            (18 - 30**0.5) / 36,
        ),
    ),
    "gauss-5": (
        (
            -(1 / 3) * (5 + 2 * (10 / 7) ** 0.5) ** 0.5,
            -(1 / 3) * (5 - 2 * (10 / 7) ** 0.5) ** 0.5,
            0.0,
            (1 / 3) * (5 - 2 * (10 / 7) ** 0.5) ** 0.5,
            (1 / 3) * (5 + 2 * (10 / 7) ** 0.5) ** 0.5,
        ),
        (
            (322 - 13 * 70**0.5) / 900,
            (322 + 13 * 70**0.5) / 900,
            128 / 225,
            (322 + 13 * 70**0.5) / 900,
            (322 - 13 * 70**0.5) / 900,
        ),
    ),
}

# Composite Simpson on [0, 1] with the endpoint pulled off zero, since the 1/t
# weight is singular at t = 0. Matches the reference implementation's choice.
_SIMPSON: tuple[tuple[float, ...], tuple[float, ...]] = (
    (0.1, 0.5, 1.0),
    (1 / 6, 4 / 6, 1 / 6),
)


def make_dllm_mask_seeds(input_ids: torch.Tensor) -> torch.Tensor:
    """Build a stable, distinct mask seed for each row of a token batch.

    The returned tensor can travel with a :class:`BatchedDataDict`, so slicing
    into different microbatch sizes preserves each sample's RNG stream.

    Args:
        input_ids: Token ids with shape ``[batch, sequence]``.

    Returns:
        An int64 seed tensor with shape ``[batch]`` on the input device.
    """
    token_positions = torch.arange(
        1, input_ids.shape[1] + 1, dtype=torch.int64, device=input_ids.device
    )
    content_hash = (input_ids.to(torch.int64) * token_positions).sum(dim=-1)
    row_offsets = torch.arange(
        input_ids.shape[0], dtype=torch.int64, device=input_ids.device
    )
    return torch.remainder(content_hash + row_offsets * 1_000_003, 2**31 - 1)


def get_quadrature(rule: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return ``(times, weights)`` on the unit interval for an integration rule.

    Args:
        rule: One of ``gauss-1`` .. ``gauss-5``, ``simpson``, or ``mc``. For
            ``mc`` the caller draws times at random, so this returns empty tuples.

    Returns:
        A ``(times, weights)`` pair. The weights sum to 1 (the length of the
        integration domain), so the weighted sum directly approximates the
        integral.

    Raises:
        ValueError: If ``rule`` is not a recognized integration rule.
    """
    if rule == "mc":
        return ((), ())
    if rule == "simpson":
        return _SIMPSON
    if rule not in _GAUSS_LEGENDRE:
        raise ValueError(
            f"Unknown dllm quadrature rule {rule!r}. Expected one of "
            f"{sorted(_GAUSS_LEGENDRE)}, 'simpson', or 'mc'."
        )
    nodes, node_weights = _GAUSS_LEGENDRE[rule]
    # Change of variable from [-1, 1] to [0, 1].
    times = tuple(0.5 * x + 0.5 for x in nodes)
    weights = tuple(0.5 * w for w in node_weights)
    return times, weights


@dataclass
class MaskPoint:
    """One masked view of a batch, to be scored by a single model forward.

    Attributes:
        input_ids: The sequence with a ``t`` fraction of scorable positions
            replaced by the mask token, shape ``[batch, seq_len]``.
        masked: Bool tensor marking the positions that were masked and therefore
            contribute to the ELBO, shape ``[batch, seq_len]``.
        coefficient: The ``w_n / (t_n * draws)`` scale applied to this point's
            log probabilities when accumulating, where ``draws`` is the number of
            mask samples averaged at this time. A ``[batch, 1]`` tensor for the
            per-row times used by Monte Carlo quadrature, otherwise a float.
        time: The mask ratio ``t_n`` this point was drawn at. A ``[batch]`` tensor
            for Monte Carlo quadrature, otherwise a float.
    """

    input_ids: torch.Tensor
    masked: torch.Tensor
    coefficient: float | torch.Tensor
    time: float | torch.Tensor


class SdmcElboEstimator:
    """Estimates a masked diffusion LM's sequence ELBO by SDMC quadrature.

    The caller drives the loop::

        estimator = SdmcElboEstimator(cfg, resolve_mask_id(cfg, model.config))
        elbo_per_position = torch.zeros_like(input_ids, dtype=torch.float32)
        for point in estimator.mask_points(input_ids, completion_mask, seed=seed):
            logprobs = score(point.input_ids)  # one forward, position-aligned
            elbo_per_position += estimator.accumulate(point, logprobs)

    ``elbo_per_position.sum(-1)`` is then the scalar ELBO per sequence, and
    ``elbo_per_position`` itself drops into the per-token log probability slot
    that the rest of the training stack already understands.
    """

    def __init__(self, cfg: DllmConfig, mask_id: int):
        """Initializes the estimator.

        Args:
            cfg: The dLLM policy configuration supplying the integration rule
                and the number of Monte Carlo mask draws.
            mask_id: The resolved mask token id, from
                :func:`nemo_rl.models.policy.dllm.config.resolve_mask_id`. Passed
                separately rather than read off ``cfg`` because ``cfg.mask_id``
                is optional -- it may still need resolving against the model.
        """
        self.cfg = cfg
        self.mask_id = mask_id
        self.times, self.weights = get_quadrature(cfg.quadrature)

    @property
    def num_forwards(self) -> int:
        """Model forward passes needed per likelihood evaluation."""
        num_points = len(self.times) if self.times else 1
        return num_points * self.cfg.mc_samples

    def mask_points(
        self,
        input_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        *,
        seed: Optional[int | torch.Tensor] = None,
    ) -> Iterator[MaskPoint]:
        """Yields the masked views whose scores make up the ELBO.

        Args:
            input_ids: Clean token ids, shape ``[batch, seq_len]``.
            completion_mask: Bool/int tensor marking positions that are part of
                the completion and thus eligible for masking and scoring, shape
                ``[batch, seq_len]``. Prompt and padding positions must be 0.
            seed: If given, masks are drawn from a generator seeded with this
                value. A ``[batch]`` tensor supplies an independent seed per row,
                preserving each sample's masks across microbatching. The
                old-policy, reference-policy, and current-policy ELBOs of a given
                rollout **must** share seeds -- otherwise their difference is
                dominated by mask noise rather than by the policy update.

        Yields:
            One :class:`MaskPoint` per (quadrature point, Monte Carlo draw).
        """
        completion_mask = completion_mask.bool()
        generators: Optional[list[torch.Generator]] = None
        if seed is not None:
            seeds = (
                seed.reshape(-1).tolist() if isinstance(seed, torch.Tensor) else [seed]
            )
            if len(seeds) not in (1, input_ids.shape[0]):
                raise ValueError(
                    f"Expected one dLLM mask seed per batch row, got {len(seeds)} "
                    f"seeds for batch size {input_ids.shape[0]}."
                )
            generators = []
            for value in seeds:
                generator = torch.Generator(device=input_ids.device)
                generator.manual_seed(int(value))
                generators.append(generator)

        # Each entry is (time, weight, draws): `draws` mask samples are averaged
        # at that time. Quadrature spends its budget on mc_samples draws per
        # fixed node; plain Monte Carlo spends it on distinct random times, one
        # draw each. Both therefore issue exactly `num_forwards` forwards.
        if self.times:
            schedule = [
                (t, w, self.cfg.mc_samples) for t, w in zip(self.times, self.weights)
            ]
        else:
            num_draws = self.cfg.mc_samples
            sampled = self._rand(
                (input_ids.shape[0], num_draws), input_ids.device, generators
            )
            # Guard the 1/t singularity independently for each sequence.
            min_time = completion_mask.sum(-1).clamp_min(1).reciprocal().float()
            sampled = torch.maximum(sampled, min_time.unsqueeze(-1))
            schedule = [
                (sampled[:, draw], 1.0 / num_draws, 1) for draw in range(num_draws)
            ]

        for time, weight, draws in schedule:
            for _ in range(draws):
                masked = self._draw_mask(completion_mask, time, generators)
                noisy = torch.where(masked, self.mask_id, input_ids)
                if self.cfg.p_mask_prompt > 0.0:
                    noisy, masked = self._mask_prompt(
                        noisy, masked, completion_mask, input_ids, generators
                    )
                coefficient = weight / (time * draws)
                if isinstance(coefficient, torch.Tensor):
                    coefficient = coefficient.unsqueeze(-1)
                yield MaskPoint(
                    input_ids=noisy,
                    masked=masked,
                    coefficient=coefficient,
                    time=time,
                )

    def accumulate(self, point: MaskPoint, logprobs: torch.Tensor) -> torch.Tensor:
        """Folds one point's log probabilities into its ELBO contribution.

        Args:
            point: The mask point the log probabilities were computed for.
            logprobs: Position-aligned log probabilities of the *clean* tokens
                under the masked input, shape ``[batch, seq_len]``. Position
                ``i`` must score token ``i`` -- see ``shift_targets=False`` in
                :func:`nemo_rl.distributed.model_utils.from_parallel_logits_to_logprobs`.

        Returns:
            The per-position contribution to the ELBO, shape
            ``[batch, seq_len]``, zero at every position this point did not mask.
        """
        return point.coefficient * logprobs * point.masked.to(logprobs.dtype)

    def _draw_mask(
        self,
        completion_mask: torch.Tensor,
        time: float | torch.Tensor,
        generators: Optional[list[torch.Generator]],
    ) -> torch.Tensor:
        """Masks a ``time`` fraction of each sequence's scorable positions.

        A fixed count per sequence is masked rather than an independent
        Bernoulli(t) draw per token. This is a stratification that removes the
        binomial spread in the number of masked tokens, and matches the
        reference implementation the published results were produced with.

        Algebraically this is the same batched exact-k selection as Automodel's
        ``nemo_automodel.components.datasets.dllm.corruption._batched_gumbel_topk``
        (ranking by noise and taking the top k is invariant to whether the noise
        is uniform or Gumbel). That helper is deliberately *not* reused: it draws
        via ``torch.rand_like`` with no ``generator`` parameter, so sharing masks
        between the old, reference, and current ELBOs would mean seeding global
        RNG inside a Ray worker -- where it is shared with dropout and other
        sampling. Threading a generator keeps that determinism local.
        """
        batch, seq_len = completion_mask.shape
        scorable = completion_mask.sum(-1)
        # At least one masked position, so a point never contributes a
        # degenerate zero (which the 1/t weight would turn into a NaN).
        # The GDPO reference uses ``int(t * completion_length)`` (floor for
        # positive values) for its exact-k mask draw.
        num_masked = (scorable.float() * time).floor().long().clamp(min=1)
        num_masked = torch.minimum(num_masked, scorable)

        # Rank scorable positions in a random order, then take the lowest
        # `num_masked` ranks. Non-scorable positions are pushed past every
        # scorable one so they can never be selected.
        noise = self._rand((batch, seq_len), completion_mask.device, generators)
        noise = noise.masked_fill(~completion_mask, float("inf"))
        ranks = noise.argsort(dim=-1).argsort(dim=-1)
        return ranks < num_masked.unsqueeze(-1)

    def _mask_prompt(
        self,
        noisy: torch.Tensor,
        masked: torch.Tensor,
        completion_mask: torch.Tensor,
        input_ids: torch.Tensor,
        generators: Optional[list[torch.Generator]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Additionally masks prompt tokens with probability ``p_mask_prompt``.

        Prompt positions are corrupted as a regularizer but are never scored, so
        they are added to the model input without entering the returned mask.
        """
        is_prompt = ~completion_mask & (input_ids != self.mask_id)
        draw = self._rand(noisy.shape, noisy.device, generators)
        corrupt = is_prompt & (draw < self.cfg.p_mask_prompt)
        return torch.where(corrupt, self.mask_id, noisy), masked

    @staticmethod
    def _rand(
        shape: torch.Size | tuple[int, ...],
        device: torch.device,
        generators: Optional[list[torch.Generator]],
    ) -> torch.Tensor:
        """Draw random values, optionally from one independent RNG per row."""
        if generators is None or len(generators) == 1:
            generator = generators[0] if generators else None
            return torch.rand(shape, generator=generator, device=device)
        if len(generators) != shape[0]:
            raise ValueError(
                f"Expected {shape[0]} row generators, got {len(generators)}."
            )
        return torch.cat(
            [
                torch.rand((1, *shape[1:]), generator=generator, device=device)
                for generator in generators
            ],
            dim=0,
        )


def accumulate_elbo_logprobs(
    estimator: SdmcElboEstimator,
    *,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    seed: Optional[int | torch.Tensor],
    score_fn: Any,
) -> torch.Tensor:
    """Runs the SDMC forwards and folds them into one per-position tensor.

    This is the whole ELBO evaluation: every quadrature point is scored and its
    contribution accumulated in place, so the caller receives a tensor with the
    same ``[batch, seq_len]`` shape an autoregressive logprob pass would return.
    Callers therefore need no dLLM-specific handling downstream.

    Gradients flow through every forward, so calling this under ``torch.no_grad``
    yields the old/reference ELBO and calling it in the training step yields the
    differentiable current ELBO -- with the same ``seed``, against the same masks.

    Args:
        estimator: The configured SDMC estimator.
        input_ids: Clean token ids, shape ``[batch, seq_len]``.
        completion_mask: Positions eligible for masking and scoring, shape
            ``[batch, seq_len]``.
        seed: Mask seed. The old, reference, and current ELBOs of one rollout
            must share it, or their differences measure mask noise rather than
            the policy update.
        score_fn: Callable taking ``(masked_input_ids, clean_target_ids)`` and
            returning position-aligned log probabilities of the clean tokens,
            shape ``[batch, seq_len]``. Injected so this loop stays independent
            of the training backend.

    Returns:
        Per-position ELBO contributions, shape ``[batch, seq_len]``. Its sum over
        the sequence is the scalar ELBO.
    """
    elbo: Optional[torch.Tensor] = None
    for point in estimator.mask_points(input_ids, completion_mask, seed=seed):
        contribution = estimator.accumulate(point, score_fn(point.input_ids, input_ids))
        elbo = contribution if elbo is None else elbo + contribution
    assert elbo is not None, "the estimator yielded no quadrature points"
    return elbo
