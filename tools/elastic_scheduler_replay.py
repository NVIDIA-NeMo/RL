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

r"""Project elastic cells against a fixed split and colocate from recorded rollouts.

Reads the per-group rollout durations a SingleController run records as
``rl.sc.generate_and_push`` spans (telemetry ``span_groups: per_step,per_prompt``),
either as OTLP/JSON (an OpenTelemetry Collector ``file`` exporter) or as the output of
``telemetry.exporter: console``. It replays those durations through three ways of
using the same ``K`` cells and reports projected step time, idle cell-seconds and
switching overhead for each:

* ``elastic``: :class:`ElasticScheduler` decides when each cell moves to training and
  back, with one optimizer update per weight publication (``mu = 1``).
* ``fixed_split``: ``--fixed-split-train-cells`` cells always train, the rest always
  serve rollouts, with ``--max-lookahead-steps`` of off-policy lookahead. Without the
  flag every split is tried and the fastest is reported, so elastic is compared with
  the best-tuned fixed split rather than an arbitrary one.
* ``colocate``: every cell serves the full batch, then every cell trains.

This is a go/no-go estimate, not a prediction. It assumes a group's duration does not
depend on where it runs or how loaded that engine is, and it holds a group's ``N``
slots until the whole group finishes, which makes cells move to training later than
per-execution accounting would. See ``docs/design-docs/elastic-scheduler.md``.

Example::

    python tools/elastic_scheduler_replay.py --trace spans.jsonl \
        --num-cells 8 --cell-capacity 144 --group-size 16 --batch-groups 64 \
        --train-step-s 3600 --num-steps 12
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nemo_rl.algorithms.single_controller_utils.elastic_scheduler import (
    CellRole,
    ElasticScheduler,
    ElasticSchedulerPolicy,
)

GROUP_SPAN_NAME = "rl.sc.generate_and_push"
STRATEGIES = ("elastic", "fixed_split", "colocate")


# ── trace loading ────────────────────────────────────────────────────────


def _iter_json_values(text: str) -> Iterator[Any]:
    """Yield every top-level JSON object in ``text``, skipping anything else.

    Handles JSON Lines, concatenated pretty-printed objects (the console exporter) and
    objects interleaved with ordinary log lines.
    """
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        yield value
        idx = text.find("{", end)


def _otlp_spans(payload: dict[str, Any]) -> Iterator[tuple[str, float, float]]:
    for resource_spans in payload.get("resourceSpans", []):
        scopes = resource_spans.get("scopeSpans") or resource_spans.get(
            "instrumentationLibrarySpans", []
        )
        for scope in scopes:
            for span in scope.get("spans", []):
                yield (
                    span.get("name", ""),
                    int(span["startTimeUnixNano"]) / 1e9,
                    int(span["endTimeUnixNano"]) / 1e9,
                )


def _console_time(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def load_group_durations(
    path: str | Path, *, span_name: str = GROUP_SPAN_NAME
) -> list[float]:
    """Read per-group rollout durations, in dispatch order, from a span export.

    Every dispatch attempt counts, substitutions included: each one held its slots.

    Raises:
        ValueError: If the file holds no ``span_name`` spans.
    """
    spans: list[tuple[float, float]] = []
    for value in _iter_json_values(Path(path).read_text()):
        if not isinstance(value, dict):
            continue
        if "resourceSpans" in value:
            for name, start, end in _otlp_spans(value):
                if name == span_name:
                    spans.append((start, end))
        elif value.get("name") == span_name and "start_time" in value:
            spans.append(
                (_console_time(value["start_time"]), _console_time(value["end_time"]))
            )
    if not spans:
        raise ValueError(
            f"no {span_name!r} spans in {path}; record them with "
            "telemetry.span_groups including per_prompt"
        )
    spans.sort()
    return [max(0.0, end - start) for start, end in spans]


# ── replay ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReplayConfig:
    """The cluster and step shape every strategy is replayed on.

    Attributes:
        num_cells: ``K``, cells in the cluster. Every strategy uses all of them.
        cell_capacity: ``C_e``, concurrent executions one cell serves in rollout.
        group_size: ``N``, executions per prompt group.
        batch_groups: ``B``, groups per training step.
        train_step_s: Seconds one cell needs to train one full step. ``m`` cells
            training together take ``train_step_s / m``.
        num_steps: Training steps to replay.
        to_training_s: Seconds a cell needs to move from rollout to training.
        to_rollout_s: Seconds a cell needs to move back, receiving the new weights.
        refit_s: Fixed-split weight sync per step, during which the trainers wait.
        min_supply_fraction: Elastic ``rho_min``.
        supply_hold_s: Elastic ``tau``.
        initial_dispatch_steps: Elastic ``phi``, extra steps of groups dispatched up
            front beyond the first step's ``B``.
        fixed_split_train_cells: Cells the fixed split dedicates to training; None
            tries every split and keeps the fastest.
        max_lookahead_steps: Steps of rollouts the fixed split runs ahead of training.
    """

    num_cells: int
    cell_capacity: int
    group_size: int
    batch_groups: int
    train_step_s: float
    num_steps: int
    to_training_s: float = 8.52
    to_rollout_s: float = 3.46
    refit_s: float = 0.0
    min_supply_fraction: float = 0.5
    supply_hold_s: float = 5.0
    initial_dispatch_steps: float = 0.5
    fixed_split_train_cells: int | None = None
    max_lookahead_steps: int = 1

    def __post_init__(self) -> None:
        positive = {
            "num_cells": self.num_cells,
            "cell_capacity": self.cell_capacity,
            "group_size": self.group_size,
            "batch_groups": self.batch_groups,
            "num_steps": self.num_steps,
        }
        if self.fixed_split_train_cells is not None:
            positive["fixed_split_train_cells"] = self.fixed_split_train_cells
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.cell_capacity < self.group_size:
            raise ValueError(
                f"cell_capacity ({self.cell_capacity}) must hold at least one group "
                f"of {self.group_size}"
            )
        non_negative = {
            "train_step_s": self.train_step_s,
            "to_training_s": self.to_training_s,
            "to_rollout_s": self.to_rollout_s,
            "refit_s": self.refit_s,
            "initial_dispatch_steps": self.initial_dispatch_steps,
            "max_lookahead_steps": self.max_lookahead_steps,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class ReplayResult:
    """Projected cost of one strategy.

    Idle and transition time are in cell-seconds. A rollout cell counts as idle in
    proportion to its unused concurrency, so a cell with half its slots empty is half
    idle; a training cell is idle while it has no micro-step to run.
    """

    strategy: str
    num_cells: int
    num_steps: int
    total_s: float
    step_end_s: tuple[float, ...]
    idle_cell_s: float
    transition_cell_s: float
    # Cells dedicated to training, for the fixed split only.
    train_cells: int | None = None

    @property
    def mean_step_s(self) -> float:
        return self.total_s / self.num_steps

    @property
    def idle_fraction(self) -> float:
        return self.idle_cell_s / (self.num_cells * self.total_s) if self.total_s else 0

    @property
    def transition_fraction(self) -> float:
        if not self.total_s:
            return 0.0
        return self.transition_cell_s / (self.num_cells * self.total_s)


class _Replay:
    """Event loop and cell-second ledger shared by the three strategies.

    A subclass sets up its state, schedules its first events in ``_start``, reacts to
    each event in ``_handle``, and reports in ``_cells`` how many cells are idle and how
    many are switching roles in its current state.
    """

    strategy = ""

    def __init__(self, durations: Sequence[float], config: ReplayConfig) -> None:
        if not durations:
            raise ValueError("need at least one group duration to replay")
        self.config = config
        self._durations = durations
        self._next_duration = 0
        self._events: list[tuple[float, int, str, int]] = []
        self._seq = 0
        self.now = 0.0
        self.idle_cell_s = 0.0
        self.transition_cell_s = 0.0
        self.step_end_s: list[float] = []

    def next_duration(self) -> float:
        """Durations in recorded order, cycling once the trace runs out."""
        duration = self._durations[self._next_duration % len(self._durations)]
        self._next_duration += 1
        return duration

    def push(self, delay: float, kind: str, payload: int = -1) -> None:
        heapq.heappush(self._events, (self.now + delay, self._seq, kind, payload))
        self._seq += 1

    def run(self) -> ReplayResult:
        """Pop events until ``num_steps`` updates, charging idle and transition time."""
        num_steps = self.config.num_steps
        self._start()
        while len(self.step_end_s) < num_steps:
            if not self._events:
                raise RuntimeError(
                    f"{self.strategy} replay stalled after {len(self.step_end_s)} of "
                    f"{num_steps} steps"
                )
            at, _, kind, payload = heapq.heappop(self._events)
            idle, transitioning = self._cells()
            self.idle_cell_s += (at - self.now) * idle
            self.transition_cell_s += (at - self.now) * transitioning
            self.now = at
            self._handle(kind, payload)
        return ReplayResult(
            strategy=self.strategy,
            num_cells=self.config.num_cells,
            num_steps=num_steps,
            total_s=self.now,
            step_end_s=tuple(self.step_end_s),
            idle_cell_s=self.idle_cell_s,
            transition_cell_s=self.transition_cell_s,
            train_cells=self._train_cells(),
        )

    def _train_cells(self) -> int | None:
        return None

    def _start(self) -> None:
        raise NotImplementedError

    def _handle(self, kind: str, payload: int) -> None:
        raise NotImplementedError

    def _cells(self) -> tuple[float, float]:
        raise NotImplementedError


class _ElasticReplay(_Replay):
    strategy = "elastic"

    def __init__(self, durations: Sequence[float], config: ReplayConfig) -> None:
        super().__init__(durations, config)
        self.scheduler = ElasticScheduler(
            policy=ElasticSchedulerPolicy(
                num_cells=config.num_cells,
                cell_capacity=config.cell_capacity,
                standalone_capacity=0,
                batch_groups=config.batch_groups,
                min_supply_fraction=config.min_supply_fraction,
                supply_hold_s=config.supply_hold_s,
                group_size=config.group_size,
                steps_per_burst=1,
            ),
            clock=lambda: self.now,
        )
        # Groups finished and not yet claimed by a training cell.
        self.ready = 0
        # Groups of the open batch claimed by, and trained on, a training cell.
        self.claimed = 0
        self.trained = 0
        self.idle_trainers: set[int] = set()
        self.returning: set[int] = set()
        self.burst_ending = False

    def _start(self) -> None:
        batch = self.config.batch_groups
        self.scheduler.record_dispatch_groups(
            round((1 + self.config.initial_dispatch_steps) * batch)
        )
        self.scheduler.open_batch()
        self._settle()

    def _open_next_batch(self) -> None:
        self.scheduler.record_dispatch_groups(self.config.batch_groups)
        self.scheduler.open_batch()

    def _end_burst(self) -> None:
        if CellRole.TO_TRAINING in self.scheduler.roles:
            return  # A late joiner finishes its transition first.
        self.burst_ending = False
        plan = self.scheduler.plan_burst_end(ready_groups=self.ready)
        for cell in plan.return_to_rollout:
            self.scheduler.begin_rollout_transition(cell)
            self.idle_trainers.discard(cell)
            self.returning.add(cell)
            self.push(self.config.to_rollout_s, "returned", cell)
        if not self.returning:
            self._open_next_batch()

    def _settle(self) -> None:
        scheduler, batch = self.scheduler, self.config.batch_groups
        launch = scheduler.launchable_groups()
        for _ in range(launch):
            self.push(self.next_duration(), "finished")
        scheduler.record_launch_groups(launch)

        cell = scheduler.next_training_cell(
            ready_groups=self.ready, groups_needed=batch - self.claimed
        )
        if cell is not None:
            scheduler.begin_training_transition(cell)
            self.push(self.config.to_training_s, "joined", cell)
        deadline = scheduler.supply_hold_deadline()
        if deadline is not None and deadline > self.now:
            self.push(deadline - self.now, "wake")

        while (
            scheduler.batch_open
            and self.idle_trainers
            and self.ready
            and self.claimed < batch
        ):
            trainer = min(self.idle_trainers)
            self.idle_trainers.discard(trainer)
            self.ready -= 1
            self.claimed += 1
            self.push(self.config.train_step_s / batch, "trained", trainer)

    def _handle(self, kind: str, payload: int) -> None:
        scheduler = self.scheduler
        if kind == "finished":
            scheduler.record_finish_groups(1)
            self.ready += 1
        elif kind == "joined":
            scheduler.finish_training_transition(payload)
            self.idle_trainers.add(payload)
            if self.burst_ending:
                self._end_burst()
        elif kind == "trained":
            self.idle_trainers.add(payload)
            self.trained += 1
            if self.trained == self.config.batch_groups:
                scheduler.close_batch()
                self.step_end_s.append(self.now)
                self.claimed = 0
                self.trained = 0
                self.burst_ending = True
                self._end_burst()
        elif kind == "returned":
            scheduler.finish_rollout_transition(payload)
            self.returning.discard(payload)
            if not self.returning:
                # Every engine now serves the published weights.
                self._open_next_batch()
        self._settle()

    def _cells(self) -> tuple[float, float]:
        scheduler = self.scheduler
        unused = scheduler.rollout_capacity - scheduler.running
        transitioning = sum(
            role in (CellRole.TO_TRAINING, CellRole.TO_ROLLOUT)
            for role in scheduler.roles
        )
        idle = unused / self.config.cell_capacity + len(self.idle_trainers)
        return idle, float(transitioning)


class _FixedSplitReplay(_Replay):
    strategy = "fixed_split"

    def __init__(
        self, durations: Sequence[float], config: ReplayConfig, train_cells: int
    ) -> None:
        super().__init__(durations, config)
        self.train_cells = train_cells
        self.rollout_cells = config.num_cells - train_cells
        if self.rollout_cells < 1:
            raise ValueError(
                f"fixed_split_train_cells ({train_cells}) must leave at least one of "
                f"{config.num_cells} cells for rollouts"
            )
        self.capacity_groups = (
            self.rollout_cells * config.cell_capacity // config.group_size
        )
        self.dispatched = 0
        self.queued = 0
        self.running = 0
        self.ready = 0
        self.published = 0
        self.trainer = "idle"

    def _train_cells(self) -> int | None:
        return self.train_cells

    def _start(self) -> None:
        self._settle()

    def _settle(self) -> None:
        config = self.config
        limit = (self.published + 1 + config.max_lookahead_steps) * config.batch_groups
        if limit > self.dispatched:
            self.queued += limit - self.dispatched
            self.dispatched = limit
        while self.queued and self.running < self.capacity_groups:
            self.queued -= 1
            self.running += 1
            self.push(self.next_duration(), "finished")
        if self.trainer == "idle" and self.ready >= config.batch_groups:
            self.ready -= config.batch_groups
            self.trainer = "training"
            self.push(config.train_step_s / self.train_cells, "trained")

    def _handle(self, kind: str, payload: int) -> None:
        if kind == "finished":
            self.running -= 1
            self.ready += 1
        elif kind == "trained":
            self.step_end_s.append(self.now)
            self.trainer = "refit"
            self.push(self.config.refit_s, "synced")
        elif kind == "synced":
            self.trainer = "idle"
            self.published += 1
        self._settle()

    def _cells(self) -> tuple[float, float]:
        config = self.config
        unused = (
            self.rollout_cells * config.cell_capacity - self.running * config.group_size
        )
        idle = unused / config.cell_capacity
        if self.trainer == "idle":
            idle += self.train_cells
        return idle, float(self.train_cells if self.trainer == "refit" else 0)


class _ColocateReplay(_Replay):
    strategy = "colocate"

    def __init__(self, durations: Sequence[float], config: ReplayConfig) -> None:
        super().__init__(durations, config)
        self.capacity_groups = (
            config.num_cells * config.cell_capacity // config.group_size
        )
        self.queued = 0
        self.running = 0
        self.finished = 0
        self.phase = "rollout"

    def _start(self) -> None:
        self.queued = self.config.batch_groups
        self.finished = 0
        self.phase = "rollout"
        self._launch()

    def _launch(self) -> None:
        while self.queued and self.running < self.capacity_groups:
            self.queued -= 1
            self.running += 1
            self.push(self.next_duration(), "finished")

    def _handle(self, kind: str, payload: int) -> None:
        config = self.config
        if kind == "finished":
            self.running -= 1
            self.finished += 1
            self._launch()
            if self.finished == config.batch_groups:
                self.phase = "to_training"
                self.push(config.to_training_s, "switched")
        elif kind == "switched":
            self.phase = "training"
            self.push(config.train_step_s / config.num_cells, "trained")
        elif kind == "trained":
            self.step_end_s.append(self.now)
            self.phase = "to_rollout"
            self.push(config.to_rollout_s, "returned")
        elif kind == "returned":
            self._start()

    def _cells(self) -> tuple[float, float]:
        config = self.config
        if self.phase == "rollout":
            used = self.running * config.group_size / config.cell_capacity
            return config.num_cells - used, 0.0
        if self.phase == "training":
            return 0.0, 0.0
        return 0.0, float(config.num_cells)


def replay_elastic(durations: Sequence[float], config: ReplayConfig) -> ReplayResult:
    """Replay with every role decision taken by :class:`ElasticScheduler`."""
    return _ElasticReplay(durations, config).run()


def replay_fixed_split(
    durations: Sequence[float], config: ReplayConfig
) -> ReplayResult:
    """Replay with a fixed set of training cells and the rest serving rollouts.

    With ``config.fixed_split_train_cells`` unset, every split from one training cell
    to ``K - 1`` is replayed and the fastest is returned.
    """
    if config.fixed_split_train_cells is not None:
        splits: Sequence[int] = (config.fixed_split_train_cells,)
    elif config.num_cells < 2:
        raise ValueError("a fixed split needs at least two cells")
    else:
        splits = range(1, config.num_cells)
    return min(
        (_FixedSplitReplay(durations, config, cells).run() for cells in splits),
        key=lambda result: result.total_s,
    )


def replay_colocate(durations: Sequence[float], config: ReplayConfig) -> ReplayResult:
    """Replay with every cell serving the full batch and then every cell training."""
    return _ColocateReplay(durations, config).run()


def replay_all(
    durations: Sequence[float], config: ReplayConfig
) -> dict[str, ReplayResult]:
    """Replay the same durations under every strategy."""
    return {
        "elastic": replay_elastic(durations, config),
        "fixed_split": replay_fixed_split(durations, config),
        "colocate": replay_colocate(durations, config),
    }


# ── command line ─────────────────────────────────────────────────────────


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)]


def format_report(durations: Sequence[float], results: dict[str, ReplayResult]) -> str:
    lines = [
        f"groups in trace: {len(durations)}  "
        f"p50 {_percentile(durations, 0.5):.1f}s  "
        f"p95 {_percentile(durations, 0.95):.1f}s  "
        f"max {max(durations):.1f}s",
        "",
        f"{'strategy':<15} {'total_h':>9} {'step_s':>9} {'idle':>7} "
        f"{'switch':>7} {'elastic_speedup':>16}",
    ]
    elastic_total = results["elastic"].total_s
    for name, result in results.items():
        speedup = result.total_s / elastic_total if elastic_total else float("nan")
        label = name if result.train_cells is None else f"{name}({result.train_cells})"
        lines.append(
            f"{label:<15} {result.total_s / 3600:>9.2f} {result.mean_step_s:>9.1f} "
            f"{result.idle_fraction:>7.1%} {result.transition_fraction:>7.1%} "
            f"{speedup:>15.2f}x"
        )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trace", required=True, help="OTLP/JSON or console spans")
    parser.add_argument("--span-name", default=GROUP_SPAN_NAME)
    parser.add_argument("--num-cells", type=int, required=True)
    parser.add_argument("--cell-capacity", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--batch-groups", type=int, required=True)
    parser.add_argument(
        "--train-step-s",
        type=float,
        required=True,
        help="seconds ONE cell needs to train one step",
    )
    parser.add_argument("--num-steps", type=int, required=True)
    defaults = ReplayConfig(
        num_cells=1,
        cell_capacity=1,
        group_size=1,
        batch_groups=1,
        train_step_s=0,
        num_steps=1,
    )
    for name in (
        "to_training_s",
        "to_rollout_s",
        "refit_s",
        "min_supply_fraction",
        "supply_hold_s",
        "initial_dispatch_steps",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name)
        )
    parser.add_argument(
        "--fixed-split-train-cells",
        type=int,
        default=None,
        help="training cells of the fixed split; default tries every split",
    )
    parser.add_argument(
        "--max-lookahead-steps", type=int, default=defaults.max_lookahead_steps
    )
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    durations = load_group_durations(args.trace, span_name=args.span_name)
    config = ReplayConfig(
        num_cells=args.num_cells,
        cell_capacity=args.cell_capacity,
        group_size=args.group_size,
        batch_groups=args.batch_groups,
        train_step_s=args.train_step_s,
        num_steps=args.num_steps,
        to_training_s=args.to_training_s,
        to_rollout_s=args.to_rollout_s,
        refit_s=args.refit_s,
        min_supply_fraction=args.min_supply_fraction,
        supply_hold_s=args.supply_hold_s,
        initial_dispatch_steps=args.initial_dispatch_steps,
        fixed_split_train_cells=args.fixed_split_train_cells,
        max_lookahead_steps=args.max_lookahead_steps,
    )
    results = replay_all(durations, config)
    if args.json:
        print(
            json.dumps(
                {
                    name: {
                        **asdict(result),
                        "mean_step_s": result.mean_step_s,
                        "idle_fraction": result.idle_fraction,
                        "transition_fraction": result.transition_fraction,
                    }
                    for name, result in results.items()
                },
                indent=2,
            )
        )
    else:
        print(format_report(durations, results))


if __name__ == "__main__":
    main()
