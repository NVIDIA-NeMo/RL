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
"""Value-level verification of the distributed Megatron -> SGLang refit.

The production distributed refit is guarded only by a *shape* check
(``NRL_REFIT_EXPORT_SHAPE_CHECK``): every exported HF tensor is compared
against the checkpoint's safetensors metadata. Shapes cannot catch a transport
or layout bug that preserves shape while corrupting values, and in a training
run such a bug presents as "the model does not learn" -- the most expensive
possible way to discover it.

This module closes that gap by driving SGLang's own weight checker
(``POST /weights_checker``) around real refits and comparing the per-parameter
content hashes it returns:

1. ``checksum`` right after engine startup -- the from-disk reference. Only
   meaningful on a fresh run; see ``compare_against_disk``.
2. refit, then ``checksum`` -- must reproduce the from-disk state, proving the
   Megatron -> HF -> NCCL -> SGLang round trip is value-preserving.
3. ``reset_tensors`` (overwrite every weight with noise), then ``checksum``.
4. refit again with the same untrained weights, then ``checksum`` -- must
   reproduce step 2.

Step 3 is what makes step 4 meaningful: without it, a refit that transferred
nothing at all would pass. It also lets step 4 separate two very different
failures, because a parameter that still hashes to its post-reset value was
never written, whereas one that hashes to a third value was written wrongly.

Every comparison is *symmetric and total*. A parameter, rank or engine that
appears on one side and not the other is a failure, never a skip: this is a
verifier, so an unexplained absence must not read as agreement. Hash equality
is required exactly, with no tolerance. Every mapping between the HF
checkpoint and Megatron is a copy, reshape, concatenation, transpose or
permutation, the refit exports the bf16 model parameters rather than the fp32
optimizer masters, and the bf16 target precision applies no quantization, so
any difference at all is a defect rather than expected rounding.

Driver-side only: it calls facade methods on ``policy_generation`` and never
imports megatron / transformer_engine. See ``megatron_refit_sglang`` for why
that matters.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_ENV_FLAG = "NRL_REFIT_VALUE_CHECK"
_MARKER = "NRL_REFIT_VALUE_CHECK"

# Enough names to identify a pattern in the log without burying the summary.
_MAX_REPORTED_NAMES = 20


class RefitValueCheckError(RuntimeError):
    """Raised when the distributed refit did not reproduce the expected values."""


def value_check_enabled() -> bool:
    """Whether ``NRL_REFIT_VALUE_CHECK=1`` requested the check."""
    return os.environ.get(_ENV_FLAG, "") == "1"


@dataclass(frozen=True)
class _RankId:
    """Identity of one engine rank, stable across ``/weights_checker`` calls.

    The HTTP layer aggregates per-rank payloads in response *arrival* order, so
    a rank's position in the list is not stable and must never be used to match
    it. ``parallelism_info`` carries a global ``rank`` plus its ``size``, which
    is; anything less is refused rather than guessed at, because a positional
    fallback turns a swapped-rank bug into a clean pass.
    """

    engine: int
    rank: int
    size: int

    def __str__(self) -> str:
        return f"engine{self.engine}/rank{self.rank}of{self.size}"


@dataclass
class _Snapshot:
    """One ``checksum`` sweep across every engine."""

    # Positional engine index -> None for engines that did not answer. The mask
    # is part of the identity of a sweep: an engine that answers one sweep and
    # not the next must not be silently dropped from both.
    answered: tuple[bool, ...]
    hashes: dict[_RankId, dict[str, str]] = field(default_factory=dict)


@dataclass
class _Diff:
    """Symmetric comparison of two checksum sweeps."""

    ranks: int = 0
    params: int = 0
    equal: int = 0
    differing: list[str] = field(default_factory=list)
    only_in_reference: list[str] = field(default_factory=list)
    only_in_current: list[str] = field(default_factory=list)
    # Populated only when the caller asks for it. The equal set is the whole
    # model in a healthy comparison, so it is not worth materialising except
    # for the reset control, where a non-empty equal set is the failure.
    equal_names: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.differing or self.only_in_reference or self.only_in_current)

    def summary(self, phase: str, *, engines: int) -> str:
        return (
            f"{_MARKER} phase={phase} engines={engines} ranks={self.ranks} "
            f"params={self.params} equal={self.equal} "
            f"differing={len(self.differing)} "
            f"missing={len(self.only_in_reference)} "
            f"extra={len(self.only_in_current)}"
        )


def _rank_id(engine: int, payload: dict[str, Any]) -> _RankId:
    """Build a stable rank identity, or refuse."""
    info = payload.get("parallelism_info") or {}
    rank = info.get("rank")
    size = info.get("size")
    if not isinstance(rank, int) or not isinstance(size, int):
        raise RefitValueCheckError(
            f"SGLang engine {engine} returned a checksum payload without a "
            "global rank/size in parallelism_info. Per-rank results cannot be "
            "matched across calls by anything other than arrival order, which "
            "is not stable, so the check refuses to run against this build."
        )
    return _RankId(engine=engine, rank=rank, size=size)


def _collect(policy_generation: Any) -> _Snapshot:
    """Take one ``checksum`` sweep and validate it structurally."""
    responses = policy_generation.check_weights(action="checksum")
    if not responses:
        raise RefitValueCheckError(
            "No SGLang engine answered the checksum request; the refit value "
            "check has nothing to compare."
        )

    answered: list[bool] = []
    hashes: dict[_RankId, dict[str, str]] = {}
    for engine, response in enumerate(responses):
        # Engines spanning several nodes answer only from their node 0; the
        # other actors return None rather than a response body.
        if response is None:
            answered.append(False)
            continue
        answered.append(True)
        if not response.get("success", False):
            raise RefitValueCheckError(
                f"SGLang engine {engine} failed the checksum request: "
                f"{response.get('message')!r}"
            )
        ranks = response.get("ranks")
        if not ranks:
            raise RefitValueCheckError(
                f"SGLang engine {engine} returned no per-rank checksums. This "
                "build's /weights_checker does not carry a payload, so the "
                "refit value check cannot run against it."
            )
        for payload in ranks:
            rank_id = _rank_id(engine, payload)
            if rank_id in hashes:
                raise RefitValueCheckError(
                    f"SGLang reported {rank_id} twice; per-rank checksums "
                    "cannot be matched across calls."
                )
            checksums = dict(payload.get("checksums") or {})
            if not checksums:
                raise RefitValueCheckError(
                    f"{rank_id} reported an empty checksum set. An unverified "
                    "rank must not be mistaken for an agreeing one."
                )
            hashes[rank_id] = checksums

    if not any(answered):
        raise RefitValueCheckError(
            "Every SGLang engine returned an empty checksum response; the "
            "refit value check has nothing to compare."
        )
    return _Snapshot(answered=tuple(answered), hashes=hashes)


def _require_same_topology(reference: _Snapshot, current: _Snapshot) -> None:
    """Refuse to compare two sweeps that do not cover the same engines/ranks."""
    if reference.answered != current.answered:
        raise RefitValueCheckError(
            "The set of SGLang engines answering /weights_checker changed "
            f"between calls ({reference.answered} -> {current.answered}). An "
            "engine was lost, replaced, or started answering late; the "
            "comparison would no longer cover the same hardware."
        )
    missing = sorted(str(r) for r in reference.hashes.keys() - current.hashes.keys())
    unexpected = sorted(str(r) for r in current.hashes.keys() - reference.hashes.keys())
    if missing or unexpected:
        raise RefitValueCheckError(
            f"Engine rank set changed between checksum calls: missing={missing} "
            f"unexpected={unexpected}."
        )


def _compare(
    *, reference: _Snapshot, current: _Snapshot, collect_equal: bool = False
) -> _Diff:
    """Compare two sweeps parameter by parameter, in both directions."""
    _require_same_topology(reference, current)

    diff = _Diff()
    for rank_id, ref_hashes in reference.hashes.items():
        cur_hashes = current.hashes[rank_id]
        diff.ranks += 1
        for name, ref_hash in ref_hashes.items():
            diff.params += 1
            cur_hash = cur_hashes.get(name)
            if cur_hash is None:
                diff.only_in_reference.append(f"{rank_id}/{name}")
            elif cur_hash == ref_hash:
                diff.equal += 1
                if collect_equal:
                    diff.equal_names.append(f"{rank_id}/{name}")
            else:
                diff.differing.append(f"{rank_id}/{name}")
        for name in cur_hashes.keys() - ref_hashes.keys():
            diff.only_in_current.append(f"{rank_id}/{name}")
    return diff


def _split_unwritten(
    *,
    reference: _Snapshot,
    after_reset: _Snapshot,
    current: _Snapshot,
) -> tuple[list[str], list[str]]:
    """Split post-reset mismatches into never-written and wrongly-written.

    A parameter whose hash still equals its randomized value was not touched by
    the refit at all; one that holds a third value was written with the wrong
    bytes. The two failures have completely different causes, so they are never
    reported as one number. Parameters absent from either sweep are handled by
    ``_compare``, which fails on them outright.
    """
    unwritten: list[str] = []
    corrupted: list[str] = []
    for rank_id, ref_hashes in reference.hashes.items():
        reset_hashes = after_reset.hashes.get(rank_id, {})
        cur_hashes = current.hashes.get(rank_id, {})
        for name, ref_hash in ref_hashes.items():
            cur_hash = cur_hashes.get(name)
            if cur_hash is None or cur_hash == ref_hash:
                continue
            label = f"{rank_id}/{name}"
            if cur_hash == reset_hashes.get(name):
                unwritten.append(label)
            else:
                corrupted.append(label)
    return unwritten, corrupted


def _report_names(kind: str, names: list[str]) -> None:
    if not names:
        return
    shown = names[:_MAX_REPORTED_NAMES]
    suffix = "" if len(names) == len(shown) else f" (+{len(names) - len(shown)} more)"
    print(f"{_MARKER} {kind}: {', '.join(shown)}{suffix}", flush=True)


def _run_reset(policy_generation: Any) -> None:
    """Randomize every engine weight, and confirm every engine did it."""
    responses = policy_generation.check_weights(action="reset_tensors")
    for engine, response in enumerate(responses or []):
        if response is None:
            continue
        if not response.get("success", False):
            raise RefitValueCheckError(
                f"SGLang engine {engine} failed reset_tensors: "
                f"{response.get('message')!r}. Without a confirmed reset on "
                "every engine, a later match would prove nothing."
            )


def _verify(
    policy_generation: Any,
    *,
    refit: Callable[[], Any],
    compare_against_disk: bool,
) -> dict[str, Any]:
    """Body of the check; see ``verify_distributed_refit_values``."""
    from_disk = _collect(policy_generation)
    engines = sum(from_disk.answered)

    refit()
    after_first = _collect(policy_generation)

    disk_differing = None
    if compare_against_disk:
        disk_vs_refit = _compare(reference=from_disk, current=after_first)
        print(disk_vs_refit.summary("from_disk_vs_refit", engines=engines), flush=True)
        _report_names("from_disk_vs_refit differing", disk_vs_refit.differing)
        _report_names("from_disk_vs_refit missing", disk_vs_refit.only_in_reference)
        _report_names("from_disk_vs_refit extra", disk_vs_refit.only_in_current)
        disk_differing = len(disk_vs_refit.differing)
        if not disk_vs_refit.ok:
            raise RefitValueCheckError(
                f"After a refit, {len(disk_vs_refit.differing)} parameters "
                "differ from the state SGLang loaded from disk "
                f"({len(disk_vs_refit.only_in_reference)} missing, "
                f"{len(disk_vs_refit.only_in_current)} unexpected). The "
                "Megatron -> HF -> SGLang round trip is not value-preserving."
            )
    else:
        print(
            f"{_MARKER} phase=from_disk_vs_refit skipped=1 reason=resumed_run",
            flush=True,
        )

    _run_reset(policy_generation)
    after_reset = _collect(policy_generation)
    reset_control = _compare(
        reference=after_first, current=after_reset, collect_equal=True
    )
    print(reset_control.summary("reset_control", engines=engines), flush=True)

    # Demand a total reset, not a mostly-complete one. SGLang excludes
    # non-persistent buffers from the checksum as well as from reset_tensors,
    # so the two sets coincide, and a parameter the control never randomized
    # proves nothing about the refit when it matches later.
    _report_names("reset_control unchanged", reset_control.equal_names)
    randomized = len(reset_control.differing)
    # A healthy control is the inverse of a healthy comparison: EVERY parameter
    # must have changed, and none may have appeared or vanished.
    if (
        randomized != reset_control.params
        or reset_control.only_in_reference
        or reset_control.only_in_current
    ):
        raise RefitValueCheckError(
            f"reset_tensors randomized only {randomized}/{reset_control.params} "
            "parameters. The control did not cover the whole model, so a "
            "subsequent match would not prove the refit wrote them."
        )

    refit()
    after_second = _collect(policy_generation)
    refit_vs_refit = _compare(reference=after_first, current=after_second)
    unwritten, corrupted = _split_unwritten(
        reference=after_first, after_reset=after_reset, current=after_second
    )
    print(refit_vs_refit.summary("refit_vs_refit", engines=engines), flush=True)
    print(
        f"{_MARKER} phase=refit_vs_refit unwritten={len(unwritten)} "
        f"corrupted={len(corrupted)}",
        flush=True,
    )
    _report_names("refit_vs_refit unwritten", unwritten)
    _report_names("refit_vs_refit corrupted", corrupted)
    _report_names("refit_vs_refit missing", refit_vs_refit.only_in_reference)
    _report_names("refit_vs_refit extra", refit_vs_refit.only_in_current)

    if not refit_vs_refit.ok:
        failures = []
        if corrupted:
            failures.append(f"{len(corrupted)} parameters refit with wrong values")
        if unwritten:
            failures.append(f"{len(unwritten)} parameters never written by the refit")
        if refit_vs_refit.only_in_reference:
            failures.append(
                f"{len(refit_vs_refit.only_in_reference)} parameters vanished "
                "from the engines"
            )
        if refit_vs_refit.only_in_current:
            failures.append(
                f"{len(refit_vs_refit.only_in_current)} unexpected parameters "
                "appeared on the engines"
            )
        raise RefitValueCheckError(
            "Distributed refit value check failed: " + "; ".join(failures)
        )

    return {
        "engines": engines,
        "ranks": refit_vs_refit.ranks,
        "params": refit_vs_refit.params,
        "from_disk_vs_refit_differing": disk_differing,
        "reset_randomized": randomized,
        "refit_vs_refit_unwritten": len(unwritten),
        "refit_vs_refit_corrupted": len(corrupted),
    }


def verify_distributed_refit_values(
    policy_generation: Any,
    *,
    refit: Callable[[], Any],
    compare_against_disk: bool = True,
) -> dict[str, Any]:
    """Prove that a distributed refit installs the expected weight values.

    Must run while the engines are quiescent: it deliberately randomizes their
    weights. It leaves them holding correctly refit weights, so the caller's
    own startup refit can proceed unchanged afterwards. Health monitoring is
    suspended for the whole verification, because a health probe that lands
    mid-checksum can time out and take an engine down.

    Args:
        policy_generation: The ``SGLangGeneration`` facade.
        refit: Zero-argument callable performing one full production refit.
        compare_against_disk: Whether the trainer's weights are expected to
            equal what SGLang loaded from disk. True only on a fresh run; a
            resumed run legitimately refits a trained checkpoint into engines
            that cold-loaded the base model.

    Returns:
        A summary dict of the phases, for logging.

    Raises:
        RefitValueCheckError: If the refit does not reproduce the expected
            per-parameter hashes, if the reset control did not take, or if the
            engines cannot be compared like for like.
    """
    print(
        f"{_MARKER} event=start compare_against_disk={compare_against_disk}", flush=True
    )
    lease = policy_generation.health_monitoring_suspend_for_refit()
    try:
        summary = _verify(
            policy_generation,
            refit=refit,
            compare_against_disk=compare_against_disk,
        )
    except BaseException as error:
        # Emit the verdict for every failure path, not just a failed
        # comparison: a validator that greps this marker must be able to tell
        # "the check failed" from "the check never finished".
        #
        # The health-monitor lease is deliberately NOT released here. A failure
        # can leave randomized weights installed, and the monitor's suspension
        # registry is a fail-closed latch: keeping it held stops health probes
        # from acting on engines whose state is unknown.
        print(
            f"{_MARKER} result=fail error_type={type(error).__name__}",
            flush=True,
        )
        raise
    policy_generation.health_monitoring_release_refit(lease)
    print(
        f"{_MARKER} result=pass engines={summary['engines']} "
        f"ranks={summary['ranks']} params={summary['params']}",
        flush=True,
    )
    return summary


def maybe_verify_distributed_refit_values(
    policy_generation: Any,
    *,
    refit: Callable[[], Any],
    colocated_inference: bool,
    compare_against_disk: bool = True,
) -> Optional[dict[str, Any]]:
    """Run the check when it is both requested and applicable."""
    if not value_check_enabled():
        return None
    if colocated_inference:
        # The colocated route transfers weights over IPC, not the NCCL
        # broadcast this check exists to cover.
        print(f"{_MARKER} event=skipped reason=colocated_inference", flush=True)
        return None
    return verify_distributed_refit_values(
        policy_generation,
        refit=refit,
        compare_against_disk=compare_against_disk,
    )
