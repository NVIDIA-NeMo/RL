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

import pytest

from nemo_rl.weight_sync.sglang_refit_value_check import (
    RefitValueCheckError,
    maybe_verify_distributed_refit_values,
    verify_distributed_refit_values,
)

_PARAMS = (
    "model.layers.0.self_attn.qkv_proj.weight",
    "model.layers.0.mlp.experts.w13_weight",
    "model.layers.0.mlp.experts.w2_weight",
    "model.layers.0.mlp.gate.weight",
    "model.embed_tokens.weight",
)

_ENGINES = 2
_TP = 2


def _state(tag):
    """Per-(engine, rank) hashes. Every rank is given distinct values so that a
    comparison which confused two ranks would not accidentally agree."""
    return {
        (engine, rank): {name: f"{tag}:e{engine}r{rank}:{name}" for name in _PARAMS}
        for engine in range(_ENGINES)
        for rank in range(_TP)
    }


def _mutate(state, *, engine=0, rank=0, name=_PARAMS[1], value="wrong"):
    out = {key: dict(value_) for key, value_ in state.items()}
    out[(engine, rank)][name] = value
    return out


class FakeGeneration:
    """Replays a scripted sequence of engine states.

    ``check_weights('checksum')`` reports the current state; ``refit`` and
    ``reset_tensors`` advance it. Rank payloads are emitted in a DIFFERENT
    order on every call, because the real HTTP layer aggregates them in
    response arrival order -- a comparison that matched positionally would
    fail here.
    """

    def __init__(self, states, *, answered=None, reset_ok=True, drop_rank_size=False):
        self._states = list(states)
        self._answered = answered or [[True] * _ENGINES] * 20
        self._reset_ok = reset_ok
        self._drop_rank_size = drop_rank_size
        self._calls = 0
        self.actions = []
        self.refits = 0
        self.lease = None
        self.released = []

    # -- SGLangGeneration facade ------------------------------------------
    def health_monitoring_suspend_for_refit(self):
        self.lease = "refit:test"
        return self.lease

    def health_monitoring_release_refit(self, lease):
        self.released.append(lease)

    def check_weights(self, action):
        self.actions.append(action)
        answered = self._answered[min(self._calls, len(self._answered) - 1)]
        if action == "reset_tensors":
            self._states.pop(0)
            return [
                {"success": self._reset_ok, "message": "Success."} if ok else None
                for ok in answered
            ]

        state = self._states[0]
        self._calls += 1
        responses = []
        for engine in range(_ENGINES):
            if not answered[engine]:
                responses.append(None)
                continue
            ranks = []
            for rank in range(_TP):
                # A key absent from the state means the rank does not report at
                # all; a key mapped to {} means it reports an empty set. The two
                # are different failures.
                if (engine, rank) not in state:
                    continue
                info = {"tp_rank": rank, "tp_size": _TP, "dp_rank": 0, "pp_rank": 0}
                if not self._drop_rank_size:
                    info["rank"] = rank
                    info["size"] = _TP
                ranks.append(
                    {
                        "checksums": dict(state[(engine, rank)]),
                        "per_gpu_checksum": f"gpu{engine}{rank}",
                        "parallelism_info": info,
                    }
                )
            # Alternate arrival order between calls.
            if self._calls % 2 == 0:
                ranks = list(reversed(ranks))
            responses.append({"success": True, "message": "Success.", "ranks": ranks})
        return responses

    def refit(self):
        self.refits += 1
        self._states.pop(0)


def _run(states, **kwargs):
    generation = FakeGeneration(states, **kwargs)
    summary = verify_distributed_refit_values(generation, refit=generation.refit)
    return generation, summary


def _clean_script():
    """disk -> refit(disk) -> reset(noise) -> refit(disk)."""
    disk = _state("disk")
    return [disk, disk, _state("noise"), disk]


# -- happy path -----------------------------------------------------------


def test_passes_when_every_refit_reproduces_the_from_disk_state():
    generation, summary = _run(_clean_script())

    assert generation.refits == 2
    assert generation.actions == [
        "checksum",
        "checksum",
        "reset_tensors",
        "checksum",
        "checksum",
    ]
    assert summary["engines"] == _ENGINES
    assert summary["ranks"] == _ENGINES * _TP
    assert summary["params"] == _ENGINES * _TP * len(_PARAMS)
    assert summary["from_disk_vs_refit_differing"] == 0
    assert summary["refit_vs_refit_corrupted"] == 0
    assert summary["refit_vs_refit_unwritten"] == 0


def test_suspends_and_releases_health_monitoring():
    generation, _ = _run(_clean_script())
    assert generation.lease == "refit:test"
    assert generation.released == ["refit:test"]


def test_holds_the_health_monitor_lease_when_the_check_fails():
    """A failure can leave randomized weights installed; the fail-closed latch
    must stay held rather than re-enabling probes against unknown state."""
    disk = _state("disk")
    generation = FakeGeneration([disk, disk, _state("noise"), _mutate(disk)])
    with pytest.raises(RefitValueCheckError):
        verify_distributed_refit_values(generation, refit=generation.refit)
    assert generation.released == []


def test_skips_the_disk_comparison_on_a_resumed_run():
    """On resume the trainer holds a trained checkpoint, so a correct refit
    must differ from what the engines cold-loaded."""
    trained = _state("trained")
    generation = FakeGeneration([_state("disk"), trained, _state("noise"), trained])
    summary = verify_distributed_refit_values(
        generation, refit=generation.refit, compare_against_disk=False
    )
    assert summary["from_disk_vs_refit_differing"] is None
    assert summary["refit_vs_refit_corrupted"] == 0


# -- value failures -------------------------------------------------------


def test_flags_a_parameter_the_refit_wrote_with_wrong_values():
    disk = _state("disk")
    with pytest.raises(RefitValueCheckError, match="wrong values"):
        _run([disk, disk, _state("noise"), _mutate(disk)])


def test_distinguishes_a_parameter_the_refit_never_wrote():
    disk = _state("disk")
    noise = _state("noise")
    # Still holds its randomized value: the refit skipped it entirely.
    partial = _mutate(disk, name=_PARAMS[3], value=noise[(0, 0)][_PARAMS[3]])
    with pytest.raises(RefitValueCheckError, match="never written"):
        _run([disk, disk, noise, partial])


def test_flags_a_refit_that_does_not_reproduce_the_checkpoint():
    disk = _state("disk")
    drifted = _mutate(disk, name=_PARAMS[4], value="drifted")
    with pytest.raises(RefitValueCheckError, match="loaded from disk"):
        _run([disk, drifted, _state("noise"), drifted])


def test_detects_a_rank_swap():
    """Two ranks whose values are exchanged must not read as agreement."""
    disk = _state("disk")
    swapped = {key: dict(value) for key, value in disk.items()}
    swapped[(0, 0)], swapped[(0, 1)] = swapped[(0, 1)], swapped[(0, 0)]
    with pytest.raises(RefitValueCheckError):
        _run([disk, disk, _state("noise"), swapped])


# -- structural failures --------------------------------------------------


def test_rejects_a_parameter_missing_from_the_final_sweep():
    disk = _state("disk")
    truncated = {key: dict(value) for key, value in disk.items()}
    del truncated[(0, 0)][_PARAMS[2]]
    with pytest.raises(RefitValueCheckError, match="vanished"):
        _run([disk, disk, _state("noise"), truncated])


def test_rejects_an_unexpected_parameter_in_the_final_sweep():
    disk = _state("disk")
    extended = {key: dict(value) for key, value in disk.items()}
    extended[(0, 0)]["model.layers.0.mlp.experts.w3_weight"] = "surprise"
    with pytest.raises(RefitValueCheckError, match="unexpected"):
        _run([disk, disk, _state("noise"), extended])


def test_rejects_a_rank_that_disappears():
    disk = _state("disk")
    fewer = {key: value for key, value in disk.items() if key != (1, 1)}
    with pytest.raises(RefitValueCheckError, match="rank set changed"):
        _run([disk, fewer, _state("noise"), fewer])


def test_rejects_an_engine_that_stops_answering():
    disk = _state("disk")
    generation = FakeGeneration(
        [disk, disk, _state("noise"), disk],
        answered=[[True, True], [True, False], [True, False], [True, False]],
    )
    with pytest.raises(RefitValueCheckError, match="engines answering"):
        verify_distributed_refit_values(generation, refit=generation.refit)


def test_rejects_an_empty_rank():
    disk = _state("disk")
    empty = {key: (dict(value) if key != (0, 1) else {}) for key, value in disk.items()}
    with pytest.raises(RefitValueCheckError, match="empty checksum set"):
        _run([empty, empty, _state("noise"), empty])


def test_refuses_a_build_without_a_global_rank():
    """Without a stable rank identity the only ordering available is arrival
    order, which is not stable; refuse rather than guess."""
    disk = _state("disk")
    generation = FakeGeneration(
        [disk, disk, _state("noise"), disk], drop_rank_size=True
    )
    with pytest.raises(RefitValueCheckError, match="global rank"):
        verify_distributed_refit_values(generation, refit=generation.refit)


def test_rejects_a_build_without_per_rank_checksums():
    class NoPayload:
        def health_monitoring_suspend_for_refit(self):
            return None

        def health_monitoring_release_refit(self, lease):
            pass

        def check_weights(self, action):
            return [{"success": True, "message": "Success."}]

    with pytest.raises(RefitValueCheckError, match="no per-rank checksums"):
        verify_distributed_refit_values(NoPayload(), refit=lambda: None)


def test_reports_an_unsuccessful_checksum_request():
    class Failing:
        def health_monitoring_suspend_for_refit(self):
            return None

        def health_monitoring_release_refit(self, lease):
            pass

        def check_weights(self, action):
            return [{"success": False, "message": "boom"}]

    with pytest.raises(RefitValueCheckError, match="failed the checksum"):
        verify_distributed_refit_values(Failing(), refit=lambda: None)


# -- control failures -----------------------------------------------------


def test_fails_when_the_reset_control_did_not_take():
    disk = _state("disk")
    with pytest.raises(RefitValueCheckError, match="did not cover"):
        _run([disk, disk, disk, disk])


def test_fails_when_the_reset_covered_only_part_of_the_model():
    """A refit that rewrote exactly the randomized half would otherwise pass
    while never touching the half the control missed."""
    disk = _state("disk")
    noise = _state("noise")
    half = {key: dict(value) for key, value in noise.items()}
    half[(0, 0)] = dict(disk[(0, 0)])
    with pytest.raises(RefitValueCheckError, match="did not cover"):
        _run([disk, disk, half, disk])


def test_fails_when_an_engine_reports_a_failed_reset():
    with pytest.raises(RefitValueCheckError, match="failed reset_tensors"):
        _run(_clean_script(), reset_ok=False)


# -- gating ---------------------------------------------------------------


def test_skipped_unless_enabled(monkeypatch):
    monkeypatch.delenv("NRL_REFIT_VALUE_CHECK", raising=False)
    called = []
    assert (
        maybe_verify_distributed_refit_values(
            object(),
            refit=lambda: called.append(1),
            colocated_inference=False,
        )
        is None
    )
    assert not called


def test_skipped_for_colocated_inference(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_VALUE_CHECK", "1")
    called = []
    assert (
        maybe_verify_distributed_refit_values(
            object(),
            refit=lambda: called.append(1),
            colocated_inference=True,
        )
        is None
    )
    assert not called


def test_runs_when_enabled_and_applicable(monkeypatch):
    monkeypatch.setenv("NRL_REFIT_VALUE_CHECK", "1")
    generation = FakeGeneration(_clean_script())

    summary = maybe_verify_distributed_refit_values(
        generation,
        refit=generation.refit,
        colocated_inference=False,
    )
    assert summary is not None
    assert summary["refit_vs_refit_corrupted"] == 0


def test_emits_a_fail_verdict_on_every_failure_path(capsys):
    """A validator greps this marker; an aborted check must not be
    indistinguishable from one that never started."""
    disk = _state("disk")
    generation = FakeGeneration([disk, disk, _state("noise"), _mutate(disk)])
    with pytest.raises(RefitValueCheckError):
        verify_distributed_refit_values(generation, refit=generation.refit)
    out = capsys.readouterr().out
    assert "NRL_REFIT_VALUE_CHECK result=fail" in out
    assert "error_type=RefitValueCheckError" in out
