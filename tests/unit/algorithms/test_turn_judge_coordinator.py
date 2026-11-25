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
import asyncio
import json
import os
import time
from types import SimpleNamespace

import pytest
from aiohttp import web

from nemo_rl.algorithms.single_controller_utils.turn_credit import (
    GOOD,
    HARMFUL,
    TurnCreditConfig,
)
from nemo_rl.algorithms.single_controller_utils.turn_judge_coordinator import (
    TurnJudgeClient,
    TurnJudgeCoordinator,
    _unpack,
    read_events,
    rollout_fingerprint,
    stash_from_env_extras,
)

N = 4


def output(n_calls):
    """A response.output with n_calls model calls (reasoning, tool call, tool output)."""
    items = []
    for _ in range(n_calls):
        items += [
            {"type": "reasoning"},
            {"type": "function_call"},
            {"type": "function_call_output"},
        ]
    return items


def env_extras(rollout_id, reward, n_calls=3, **kw):
    return {
        "ng_rollout_id": rollout_id,
        "reward": reward,
        "response": {"output": output(n_calls)},
        "opencode_results_fpath": f"/artifacts/{rollout_id}/generation.json",
        "test_output": "x" * 1000,
        "instance_config": {"mask_sample": False},
        **kw,
    }


def request(gid, rewards, rollout_ids=None):
    rollout_ids = rollout_ids or [f"{gid}-r{i}" for i in range(len(rewards))]
    return SimpleNamespace(
        group_id=gid,
        rollout_ids=tuple(rollout_ids),
        canonical_sample_ids=tuple(f"{gid}_g{i}" for i in range(len(rewards))),
        rewards=tuple(rewards),
    )


PROMPT = {
    "extra_env_info": {
        "instance_id": "task-1",
        "agent_ref": {"name": "swe_agent"},
        "_ng_group": "x",
    }
}


class FakeService:
    """The judge service's HTTP surface with scripted behaviour."""

    def __init__(self, mode="done", polls_before_done=1, result=None):
        self.mode = mode
        self.polls_before_done = polls_before_done
        self.result = result
        self.posts, self.gets, self.deletes = [], [], []
        self.lost_once = False

    def app(self):
        app = web.Application(client_max_size=64 * 1024**2)
        app.router.add_post("/judge_group", self.post)
        app.router.add_get("/judge_group/{rid}", self.get)
        app.router.add_delete("/judge_group/{rid}", self.delete)
        return app

    def _body(self, rid):
        if (
            self.mode in ("done", "lose_once")
            and len([g for g in self.gets if g == rid]) >= self.polls_before_done
        ):
            return {"request_id": rid, "state": "done", "result": self.result}
        return {"request_id": rid, "state": "running", "result": None}

    async def post(self, request):
        body = await request.json()
        self.posts.append(body)
        if self.mode == "reject":
            return web.json_response({"detail": "bad payload"}, status=422)
        return web.json_response(self._body(body["request_id"]))

    async def get(self, request):
        rid = request.match_info["rid"]
        self.gets.append(rid)
        if self.mode == "lose_once" and not self.lost_once:
            self.lost_once = True
            return web.json_response({"detail": "unknown"}, status=404)
        return web.json_response(self._body(rid))

    async def delete(self, request):
        self.deletes.append(request.match_info["rid"])
        return web.json_response({"ok": True})


async def serve(service, tmp_path):
    runner = web.AppRunner(service.app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    (tmp_path / "judge.env").write_text(
        f"export TURN_JUDGE_URL=http://127.0.0.1:{port}\n"
    )
    return runner


def config(tmp_path, **judge):
    return TurnCreditConfig(
        enabled=True,
        judge={
            "endpoint_file": str(tmp_path / "judge.env"),
            "stage1_timeout_s": 0.1,
            "pair_timeout_s": 0.1,
            "budget_s": 1.0,
            "poll_interval_s": 0.02,
            "request_timeout_s": 2.0,
            "breaker_failures": 2,
            "breaker_cooloff_s": 60.0,
            **judge,
        },
    )


def result(labels, status="ok", n_calls=3):
    return {
        "status": status,
        "labels": {
            str(i): {
                "n_calls": n_calls,
                "labels": [{"idx": 3 * k, "call": k, "label": lab} for k, lab in labs],
            }
            for i, labs in labels.items()
        },
        "info": {"pairs_total": 3, "pairs_ok": 3, "sessions": 4},
    }


def observe_group(coord, gid, rewards, n_calls=3):
    for i, r in enumerate(rewards):
        coord.observe_completion(
            gid, i, env_extras(f"{gid}-r{i}", r, n_calls=n_calls), r
        )


def run(coro):
    return asyncio.run(coro)


def test_stash_fields():
    s = stash_from_env_extras(
        env_extras(
            "r0",
            1.0,
            n_calls=5,
            opencode_run_stderr="... signal: killed",
            opencode_context_limited=True,
        ),
        0.5,
    )
    assert (
        s["n_calls"] == 5
        and s["reward"] == 1.0
        and s["cap_killed"]
        and s["context_limited"]
    )
    assert s["response_output"] is None and s["opencode_results_fpath"].endswith(
        "generation.json"
    )
    # Without a generation.json path the trajectory travels inline.
    s = stash_from_env_extras(
        {"ng_rollout_id": "r", "response": {"output": output(2)}}, 0.0
    )
    assert s["response_output"] == output(2) and s["reward"] == 0.0


def test_judged_group_gets_labels(tmp_path):
    rewards = [1.0, 0.0, 0.0, 1.0]
    service = FakeService(
        result=result({1: [(0, "harmful"), (2, "good")], 3: [(1, "good")]})
    )

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(
            config(tmp_path), state_dir=str(tmp_path / "state"), job_id="j1"
        )
        coord.resume([], [])
        observe_group(coord, "g", rewards)
        status = await coord.judge_sealed_group(
            request("g", rewards), PROMPT, "swe_agent"
        )
        labels = coord.labels_for([f"g_g{i}" for i in range(N)], step=1)
        metrics = coord.pop_step_metrics()
        await coord.close()
        await runner.cleanup()
        return status, labels, metrics

    status, labels, metrics = run(go())
    assert status == "ok"
    assert set(labels) == {"g_g1", "g_g3"}
    assert (
        labels["g_g1"].labels == ((0, HARMFUL), (2, GOOD))
        and labels["g_g1"].n_calls == 3
    )
    assert labels["g_g3"].labels == ((1, GOOD),)
    body = service.posts[0]
    assert (
        body["request_id"] == f"g:{rollout_fingerprint([f'g-r{i}' for i in range(N)])}"
    )
    assert body["task"] == {
        "instance_id": "task-1",
        "agent_ref": {"name": "swe_agent"},
    }  # _ng* dropped
    assert [r["generation_index"] for r in body["rollouts"]] == list(range(N))
    assert all("gate_rollout_id" not in r for r in body["rollouts"])
    assert (
        metrics["turn_judge/status/ok"] == 1
        and metrics["turn_judge/labels/good_calls"] == 2
    )
    assert metrics["turn_judge/batch/groups_labeled"] == 1


def test_skips(tmp_path):
    service = FakeService(result=result({}))

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(
            config(tmp_path, agents=["swe_agent"]), state_dir=None
        )
        coord.resume([], [])
        out = []
        observe_group(coord, "u", [1.0] * N)
        out.append(
            await coord.judge_sealed_group(request("u", [1.0] * N), PROMPT, "swe_agent")
        )
        out.append(
            await coord.judge_sealed_group(
                request("a", [1.0, 0.0, 0.0, 0.0]), PROMPT, "tb_agent"
            )
        )
        observe_group(coord, "m", [1.0, 0.0, 0.0, 0.0])
        coord._stash["m"].pop(2)
        out.append(
            await coord.judge_sealed_group(
                request("m", [1.0, 0.0, 0.0, 0.0]), PROMPT, "swe_agent"
            )
        )
        # A regenerated sibling (new gate rollout id) whose completion was not observed.
        observe_group(coord, "r", [1.0, 0.0, 0.0, 0.0])
        ids = [f"r-r{i}" for i in range(N)]
        ids[1] = "r-r1-retry"
        out.append(
            await coord.judge_sealed_group(
                request("r", [1.0, 0.0, 0.0, 0.0], ids), PROMPT, "swe_agent"
            )
        )
        await coord.close()
        await runner.cleanup()
        return out

    assert run(go()) == [
        "skipped:uniform",
        "skipped:agent",
        "skipped:incomplete_payload",
        "skipped:incomplete_payload",
    ]
    assert service.posts == []


def test_n_calls_mismatch_and_conflicts_drop_labels(tmp_path):
    rewards = [1.0, 0.0, 0.0, 1.0]
    res = result({0: [(0, "good")], 1: [(1, "good"), (1, "harmful"), (2, "harmful")]})
    res["labels"]["0"]["n_calls"] = (
        7  # the service saw another trajectory than the trainer
    )
    service = FakeService(result=res)

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
        coord.resume([], [])
        observe_group(coord, "g", rewards)
        await coord.judge_sealed_group(request("g", rewards), PROMPT, "swe_agent")
        labels = coord.labels_for([f"g_g{i}" for i in range(N)], step=1)
        m = coord.pop_step_metrics()
        await coord.close()
        await runner.cleanup()
        return labels, m

    labels, m = run(go())
    assert set(labels) == {"g_g1"} and labels["g_g1"].labels == ((2, HARMFUL),)
    assert (
        m["turn_judge/labels/n_calls_mismatch"] == 1
        and m["turn_judge/labels/conflicting_call"] == 1
    )


@pytest.mark.parametrize("status", ["ill_posed", "stage1_failed", "no_pairs"])
def test_unlabeled_statuses(tmp_path, status):
    rewards = [1.0, 0.0, 0.0, 1.0]
    service = FakeService(result={"status": status, "labels": {}, "info": {}})

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
        coord.resume([], [])
        observe_group(coord, "g", rewards)
        out = await coord.judge_sealed_group(request("g", rewards), PROMPT, "swe_agent")
        labels = coord.labels_for(["g_g0"], step=1)
        m = coord.pop_step_metrics()
        await coord.close()
        await runner.cleanup()
        return out, labels, m

    out, labels, m = run(go())
    assert (
        out == status and labels == {} and m[f"turn_judge/batch/groups_{status}"] == 1
    )


def test_timeout_cancels_at_the_service(tmp_path):
    rewards = [1.0, 0.0, 0.0, 1.0]
    service = FakeService(mode="never")

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(config(tmp_path, budget_s=0.4), state_dir=None)
        coord.resume([], [])
        observe_group(coord, "g", rewards)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        out = await coord.judge_sealed_group(request("g", rewards), PROMPT, "swe_agent")
        waited = loop.time() - t0
        m = coord.pop_step_metrics()
        await coord.close()
        await runner.cleanup()
        return out, waited, m

    out, waited, m = run(go())
    assert out == "timeout" and 0.35 < waited < 2.0
    assert len(service.deletes) == 1 and m["turn_judge/timeout_rate"] == 1.0


def test_lost_request_is_resent(tmp_path):
    rewards = [1.0, 0.0, 0.0, 1.0]
    service = FakeService(
        mode="lose_once", polls_before_done=2, result=result({0: [(0, "good")]})
    )

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
        coord.resume([], [])
        observe_group(coord, "g", rewards)
        out = await coord.judge_sealed_group(request("g", rewards), PROMPT, "swe_agent")
        await coord.close()
        await runner.cleanup()
        return out

    assert run(go()) == "ok"
    assert (
        len(service.posts) == 2
        and service.posts[0]["request_id"] == service.posts[1]["request_id"]
    )


def test_rejected_request_returns_at_once(tmp_path):
    rewards = [1.0, 0.0, 0.0, 1.0]
    service = FakeService(mode="reject")

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(config(tmp_path, budget_s=5.0), state_dir=None)
        coord.resume([], [])
        observe_group(coord, "g", rewards)
        out = await coord.judge_sealed_group(request("g", rewards), PROMPT, "swe_agent")
        await coord.close()
        await runner.cleanup()
        return out

    assert run(go()) == "rejected"


def test_transport_errors_open_the_breaker(tmp_path):
    (tmp_path / "judge.env").write_text(
        "export TURN_JUDGE_URL=http://127.0.0.1:9\n"
    )  # nothing listens
    rewards = [1.0, 0.0, 0.0, 1.0]

    async def go():
        coord = TurnJudgeCoordinator(config(tmp_path, budget_s=0.5), state_dir=None)
        coord.resume([], [])
        out = []
        for gid in ("a", "b"):
            observe_group(coord, gid, rewards)
            out.append(
                await coord.judge_sealed_group(
                    request(gid, rewards), PROMPT, "swe_agent"
                )
            )
        m = coord.pop_step_metrics()
        await coord.close()
        return out, m

    out, m = run(go())
    assert out == ["transport_error", "skipped:breaker"]
    assert m["turn_judge/breaker_open"] == 1.0


def test_restart_restores_live_groups_and_supersedes_old_files(tmp_path):
    rewards = [1.0, 0.0, 0.0, 1.0]
    state = tmp_path / "state"
    service = FakeService(result=result({1: [(0, "harmful")]}))

    async def job1():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(
            config(tmp_path), state_dir=str(state), job_id="1", clock=lambda: 1000.0
        )
        coord.resume([], [])
        for gid in ("done", "dropped"):
            observe_group(coord, gid, rewards)
            await coord.judge_sealed_group(request(gid, rewards), PROMPT, "swe_agent")
        # In flight at the crash: two of four rollouts sealed.
        for i in range(2):
            coord.observe_completion(
                "inflight", i, env_extras(f"inflight-r{i}", rewards[i]), rewards[i]
            )
        await coord.close()
        await runner.cleanup()

    run(job1())
    # Job 2 restores "done" (replay buffer) and "inflight" (recovery ledger); "dropped" is not restored.
    coord2 = TurnJudgeCoordinator(
        config(tmp_path), state_dir=str(state), job_id="2", clock=lambda: 2000.0
    )
    coord2.resume(["done", None], ["inflight"])
    assert set(coord2._results) == {"done"} and set(coord2._stash) == {"inflight"}
    assert (
        _unpack(coord2._stash["inflight"][0]["test_output"]) == "x" * 1000
    )  # kept compressed
    labels = coord2.labels_for(["done_g1", "done_g0"], step=5)
    assert labels["done_g1"].labels == ((0, HARMFUL),)
    m = coord2.pop_step_metrics()
    assert (
        m["turn_judge/state_restored_results"] == 1
        and m["turn_judge/state_restored_stash"] == 2
    )
    run(coord2.close())
    # The restored in-flight group seals after the restart: its two reused rollouts come from the log,
    # the two regenerated ones are observed now, and a re-sealed judged group reuses its result.
    names = sorted(os.listdir(state))
    assert names == ["events.1.1000.jsonl", "events.2.2000.jsonl"]
    events = read_events(str(state))
    assert (
        events[0]["e"] == "snapshot_start" and events[0]["job"] == "2"
    )  # job 1's file is superseded
    coord3 = TurnJudgeCoordinator(
        config(tmp_path), state_dir=str(state), job_id="3", clock=lambda: 3000.0
    )
    coord3.resume(["done"], ["inflight"])
    for i in range(2, N):
        coord3.observe_completion(
            "inflight", i, env_extras(f"inflight-r{i}", rewards[i]), rewards[i]
        )
    assert sorted(coord3._stash["inflight"]) == [0, 1, 2, 3]

    async def reseal():
        return await coord3.judge_sealed_group(
            request("done", rewards), PROMPT, "swe_agent"
        )

    assert run(reseal()) == "reused"
    run(coord3.close())


def test_regenerated_group_is_judged_again(tmp_path):
    rewards = [1.0, 0.0, 0.0, 1.0]
    service = FakeService(result=result({1: [(0, "harmful")]}))

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
        coord.resume([], [])
        observe_group(coord, "g", rewards)
        first = await coord.judge_sealed_group(
            request("g", rewards), PROMPT, "swe_agent"
        )
        coord.observe_completion("g", 2, env_extras("g-r2-new", 0.0), 0.0)
        ids = [f"g-r{i}" for i in range(N)]
        ids[2] = "g-r2-new"
        second = await coord.judge_sealed_group(
            request("g", rewards, ids), PROMPT, "swe_agent"
        )
        await coord.close()
        await runner.cleanup()
        return first, second

    assert run(go()) == ("ok", "ok")
    assert (
        len(service.posts) == 2
        and service.posts[0]["request_id"] != service.posts[1]["request_id"]
    )


def test_observer_never_raises_and_consumed_groups_are_pruned(tmp_path):
    coord = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
    coord.resume([], [])
    coord.observe_completion("g", 0, {"response": object()}, 1.0)
    coord.observe_completion("g", "x", {}, 1.0)  # bad generation index
    m = coord.pop_step_metrics()
    assert m["turn_judge/observer_errors"] == 1
    coord._results["old"] = SimpleNamespace(fingerprint="f", status="ok", labels={})
    coord.labels_for(["old_g0"], step=1)
    coord.labels_for([], step=3)
    assert "old" in coord._results
    coord.labels_for([], step=4)
    assert "old" not in coord._results


def test_side_log_records(tmp_path):
    coord = TurnJudgeCoordinator(
        config(tmp_path),
        state_dir=str(tmp_path / "state"),
        job_id="7",
        clock=lambda: 5.0,
    )
    coord.resume([], [])
    coord.record_shaping(
        3, [{"sample_id": "g_g0", "status": "shaped", "lambda_eff": 0.5}]
    )
    run(coord.close())
    lines = [
        json.loads(x)
        for x in (tmp_path / "state" / "events.7.5.jsonl").read_text().splitlines()
    ]
    assert lines[-1]["e"] == "shaped" and lines[-1]["rows"][0]["lambda_eff"] == 0.5


def test_missing_endpoint_file_waits_without_opening_the_breaker(tmp_path):
    rewards = [1.0, 0.0, 0.0, 1.0]
    service = FakeService(result=result({0: [(0, "good")]}))

    async def go():
        coord = TurnJudgeCoordinator(config(tmp_path, budget_s=0.3), state_dir=None)
        coord.resume([], [])
        observe_group(coord, "a", rewards)
        first = await coord.judge_sealed_group(
            request("a", rewards), PROMPT, "swe_agent"
        )  # no file at all
        runner = await serve(
            service, tmp_path
        )  # the service comes up and writes its file
        observe_group(coord, "b", rewards)
        second = await coord.judge_sealed_group(
            request("b", rewards), PROMPT, "swe_agent"
        )
        m = coord.pop_step_metrics()
        await coord.close()
        await runner.cleanup()
        return first, second, m

    first, second, m = run(go())
    assert (
        first == "no_service" and second == "ok" and m["turn_judge/breaker_open"] == 0.0
    )


def test_checkpoint_state_round_trip(tmp_path):
    """The restart state a full checkpoint carries: results of its replay and in-flight groups, stash of in-flight."""
    rewards = [1.0, 0.0, 0.0, 1.0]
    service = FakeService(result=result({1: [(0, "harmful")], 2: [(2, "good")]}))

    async def job1():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
        coord.resume([], [])
        for gid in ("committed", "judged_not_committed", "consumed"):
            observe_group(coord, gid, rewards)
            await coord.judge_sealed_group(request(gid, rewards), PROMPT, "swe_agent")
        coord.committed("committed")
        for i in range(2):
            coord.observe_completion(
                "inflight", i, env_extras(f"inflight-r{i}", rewards[i]), rewards[i]
            )
        state = coord.state_dict(
            ["committed", None], ["judged_not_committed", "inflight"]
        )
        await coord.close()
        await runner.cleanup()
        return state

    state = run(job1())
    assert set(state["results"]) == {"committed", "judged_not_committed"}
    assert set(state["stash"]) == {"judged_not_committed", "inflight"}
    import io

    import torch

    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    coord2 = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
    coord2.resume(["committed"], ["judged_not_committed", "inflight"], loaded)
    labels = coord2.labels_for(["committed_g1", "committed_g2", "committed_g0"], step=9)
    assert labels["committed_g1"].labels == ((0, HARMFUL),) and labels[
        "committed_g2"
    ].labels == ((2, GOOD),)
    assert sorted(coord2._stash["inflight"]) == [0, 1]

    async def reseal():
        # Sealed again after the restart with the same rollouts: the result is reused, nothing is sent.
        return await coord2.judge_sealed_group(
            request("judged_not_committed", rewards), PROMPT, "swe_agent"
        )

    assert run(reseal()) == "reused"
    run(coord2.close())


def test_service_alignment_proof_gates_labels(tmp_path):
    rewards = [1.0, 0.0, 0.0, 1.0]
    res = result({0: [(0, "good")], 1: [(1, "harmful")], 2: [(2, "harmful")]})
    res["labels"]["0"].update(
        aligned=False, trailing_empty_calls=0, alignment_reason="calls 2 != steps 3"
    )
    res["labels"]["1"].update(
        aligned=True, trailing_empty_calls=1, alignment_reason="ok"
    )
    # rollout 2: an older service without the proof fields
    service = FakeService(result=res)

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
        coord.resume([], [])
        observe_group(coord, "g", rewards)
        await coord.judge_sealed_group(request("g", rewards), PROMPT, "swe_agent")
        state = coord.state_dict(["g"], [])
        labels = coord.labels_for([f"g_g{i}" for i in range(N)], step=1)
        m = coord.pop_step_metrics()
        await coord.close()
        await runner.cleanup()
        return labels, m, state

    labels, m, state = run(go())
    assert set(labels) == {"g_g1", "g_g2"}
    assert labels["g_g1"].aligned is True and labels["g_g1"].trailing_empty_calls == 1
    assert labels["g_g2"].aligned is None and labels["g_g2"].trailing_empty_calls == 0
    assert (
        m["turn_judge/labels/alignment_not_proven"] == 1
        and m["turn_judge/labels/trailing_empty_rollouts"] == 1
    )
    coord2 = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
    coord2.resume(["g"], [], state)
    again = coord2.labels_for(["g_g1", "g_g2"], step=2)
    assert again == {"g_g1": labels["g_g1"], "g_g2": labels["g_g2"]}
    # Checkpoints written before the proof fields existed load with the old meaning.
    legacy = {
        "results": {
            "g": {"fp": "f", "status": "ok", "labels": {"g_g1": [1, 3, [[1, -1]]]}}
        },
        "stash": {},
    }
    coord3 = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
    coord3.resume(["g"], [], legacy)
    lab = coord3.labels_for(["g_g1"], step=3)["g_g1"]
    assert (
        lab.aligned is None
        and lab.trailing_empty_calls == 0
        and lab.labels == ((1, -1),)
    )
    run(coord2.close())
    run(coord3.close())


def test_step_output_tokens_reach_the_labels_and_the_checkpoint(tmp_path):
    """The service's per-call output tokens ride with the labels (run-by-run identity check, LOG 094)."""
    rewards = [1.0, 0.0, 0.0, 1.0]
    res = result({0: [(0, "good")], 1: [(1, "harmful")], 2: [(2, "harmful")]})
    res["labels"]["0"].update(
        aligned=True, trailing_empty_calls=0, step_output_tokens=[120, 7, 33]
    )
    res["labels"]["1"].update(
        aligned=True, trailing_empty_calls=1, step_output_tokens=[5, 6, 7, None]
    )
    res["labels"]["2"].update(
        aligned=True, trailing_empty_calls=0, step_output_tokens=[5, "6", 7]
    )  # malformed
    service = FakeService(result=res)

    async def go():
        runner = await serve(service, tmp_path)
        coord = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
        coord.resume([], [])
        observe_group(coord, "g", rewards)
        await coord.judge_sealed_group(request("g", rewards), PROMPT, "swe_agent")
        state = coord.state_dict(["g"], [])
        labels = coord.labels_for([f"g_g{i}" for i in range(N)], step=1)
        m = coord.pop_step_metrics()
        await coord.close()
        await runner.cleanup()
        return labels, m, state

    labels, m, state = run(go())
    assert labels["g_g0"].step_tokens == (120, 7, 33) and labels[
        "g_g1"
    ].step_tokens == (5, 6, 7, None)
    assert (
        labels["g_g2"].step_tokens is None
        and m["turn_judge/labels/no_step_tokens"] == 1
    )
    coord2 = TurnJudgeCoordinator(config(tmp_path), state_dir=None)
    coord2.resume(["g"], [], state)
    again = coord2.labels_for(["g_g0", "g_g1", "g_g2"], step=2)
    assert again == {k: labels[k] for k in ("g_g0", "g_g1", "g_g2")}
    run(coord2.close())


def test_slow_polls_of_a_submitted_request_never_open_the_breaker(tmp_path):
    """A service whose event loop stalls past the request timeout after accepting requests (LOG 096): the polls
    fail, the request keeps polling and gets its result, and new groups are still sent."""
    (tmp_path / "judge.env").write_text("export TURN_JUDGE_URL=http://127.0.0.1:9\n")
    client = TurnJudgeClient(config(tmp_path, breaker_failures=2).judge)
    calls = []

    async def fake_request(method, path, body=None):
        calls.append(method)
        if method == "POST":
            return 200, {"state": "queued"}
        if calls.count("GET") <= 4:  # the stall: four polls in a row time out
            raise asyncio.TimeoutError()
        return 200, {"state": "done", "result": {"status": "ok"}}

    client._request = fake_request
    outcome = run(client.judge({"request_id": "g:fp"}, deadline=time.time() + 30))
    assert outcome == ("done", {"status": "ok"})
    assert not client.breaker_open() and client.consecutive_failures == 0

    # Failed submissions still open it.
    async def down(method, path, body=None):
        raise ConnectionRefusedError()

    client._request = down
    assert run(client.judge({"request_id": "h:fp"}, deadline=time.time() + 30)) == (
        "transport_error",
        None,
    )
    assert client.breaker_open()
