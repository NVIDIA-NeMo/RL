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
"""``nrl-k8s`` command-line entry point.

Hydra-style overrides (``infra.scheduler.queue=x``) are collected via
``click.UNPROCESSED`` — any ``key=value`` token after the recipe path is
passed to :func:`nrl_k8s.config.load_recipe_with_infra` as an override.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import click
import yaml
from kubernetes.client.exceptions import ApiException
from omegaconf import OmegaConf

from . import __version__
from .config import LoadedConfig, load_recipe_with_infra
from .orchestrate import ALL_ROLES
from .schema import ClusterSpec, CodeSource, RunMode, SubmitterMode

_INFRA_OPTION = click.option(
    "--infra",
    "infra_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a standalone infra YAML. When set, the recipe must not "
    "contain an `infra:` key.",
)
_ROLE_CHOICE = click.Choice(list(ALL_ROLES))
_MODE_CHOICE = click.Choice([m.value for m in RunMode])
_SUBMITTER_CHOICE = click.Choice([m.value for m in SubmitterMode])
_CODE_SOURCE_CHOICE = click.Choice([m.value for m in CodeSource])


# Macro -> (submitter, codeSource, no_wait). CLI --mode wins over
# infra.launch.runMode; explicit --submitter / --code-source /
# --wait/--no-wait flags win over both.
_MODE_DEFAULTS: dict[RunMode, tuple[SubmitterMode, CodeSource, bool]] = {
    RunMode.INTERACTIVE: (SubmitterMode.PORT_FORWARD, CodeSource.UPLOAD, False),
    RunMode.BATCH:       (SubmitterMode.EXEC,         CodeSource.IMAGE,  True),
}


def _resolve_mode_defaults(
    *,
    cli_mode: str | None,
    infra_mode: RunMode,
    cli_submitter: str | None,
    cli_code_source: str | None,
    cli_wait: bool | None,
) -> tuple[RunMode, SubmitterMode, CodeSource, bool]:
    """Return (resolved_mode, submitter, code_source, no_wait).

    ``cli_wait`` is the tri-state carried by click's ``--wait/--no-wait``
    flag pair (True / False / None=unset).
    """
    mode = RunMode(cli_mode) if cli_mode else infra_mode
    default_submitter, default_code_src, default_no_wait = _MODE_DEFAULTS[mode]
    submitter = SubmitterMode(cli_submitter) if cli_submitter else default_submitter
    code_src = CodeSource(cli_code_source) if cli_code_source else default_code_src
    if cli_wait is None:
        no_wait = default_no_wait
    else:
        no_wait = not cli_wait
    return mode, submitter, code_src, no_wait


def _apply_mode_overrides(
    loaded: LoadedConfig,
    *,
    submitter: SubmitterMode,
    code_source: CodeSource,
    code_path: str | None,
) -> None:
    """Mutate the loaded InfraConfig so downstream sees the resolved values.

    `_resolve_mode_defaults` produces the final submitter/codeSource; we
    push those into the pydantic model so `orchestrate.submit_training`
    and `build_submitter` read one source of truth. `code_path` overrides
    `launch.codePath` when set; else the infra YAML keeps its value.
    """
    # Pydantic models are immutable by default; use model_copy via deep set.
    infra = loaded.infra
    infra.submit.submitter = submitter
    infra.launch.codeSource = code_source
    if code_path is not None:
        infra.launch.codePath = code_path
    # Re-run the validator manually so codePath-required rule fires with
    # the effective values.
    if code_source in (CodeSource.IMAGE, CodeSource.LUSTRE) and not infra.launch.codePath:
        _cli_error(
            f"--code-source {code_source.value} requires --code-path (or infra.launch.codePath)",
            hint="pass --code-path /opt/nemo-rl (image default) or a Lustre mount path",
        )


# Shared decorator factory for the flag block added to both launch and run.
def _mode_options(fn):
    fn = click.option(
        "--mode",
        "cli_mode",
        type=_MODE_CHOICE,
        default=None,
        help="Macro: interactive = port-forward + working_dir upload + tail "
        "(dev default). batch = kubectl exec + code from image + no wait "
        "(production). Overrides infra.launch.runMode.",
    )(fn)
    fn = click.option(
        "--submitter",
        "cli_submitter",
        type=_SUBMITTER_CHOICE,
        default=None,
        help="Transport for the training entrypoint. Overrides --mode's default.",
    )(fn)
    fn = click.option(
        "--code-source",
        "cli_code_source",
        type=_CODE_SOURCE_CHOICE,
        default=None,
        help="Where the code lives. `upload` stages a working_dir from the "
        "laptop; `image` / `lustre` expect code on disk inside the pod.",
    )(fn)
    fn = click.option(
        "--code-path",
        "cli_code_path",
        type=str,
        default=None,
        help="Absolute container path for code when --code-source is "
        "image or lustre. Overrides infra.launch.codePath.",
    )(fn)
    fn = click.option(
        "--run-id",
        "cli_run_id",
        type=str,
        default=None,
        help="Human-readable tag for this run. Used as the Ray submission "
        "id (port-forward) or pidfile directory name (exec). Defaults to "
        "`training-<timestamp>`.",
    )(fn)
    fn = click.option(
        "--wait/--no-wait",
        "cli_wait",
        default=None,
        help="Override mode's wait default: --wait tails logs and exits "
        "on terminal state; --no-wait returns immediately after submit.",
    )(fn)
    return fn


# =============================================================================
# Root group
# =============================================================================


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="nrl-k8s")
def main() -> None:
    """Launch NeMo-RL recipes on Kubernetes."""


# =============================================================================
# check — load + validate + (optionally) render manifests
# =============================================================================


@main.command()
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the full resolved config + rendered RayCluster manifests to "
    "this file (yaml or json — extension picks the format). Omit to print "
    "only a one-page summary.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["yaml", "json"]),
    default=None,
    help="Override the format when using --output. Defaults to the extension.",
)
def check(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    output_path: Path | None,
    output_format: str | None,
) -> None:
    """Load + validate a recipe/infra pair and print a one-line summary per
    role (or dump the fully-resolved config + rendered RayCluster manifests
    to a file with ``-o``). Replaces the former ``validate`` + ``plan``.
    """
    from .manifest import build_raycluster_manifest

    try:
        loaded = load_recipe_with_infra(
            recipe, overrides=list(overrides), infra_path=infra_path
        )
    except Exception as exc:  # noqa: BLE001 — surface the full message to the user
        _explain_and_exit(exc, context="failed to load recipe")

    manifests: dict[str, dict] = {}
    for role in ALL_ROLES:
        cluster = getattr(loaded.infra.clusters, role)
        if cluster is None:
            continue
        manifests[role] = build_raycluster_manifest(cluster, loaded.infra, role=role)

    if output_path is not None:
        _dump_check_output(loaded, manifests, output_path, output_format)
        click.echo(f"wrote full config + {len(manifests)} manifest(s) to {output_path}")
        return

    _print_check_summary(loaded, manifests)


def _print_check_summary(loaded: LoadedConfig, manifests: dict[str, dict]) -> None:
    """One-page overview — namespace, image, launch/attach, per-role highlights."""
    infra = loaded.infra
    click.echo(f"namespace:   {infra.namespace}")
    click.echo(f"image:       {infra.image}")
    if infra.imagePullSecrets:
        click.echo(f"pullSecrets: {', '.join(infra.imagePullSecrets)}")
    if infra.serviceAccount:
        click.echo(f"sa:          {infra.serviceAccount}")
    click.echo(
        f"scheduler:   {infra.scheduler.kind.value}"
        + (f" (queue={infra.scheduler.queue})" if infra.scheduler.queue else "")
    )
    click.echo(f"launch.mode: {infra.launch.mode.value}")
    if infra.launch.entrypoint:
        click.echo("entrypoint:")
        _print_block(infra.launch.entrypoint)

    click.echo("")
    click.echo("CLUSTERS")
    click.echo("--------")
    if not manifests:
        click.echo("  (none declared)")
        return
    for role, m in manifests.items():
        spec = m["spec"]
        name = m["metadata"]["name"]
        head = spec.get("headGroupSpec", {}).get("template", {}).get("spec", {})
        head_res = (
            head.get("containers", [{}])[0].get("resources", {}).get("limits") or {}
        )
        workers = spec.get("workerGroupSpecs") or []
        wrep = sum(int(w.get("replicas", 0)) for w in workers)
        wgpu = 0
        wcpu = wmem = "—"
        if workers:
            w_res = (
                workers[0]
                .get("template", {})
                .get("spec", {})
                .get("containers", [{}])[0]
                .get("resources", {})
                .get("limits")
                or {}
            )
            wgpu = int(w_res.get("nvidia.com/gpu", 0)) * wrep
            wcpu = w_res.get("cpu", "—")
            wmem = w_res.get("memory", "—")
        daemon = loaded.infra.clusters.__dict__[role].daemon
        daemon_id = daemon.submissionId if daemon else "—"

        click.echo(f"  {role}: {name}")
        click.echo(
            f"    head    cpu={head_res.get('cpu', '—')} mem={head_res.get('memory', '—')}"
        )
        if workers:
            click.echo(f"    workers {wrep}x cpu={wcpu} mem={wmem} gpu={wgpu}")
        else:
            click.echo("    workers (none — head-only)")
        click.echo(f"    daemon  {daemon_id}")
        if daemon and daemon.entrypoint:
            click.echo("    entrypoint:")
            _print_block(daemon.entrypoint, indent="      ")


def _print_block(text: str, *, indent: str = "  ") -> None:
    """Print a multi-line shell/script body with consistent indent."""
    # Trim surrounding blank lines but keep internal formatting.
    lines = text.rstrip("\n").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    for line in lines:
        click.echo(f"{indent}{line}")


def _dump_check_output(
    loaded: LoadedConfig,
    manifests: dict[str, dict],
    path: Path,
    fmt_override: str | None,
) -> None:
    fmt = fmt_override or ("json" if path.suffix == ".json" else "yaml")
    bundle = {
        "infra": loaded.infra.model_dump(mode="json"),
        "recipe": OmegaConf.to_container(loaded.recipe, resolve=True),
        "manifests": manifests,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    else:
        path.write_text(yaml.safe_dump(bundle, sort_keys=False))


# =============================================================================
# Deprecated aliases — kept only where scripts/docs still reference them.
# Unimplemented stub commands (doctor/dashboard/dev) were removed; see
# infra/nrl_k8s/docs/roadmap.md for the planned work.
# =============================================================================


@main.command(hidden=True)
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
@click.pass_context
def validate(ctx, recipe, overrides, infra_path) -> None:
    """Deprecated: use ``check``. Prints the summary for backwards compat."""
    click.echo("note: `validate` is deprecated — use `check`.", err=True)
    ctx.invoke(
        check,
        recipe=recipe,
        overrides=overrides,
        infra_path=infra_path,
        output_path=None,
        output_format=None,
    )


@main.command()
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
@click.option(
    "--repo-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    show_default="cwd",
    help="NeMo-RL repo root used to source files for the working_dir upload.",
)
@click.option(
    "--replace",
    is_flag=True,
    help="Long-lived mode only: stop any running daemon/training job before "
    "submitting new ones.",
)
@click.option(
    "--recreate",
    is_flag=True,
    help="Long-lived mode only: delete + re-apply any RayCluster whose live "
    "spec has drifted from the rendered manifest.",
)
@click.option(
    "--skip-daemons",
    is_flag=True,
    help="Long-lived mode only: bring up every declared cluster but only "
    "submit training — skip daemons on gym/generation roles.",
)
@click.option(
    "--rayjob/--raycluster",
    "as_rayjob",
    default=True,
    help="--rayjob (default): submit as an ephemeral KubeRay RayJob that "
    "auto-tears down the cluster when the job finishes. --raycluster: "
    "attach to a long-lived RayCluster (supports --replace/--recreate/"
    "--skip-daemons).",
)
@click.option(
    "--rayjob-name",
    "rayjob_name",
    type=str,
    default=None,
    help="[--rayjob only] RayJob metadata name. Defaults to the training "
    "cluster name.",
)
@click.option(
    "--shutdown/--no-shutdown",
    "rayjob_shutdown",
    default=True,
    show_default=True,
    help="[--rayjob only] Delete the ephemeral RayCluster once the Ray Job "
    "reaches a terminal state (KubeRay's shutdownAfterJobFinishes).",
)
@click.option(
    "--ttl",
    "rayjob_ttl",
    type=int,
    default=3600,
    show_default=True,
    help="[--rayjob only] Seconds to keep the RayJob object after the run "
    "finishes (ttlSecondsAfterFinished). Useful for post-mortem log access.",
)
@click.option(
    "--timeout",
    "rayjob_timeout",
    type=int,
    default=86400,
    show_default=True,
    help="[--rayjob only] Seconds to wait for the RayJob to reach a terminal "
    "state when --wait is set.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="[--rayjob only] Render the RayJob manifest and print it; do not apply.",
)
@_mode_options
def run(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    repo_root: Path,
    replace: bool,
    recreate: bool,
    skip_daemons: bool,
    as_rayjob: bool,
    rayjob_name: str | None,
    rayjob_shutdown: bool,
    rayjob_ttl: int,
    rayjob_timeout: int,
    dry_run: bool,
    cli_mode: str | None,
    cli_submitter: str | None,
    cli_code_source: str | None,
    cli_code_path: str | None,
    cli_run_id: str | None,
    cli_wait: bool | None,
) -> None:
    """Submit a recipe to the cluster. Ephemeral by default, long-lived with ``--raycluster``.

    **Ephemeral mode (``--rayjob``, default)** — submits the recipe as a
    KubeRay RayJob. KubeRay creates the RayCluster, submits
    ``infra.launch.entrypoint`` over the dashboard HTTP API, polls until
    the driver is terminal, then tears the cluster down.
    ``shutdownAfterJobFinishes=true`` by default. Pass ``--no-wait`` to
    return as soon as the RayJob is applied, ``--dry-run`` to render the
    manifest without applying.

    **Long-lived mode (``--raycluster``)** — idempotent: for each declared
    role, reuse the live RayCluster when its spec matches the rendered
    manifest, apply when it is absent, warn + reuse on drift (pass
    ``--recreate`` to delete + re-apply). Then submit daemons and the
    training entrypoint. Cluster stays up for subsequent ``nrl-k8s run``
    invocations. ``--mode interactive`` (default) uses port-forward +
    working_dir upload and tails logs; ``--mode batch`` uses kubectl exec +
    in-image code and returns as soon as the driver is running via nohup.
    """
    from . import orchestrate
    from . import submit as submit_mod

    loaded = _load_or_exit(recipe, overrides, infra_path)
    if not loaded.infra.launch.entrypoint:
        _cli_error(
            "infra.launch.entrypoint is empty",
            hint="`nrl-k8s run` requires infra.launch.entrypoint; see docs/recipes.md",
        )

    if as_rayjob:
        _run_rayjob(
            loaded,
            recipe=recipe,
            name=rayjob_name,
            shutdown_after=rayjob_shutdown,
            ttl_seconds=rayjob_ttl,
            timeout_s=rayjob_timeout,
            dry_run=dry_run,
            cli_wait=cli_wait,
        )
        return

    if dry_run:
        _cli_error(
            "--dry-run is only valid with --rayjob",
            hint="for long-lived clusters, use `nrl-k8s cluster up --role training --dry-run`",
        )

    if not submit_mod.is_in_cluster():
        _preflight_or_exit(loaded.infra.namespace)

    mode, submitter, code_src, no_wait = _resolve_mode_defaults(
        cli_mode=cli_mode,
        infra_mode=loaded.infra.launch.runMode,
        cli_submitter=cli_submitter,
        cli_code_source=cli_code_source,
        cli_wait=cli_wait,
    )
    _apply_mode_overrides(loaded, submitter=submitter, code_source=code_src, code_path=cli_code_path)
    click.echo(
        f"[run] mode={mode.value} submitter={submitter.value} "
        f"code_source={code_src.value} no_wait={no_wait} "
        f"recreate={recreate} skip_daemons={skip_daemons}",
        err=True,
    )

    try:
        result = orchestrate.run(
            loaded,
            log=click.echo,
            repo_root=repo_root.resolve(),
            replace=replace,
            run_id=cli_run_id,
            skip_daemons=skip_daemons,
            recreate=recreate,
        )
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="run failed")

    _emit_handle(result.handle)
    if not no_wait:
        _follow_handle(result.handle)


def _run_rayjob(
    loaded: LoadedConfig,
    *,
    recipe: Path,
    name: str | None,
    shutdown_after: bool,
    ttl_seconds: int,
    timeout_s: int,
    dry_run: bool,
    cli_wait: bool | None,
) -> None:
    """``nrl-k8s run --rayjob`` path. KubeRay owns the RayCluster lifecycle."""
    from . import k8s
    from . import orchestrate
    from . import submit as submit_mod
    from .rayjob import build_rayjob_manifest

    cluster = _pick_cluster_or_exit(loaded, "training")
    manifest = build_rayjob_manifest(
        cluster,
        loaded.infra,
        entrypoint=loaded.infra.launch.entrypoint,
        role="training",
        name=name,
        shutdown_after_finishes=shutdown_after,
        ttl_seconds_after_finished=ttl_seconds,
    )
    job_name = manifest["metadata"]["name"]
    namespace = loaded.infra.namespace

    if dry_run:
        click.echo(yaml.safe_dump(manifest, sort_keys=False).rstrip())
        return

    if not submit_mod.is_in_cluster():
        _preflight_or_exit(namespace)

    _check_stale_rayjobs(loaded, namespace)

    orchestrate.ensure_dra_resources("training", loaded, log=click.echo)
    click.echo(f"[run --rayjob] applying RayJob {job_name} in {namespace}")
    try:
        k8s.apply_rayjob(manifest, namespace)
    except ApiException as exc:
        _explain_and_exit(exc, context=f"rayjob {job_name} apply failed")
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context=f"rayjob {job_name} apply failed")

    job_id_cmd = f"$(kubectl get rayjob {job_name} -n {namespace} -o jsonpath='{{.status.jobId}}')"
    click.echo(
        f"follow:  kubectl get rayjob {job_name} -n {namespace} -w\n"
        f"logs:    nrl-k8s job logs {job_id_cmd} {recipe} --infra <infra> --role training -f",
    )
    # Default is wait unless user passed --no-wait.
    if cli_wait is False:
        return

    click.echo(f"[run --rayjob] waiting for {job_name} to reach a terminal state ...")

    def _on_update(deployment: str | None, job: str | None) -> None:
        click.echo(f"[run --rayjob] {job_name} deployment={deployment} job={job}")

    try:
        final = k8s.wait_for_rayjob_terminal(
            job_name, namespace, timeout_s=timeout_s, on_update=_on_update
        )
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context=f"rayjob {job_name} wait failed")

    status = final.get("status") or {}
    dep = status.get("jobDeploymentStatus")
    job_status = status.get("jobStatus")
    message = (status.get("message") or "").strip()
    click.echo(f"[run --rayjob] {job_name} finished: deployment={dep} job={job_status}")
    if message:
        click.echo(f"[run --rayjob] message: {message}")
    orchestrate.delete_dra_resources("training", loaded, log=click.echo)
    sys.exit(0 if dep == "Complete" else 1)


@main.command()
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
def status(recipe: Path, overrides: tuple[str, ...], infra_path: Path | None) -> None:
    """Summarise every cluster declared in the recipe.

    Prints, per role: RayCluster state, head pod phase, worker pod phases,
    and (if a daemon is declared) its Ray Job status.
    """
    from . import inspect as ins

    loaded = _load_or_exit(recipe, overrides, infra_path)
    rows = ins.collect_status(loaded)
    if not rows:
        click.echo("(no clusters declared in recipe)")
        return

    header = (
        f"{'ROLE':<11} {'NAME':<36} {'STATE':<9} {'HEAD':<9} {'WORKERS':<20} DAEMON"
    )
    click.echo(header)
    click.echo("-" * len(header))
    for row in rows:
        workers = ",".join(row.worker_phases) or "—"
        daemon = (
            f"{row.daemon_submission_id}={row.daemon_status or 'unknown'}"
            if row.daemon_submission_id
            else "—"
        )
        click.echo(
            f"{row.role:<11} {row.name:<36} {row.state:<9} "
            f"{(row.head_phase or '—'):<9} {workers:<20} {daemon}"
        )


@main.command()
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
@click.option(
    "--role",
    type=_ROLE_CHOICE,
    required=True,
    help="Which cluster's logs to tail.",
)
@click.option(
    "--source",
    type=click.Choice(["auto", "daemon", "head", "worker"]),
    default="auto",
    show_default=True,
    help="'auto' = daemon Ray Job if the role has one, else head pod.",
)
@click.option("-f", "--follow", is_flag=True, help="Stream new output until Ctrl+C.")
@click.option(
    "--tail",
    "tail_lines",
    type=int,
    default=200,
    show_default=True,
    help="Number of trailing lines to show before following.",
)
def logs(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    role: str,
    source: str,
    follow: bool,
    tail_lines: int,
) -> None:
    """Stream logs from a role's cluster.

    When the role has a daemon (generation / gym), ``--source auto`` shows
    the daemon's Ray Job logs via the dashboard. Otherwise it falls back
    to the head pod's container logs via kubectl.
    """
    from . import inspect as ins

    loaded = _load_or_exit(recipe, overrides, infra_path)
    cluster = _pick_cluster_or_exit(loaded, role)
    namespace = loaded.infra.namespace

    effective = source
    if effective == "auto":
        effective = "daemon" if cluster.daemon is not None else "head"

    if effective == "daemon":
        if cluster.daemon is None or not cluster.daemon.submissionId:
            _cli_error(
                f"role {role} has no daemon submissionId",
                hint=f"use --source head|worker, or declare `clusters.{role}.daemon.submissionId`",
            )
        _tail_daemon(cluster.name, namespace, cluster.daemon.submissionId)
        return

    # Pod logs — head or a worker.
    if effective == "head":
        pod_name = ins.head_pod_name(cluster.name, namespace)
    else:
        pod_name = _first_worker_pod_or_exit(cluster.name, namespace)

    for line in ins.stream_pod_logs(
        pod_name, namespace, follow=follow, tail_lines=tail_lines
    ):
        click.echo(line, nl=False)


# ---- `cluster` group ----------------------------------------------------


@main.group()
def cluster() -> None:
    """Manage long-lived RayClusters (generation, gym, training)."""


@cluster.command("up")
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
@click.option(
    "--role",
    type=_ROLE_CHOICE,
    required=True,
)
@click.option(
    "--wait/--no-wait",
    default=True,
    help="Wait for the cluster to reach state=ready before returning.",
)
@click.option(
    "--timeout",
    default=900,
    show_default=True,
    help="Seconds to wait for readiness when --wait is set.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Render the RayCluster manifest for the role and print it; do not apply.",
)
def cluster_up(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    role: str,
    wait: bool,
    timeout: int,
    dry_run: bool,
) -> None:
    """Bring up a RayCluster, then submit its daemon if the recipe has one."""
    from . import orchestrate
    from .manifest import build_raycluster_manifest

    loaded = _load_or_exit(recipe, overrides, infra_path)
    cluster_spec = _pick_cluster_or_exit(loaded, role)
    if dry_run:
        manifest = build_raycluster_manifest(cluster_spec, loaded.infra, role=role)
        click.echo(yaml.safe_dump(manifest, sort_keys=False).rstrip())
        return

    try:
        name = orchestrate.bring_up_cluster(
            role, loaded, log=click.echo, wait_ready=wait, ready_timeout_s=timeout
        )
        if wait:
            # Only submit the daemon once the cluster is ready (matches
            # the `run` flow — same code path).
            orchestrate.submit_daemon(
                role,
                loaded,
                name,
                log=click.echo,
                repo_root=Path.cwd(),
            )
    except ApiException as exc:
        if exc.status == 403:
            _cli_error(
                f"forbidden to create RayCluster in {loaded.infra.namespace}",
                hint="missing RBAC — run `nrl-k8s doctor` or ask an admin to grant the edit role.",
            )
        _explain_and_exit(exc, context=f"cluster up ({role}) failed")
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context=f"cluster up ({role}) failed")


@cluster.command("down")
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
@click.option(
    "--role",
    type=_ROLE_CHOICE,
    help="Delete the cluster for this role (uses recipe to resolve name).",
)
@click.option(
    "--name",
    "name_opt",
    help="Delete a RayCluster by name directly (overrides --role).",
)
@click.option(
    "--wait/--no-wait",
    default=True,
    help="Wait for the RayCluster object to disappear.",
)
def cluster_down(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    role: str | None,
    name_opt: str | None,
    wait: bool,
) -> None:
    """Delete a managed RayCluster by role or by name."""
    from . import k8s

    loaded = _load_or_exit(recipe, overrides, infra_path)
    namespace = loaded.infra.namespace

    if name_opt:
        target = name_opt
    elif role:
        cluster = _pick_cluster_or_exit(loaded, role)
        target = cluster.name
    else:
        _cli_error(
            "pass --role or --name",
            hint="e.g. `nrl-k8s cluster down recipe.yaml --role training`",
            exit_code=2,
        )

    click.echo(f"deleting RayCluster {target} in {namespace} ...")
    k8s.delete_raycluster(target, namespace)
    if wait:
        k8s.wait_for_raycluster_gone(target, namespace)
    click.echo(f"RayCluster {target} deleted.")

    if role:
        from . import orchestrate

        orchestrate.delete_dra_resources(role, loaded, log=click.echo)


@cluster.command("list")
@click.option(
    "--namespace",
    "-n",
    default=None,
    help="Kubernetes namespace to list. Defaults to the current kube context's namespace.",
)
def cluster_list(namespace: str | None) -> None:
    """List RayClusters in a namespace and their state."""
    from . import k8s
    from .config import _infer_kube_namespace

    ns = namespace or _infer_kube_namespace()
    rows = k8s.list_rayclusters(ns)
    if not rows:
        click.echo(f"(no RayClusters in {ns})")
        return
    for obj in rows:
        name = obj["metadata"]["name"]
        state = obj.get("status", {}).get("state", "—")
        click.echo(f"{name}\t{state}")


@cluster.command("dashboard")
@click.argument("name")
@click.option(
    "--namespace",
    "-n",
    default=None,
    help="Kubernetes namespace. Defaults to the current kube context's namespace.",
)
@click.option(
    "--port",
    "local_port",
    type=int,
    default=8265,
    show_default=True,
    help="Local port to bind the forward to.",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    show_default=True,
    help="Open the dashboard URL in a browser once the forward is up.",
)
@click.option(
    "--fix/--no-fix",
    "auto_fix",
    default=True,
    show_default=True,
    help="If Ray's dashboard static assets are symlinks on the head pod "
    "(uv install default), reinstall ray[default] with --link-mode=copy "
    "before forwarding. Pass --no-fix on images already built with "
    "UV_LINK_MODE=copy.",
)
def cluster_dashboard(
    name: str,
    namespace: str | None,
    local_port: int,
    open_browser: bool,
    auto_fix: bool,
) -> None:
    """Port-forward a RayCluster's dashboard (and fix it if blank).

    ``NAME`` is the RayCluster name (as shown by ``nrl-k8s cluster list``
    or ``kubectl get rayclusters``). No recipe / infra YAML required.

    Does everything in one go:

    1. Resolve the head pod for ``NAME``.
    2. If ``--fix`` (default): check for symlinked dashboard assets and,
       if any are present, ``uv pip install --reinstall --link-mode=copy
       ray[default]`` on the head pod so aiohttp can actually serve the
       JS/CSS (the assets are otherwise 404 → blank page).
    3. ``kubectl port-forward svc/<cluster>-head-svc <port>:8265``.
    4. Open ``http://localhost:<port>`` in the default browser.
    5. Ctrl+C kills the forward; the cluster keeps running.

    The permanent fix is in the image build — ``ENV UV_LINK_MODE=copy``
    before the first ``uv pip install`` step in your Dockerfile. The
    auto-fix here is a convenience for images without that flag.
    """
    import time
    import webbrowser
    from . import submit as submit_mod
    from .config import _infer_kube_namespace

    ns = namespace or _infer_kube_namespace()
    if not submit_mod.is_in_cluster():
        _preflight_or_exit(ns)

    if auto_fix:
        _reinstall_ray_if_symlinked(name, ns)

    url = f"http://localhost:{local_port}"
    pf = submit_mod._PortForward(name, ns, local_port)
    click.echo(f"[dashboard] forwarding {name} head :8265 → {url}")
    try:
        pf.start()
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="dashboard port-forward failed")
    if open_browser:
        webbrowser.open(url)
    click.echo("[dashboard] Ctrl+C to stop.")
    try:
        while pf.alive():
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\n[dashboard] stopping forward.")
    finally:
        pf.stop()


def _reinstall_ray_if_symlinked(cluster_name: str, namespace: str) -> None:
    """Reinstall ray[default] in copy mode when its assets are symlinks.

    Checks for any symlink under Ray's dashboard build dir and, if one
    exists, runs ``uv pip install --reinstall --link-mode=copy
    ray[default]==<current>`` to replace every symlink in the package
    with a real file. Idempotent: no-op when the dir has no symlinks.
    """
    import subprocess as _sp

    from . import k8s

    pod = k8s.get_head_pod(cluster_name, namespace)
    script = (
        "set -eu\n"
        "BUILD=$(python3 -c 'import ray, os; "
        "print(os.path.join(os.path.dirname(ray.__file__),"
        '"dashboard/client/build"))\' 2>/dev/null)\n'
        'if [ -z "$BUILD" ] || [ ! -d "$BUILD" ]; then\n'
        '  echo "[dashboard] ray install not found on pod; skipping fix"\n'
        "  exit 0\n"
        "fi\n"
        'if ! find "$BUILD" -type l -print -quit 2>/dev/null | grep -q .; then\n'
        '  echo "[dashboard] assets already real files; no fix needed"\n'
        "  exit 0\n"
        "fi\n"
        "VER=$(python3 -c 'import ray; print(ray.__version__)')\n"
        'echo "[dashboard] reinstalling ray[default]==$VER with --link-mode=copy (~30s) ..."\n'
        "UV=$(command -v uv || echo /opt/nemo_rl_venv/bin/uv)\n"
        '"$UV" pip install --reinstall --link-mode=copy --quiet "ray[default]==$VER"\n'
        'echo "[dashboard] reinstall complete."\n'
    )
    cmd = [
        "kubectl", "exec", "-n", namespace, pod.metadata.name, "--",
        "bash", "-c", script,
    ]
    try:
        res = _sp.run(cmd, check=False, capture_output=True, text=True, timeout=180)
    except (_sp.TimeoutExpired, FileNotFoundError) as exc:
        click.echo(f"[dashboard] fix skipped: {exc}", err=True)
        return
    for line in (res.stdout or "").splitlines():
        click.echo(line)
    if res.returncode != 0 and (res.stderr or "").strip():
        click.echo(f"[dashboard] stderr: {res.stderr.strip()}", err=True)


# ---- `job` group --------------------------------------------------------


@main.group()
def job() -> None:
    """Inspect and control Ray jobs on managed clusters."""


@job.command("list")
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
@click.option(
    "--role",
    type=_ROLE_CHOICE,
    required=True,
    help="Which cluster's Ray jobs to list.",
)
def job_list(
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    role: str,
) -> None:
    """List Ray Jobs currently registered on a role's RayCluster."""
    from ray.job_submission import JobSubmissionClient

    from . import submit

    loaded = _load_or_exit(recipe, overrides, infra_path)
    cluster = _pick_cluster_or_exit(loaded, role)
    namespace = loaded.infra.namespace

    try:
        with submit.dashboard_url(cluster.name, namespace) as dash:
            clnt = JobSubmissionClient(dash)
            jobs = clnt.list_jobs()
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="list jobs failed")

    if not jobs:
        click.echo(f"(no Ray jobs on {cluster.name})")
        return
    click.echo(f"{'SUBMISSION':<40} {'STATUS':<12} ENTRYPOINT")
    for j in jobs:
        entry = (j.entrypoint or "").splitlines()[0][:80]
        click.echo(f"{j.submission_id:<40} {j.status.value:<12} {entry}")


@job.command("logs")
@click.argument("submission_id")
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
@click.option(
    "--role",
    type=_ROLE_CHOICE,
    required=True,
    help="Which cluster hosts the job.",
)
@click.option(
    "-f", "--follow", is_flag=True, help="Stream new output until Ctrl+C."
)
def job_logs(
    submission_id: str,
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    role: str,
    follow: bool,
) -> None:
    """Stream logs for a submitted run by its id on a given role's cluster.

    Dispatches on the cached handle (``~/.cache/nrl-k8s/runs/<id>.json``):
    port-forward handles go through Ray's log tail API; exec handles go
    through ``kubectl exec … tail -F`` on the head pod's stdout file.

    When no cached handle exists we fall back to the Ray dashboard — so
    this command keeps working against jobs submitted by older CLI
    versions or by ``ray job submit`` directly.
    """
    del follow  # always follows — flag kept for back-compat / readability
    from .submitters import load_handle

    loaded = _load_or_exit(recipe, overrides, infra_path)
    cluster = _pick_cluster_or_exit(loaded, role)

    handle = load_handle(submission_id)
    if handle is not None and handle.kind == "exec":
        _follow_handle(handle)
        return
    _tail_daemon(cluster.name, loaded.infra.namespace, submission_id)


@job.command("stop")
@click.argument("submission_id")
@click.argument("recipe", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
@_INFRA_OPTION
@click.option(
    "--role",
    type=_ROLE_CHOICE,
    required=True,
    help="Which cluster hosts the job.",
)
@click.option(
    "--force", is_flag=True, help="Exec mode only: send SIGKILL instead of SIGTERM."
)
def job_stop(
    submission_id: str,
    recipe: Path,
    overrides: tuple[str, ...],
    infra_path: Path | None,
    role: str,
    force: bool,
) -> None:
    """Stop a submitted run by id.

    Transport-aware via the cached handle — Ray jobs go through
    ``stop_job``; exec runs are killed with SIGTERM (or SIGKILL with
    ``--force``). Falls back to Ray's API when no cached handle exists.
    """
    from .submitters import load_handle

    loaded = _load_or_exit(recipe, overrides, infra_path)
    cluster = _pick_cluster_or_exit(loaded, role)

    handle = load_handle(submission_id)
    if handle is not None and handle.kind == "exec":
        from .submitters.exec_ import ExecSubmitter

        tmp_root = (handle.tmp_dir or "/tmp/nrl-x").rsplit("/", 1)[0] or "/tmp"
        try:
            ExecSubmitter(exec_tmp_dir=tmp_root).stop(handle, force=force)
        except Exception as exc:  # noqa: BLE001
            _explain_and_exit(exc, context=f"stop {submission_id} failed")
        click.echo(f"stopped {submission_id} (exec)")
        return

    from ray.job_submission import JobSubmissionClient

    from . import submit

    try:
        with submit.dashboard_url(cluster.name, loaded.infra.namespace) as dash:
            clnt = JobSubmissionClient(dash)
            clnt.stop_job(submission_id)
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context=f"stop {submission_id} failed")
    click.echo(f"stopped {submission_id}")


# =============================================================================
# Dev pod
# =============================================================================


@main.group()
def dev():
    """Manage a lightweight dev pod on the cluster."""


@dev.command("connect")
@click.option(
    "--image",
    default="nvcr.io/nvidian/nemo-rl:nightly",
    help="Container image for the dev pod.",
)
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
def dev_connect(image: str, namespace: str | None) -> None:
    """Create a dev pod (if needed) and exec into it."""
    import subprocess
    import time

    from . import k8s
    from .config import get_username
    from .dev import build_dev_pod_manifest

    user = get_username()
    pod_name = f"{user}-dev-pod"
    if namespace is None:
        namespace = _infer_namespace()

    _check_dev_pod_rbac(namespace)

    phase = k8s.get_pod_phase(pod_name, namespace)
    if phase is None:
        click.echo(f"creating dev pod {pod_name} in {namespace} ...")
        manifest = build_dev_pod_manifest(user, namespace, image)
        k8s.create_pod(manifest, namespace)
        phase = "Pending"
    else:
        running_image = k8s.get_pod_image(pod_name, namespace)
        if running_image and running_image != image:
            click.echo(
                f"warning: dev pod is using image {running_image}, "
                f"not {image} — stop and reconnect to switch",
                err=True,
            )

    if phase != "Running":
        click.echo(f"waiting for {pod_name} to be Running ...")
        for _ in range(120):
            time.sleep(2)
            phase = k8s.get_pod_phase(pod_name, namespace)
            if phase == "Running":
                break
            if phase in ("Failed", "Succeeded"):
                _cli_error(f"dev pod reached phase {phase} — check `kubectl describe pod {pod_name} -n {namespace}`")
        else:
            _cli_error(f"dev pod did not reach Running after 240s (phase={phase})")

    click.echo(f"connecting to {pod_name} ...")
    subprocess.run(
        ["kubectl", "exec", "-it", "-n", namespace, pod_name, "--", "bash"],
    )
    click.echo(f"\npod {pod_name} is still running — stop with: nrl-k8s dev stop")


@dev.command("stop")
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
def dev_stop(namespace: str | None) -> None:
    """Delete your dev pod."""
    from . import k8s
    from .config import get_username

    user = get_username()
    pod_name = f"{user}-dev-pod"
    if namespace is None:
        namespace = _infer_namespace()

    phase = k8s.get_pod_phase(pod_name, namespace)
    if phase is None:
        click.echo(f"no dev pod {pod_name} found in {namespace}")
        return

    click.echo(f"deleting {pod_name} ...")
    k8s.delete_pod(pod_name, namespace)
    click.echo(f"{pod_name} deleted.")


_REQUIRED_FIRST_TIME = ("HF_TOKEN", "WANDB_API_KEY")


@dev.command("setup-secrets")
@click.argument("kvs", nargs=-1)
@click.option(
    "--ssh-key",
    type=click.Path(exists=True),
    help="Path to an SSH private key.",
)
@click.option(
    "--add-rclone",
    is_flag=True,
    help="Read ~/.config/rclone/rclone.conf and store it in the secret.",
)
@click.option("--namespace", "-n", default=None, help="Kubernetes namespace.")
def dev_setup_secrets(
    kvs: tuple[str, ...],
    ssh_key: str | None,
    add_rclone: bool,
    namespace: str | None,
) -> None:
    """Create or update your user secrets.

    Pass token values as NAME=VAL positional args and SSH keys via --ssh-key.

    First-time usage requires HF_TOKEN, WANDB_API_KEY, and --ssh-key:

    \b
      nrl-k8s dev setup-secrets \\
        HF_TOKEN=hf_xxx WANDB_API_KEY=key_yyy \\
        --ssh-key ~/.ssh/id_ed25519 --add-rclone

    Subsequent runs accept any subset to update individual keys.
    """
    from pathlib import Path

    from . import k8s
    from .config import get_username

    user = get_username()
    secret_name = f"{user}-secrets"
    if namespace is None:
        namespace = _infer_namespace()

    data: dict[str, str] = {}
    for kv in kvs:
        if "=" not in kv:
            _cli_error(f"invalid argument {kv!r} — expected NAME=VAL")
        name, val = kv.split("=", 1)
        data[name] = val

    if ssh_key:
        p = Path(ssh_key)
        data["SSH_KEY_NAME"] = p.name
        data["SSH_KEY_CONTENT"] = p.read_text()

    if add_rclone:
        rclone_conf = Path.home() / ".config" / "rclone" / "rclone.conf"
        if not rclone_conf.exists():
            _cli_error(
                f"rclone config not found at {rclone_conf}",
                hint="install rclone and run `rclone config` first",
            )
        data["RCLONE_CONF"] = rclone_conf.read_text()

    is_new = not k8s.secret_exists(secret_name, namespace)
    if is_new:
        missing = [k for k in _REQUIRED_FIRST_TIME if k not in data]
        if missing:
            _cli_error(
                f"first-time setup requires: {', '.join(missing)}",
                hint=f"nrl-k8s dev setup-secrets {' '.join(f'{k}=<value>' for k in missing)} --ssh-key ~/.ssh/id_ed25519",
            )
        if not ssh_key:
            _cli_error(
                "first-time setup requires --ssh-key",
                hint="nrl-k8s dev setup-secrets ... --ssh-key ~/.ssh/id_ed25519",
            )

    k8s.create_or_update_secret(secret_name, namespace, data)
    action = "created" if is_new else "updated"
    click.echo(f"{action} secret {secret_name} in {namespace} (keys: {', '.join(sorted(data))})")


def _infer_namespace() -> str:
    from .config import _infer_kube_namespace

    return _infer_kube_namespace()


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class _StaleResource:
    kind: str  # e.g. "rayjob", "raycluster", "pod"
    name: str
    status: str


def _find_stale_resources(
    checks: list[tuple[str, str, callable]],
) -> list[_StaleResource]:
    """Probe a list of ``(kind, name, getter)`` and return those that exist.

    ``getter(name)`` should return a status string if the resource exists,
    or ``None`` if it doesn't. Generic so callers can check any resource type.
    """
    stale: list[_StaleResource] = []
    for kind, name, getter in checks:
        status = getter(name)
        if status is not None:
            stale.append(_StaleResource(kind=kind, name=name, status=status))
    return stale


def _error_on_stale(stale: list[_StaleResource], namespace: str) -> None:
    """If ``stale`` is non-empty, print all resources and their delete commands, then exit."""
    if not stale:
        return

    lines = ["stale resources from a previous run exist:\n"]
    for r in stale:
        lines.append(f"  {r.kind}/{r.name} (status={r.status})")
    lines.append("\ndelete them and resubmit:")
    for r in stale:
        lines.append(f"  kubectl delete {r.kind} {r.name} -n {namespace}")

    _cli_error(
        "\n".join(lines),
        hint="once deleted, re-run the same nrl-k8s run command",
    )


def _check_stale_rayjobs(loaded: LoadedConfig, namespace: str) -> None:
    """Check all roles for existing RayJobs upfront.

    Reports every stale RayJob at once so the user can clean up in one pass
    rather than hitting them one at a time in a loop.

    Stale DRA resources (ComputeDomain, RoCE ResourceClaimTemplate) are not
    checked — they are 1:1 with the RayJob by design (named after the
    cluster + role), so deleting the RayJob and resubmitting will recreate
    them idempotently. TODO: add DRA garbage collection to a future
    ``nrl-k8s clean`` command.
    """
    from . import k8s
    from .orchestrate import ALL_ROLES, _get_cluster

    def _rayjob_status(name: str) -> str | None:
        existing = k8s.get_rayjob(name, namespace)
        if existing is None:
            return None
        return (existing.get("status") or {}).get("jobDeploymentStatus", "Pending")

    checks = []
    for role in ALL_ROLES:
        cluster = _get_cluster(loaded.infra, role)
        if cluster is None:
            continue
        checks.append(("rayjob", cluster.name, _rayjob_status))

    _error_on_stale(_find_stale_resources(checks), namespace)


def _check_dev_pod_rbac(namespace: str) -> None:
    """Verify the default SA has edit access so kubectl works inside the dev pod."""
    import subprocess

    sa = f"system:serviceaccount:{namespace}:default"
    result = subprocess.run(
        ["kubectl", "auth", "can-i", "get", "pods",
         f"--as={sa}", "-n", namespace],
        capture_output=True, text=True,
    )
    if result.stdout.strip() == "yes":
        return
    rolebinding = (
        f"apiVersion: rbac.authorization.k8s.io/v1\n"
        f"kind: RoleBinding\n"
        f"metadata:\n"
        f"  name: default-sa-edit\n"
        f"  namespace: {namespace}\n"
        f"subjects:\n"
        f"  - kind: ServiceAccount\n"
        f"    name: default\n"
        f"    namespace: {namespace}\n"
        f"roleRef:\n"
        f"  kind: ClusterRole\n"
        f"  name: edit\n"
        f"  apiGroup: rbac.authorization.k8s.io"
    )
    _cli_error(
        f"the default service account in {namespace} lacks edit permissions — "
        f"kubectl won't work inside the dev pod",
        hint=f"apply this RoleBinding, then retry:\n\n{rolebinding}",
    )


def _preflight_or_exit(namespace: str) -> None:
    """Fail fast when kubectl is missing or RBAC is wrong — before we spawn anything."""
    from . import submit

    try:
        submit.kubectl_preflight(namespace)
    except RuntimeError as exc:
        _cli_error(str(exc), hint="see `nrl-k8s doctor` for cluster access checks")


def _cli_error(msg: str, *, hint: str | None = None, exit_code: int = 1) -> NoReturn:
    """Emit a stderr error with an optional actionable hint, then exit."""
    click.echo(f"error: {msg}", err=True)
    if hint:
        click.echo(f"hint: {hint}", err=True)
    sys.exit(exit_code)


def _explain_and_exit(exc: BaseException, *, context: str) -> NoReturn:
    """Map common exceptions to an actionable hint before exiting."""
    hint: str | None = None
    if isinstance(exc, ApiException):
        if exc.status == 403:
            hint = (
                "missing RBAC for this action; run `nrl-k8s doctor` or ask an "
                "admin to grant the edit role on the namespace."
            )
        elif exc.status == 401:
            hint = "kubectl credentials rejected; try `aws sso login`."
        elif exc.status in (500, 502, 503, 504):
            hint = "control-plane 5xx — retry in a few seconds."
    elif isinstance(exc, ConnectionRefusedError):
        hint = (
            "connection refused — kubectl port-forward to the dashboard failed; "
            "is kubectl authenticated? (try `aws sso login`)"
        )
    elif isinstance(exc, ValueError) and "launch.entrypoint" in str(exc):
        hint = "set infra.launch.entrypoint in your recipe; see docs/recipes.md."
    _cli_error(f"{context}: {exc}", hint=hint)


def _load_or_exit(
    recipe: Path, overrides: tuple[str, ...], infra_path: Path | None = None
) -> LoadedConfig:
    try:
        return load_recipe_with_infra(
            recipe, overrides=list(overrides), infra_path=infra_path
        )
    except Exception as exc:  # noqa: BLE001
        _explain_and_exit(exc, context="failed to load recipe")


def _pick_cluster_or_exit(loaded: LoadedConfig, role: str) -> ClusterSpec:
    cluster = getattr(loaded.infra.clusters, role)
    if cluster is None:
        _cli_error(
            f"infra.clusters.{role} is not defined in {loaded.source_path}",
            hint=f"declare a `clusters.{role}` block in the recipe or pass a different --role",
        )
    return cluster


def _tail(dashboard: str, job_id: str) -> None:
    """Stream Ray Job logs to stdout until terminal or Ctrl+C."""
    from . import submit as submit_mod

    try:
        for line in submit_mod.tail_job_logs(dashboard, job_id):
            click.echo(line, nl=False)
    except KeyboardInterrupt:
        click.echo("\n(interrupted — job continues running)", err=True)


def _tail_daemon(cluster_name: str, namespace: str, submission_id: str) -> None:
    """Open a dashboard port-forward and tail a Ray Job by submission_id."""
    from . import inspect as ins
    from . import submit as submit_mod

    try:
        with submit_mod.dashboard_url(cluster_name, namespace) as dash:
            click.echo(f"# tailing {submission_id} via {dash}", err=True)
            _tail(dash, submission_id)
    except KeyboardInterrupt:
        click.echo("\n(interrupted — job continues running)", err=True)
    except Exception as exc:  # noqa: BLE001
        hint = _diagnose_port_forward_failure(cluster_name, namespace)
        if hint:
            _cli_error(f"tailing {cluster_name} failed: {exc}", hint=hint)
        _explain_and_exit(exc, context=f"tailing {submission_id} failed")


def _diagnose_port_forward_failure(cluster_name: str, namespace: str) -> str | None:
    """Check head pod state to produce a more helpful error message."""
    from . import k8s

    try:
        head_pod = f"{cluster_name}-head"
        pods = k8s.list_pods_by_label(
            f"ray.io/cluster={cluster_name},ray.io/node-type=head", namespace
        )
        if not pods:
            return (
                f"no head pod found for {cluster_name} — "
                f"the cluster may still be provisioning. "
                f"check: kubectl get pods -l ray.io/cluster={cluster_name} -n {namespace}"
            )
        pod = pods[0]
        phase = pod.status.phase if pod.status else "Unknown"
        if phase != "Running":
            return (
                f"head pod is {phase}, not Running yet — "
                f"wait for the cluster to be ready and retry. "
                f"check: kubectl get pods -l ray.io/cluster={cluster_name} -n {namespace}"
            )
    except Exception:
        pass
    return None


def _emit_handle(handle) -> None:  # type: ignore[no-untyped-def]
    """Print the resolved handle + next-step commands to stdout.

    Kept close to the submit call sites so the user sees a coherent
    "here's what you submitted, here's how to follow it" block in both
    interactive and batch flows.
    """
    click.echo(f"run id:  {handle.run_id}")
    click.echo(f"kind:    {handle.kind}")
    click.echo(f"cluster: {handle.cluster_name}  (ns={handle.namespace})")
    if handle.kind == "exec":
        click.echo(f"pod:     {handle.pod}")
        click.echo(f"tmp:     {handle.tmp_dir}")
    click.echo(
        f"follow:  nrl-k8s job logs {handle.run_id} "
        f"<recipe> --role training -f"
    )
    click.echo(
        f"stop:    nrl-k8s job stop {handle.run_id} "
        f"<recipe> --role training"
    )


def _follow_handle(handle) -> None:  # type: ignore[no-untyped-def]
    """Stream logs for a handle using whichever transport submitted it."""
    from .submitters import build_submitter
    from .schema import SubmitterMode

    class _Stub:  # minimal infra shim so build_submitter picks the right transport
        class submit:
            submitter = SubmitterMode.EXEC if handle.kind == "exec" else SubmitterMode.PORT_FORWARD
            execTmpDir = handle.tmp_dir.rsplit("/", 1)[0] if (handle.kind == "exec" and handle.tmp_dir) else "/tmp"

    submitter = build_submitter(_Stub)  # type: ignore[arg-type]
    try:
        for line in submitter.follow(handle):
            click.echo(line, nl=False)
    except KeyboardInterrupt:
        click.echo("\n(interrupted — run continues)", err=True)


def _first_worker_pod_or_exit(cluster_name: str, namespace: str) -> str:
    from . import inspect as ins

    pods = ins.list_cluster_pods(cluster_name, namespace)
    if not pods.worker_names:
        _cli_error(
            f"no worker pods for {cluster_name} in {namespace}",
            hint="is the RayCluster still scheduling? check `nrl-k8s status` first.",
        )
    return pods.worker_names[0]


__all__ = ["main"]
