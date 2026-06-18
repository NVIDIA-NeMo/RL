import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INSTANCE_FILE = WORKSPACE_ROOT / "r2e_gym_instance_ids.txt"
DEFAULT_CONTEXTS_DIR = WORKSPACE_ROOT / ".arm64_build_contexts"
DEFAULT_REPO_CACHE_DIR = WORKSPACE_ROOT / ".arm64_repo_cache"
DEFAULT_STATE_FILE = WORKSPACE_ROOT / "arm64_build_state.json"
DEFAULT_LOGS_DIR = WORKSPACE_ROOT / "logs" / "build_images" / "instances"
DEFAULT_HF_DATASET = "R2E-Gym/R2E-Gym-Lite"


REPO_ALIASES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "datalad": "datalad",
    "scrapy": "scrapy",
    "aiohttp": "aiohttp",
    "tornado": "tornado",
    "pillow": "pillow",
    "orange3": "orange3",
    "sympy": "sympy",
    "pyramid": "pyramid",
    "coveragepy": "coveragepy",
    "bokeh": "bokeh",
}

DEFAULT_TESTS_CMD = {
    "sympy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "pandas": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "pillow": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "scrapy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "pyramid": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "datalad": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "aiohttp": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "coveragepy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "numpy": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "orange3": "QT_QPA_PLATFORM=minimal PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' xvfb-run --auto-servernum .venv/bin/python -W ignore -m pytest -rA r2e_tests",
    "bokeh": "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' .venv/bin/python -W ignore -m pytest -rA r2e_tests",
}


@dataclass
class InstanceRef:
    instance_id: str
    owner: str
    repo_raw: str
    repo_key: str
    commit_hash: str


@dataclass
class PreparedInstance:
    ref: InstanceRef
    old_commit_hash: str
    test_file_codes: list[str]
    test_file_names: list[str]
    tests_cmd: str
    image_tag: str
    env_group_key: str
    repo_dir: Optional[Path] = None


class LoggedCommandError(RuntimeError):
    def __init__(self, message: str, log_file: Path):
        super().__init__(message)
        self.log_file = log_file

    def __str__(self) -> str:
        return f"{super().__str__()} (see log: {self.log_file})"


def load_state(state_file: Path) -> dict:
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"built": [], "pushed": [], "failed": []}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=state_file.parent, suffix=".tmp", prefix=state_file.stem
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.replace(tmp_path, state_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def run_cmd(cmd: list[str], cwd: Optional[Path] = None) -> str:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def run_logged_cmd(
    cmd: list[str],
    log_file: Path,
    cwd: Optional[Path] = None,
    quiet: bool = False,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"$ {' '.join(shlex.quote(part) for part in cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if not quiet:
                print(line, end="")
            lf.write(line)
        proc.wait()
        lf.write(f"\n[exit_code] {proc.returncode}\n\n")
    if proc.returncode != 0:
        raise LoggedCommandError(
            f"Command failed with exit code {proc.returncode}",
            log_file,
        )


def parse_instance_id(instance_id: str) -> InstanceRef:
    cleaned = instance_id.strip()
    if not cleaned:
        raise ValueError("Empty instance id")
    owner_repo, commit_hash = cleaned.rsplit("-", 1)
    owner, repo_raw = owner_repo.split("__", 1)
    repo_key_raw = repo_raw.lower()
    if repo_key_raw not in REPO_ALIASES:
        raise ValueError(f"Unsupported repository '{repo_raw}' in instance id '{cleaned}'")
    return InstanceRef(
        instance_id=cleaned,
        owner=owner,
        repo_raw=repo_raw,
        repo_key=REPO_ALIASES[repo_key_raw],
        commit_hash=commit_hash,
    )


def load_instance_ids(
    instance_file: Path, start: int, limit: Optional[int]
) -> list[InstanceRef]:
    refs: list[InstanceRef] = []
    with open(instance_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            refs.append(parse_instance_id(line))
    refs = refs[start:]
    if limit is not None:
        refs = refs[:limit]
    return refs


def ensure_repo_checkout(ref: InstanceRef, repo_cache_dir: Path) -> Path:
    repo_dir = repo_cache_dir / f"{ref.owner}__{ref.repo_raw}"
    remote_url = f"https://github.com/{ref.owner}/{ref.repo_raw}.git"
    repo_cache_dir.mkdir(parents=True, exist_ok=True)

    lock_file = repo_cache_dir / f"{ref.owner}__{ref.repo_raw}.lock"
    with open(lock_file, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if not repo_dir.exists():
                run_cmd(
                    ["git", "clone", "--filter=blob:none", remote_url, repo_dir.as_posix()]
                )
            else:
                run_cmd(["git", "fetch", "--all", "--tags"], cwd=repo_dir)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return repo_dir


def resolve_old_commit(
    ref: InstanceRef,
    repo_dir: Path,
    workspace_root: Path,
) -> str:
    # Prefer local parsed commit data when available.
    commit_json = workspace_root / "commit_data" / ref.repo_key / f"{ref.commit_hash}.json"
    if commit_json.exists():
        with open(commit_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        old_hash = data.get("old_commit_hash")
        if old_hash:
            return old_hash

    # Fallback: first parent of the tagged commit.
    run_cmd(["git", "cat-file", "-e", f"{ref.commit_hash}^{{commit}}"], cwd=repo_dir)
    return run_cmd(["git", "rev-parse", f"{ref.commit_hash}^"], cwd=repo_dir)


def load_hf_test_index(
    dataset_name: str,
    split: str = "train",
) -> dict[tuple[str, str], dict]:
    """Load the HuggingFace dataset and build a (repo_key, commit_hash) -> test data index."""
    from datasets import load_dataset as hf_load_dataset

    print(f"Loading HuggingFace dataset '{dataset_name}' (split={split}) ...")
    ds = hf_load_dataset(dataset_name, split=split)
    index: dict[tuple[str, str], dict] = {}
    for row in ds:
        repo_name = row.get("repo_name", "")
        commit_hash = row.get("commit_hash", "")
        repo_key = REPO_ALIASES.get(repo_name.lower(), repo_name.lower())
        exec_str = row.get("execution_result_content", "")
        if not exec_str:
            continue
        try:
            exec_result = json.loads(exec_str)
        except json.JSONDecodeError:
            continue
        codes = exec_result.get("test_file_codes", []) or []
        names = exec_result.get("test_file_names", []) or []
        if codes:
            index[(repo_key, commit_hash)] = {
                "test_file_codes": codes,
                "test_file_names": names,
            }
    print(f"  indexed {len(index)} instances with test data from dataset")
    return index


def load_test_assets(
    ref: InstanceRef,
    workspace_root: Path,
    hf_index: dict[tuple[str, str], dict] | None = None,
) -> tuple[list[str], list[str], str]:
    test_codes: list[str] = []
    test_names: list[str] = []

    # 1. Try local test_data/ files first.
    test_json = workspace_root / "test_data" / ref.repo_key / f"{ref.commit_hash}.json"
    if test_json.exists():
        with open(test_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        test_codes = data.get("test_file_codes", []) or []
        test_names = data.get("test_file_names", []) or []

    # 2. Fall back to HuggingFace dataset index.
    if not test_codes and hf_index is not None:
        entry = hf_index.get((ref.repo_key, ref.commit_hash))
        if entry:
            test_codes = entry["test_file_codes"]
            test_names = entry["test_file_names"]

    # Keep parity with the extraction pipeline for custom runners.
    if ref.repo_key == "tornado":
        tests_cmd = ".venv/bin/python -W ignore r2e_tests/tornado_unittest_runner.py"
    elif ref.repo_key == "pillow" and any("unittest" in code for code in test_codes):
        tests_cmd = ".venv/bin/python -W ignore r2e_tests/unittest_custom_runner.py"
    else:
        tests_cmd = DEFAULT_TESTS_CMD[ref.repo_key]

    if not test_codes:
        raise ValueError(f"No test data found for {ref.instance_id} (checked local test_data/ and HF dataset)")

    return test_codes, test_names, tests_cmd


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def prepare_aiohttp_makefile(repo_dir: Path, old_commit_hash: str, dest: Path) -> None:
    """Checkout the Makefile from the repo at old_commit_hash and apply pip->uv rewrite."""
    try:
        content = run_cmd(
            ["git", "show", f"{old_commit_hash}:Makefile"], cwd=repo_dir
        )
    except subprocess.CalledProcessError:
        content = "# placeholder – original Makefile not found at this commit\n"

    content = content.replace("python -m pip install", "pip install")
    content = content.replace("pip install", "uv pip install")
    dest.write_text(content, encoding="utf-8")


def prepare_context(
    ref: InstanceRef,
    old_commit_hash: str,
    test_file_codes: list[str],
    test_file_names: list[str],
    tests_cmd: str,
    contexts_dir: Path,
    workspace_root: Path,
    repo_dir: Optional[Path] = None,
) -> Path:
    context_dir = contexts_dir / ref.instance_id
    tests_dir = context_dir / "r2e_tests"
    if tests_dir.exists():
        shutil.rmtree(tests_dir)
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")

    dockerfile_src = (
        workspace_root
        / "src"
        / "r2egym"
        / "repo_analysis"
        / "base_dockerfiles"
        / f"Dockerfile.{ref.repo_key}"
    )
    install_src = (
        workspace_root / "src" / "r2egym" / "install_utils" / f"{ref.repo_key}_install.sh"
    )
    if not dockerfile_src.exists():
        raise FileNotFoundError(f"Missing dockerfile template: {dockerfile_src}")
    if not install_src.exists():
        raise FileNotFoundError(f"Missing install script: {install_src}")

    (context_dir / "Dockerfile").write_text(
        dockerfile_src.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (context_dir / "install.sh").write_text(
        install_src.read_text(encoding="utf-8"), encoding="utf-8"
    )
    run_tests_content = (
        "#!/usr/bin/env bash\n"
        "set -euxo pipefail\n"
        "cd /testbed\n"
        f"{tests_cmd}\n"
    )
    (context_dir / "run_tests.sh").write_text(run_tests_content, encoding="utf-8")

    # Needed for custom runner commands.
    copy_if_exists(
        workspace_root / "src" / "r2egym" / "install_utils" / "unittest_custom_runner.py",
        tests_dir / "unittest_custom_runner.py",
    )
    copy_if_exists(
        workspace_root / "src" / "r2egym" / "install_utils" / "tornado_unittest_runner.py",
        tests_dir / "tornado_unittest_runner.py",
    )

    # aiohttp needs its Makefile (with pip->uv rewrite) and asyncio compat script.
    if ref.repo_key == "aiohttp":
        copy_if_exists(
            workspace_root / "src" / "r2egym" / "install_utils" / "process_aiohttp_updateasyncio.py",
            context_dir / "process_aiohttp_updateasyncio.py",
        )
        if repo_dir is not None:
            prepare_aiohttp_makefile(repo_dir, old_commit_hash, context_dir / "Makefile")

    # datalad needs its conftest.py for fixtures.
    if ref.repo_key == "datalad":
        conftest_src = (
            workspace_root / "src" / "r2egym" / "install_utils" / "datalads_conftest.py"
        )
        if conftest_src.exists():
            has_conftest = any(n == "conftest.py" for n in test_file_names)
            if not has_conftest:
                (tests_dir / "conftest.py").write_text(
                    conftest_src.read_text(encoding="utf-8"), encoding="utf-8"
                )

    # Pillow tests import from .helper; extract helper.py from the repo checkout.
    if ref.repo_key == "pillow" and repo_dir is not None:
        try:
            helper_content = run_cmd(
                ["git", "show", f"{old_commit_hash}:Tests/helper.py"], cwd=repo_dir
            )
            (tests_dir / "helper.py").write_text(helper_content, encoding="utf-8")
        except subprocess.CalledProcessError:
            pass

    for idx, (code, name) in enumerate(zip(test_file_codes, test_file_names)):
        safe_name = name if name.endswith(".py") else f"test_{idx + 1}.py"
        (tests_dir / safe_name).write_text(code, encoding="utf-8")

    (context_dir / "old_commit.txt").write_text(old_commit_hash + "\n", encoding="utf-8")
    return context_dir


def make_registry_tag(registry: str, local_tag: str) -> str:
    """registry:instance_id from local prefix:instance_id."""
    if ":" not in local_tag:
        raise ValueError(f"Expected image tag with ':': {local_tag}")
    _name, tag = local_tag.rsplit(":", 1)
    return f"{registry.rstrip('/')}:{tag}"


def make_image_tag(tag_prefix: str, instance_id: str) -> str:
    # Keep the full instance id as the Docker tag.
    return f"{tag_prefix}:{instance_id}"


def make_env_group_key(ref: InstanceRef, old_commit_hash: str, tests_cmd: str) -> str:
    # Group instances with matching expensive Docker build layers.
    tests_cmd_digest = hashlib.sha1(tests_cmd.encode("utf-8")).hexdigest()[:10]
    return (
        f"{ref.owner}__{ref.repo_raw}"
        f"|old={old_commit_hash}"
        f"|tests_cmd={tests_cmd_digest}"
    )


def push_image(
    local_tag: str, registry_tag: str, log_dir: Path, quiet: bool = False
) -> None:
    push_log = log_dir / "push.log"
    run_logged_cmd(["docker", "tag", local_tag, registry_tag], log_file=push_log, quiet=quiet)
    run_logged_cmd(["docker", "push", registry_tag], log_file=push_log, quiet=quiet)


def remove_local_image(image_tag: str, log_dir: Path, quiet: bool = False) -> None:
    cleanup_log = log_dir / "cleanup.log"
    run_logged_cmd(["docker", "image", "rm", image_tag], log_file=cleanup_log, quiet=quiet)


def build_image(
    context_dir: Path,
    old_commit_hash: str,
    image_tag: str,
    use_buildx: bool,
    platform: str,
    log_dir: Path,
    quiet: bool = False,
) -> None:
    if use_buildx:
        cmd = [
            "docker",
            "buildx",
            "build",
            "--platform",
            platform,
            "--load",
            "-t",
            image_tag,
            ".",
            "--build-arg",
            f"OLD_COMMIT={old_commit_hash}",
        ]
    else:
        cmd = [
            "docker",
            "build",
            "-t",
            image_tag,
            ".",
            "--build-arg",
            f"OLD_COMMIT={old_commit_hash}",
        ]
    run_logged_cmd(cmd, cwd=context_dir, log_file=log_dir / "build.log", quiet=quiet)


def process_one_instance(
    item: PreparedInstance,
    args: argparse.Namespace,
    quiet: bool = False,
) -> tuple[str, str, bool]:
    ref = item.ref
    log_dir = args.logs_dir / ref.instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "meta.json").write_text(
        json.dumps(
            {
                "instance_id": ref.instance_id,
                "repo": f"{ref.owner}/{ref.repo_raw}",
                "old_commit_hash": item.old_commit_hash,
                "image_tag": item.image_tag,
                "env_group_key": item.env_group_key,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    context_dir = prepare_context(
        ref=ref,
        old_commit_hash=item.old_commit_hash,
        test_file_codes=item.test_file_codes,
        test_file_names=item.test_file_names,
        tests_cmd=item.tests_cmd,
        contexts_dir=args.contexts_dir,
        workspace_root=WORKSPACE_ROOT,
        repo_dir=item.repo_dir,
    )
    build_image(
        context_dir=context_dir,
        old_commit_hash=item.old_commit_hash,
        image_tag=item.image_tag,
        use_buildx=args.use_buildx,
        platform=args.platform,
        log_dir=log_dir,
        quiet=quiet,
    )

    pushed = False
    if args.push:
        registry_tag = make_registry_tag(args.registry, item.image_tag)
        push_image(item.image_tag, registry_tag, log_dir=log_dir, quiet=quiet)
        pushed = True
        if args.cleanup_local:
            remove_local_image(item.image_tag, log_dir=log_dir, quiet=quiet)

    return ref.instance_id, item.image_tag, pushed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ARM64 Docker images from r2e gym instance ids."
    )
    parser.add_argument(
        "--instance-file",
        type=Path,
        default=DEFAULT_INSTANCE_FILE,
        help="Path to instance-id file (default: r2e_gym_instance_ids.txt)",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index in instance list")
    parser.add_argument("--limit", type=int, default=None, help="Max instances to process")
    parser.add_argument(
        "--repos",
        type=str,
        default="",
        help="Comma-separated repo keys to include (e.g. pandas,numpy)",
    )
    parser.add_argument(
        "--contexts-dir",
        type=Path,
        default=DEFAULT_CONTEXTS_DIR,
        help="Directory for generated build contexts",
    )
    parser.add_argument(
        "--repo-cache-dir",
        type=Path,
        default=DEFAULT_REPO_CACHE_DIR,
        help="Directory to clone/fetch source repos",
    )
    parser.add_argument(
        "--tag-prefix",
        type=str,
        default="r2egym-arm64",
        help="Image repository/name prefix, final format: <prefix>:<instance_id>",
    )
    parser.add_argument(
        "--use-buildx",
        action="store_true",
        help="Use docker buildx (recommended for cross-platform or multi-arch output)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default="linux/arm64",
        help="Target platform when using --use-buildx",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be built without running docker build",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help="State file for checkpoint/recovery (built/pushed/failed)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Ignore previous built state and rebuild images",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry entries previously marked as failed in the state file",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push built images to --registry and checkpoint push progress",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="",
        help="Registry/repository path for push, e.g. registry.example.com/team/r2egym-arm64",
    )
    parser.add_argument(
        "--cleanup-local",
        action="store_true",
        help="Remove local image after successful push",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Parallel workers for build (and optional push) operations",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Directory for per-instance build/push/cleanup logs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_HF_DATASET,
        help="HuggingFace dataset for test data when local test_data/ is missing "
             "(default: %(default)s; set to empty string to disable)",
    )
    args = parser.parse_args()

    if args.push and not args.registry:
        raise ValueError("--registry is required when --push is enabled")

    refs = load_instance_ids(args.instance_file, args.start, args.limit)
    if args.repos:
        selected = {r.strip().lower() for r in args.repos.split(",") if r.strip()}
        refs = [r for r in refs if r.repo_key in selected]

    if not refs:
        print("No instances selected.")
        return

    state = load_state(args.state_file)
    state.setdefault("built", [])
    state.setdefault("pushed", [])
    state.setdefault("failed", [])
    built_set = set(state["built"])
    pushed_set = set(state["pushed"])
    failed_set = set(state["failed"])

    print(f"Selected {len(refs)} instances")
    refs_to_process: list[InstanceRef] = []
    for i, ref in enumerate(refs, start=1):
        print(f"[{i}/{len(refs)}] {ref.instance_id}")
        if ref.instance_id in failed_set and not args.retry_failed:
            print("  skipped (previously failed; pass --retry-failed to retry)")
            continue
        if args.push:
            if ref.instance_id in pushed_set and not args.rebuild:
                print("  skipped (already pushed)")
                continue
        elif ref.instance_id in built_set and not args.rebuild:
            print("  skipped (already built)")
            continue
        refs_to_process.append(ref)

    if not refs_to_process:
        print("No pending instances after state filtering.")
        print(f"  state:  {args.state_file}")
        return

    if args.dry_run:
        for ref in refs_to_process:
            image_tag = make_image_tag(args.tag_prefix, ref.instance_id)
            print(f"  dry-run build -> {image_tag}")
        print("  note: env-group scheduling is applied during real builds")
        return

    # Pre-checkout unique repos once to avoid clone/fetch races in workers.
    repo_dir_by_key: dict[str, Path] = {}
    for ref in refs_to_process:
        repo_cache_key = f"{ref.owner}__{ref.repo_raw}"
        if repo_cache_key not in repo_dir_by_key:
            repo_dir_by_key[repo_cache_key] = ensure_repo_checkout(ref, args.repo_cache_dir)

    # Load HuggingFace dataset index for test data fallback.
    hf_index: dict[tuple[str, str], dict] | None = None
    if args.dataset:
        hf_index = load_hf_test_index(args.dataset)

    # Precompute per-instance build metadata and group by env key to improve cache reuse.
    prepared_items: list[PreparedInstance] = []
    env_groups: dict[str, list[PreparedInstance]] = defaultdict(list)
    for ref in refs_to_process:
        repo_cache_key = f"{ref.owner}__{ref.repo_raw}"
        repo_dir = repo_dir_by_key[repo_cache_key]
        old_commit_hash = resolve_old_commit(ref, repo_dir, WORKSPACE_ROOT)
        try:
            test_codes, test_names, tests_cmd = load_test_assets(ref, WORKSPACE_ROOT, hf_index)
        except ValueError as e:
            print(f"  SKIP {ref.instance_id}: {e}")
            if ref.instance_id not in failed_set:
                state["failed"].append(ref.instance_id)
                failed_set.add(ref.instance_id)
                save_state(args.state_file, state)
            continue
        image_tag = make_image_tag(args.tag_prefix, ref.instance_id)
        env_group_key = make_env_group_key(ref, old_commit_hash, tests_cmd)
        item = PreparedInstance(
            ref=ref,
            old_commit_hash=old_commit_hash,
            test_file_codes=test_codes,
            test_file_names=test_names,
            tests_cmd=tests_cmd,
            image_tag=image_tag,
            env_group_key=env_group_key,
            repo_dir=repo_dir,
        )
        prepared_items.append(item)
        env_groups[env_group_key].append(item)

    sorted_groups = sorted(env_groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    print(f"\nEnv-group scheduling: {len(sorted_groups)} groups")
    for idx, (group_key, items) in enumerate(sorted_groups, start=1):
        print(f"  [{idx}/{len(sorted_groups)}] {group_key} -> {len(items)} instances")

    # Flatten groups into a single ordered list (cache-friendly order preserved).
    ordered_items: list[PreparedInstance] = []
    for _group_key, group_items in sorted_groups:
        ordered_items.extend(group_items)

    total = len(ordered_items)
    done = 0
    quiet = args.max_workers > 1
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(process_one_instance, item, args, quiet=quiet): item
            for item in ordered_items
        }

        for future in as_completed(futures):
            item = futures[future]
            done += 1
            try:
                instance_id, image_tag, pushed = future.result()
                print(f"[{done}/{total}] built -> {image_tag}")

                if instance_id not in built_set:
                    state["built"].append(instance_id)
                    built_set.add(instance_id)
                if pushed and instance_id not in pushed_set:
                    registry_tag = make_registry_tag(args.registry, image_tag)
                    print(f"[{done}/{total}] pushed -> {registry_tag}")
                    state["pushed"].append(instance_id)
                    pushed_set.add(instance_id)

                if instance_id in failed_set:
                    failed_set.remove(instance_id)
                    state["failed"] = [x for x in state["failed"] if x != instance_id]
                save_state(args.state_file, state)
            except Exception as e:
                print(f"[{done}/{total}] failed -> {item.ref.instance_id}: {e}")
                if item.ref.instance_id not in failed_set:
                    state["failed"].append(item.ref.instance_id)
                    failed_set.add(item.ref.instance_id)
                save_state(args.state_file, state)

    print("\nDone.")
    print(f"  built:  {len(state.get('built', []))}")
    print(f"  pushed: {len(state.get('pushed', []))}")
    print(f"  failed: {len(state.get('failed', []))}")
    print(f"  state:  {args.state_file}")


if __name__ == "__main__":
    main()
