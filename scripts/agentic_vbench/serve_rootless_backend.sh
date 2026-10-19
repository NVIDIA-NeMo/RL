#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Remote rootless Podman backend for Agentic-VBench task containers.
#
# A gym-native driver runs inside a Pyxis/enroot container, where nested rootless
# Podman cannot write cgroups, so the tasks' Docker-API traffic goes to this job
# instead: a rootless `podman system service` behind an mTLS socat proxy on a CPU
# node. It writes `client.env` (DOCKER_HOST, DOCKER_TLS_VERIFY, DOCKER_CERT_PATH)
# beside the TLS material in AVB_BACKEND_DIR and removes it on exit, so a present
# client.env is the liveness signal the agent waits for. Use one backend per
# evaluation job: ~12 concurrent trials saturate a 64-core node.
#
#   AGENTIC_VBENCH_BACKEND_DIR=... bash scripts/agentic_vbench/submit_backend.sh
#
# This copy comes from the eval-factory benchmarking repo
# (scripts/agentic_vbench/serve_rootless_backend.sbatch), whose backend.py also
# polls, inspects and garbage-collects backends. Keep AVB_BACKEND_DIR on the shared
# filesystem the training job can read; submit_backend.sh maps
# AGENTIC_VBENCH_BACKEND_DIR onto it.
#SBATCH --job-name=avb-backend
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=avb-backend.%j.log
set -euo pipefail
umask 077
: "${AVB_BACKEND_DIR:?Set AVB_BACKEND_DIR to the directory the driver reads client.env from}"
mkdir -p "$AVB_BACKEND_DIR"
backend_dir=$(cd "$AVB_BACKEND_DIR" && pwd)
if [[ -f $backend_dir/client.env ]]; then
  echo "A backend already publishes $backend_dir/client.env; refusing to start a second one" >&2
  exit 1
fi
# A portable bundle keeps the native Harbor backend usable on Slurm nodes
# without a system Podman installation. All writable state remains job-local.
if [[ -n ${PODMAN_BUNDLE_ROOT:-} ]]; then
  PODMAN_BUNDLE_ROOT=$(cd "$PODMAN_BUNDLE_ROOT" && pwd)
  export PODMAN_BUNDLE_ROOT
  export PATH="$PODMAN_BUNDLE_ROOT/usr/local/bin:$PATH"
  export LD_LIBRARY_PATH="$PODMAN_BUNDLE_ROOT/usr/local/lib:${LD_LIBRARY_PATH:-}"
fi
for tool in podman socat openssl; do
  command -v "$tool" >/dev/null || { echo "missing $tool on $(hostname)" >&2; exit 1; }
done
# Task images and containers accumulate here. Prefer node-local disk over /tmp, a
# RAM-backed tmpfs on CPU workers; keep the prefix short for Podman socket paths.
runtime_parent=/tmp
if [[ -d /raid/scratch && -w /raid/scratch ]]; then
  runtime_parent="/raid/scratch/$(id -u)"
  mkdir -p "$runtime_parent"
fi
runtime=$(mktemp -d "$runtime_parent/avb-backend.XXXXXXXX")
# sbatch --export=ALL preserves the submitter's TMPDIR. Interactive tooling can
# point it at Lustre, where rootless Buildah cannot create the mount namespace
# directories used by `RUN` layers. Keep all Podman/Buildah temporary state on
# the backend node beside its storage and run roots.
mkdir -p "$runtime/tmp" "$runtime/config" "$runtime/cache"
export TMPDIR="$runtime/tmp"
# Rootless Podman otherwise writes CNI network definitions under
# ~/.config/cni/net.d even though its storage and run roots are node-local.
# Agentic rollouts create a fresh compose network per trial, so a full home
# quota turns a valid episode into a missing-verifier-reward retry loop.  Keep
# every disposable Podman config/cache artifact with this backend's runtime;
# cleanup removes the whole directory when the backend exits.
export XDG_CONFIG_HOME="$runtime/config"
export XDG_CACHE_HOME="$runtime/cache"
if [[ -n ${PODMAN_BUNDLE_ROOT:-} ]]; then
  export XDG_RUNTIME_DIR="$runtime/xdg-runtime"
  mkdir -m 700 "$XDG_RUNTIME_DIR"
  # Podman's image/build library looks in the normal home directory for its
  # policy even when XDG_CONFIG_HOME is set. Preserve the user's trust policy.
  [[ -r $HOME/.config/containers/policy.json || -r /etc/containers/policy.json ]] || {
    echo 'Missing Podman policy.json; install the bundle policy in the normal user config before launching the backend' >&2
    exit 1
  }
  python3 - "$PODMAN_BUNDLE_ROOT" "$runtime" <<'PODMAN_CONFIG_PY'
import json
from pathlib import Path
import sys

bundle, runtime = map(Path, sys.argv[1:])
quote = lambda path: json.dumps(str(path))
helpers = bundle / "usr/local/lib/podman"
# Rootless cgroup v1 cannot enforce per-container limits. Keep the service in
# Slurm's cgroup: otherwise Podman moves it to a transient user-systemd scope
# whose manager exits when the batch node has no interactive login session.
cgroup_config = '' if Path('/sys/fs/cgroup/cgroup.controllers').exists() else 'cgroups = "disabled"\n'
(runtime / "containers.conf").write_text(
    '[containers]\n'
    + cgroup_config +
    f'seccomp_profile = {quote(bundle / "etc/containers/seccomp.json")}\n'
    '[engine]\ncgroup_manager = "cgroupfs"\nevents_logger = "file"\nruntime = "crun"\n'
    f'conmon_path = [{quote(helpers / "conmon")}]\n'
    f'helper_binaries_dir = [{quote(helpers)}, {quote(bundle / "usr/local/bin")}]\n'
    '[engine.runtimes]\n'
    f'crun = [{quote(bundle / "usr/local/bin/crun")}]\n'
)
(runtime / "storage.conf").write_text(
    '[storage]\ndriver = "overlay"\n'
    f'runroot = {quote(runtime / "run")}\n'
    f'graphroot = {quote(runtime / "storage")}\n'
    '[storage.options.overlay]\nignore_chown_errors = "true"\n'
    f'mount_program = {quote(bundle / "usr/local/bin/fuse-overlayfs")}\n'
)
PODMAN_CONFIG_PY
  export CONTAINERS_CONF="$runtime/containers.conf"
  export CONTAINERS_STORAGE_CONF="$runtime/storage.conf"
  export CONTAINERS_REGISTRIES_CONF="$PODMAN_BUNDLE_ROOT/etc/containers/registries.conf"
  if [[ ! -f /sys/fs/cgroup/cgroup.controllers ]]; then
    # The API command has no --cgroups flag; its rootless setup still attempts
    # a user-systemd scope even with cgroupfs selected. No session bus is needed
    # on cgroup v1. Prevent that optional move out of the Slurm allocation.
    export DBUS_SESSION_BUS_ADDRESS="unix:path=$runtime/no-session-bus"
  fi
fi
port=${AVB_BACKEND_PORT:-18443}
# Certificates must outlive the job: the walltime in days, plus one.
walltime_seconds=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null | awk -F'[-:]' '
  NF==4 {print $1*86400+$2*3600+$3*60+$4} NF==3 {print $1*3600+$2*60+$3} NF==2 {print $1*60+$2}')
tls_days=${AVB_BACKEND_TLS_DAYS:-$(( ${walltime_seconds:-86400} / 86400 + 2 ))}
backend_host=$(hostname -f)
cleanup() {
  rm -f "$backend_dir/client.env"
  # Containers of a cancelled evaluation would otherwise outlive it on this node.
  podman --root "$runtime/storage" --runroot "$runtime/run" rm -af >/dev/null 2>&1 || true
  [[ -z ${proxy_pid:-} ]] || kill "$proxy_pid" 2>/dev/null || true
  [[ -z ${service_pid:-} ]] || kill "$service_pid" 2>/dev/null || true
  rm -rf "$runtime"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
openssl req -x509 -newkey rsa:2048 -nodes -days "$tls_days" -subj /CN=avb-backend-ca \
  -keyout "$backend_dir/ca-key.pem" -out "$backend_dir/ca.pem" 2>/dev/null
for name in server client; do
  openssl req -newkey rsa:2048 -nodes -subj "/CN=avb-$name" \
    -keyout "$backend_dir/$name-key.pem" -out "$backend_dir/$name.csr" 2>/dev/null
  if [[ $name == server ]]; then
    printf 'subjectAltName=DNS:%s\nextendedKeyUsage=serverAuth\n' "$backend_host" > "$backend_dir/extensions.cnf"
  else
    printf 'extendedKeyUsage=clientAuth\n' > "$backend_dir/extensions.cnf"
  fi
  openssl x509 -req -in "$backend_dir/$name.csr" -CA "$backend_dir/ca.pem" \
    -CAkey "$backend_dir/ca-key.pem" -CAcreateserial -days "$tls_days" \
    -extfile "$backend_dir/extensions.cnf" -out "$backend_dir/$name-cert.pem" 2>/dev/null
done
# The Docker client reads ca.pem, cert.pem and key.pem from DOCKER_CERT_PATH.
cp "$backend_dir/client-cert.pem" "$backend_dir/cert.pem"
cp "$backend_dir/client-key.pem" "$backend_dir/key.pem"
# Initialize rootless namespaces before backgrounding the API. Otherwise the
# initial Podman invocation can re-exec and leave the supervisor tracking the
# bootstrap process rather than the server (Podman's own API tests do this).
podman --root "$runtime/storage" --runroot "$runtime/run" unshare true \
  > "$backend_dir/bootstrap.log" 2>&1
service_trace=()
if [[ ${AVB_TRACE_SERVICE:-0} == 1 ]]; then
  service_trace=("$(command -v strace)" -f -e trace=signal -o "$backend_dir/service-signals.log")
fi
podman_binary=$(command -v podman)
service_path=$PATH
if [[ -n ${PODMAN_BUNDLE_ROOT:-} && ! -f /sys/fs/cgroup/cgroup.controllers ]]; then
  # Netavark 1.17 starts its DNS daemon through systemd-run whenever it finds
  # that binary on PATH. Use only the portable runtime's complete helper set
  # so DNS starts directly and remains in Slurm's allocation, like the API.
  service_path="$PODMAN_BUNDLE_ROOT/usr/local/bin:$PODMAN_BUNDLE_ROOT/usr/local/lib/podman"
fi
PATH="$service_path" "${service_trace[@]}" "$podman_binary" --log-level="${AVB_PODMAN_LOG_LEVEL:-info}" --root "$runtime/storage" --runroot "$runtime/run" \
  --storage-opt overlay.ignore_chown_errors=true system service --time=0 "unix:$runtime/api.sock" \
  > "$backend_dir/podman.log" 2>&1 &
service_pid=$!
for attempt in {1..30}; do
  [[ ! -S $runtime/api.sock ]] || break
  kill -0 "$service_pid"
  sleep 1
done
test -S "$runtime/api.sock"
# Docker half-closes stdin while continuing to read exec output. Keep that
# half-closed connection alive until the command finishes (socat defaults to 0.5s).
# Must exceed the longest single exec: verifiers run up to 1800 s x the Harbor
# verifier timeout multiplier (3 -> 5400 s).
socat -t "${AVB_BACKEND_HALFCLOSE_SECONDS:-14400}" "OPENSSL-LISTEN:$port,reuseaddr,fork,cert=$backend_dir/server-cert.pem,key=$backend_dir/server-key.pem,cafile=$backend_dir/ca.pem,verify=1" \
  "UNIX-CONNECT:$runtime/api.sock" > "$backend_dir/proxy.log" 2>&1 &
proxy_pid=$!
sleep 1
kill -0 "$proxy_pid"
{
  printf '# Written by serve_rootless_backend.sbatch (Slurm job %s on %s); removed when the job ends.\n' "${SLURM_JOB_ID:-?}" "$backend_host"
  printf 'export DOCKER_HOST=%q\n' "tcp://$backend_host:$port"
  printf 'export DOCKER_TLS_VERIFY=1\n'
  printf 'export DOCKER_CERT_PATH=%q\n' "$backend_dir"
  printf 'export DOCKER_BUILDKIT=0\n'
} > "$backend_dir/client.env"
echo "Backend ready: source $backend_dir/client.env"
if [[ ${AVB_TRACE_SERVICE:-0} == 1 ]]; then
  ps -u "$(id -u)" -o pid,ppid,comm > "$backend_dir/processes.log"
  printf 'batch=%s parent=%s service=%s proxy=%s\n' "$$" "$PPID" "$service_pid" "$proxy_pid" >> "$backend_dir/processes.log"
fi
backend_status=0
wait -n -p exited_pid "$proxy_pid" "$service_pid" || backend_status=$?
echo "Backend process ${exited_pid:-unknown} exited unexpectedly (status=$backend_status; proxy=$proxy_pid service=$service_pid)" >&2
# A clean daemon exit still means the long-lived backend failed.
(( backend_status != 0 )) || backend_status=1
exit "$backend_status"
