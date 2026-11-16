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

# Submit-time checks for where a launcher keeps Gym server venvs.
#
# Sourced by the launchers that pass GYM_VENV_DIR to training. See "Python Environments on Shared
# Filesystems" in docs/cluster.md.

# gym_venv_launch_checks <gym venv dir> <archive or empty> <mounts> <snapshot dir>
#
# Exits the launcher if the venv directory is on a shared filesystem or the archive cannot be used.
# With an archive, appends its unpack command to SETUP_COMMAND and exports it.
# Set GYM_VENV_ALLOW_SHARED=1 to accept a venv directory on a filesystem this check rejects.
gym_venv_launch_checks() {
    local venv_dir="$1" archive="$2" mounts="$3" snapshot_dir="$4"
    local spec mount_host mount_container host_dir="" probe fs suffix unpack
    local -a specs

    # Venvs on a shared filesystem make every node import and check thousands of files there, which
    # overloads its metadata servers. A mount that covers the venv directory is checked at its host
    # path, even under /opt; the image's own unmounted /opt is not. The check looks at the nearest
    # existing directory, because Gym creates the venv directory on first use.
    [[ "${venv_dir}" != /opt/* ]] && host_dir="${venv_dir}"
    IFS=',' read -ra specs <<< "${mounts}"
    for spec in "${specs[@]}"; do
        IFS=: read -r mount_host mount_container _ <<< "${spec}"
        mount_container="${mount_container:-${mount_host}}"
        mount_container="${mount_container%/}"
        if [[ -n "${mount_host}" && ( "${venv_dir}" == "${mount_container}" || "${venv_dir}" == "${mount_container}/"* ) ]]; then
            host_dir="${mount_host%/}${venv_dir#"${mount_container}"}"
            break
        fi
    done
    if [[ -n "${host_dir}" && "${GYM_VENV_ALLOW_SHARED:-0}" != 1 ]]; then
        probe="${host_dir}"
        while [[ ! -e "${probe}" && "${probe}" != / ]]; do
            probe="$(dirname "${probe}")"
        done
        fs="$(stat -f -c %T "${probe}" 2>/dev/null || true)"
        # stat prints UNKNOWN for a filesystem it cannot name; treat it as shared rather than guess.
        case "${fs}" in
            lustre|nfs*|gpfs|fhgfs|beegfs|ceph*|fuse*|cifs|smb*|UNKNOWN*)
                echo "Error: GYM_VENV_DIR=${venv_dir} is on a shared filesystem (${fs}, host path ${host_dir})." >&2
                echo "  Keep Gym venvs on each node's local disk: leave GYM_VENV_DIR at /opt/gym_venvs and" >&2
                echo "  set GYM_VENV_ARCHIVE to reuse prebuilt venvs (docs/cluster.md)." >&2
                echo "  If this filesystem is node-local, set GYM_VENV_ALLOW_SHARED=1 to skip this check." >&2
                exit 1
                ;;
        esac
    fi
    if [[ "${venv_dir}" != /opt/* ]] && ! gym_path_identity_mounted "${venv_dir}" "${mounts}"; then
        gym_mount_error GYM_VENV_DIR "${venv_dir}"
    fi

    [[ -n "${archive}" ]] || return 0
    if [[ "${venv_dir}" != /opt/gym_venvs ]]; then
        echo "Error: GYM_VENV_ARCHIVE unpacks to /opt/gym_venvs, but GYM_VENV_DIR=${venv_dir}." >&2
        exit 1
    fi
    if [[ "${archive}" != /* ]]; then
        echo "Error: GYM_VENV_ARCHIVE=${archive} must be an absolute path." >&2
        exit 1
    fi
    for suffix in "" .size .servers; do
        if [[ ! -f "${archive}${suffix}" ]]; then
            echo "Error: ${archive}${suffix} does not exist; gym_venv_archive.sh build writes the archive" >&2
            echo "  with its .size and .servers files next to it." >&2
            exit 1
        fi
    done
    gym_path_identity_mounted "${archive}" "${mounts}" || gym_mount_error GYM_VENV_ARCHIVE "${archive}"
    # ray.sub runs SETUP_COMMAND on every node before Ray starts and ignores its exit status; the
    # unpack ends the job through ray.sub's ENDED file instead.
    unpack="NEMO_GYM_VENV_DIR=$(printf %q "${venv_dir}") bash $(printf %q "${snapshot_dir}/examples/nemo_gym/gym_venv_archive.sh") unpack"
    unpack+=" $(printf %q "${archive}") \"\$(dirname \"\$0\")/ENDED\""
    export SETUP_COMMAND="${SETUP_COMMAND:+${SETUP_COMMAND}$'\n'}${unpack}"
}

# gym_path_identity_mounted <path> <mounts>
# The path is passed into the container as is, so only a mount that keeps it at the same path counts.
gym_path_identity_mounted() {
    local path="$1" spec host_path container_path
    local -a specs
    IFS=',' read -ra specs <<< "$2"
    for spec in "${specs[@]}"; do
        IFS=: read -r host_path container_path _ <<< "${spec}"
        [[ -z "${container_path}" || "${container_path%/}" == "${host_path%/}" ]] || continue
        [[ -n "${host_path}" && ( "${path}" == "${host_path%/}" || "${path}" == "${host_path%/}/"* ) ]] && return 0
    done
    return 1
}

# gym_mount_error <variable name> <path>
gym_mount_error() {
    echo "Error: $1=$2" >&2
    echo "  is not under any path in MOUNTS that keeps the same path inside the container," >&2
    echo "  so it will not exist there under that name. Add its filesystem to EXTRA_MOUNTS" >&2
    echo "  with matching host and container paths, e.g." >&2
    echo "    EXTRA_MOUNTS=/host/path:/host/path,/other/path:/other/path" >&2
    exit 1
}
