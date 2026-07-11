#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

readonly -a CUTEDSL_REQUIRED_EVENT_PHASES=(
    preflight
    image_hash
    runtime_bootstrap
    config_validation
    focused_tests
    gpu_smoke
    functional_grpo
    timing
    profile
    metrics_export
    complete
)

_cutedsl_json_escape() {
    local value=${1-}
    value=${value//\/\\}
    value=${value//\"/\\\"}
    value=${value//$'\n'/\\n}
    value=${value//$'\r'/\\r}
    value=${value//$'\t'/\\t}
    printf '%s' "${value}"
}

_cutedsl_json_string_or_null() {
    local value=${1-}
    if [[ -z "${value}" ]]; then
        printf 'null'
    else
        printf '"%s"' "$(_cutedsl_json_escape "${value}")"
    fi
}

cutedsl_events_init() {
    local result_dir="${1:?cutedsl_events_init requires a result directory}"
    mkdir -p "${result_dir}"
    CUTEDSL_EVENTS_FILE="${result_dir}/events.jsonl"
    export CUTEDSL_EVENTS_FILE
    : > "${CUTEDSL_EVENTS_FILE}"
}

cutedsl_write_event() {
    local phase="${1:?cutedsl_write_event requires a phase}"
    local status="${2:?cutedsl_write_event requires a status}"
    local exit_code=${3-}
    local message=${4-}
    local artifact=${5-}
    local exit_code_json=null

    if [[ -n "${exit_code}" ]]; then
        if [[ ! "${exit_code}" =~ ^[0-9]+$ ]]; then
            echo "[ERROR] Event exit code must be empty or a non-negative integer: ${exit_code}" >&2
            return 2
        fi
        exit_code_json=${exit_code}
    fi
    if [[ -z "${CUTEDSL_EVENTS_FILE:-}" ]]; then
        echo "[ERROR] cutedsl_events_init must run before writing events." >&2
        return 2
    fi

    printf '{"timestamp_utc":"%s","cluster":"%s","job_id":"%s","phase":"%s","status":"%s","exit_code":%s,"message":"%s","artifact":%s}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(_cutedsl_json_escape "${CUTEDSL_EVENT_CLUSTER:-unknown}")" \
        "$(_cutedsl_json_escape "${CUTEDSL_EVENT_JOB_ID:-unknown}")" \
        "$(_cutedsl_json_escape "${phase}")" \
        "$(_cutedsl_json_escape "${status}")" \
        "${exit_code_json}" \
        "$(_cutedsl_json_escape "${message}")" \
        "$(_cutedsl_json_string_or_null "${artifact}")" \
        >> "${CUTEDSL_EVENTS_FILE}"

    if [[ "${status}" == "start" ]]; then
        CUTEDSL_CURRENT_PHASE=${phase}
        CUTEDSL_CURRENT_ARTIFACT=${artifact}
        export CUTEDSL_CURRENT_PHASE CUTEDSL_CURRENT_ARTIFACT
    fi
}

cutedsl_write_root_cause() {
    local symptom="${1:?cutedsl_write_root_cause requires a symptom}"
    local evidence="${2:?cutedsl_write_root_cause requires evidence}"
    local root_cause="${3:?cutedsl_write_root_cause requires a root cause}"
    local fix_commit="${4:?cutedsl_write_root_cause requires a fix commit}"
    local verification_job="${5:?cutedsl_write_root_cause requires a verification job}"
    local root_status=resolved

    if [[ "${root_cause}" == "Pending investigation" ]]; then
        root_status=pending
    fi
    printf '{"timestamp_utc":"%s","cluster":"%s","job_id":"%s","phase":"root_cause","status":"%s","exit_code":null,"message":"Root-cause record","artifact":"%s","symptom":"%s","evidence":"%s","root_cause":"%s","fix_commit":"%s","verification_job":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(_cutedsl_json_escape "${CUTEDSL_EVENT_CLUSTER:-unknown}")" \
        "$(_cutedsl_json_escape "${CUTEDSL_EVENT_JOB_ID:-unknown}")" \
        "${root_status}" \
        "$(_cutedsl_json_escape "${evidence}")" \
        "$(_cutedsl_json_escape "${symptom}")" \
        "$(_cutedsl_json_escape "${evidence}")" \
        "$(_cutedsl_json_escape "${root_cause}")" \
        "$(_cutedsl_json_escape "${fix_commit}")" \
        "$(_cutedsl_json_escape "${verification_job}")" \
        >> "${CUTEDSL_EVENTS_FILE}"
}

cutedsl_write_status() {
    local exit_code="${1:?cutedsl_write_status requires an exit code}"
    if [[ ! "${exit_code}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Status exit code must be a non-negative integer: ${exit_code}" >&2
        return 2
    fi
    printf '{\n  "run_id": "%s",\n  "job_id": "%s",\n  "exit_code": %d,\n  "finished_at_utc": "%s"\n}\n' \
        "$(_cutedsl_json_escape "${RUN_ID:-unknown}")" \
        "$(_cutedsl_json_escape "${SLURM_JOB_ID:-unknown}")" \
        "${exit_code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${RESULT_DIR:?RESULT_DIR must be set}/status.json"
}
