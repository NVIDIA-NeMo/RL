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
    local output=
    local character
    local codepoint
    local escaped
    local LC_ALL=C

    while [[ -n "${value}" ]]; do
        character=${value:0:1}
        value=${value:1}
        case "${character}" in
            $'\\') output+='\\' ;;
            '"') output+='\"' ;;
            $'\b') output+='\b' ;;
            $'\f') output+='\f' ;;
            $'\n') output+='\n' ;;
            $'\r') output+='\r' ;;
            $'\t') output+='\t' ;;
            *)
                printf -v codepoint '%d' "'${character}"
                if ((codepoint >= 1 && codepoint <= 31)); then
                    printf -v escaped '\\u%04x' "${codepoint}"
                    output+="${escaped}"
                else
                    output+="${character}"
                fi
                ;;
        esac
    done
    printf '%s' "${output}"
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
    unset CUTEDSL_FINALIZATION_STARTED CUTEDSL_TERMINATING_SIGNAL
    rm -f "${result_dir}/status.json" "${result_dir}/report.html"
    : > "${CUTEDSL_EVENTS_FILE}"
}

_cutedsl_exit_for_signal() {
    local exit_code="${1:?signal exit code is required}"
    local signal_name="${2:?signal name is required}"
    CUTEDSL_TERMINATING_SIGNAL="${signal_name}"
    export CUTEDSL_TERMINATING_SIGNAL
    trap '' TERM INT
    exit "${exit_code}"
}

cutedsl_install_signal_traps() {
    trap '_cutedsl_exit_for_signal 143 TERM' TERM
    trap '_cutedsl_exit_for_signal 130 INT' INT
}

cutedsl_begin_finalization() {
    if [[ "${CUTEDSL_FINALIZATION_STARTED:-0}" == "1" ]]; then
        return 1
    fi
    CUTEDSL_FINALIZATION_STARTED=1
    export CUTEDSL_FINALIZATION_STARTED
    trap - EXIT
    trap '' TERM INT
    return 0
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
    local reproduction="${6:?cutedsl_write_root_cause requires reproduction evidence}"
    local hypothesis="${7:?cutedsl_write_root_cause requires a hypothesis}"
    local tested_change="${8:?cutedsl_write_root_cause requires one tested change}"
    local verification_evidence="${9:?cutedsl_write_root_cause requires verification evidence}"
    local root_status=resolved

    if [[ "${root_cause}" == "Pending investigation" ]]; then
        root_status=pending
    fi
    printf '{"timestamp_utc":"%s","cluster":"%s","job_id":"%s","phase":"root_cause","status":"%s","exit_code":null,"message":"Root-cause record","artifact":"%s","symptom":"%s","evidence":"%s","root_cause":"%s","fix_commit":"%s","verification_job":"%s","reproduction":"%s","hypothesis":"%s","tested_change":"%s","verification_evidence":"%s"}\n' \
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
        "$(_cutedsl_json_escape "${reproduction}")" \
        "$(_cutedsl_json_escape "${hypothesis}")" \
        "$(_cutedsl_json_escape "${tested_change}")" \
        "$(_cutedsl_json_escape "${verification_evidence}")" \
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

_cutedsl_phase_has_terminal_event() {
    local phase="${1:?phase is required}"
    grep -Eq \
        "\"phase\":\"${phase}\".*\"status\":\"(pass|fail|skip)\"" \
        "${CUTEDSL_EVENTS_FILE}"
}

cutedsl_finalize_events() {
    local exit_code="${1:?cutedsl_finalize_events requires an exit code}"
    local phase

    if [[ ${exit_code} -ne 0 ]]; then
        if [[ -n "${CUTEDSL_CURRENT_PHASE:-}" ]] && \
            ! _cutedsl_phase_has_terminal_event "${CUTEDSL_CURRENT_PHASE}"; then
            cutedsl_write_event "${CUTEDSL_CURRENT_PHASE}" fail "${exit_code}" \
                "Phase failed before completion" "${CUTEDSL_CURRENT_ARTIFACT:-slurm.out}"
        fi
    fi

    for phase in "${CUTEDSL_REQUIRED_EVENT_PHASES[@]}"; do
        if [[ "${phase}" == "complete" ]]; then
            continue
        fi
        if ! grep -Fq "\"phase\":\"${phase}\"" "${CUTEDSL_EVENTS_FILE}"; then
            cutedsl_write_event "${phase}" skip 0 "Phase not reached" ''
        fi
    done
}

cutedsl_finalize_run() {
    local original_exit_code="${1:?cutedsl_finalize_run requires an exit code}"
    local completion_message="${2:?cutedsl_finalize_run requires a message}"
    local renderer="${3:?cutedsl_finalize_run requires the renderer path}"
    local renderer_exit_code=0
    local final_exit_code=${original_exit_code}
    local completion_artifact=report.html
    local completion_status=pass
    local failure_symptom="Run exited with code ${original_exit_code} during ${CUTEDSL_CURRENT_PHASE:-preflight}"

    if [[ -n "${CUTEDSL_TERMINATING_SIGNAL:-}" ]]; then
        failure_symptom="Run received SIG${CUTEDSL_TERMINATING_SIGNAL} and exited with code ${original_exit_code} during ${CUTEDSL_CURRENT_PHASE:-preflight}"
    fi

    if [[ ${original_exit_code} -ne 0 ]] && \
        ! grep -Fq '"phase":"root_cause"' "${CUTEDSL_EVENTS_FILE}"; then
        cutedsl_write_root_cause \
            "${failure_symptom}" \
            "${CUTEDSL_CURRENT_ARTIFACT:-slurm.out}" \
            "Pending investigation" "pending" "pending" \
            "Pending reproduction" "Pending hypothesis" "Pending tested change" \
            "Pending verification evidence"
    fi

    cutedsl_finalize_events "${original_exit_code}"
    cutedsl_write_status "${original_exit_code}"
    "${CUTEDSL_REPORT_PYTHON:-python3}" "${renderer}" --run-dir "${RESULT_DIR}"
    renderer_exit_code=$?
    if [[ ${renderer_exit_code} -ne 0 ]]; then
        completion_status=fail
        completion_message="Report rendering failed with code ${renderer_exit_code}"
        completion_artifact=$(basename "${renderer}")
        if [[ ${original_exit_code} -eq 0 ]]; then
            final_exit_code=${renderer_exit_code}
        fi
    elif [[ ${original_exit_code} -ne 0 ]]; then
        completion_status=fail
    fi

    cutedsl_write_event complete "${completion_status}" "${final_exit_code}" \
        "${completion_message}" "${completion_artifact}"
    cutedsl_write_status "${final_exit_code}"

    if [[ ${renderer_exit_code} -eq 0 ]]; then
        "${CUTEDSL_REPORT_PYTHON:-python3}" "${renderer}" --run-dir "${RESULT_DIR}"
    fi
    return "${final_exit_code}"
}
