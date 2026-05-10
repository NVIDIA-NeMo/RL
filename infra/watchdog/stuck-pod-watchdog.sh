#!/usr/bin/env bash
# stuck-pod-watchdog.sh — reap RayCluster worker pods stuck in
# PodInitializing because the DRA driver couldn't prepare RoCE /
# ComputeDomain channels (FailedPrepareDynamicResources, DeadlineExceeded).
#
# Symptom on GB300: a pod schedules onto a node whose IMEX daemon is
# wedged from a prior SIGKILL'd workload. The pod sits at Init:0/N
# forever waiting for DRA prep that will never succeed. KubeRay's
# autoscaler doesn't reap it because the pod isn't "failed", just
# stuck.
#
# This watchdog deletes such pods so KubeRay's RayCluster controller
# respawns them — usually onto a different node where the DRA driver
# is healthy. If a node is persistently bad, the watchdog will keep
# delete-and-respawning until k8s eventually places the pod elsewhere.
#
# Run as a Deployment in the same namespace as the RayClusters. The
# pod needs RBAC for: get/list/watch/delete pods and get events.

set -u

NAMESPACE=${NAMESPACE:-default}
LABEL_SELECTOR=${LABEL_SELECTOR:-ray.io/cluster}
STUCK_AGE_SECONDS=${STUCK_AGE_SECONDS:-180}  # 3 min — covers normal init
POLL_INTERVAL=${POLL_INTERVAL:-30}

echo "[watchdog] namespace=${NAMESPACE} selector=${LABEL_SELECTOR} stuck_age=${STUCK_AGE_SECONDS}s interval=${POLL_INTERVAL}s"

while true; do
  # JSON path: pods owned by a RayCluster, currently in initialising
  # state for longer than $STUCK_AGE_SECONDS. We don't filter on
  # event reason here (events can be ephemeral); we filter purely
  # on duration-in-PodInitializing and then confirm with an events
  # check before deleting.
  candidates=$(
    kubectl get pods -n "${NAMESPACE}" -l "${LABEL_SELECTOR}" -o json 2>/dev/null \
      | jq -r --argjson stuck_age "${STUCK_AGE_SECONDS}" '
          .items[]
          | select(.metadata.creationTimestamp != null)
          | select((now - (.metadata.creationTimestamp | fromdateiso8601)) > $stuck_age)
          | select(
              (.status.initContainerStatuses // []) | any(
                .state.waiting.reason == "PodInitializing"
              )
            )
          | .metadata.name
        ' 2>/dev/null
  )
  if [ -z "${candidates}" ]; then
    sleep "${POLL_INTERVAL}"
    continue
  fi

  echo "${candidates}" | while IFS= read -r podname; do
    [ -z "${podname}" ] && continue
    # Confirm by checking the pod's events for FailedPrepareDynamicResources
    # OR FailedCreatePodSandBox (the two known DRA-prep failure paths).
    # Without this guard, we'd reap pods that are just slow to pull
    # an image or finish their initContainer.
    events=$(kubectl get events -n "${NAMESPACE}" \
                --field-selector "involvedObject.name=${podname}" \
                --sort-by='.lastTimestamp' \
                -o json 2>/dev/null \
              | jq -r '.items[] | "\(.reason) \(.message)"' 2>/dev/null || true)
    if echo "${events}" | grep -qE "FailedPrepareDynamicResources|FailedCreatePodSandBox|context deadline exceeded"; then
      echo "[watchdog] $(date -u +%H:%M:%S) reaping ${podname} (stuck > ${STUCK_AGE_SECONDS}s, DRA/sandbox prep failure):"
      echo "${events}" | tail -2 | sed 's/^/    /'
      kubectl delete pod "${podname}" -n "${NAMESPACE}" --wait=false 2>&1 | sed 's/^/    /' || true
    fi
  done

  sleep "${POLL_INTERVAL}"
done
