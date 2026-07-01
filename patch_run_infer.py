#!/usr/bin/env python3
"""
Apply RC6+RC7+RC8 patches to run_infer.py in the enroot overlay.

CRITICAL: Must run on ALL THREE NODES — ray-head AND both ray-workers.
SIF agent containers are spawned by Ray WORKER tasks on worker nodes (not head node).
Each node has a SEPARATE enroot overlay. Patching only ray-head leaves workers unpatched
→ 10/16 instances fail with old source/pid=-1 behavior (confirmed Run 17, 2026-07-15).

  JOBID=<JOBID>; HEAD=<lyrisXXXX>; W1=<lyrisXXXX>; W2=<lyrisXXXX>
  # Head node:
  srun --jobid=$JOBID --overlap -w $HEAD -N1 --ntasks=1 \
    --container-name=ray-head --no-container-mount-home \
    python3 /lustre/fsw/coreai_comparch_trtllm/erinh/RL/patch_run_infer.py
  # Worker nodes:
  for node in $W1 $W2; do
    srun --jobid=$JOBID --overlap -w $node -N1 --ntasks=1 \
      --container-name=ray-worker --no-container-mount-home \
      python3 /lustre/fsw/coreai_comparch_trtllm/erinh/RL/patch_run_infer.py
  done

RC6: export SWE_INSTANCE_ID directly in initial_setup_cmd (not just via .bashrc)
RC7: source /swe_util/... → bash /swe_util/... (subprocess so exit 1 doesn't kill BashSession)
RC7b: add conda activate testbed after bash entry script (subprocess env doesn't propagate)
RC8: rm -f /usr/local/bin/jq first (agent_script.sh copies x86_64 jq shadowing arm64 /usr/bin/jq)
"""

RUN_INFER = (
    "/opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/"
    "swe_openhands_setup/OpenHands/evaluation/benchmarks/swe_bench/run_infer.py"
)

with open(RUN_INFER, "r") as f:
    content = f.read()

applied = []

# ---------- RC6+RC8: inject rm jq + direct export at start of initial_setup_cmd ----------
# Handles both the original baseline (echo '...' >> .bashrc as first line)
# and a version where RC6 was previously applied (export ... as first line)

if "rm -f /usr/local/bin/jq" in content:
    applied.append("RC6+RC8: already applied (skipped)")
else:
    # Original baseline: first line after f"""\n is the echo >> .bashrc
    OLD_SETUP_ORIG = (
        "    initial_setup_cmd = f\"\"\"\n"
        "echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && \\"
    )
    NEW_SETUP_ORIG = (
        "    initial_setup_cmd = f\"\"\"\n"
        "rm -f /usr/local/bin/jq && \\\n"
        "export SWE_INSTANCE_ID={instance['instance_id']} && \\\n"
        "echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && \\"
    )
    # RC6-patched: first line is direct export
    OLD_SETUP_RC6 = (
        "    initial_setup_cmd = f\"\"\"\n"
        "export SWE_INSTANCE_ID={instance['instance_id']} && \\\n"
        "echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && \\"
    )
    NEW_SETUP_RC6 = (
        "    initial_setup_cmd = f\"\"\"\n"
        "rm -f /usr/local/bin/jq && \\\n"
        "export SWE_INSTANCE_ID={instance['instance_id']} && \\\n"
        "echo 'export SWE_INSTANCE_ID={instance['instance_id']}' >> ~/.bashrc && \\"
    )

    if OLD_SETUP_ORIG in content:
        content = content.replace(OLD_SETUP_ORIG, NEW_SETUP_ORIG, 1)
        applied.append("RC6+RC8: rm jq + direct SWE_INSTANCE_ID export added (from original baseline)")
    elif OLD_SETUP_RC6 in content:
        content = content.replace(OLD_SETUP_RC6, NEW_SETUP_RC6, 1)
        applied.append("RC6+RC8: rm jq added before existing RC6 export")
    else:
        print("ERROR: RC6+RC8 target not found — check run_infer.py initial_setup_cmd manually")

# ---------- RC7: source → bash for instance_swe_entry.sh ----------
OLD_SOURCE = "action = CmdRunAction(command=f'source /swe_util/{entry_script_path}')"
NEW_SOURCE = "action = CmdRunAction(command=f'bash /swe_util/{entry_script_path}')"

if OLD_SOURCE in content:
    content = content.replace(OLD_SOURCE, NEW_SOURCE, 1)
    applied.append("RC7: source → bash for entry script")
elif NEW_SOURCE in content:
    applied.append("RC7: already applied (skipped)")
else:
    print("ERROR: RC7 target not found — source/bash entry script line missing")

# ---------- RC7b: conda activation after bash entry script ----------
CONDA_MARKER = "# Activate conda testbed in BashSession (subprocess env changes don't propagate back)"

if CONDA_MARKER in content:
    applied.append("RC7b: conda activation already present (skipped)")
else:
    OLD_ASSERT = (
        "        assert_and_raise(\n"
        "            obs.exit_code == 0,\n"
        "            f'Failed to source /swe_util/{entry_script_path}: {str(obs)}',\n"
        "        )\n"
        "    elif DATASET_TYPE == 'SWE-Gym':"
    )
    NEW_ASSERT = (
        "        assert_and_raise(\n"
        "            obs.exit_code == 0,\n"
        "            f'Failed to source /swe_util/{entry_script_path}: {str(obs)}',\n"
        "        )\n"
        "        # Activate conda testbed in BashSession (subprocess env changes don't propagate back)\n"
        "        action = CmdRunAction(command='if [ -d /opt/miniconda3 ]; then . /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed; fi')\n"
        "        action.set_hard_timeout(60)\n"
        "        obs = runtime.run_action(action)\n"
        "        logger.info(obs, extra={'msg_type': 'OBSERVATION'})\n"
        "    elif DATASET_TYPE == 'SWE-Gym':"
    )
    if OLD_ASSERT in content:
        content = content.replace(OLD_ASSERT, NEW_ASSERT, 1)
        applied.append("RC7b: conda activation after entry script")
    else:
        print("WARNING: RC7b target not found (may differ or already present)")

with open(RUN_INFER, "w") as f:
    f.write(content)

print("Patches applied:")
for a in applied:
    print(f"  - {a}")
print("Done.")
