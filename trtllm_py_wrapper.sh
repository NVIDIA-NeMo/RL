#!/bin/bash
# Ray spawns TrtllmAsyncGenerationWorker actors as plain Python subprocesses, so
# OpenMPI's singleton bootstrap is never triggered properly (opal_init fails on
# v13's OMPI 4.1.9a1). But `mpirun -n 1` cleanly bootstraps a 1-process MPI world.
#
# Since nemo-rl uses orchestrator_type="ray" for trtllm, each TP rank IS its own
# Ray actor / Python process, so a singleton MPI world per actor is what we want.
#
# Set NEMO_RL_PY_EXECUTABLES_TRTLLM to this wrapper so Ray launches each actor
# under `mpirun -n 1`. Inside the python process, MPI_Init then succeeds and
# trtllm's MPI_Comm_split_type call works on the singleton WORLD.
#
# IMPORTANT: ray.sub uses `srun --mpi=pmix` to bring up the cluster, which sets
# PMIX_* and OMPI_* env vars on every spawned process. mpirun sees those and
# aborts with "mpirun does not support recursive calls". Clear them before
# invoking mpirun so it treats this as a fresh launch.
for v in $(env | grep -oE '^(OMPI|PMI|PMIX|MV2|SLURM)_[A-Za-z0-9_]+'); do
    # Keep our own allow-run-as-root settings + the wrapper override.
    # SLURM_* must be unset too: mpirun's plm picks "slurm" as the process
    # launcher when SLURM_* is present and then tries to exec srun, which
    # isn't on PATH inside enroot containers → "unable to locate srun" abort.
    case "$v" in
        OMPI_ALLOW_RUN_AS_ROOT|OMPI_ALLOW_RUN_AS_ROOT_CONFIRM) continue ;;
        *) unset "$v" ;;
    esac
done
exec /usr/local/mpi/bin/mpirun -n 1 --allow-run-as-root /opt/nemo_rl_venv/bin/python3 "$@"
