#!/bin/bash
# Submits the full grid: baseline+emotional array first, then the yoked array,
# which is only allowed to start once every task in the first array has
# completed successfully (afterok). This enforces the ordering the yoked
# agent's trace-loading depends on, at the scheduler level rather than by
# manually timing submissions.

set -e

BE_JOBID=$(sbatch --parsable run_reversal_array.slurm)
echo "Submitted baseline+emotional array: job $BE_JOBID"

# afterok:<jobid> on an array job waits for ALL array tasks to finish successfully.
YOKED_JOBID=$(sbatch --parsable --dependency=afterok:${BE_JOBID} run_reversal_yoked_array.slurm)
echo "Submitted yoked array: job $YOKED_JOBID (waiting on $BE_JOBID)"

echo ""
echo "Check status with: squeue -u \$USER"
echo "The yoked array will show as PD (pending, dependency) until $BE_JOBID fully completes."
