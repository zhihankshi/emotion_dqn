#!/bin/bash
#SBATCH --job-name=reversal_test
#SBATCH --account=zshi47          # TODO: from `sacctmgr show associations user=$USER`
#SBATCH --partition=a100                     # a100 or l40s; no need for h100/h200/b200 here
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00                      # generous first-pass estimate; tighten once you've timed one run
#SBATCH --output=logs/reversal_test_%j.out
#SBATCH --error=logs/reversal_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zshi47@jhu.edu         # TODO

mkdir -p logs

module reset
module load anaconda3/2024.02-1              # TODO: match to `module avail anaconda3` output
conda activate honeylab

# TEMPORARY: scratch access at /weka/scratch/jhu/$USER isn't provisioned yet (permission
# denied as of last check) — falling back to home. 86G quota is fine for a pilot/first pass.
# Once ARCH provisions scratch access, switch this back to:
#   SCRATCH_ROOT=/weka/scratch/jhu/$USER
SCRATCH_ROOT=$HOME/emotional_dqn_runs
OUTPUT_DIR=${SCRATCH_ROOT}/reversal_runs/test_emotional_seed0
mkdir -p "$OUTPUT_DIR"

cd "$SLURM_SUBMIT_DIR"

python scripts/train_reversal.py \
    --agent-type emotional \
    --seed 0 \
    --maze-acquisition shield_trap \
    --maze-a shield_trap \
    --maze-b shield_avoidance \
    --reversal-period 150 \
    --num-reversals 8 \
    --mood-delta-source online \
    --eta 0.5 \
    --lambda 0.8 \
    --reward-scale 0.02 \
    --epsilon-floor 0.05 \
    --buffer-size 12000 \
    --output-dir "$OUTPUT_DIR"

echo "Test run finished. Output in $OUTPUT_DIR"
