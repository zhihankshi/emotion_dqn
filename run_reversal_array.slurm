#!/bin/bash
#SBATCH --job-name=reversal_be
#SBATCH --account=zshi47          
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00                      # TODO: tighten once run_reversal_single.slurm tells you a real number
#SBATCH --array=0-9                          # 2 agent types x 5 seeds = 10 tasks (indices 0-9)
#SBATCH --output=logs/reversal_be_%A_%a.out
#SBATCH --error=logs/reversal_be_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zshi47@jhu.edu         

mkdir -p logs

module reset
module load anaconda3/2024.02-1
conda activate honeylab

cd "$SLURM_SUBMIT_DIR"

# --- map SLURM_ARRAY_TASK_ID -> (agent_type, seed) ---
AGENT_TYPES=(baseline emotional)
SEEDS=(0 1 2 3 4)

N_SEEDS=${#SEEDS[@]}
AGENT_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))

AGENT_TYPE=${AGENT_TYPES[$AGENT_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# TEMPORARY: scratch access at /weka/scratch/jhu/$USER isn't provisioned yet — using home.
# Switch back to /weka/scratch/jhu/$USER once ARCH confirms access.
SCRATCH_ROOT=$HOME/emotional_dqn_runs
OUTPUT_DIR=${SCRATCH_ROOT}/reversal_runs/${AGENT_TYPE}_seed${SEED}
mkdir -p "$OUTPUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID -> agent_type=$AGENT_TYPE seed=$SEED -> $OUTPUT_DIR"

python scripts/train_reversal.py \
    --agent-type "$AGENT_TYPE" \
    --seed "$SEED" \
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
