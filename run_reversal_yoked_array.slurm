#!/bin/bash
#SBATCH --job-name=reversal_yoked
#SBATCH --account=zshi47          
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-4                          # 5 seeds
#SBATCH --output=logs/reversal_yoked_%A_%a.out
#SBATCH --error=logs/reversal_yoked_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zshi47@jhu.edu         

mkdir -p logs

module reset
module load anaconda3/2024.02-1
conda activate honeylab

cd ~/emotional_dqn

SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Yoke against a DIFFERENT seed's completed emotional trace, not this seed's own —
# otherwise the "control" trivially matches itself. Simple offset by one, wrapping around.
N_SEEDS=${#SEEDS[@]}
TRACE_SEED_IDX=$(( (SLURM_ARRAY_TASK_ID + 1) % N_SEEDS ))
TRACE_SEED=${SEEDS[$TRACE_SEED_IDX]}

# TEMPORARY: scratch access at /weka/scratch/jhu/$USER isn't provisioned yet — using home.
# Switch back to /weka/scratch/jhu/$USER once ARCH confirms access.
SCRATCH_ROOT=$HOME/emotional_dqn_runs
RUNS_ROOT=${SCRATCH_ROOT}/reversal_runs
MOOD_TRACE_PATH="${RUNS_ROOT}/emotional_seed${TRACE_SEED}/mood_trace.json"

if [ ! -f "$MOOD_TRACE_PATH" ]; then
    echo "ERROR: expected mood trace not found at $MOOD_TRACE_PATH" >&2
    echo "This job depends on the emotional array having completed successfully first." >&2
    exit 1
fi

OUTPUT_DIR=${RUNS_ROOT}/yoked_seed${SEED}
mkdir -p "$OUTPUT_DIR"

echo "Task $SLURM_ARRAY_TASK_ID -> yoked seed=$SEED, tracing emotional seed=$TRACE_SEED -> $OUTPUT_DIR"

python scripts/train_reversal.py \
    --agent-type yoked \
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
    --yoked-mode replay_trace \
    --mood-trace-path "$MOOD_TRACE_PATH" \
    --output-dir "$OUTPUT_DIR"
