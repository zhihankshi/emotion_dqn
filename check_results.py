# check_results.py
import json
from pathlib import Path

# Find the latest experiment
exp_dir = Path("experiments")
latest = sorted(exp_dir.iterdir())[-1]
print(f"Checking: {latest}")

# Load comparison
comparison_file = list(latest.glob("*_comparison.json"))[0]
with open(comparison_file) as f:
    data = json.load(f)

print("\nAll runs:")
for agent_type, runs in data['all_runs'].items():
    print(f"\n{agent_type}:")
    for run in runs:
        print(f"  Run {run['run_id']}: first_success={run['first_success_episode']}, "
              f"success_rate={run['final_success_rate']:.2%}")