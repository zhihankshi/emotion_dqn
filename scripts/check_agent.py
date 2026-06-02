"""Check agent checkpoint details."""
import torch
import glob

# Find most recent agents
baseline_files = sorted(glob.glob('test_runs/complex_*/baseline_agent.pt'))
emotional_files = sorted(glob.glob('test_runs/complex_*/emotional_agent.pt'))

if baseline_files:
    baseline = baseline_files[-1]
    print(f"Baseline: {baseline}")
    b = torch.load(baseline, map_location='cpu')
    print(f"  Steps: {b.get('steps', 'unknown')}")
    print(f"  Updates: {b.get('updates', 'unknown')}")
    print(f"  Epsilon: {b.get('epsilon', 'unknown')}")

if emotional_files:
    emotional = emotional_files[-1]
    print(f"\nEmotional: {emotional}")
    e = torch.load(emotional, map_location='cpu')
    print(f"  Steps: {e.get('steps', 'unknown')}")
    print(f"  Updates: {e.get('updates', 'unknown')}")
    print(f"  Epsilon: {e.get('epsilon', 'unknown')}")