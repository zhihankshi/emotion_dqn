"""Check epsilon values during successful training episodes."""
import pandas as pd
import glob

files = sorted(glob.glob('test_runs/complex_*/emotional_run0_episodes.csv'))

if files:
    df = pd.read_csv(files[-1])
    
    # Look at successful episodes
    successes = df[df['success'] == True]
    
    print("Successful episodes:")
    print(successes[['episode', 'epsilon', 'steps', 'total_reward']].head(20))
    
    print(f"\nEpsilon range during successes:")
    print(f"  Min epsilon: {successes['epsilon'].min():.4f}")
    print(f"  Max epsilon: {successes['epsilon'].max():.4f}")
    print(f"  Mean epsilon: {successes['epsilon'].mean():.4f}")
    
    print(f"\nTotal successes: {len(successes)}")
    print(f"Success rate: {len(successes) / len(df) * 100:.1f}%")
else:
    print("No training files found")