"""Check environment info and success detection."""
import sys
sys.path.insert(0, '..')  # Add parent directory

from environments import VisualMazeEnv

env = VisualMazeEnv('complex')
obs, info = env.reset()

print("Initial info keys:", list(info.keys()))
print("Initial info:", info)

# Take random actions and look for rewards
print("\nTaking random actions:")
for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if reward != -0.04:  # Not just step penalty
        print(f"  Step {i}: reward={reward:.2f}, terminated={terminated}, info={info}")
    
    if terminated or truncated:
        print(f"  Episode ended at step {i}")
        break