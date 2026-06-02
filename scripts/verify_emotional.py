"""Verify emotional agent is working correctly."""
import numpy as np
from environments import VisualMazeEnv
from agents import EmotionalDQNAgent

print("Creating environment and agent...")
env = VisualMazeEnv(maze_name="minimal")
agent = EmotionalDQNAgent(
    observation_shape=(64, 64, 3),
    n_actions=4,
    buffer_size=10000,
    batch_size=32,
    lambda_mood=0.8,
    beta=1.0,
    seed=42
)

print(f"Agent beta: {agent.beta}")
print(f"Agent lambda_mood: {agent.mood_tracker.lambda_mood}")

print("\nRunning 2 episodes...")
all_moods = []

for ep in range(2):
    obs, info = env.reset()
    done = False
    ep_moods = []
    
    while not done:
        action = agent.select_action(obs, training=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # This is what train.py does
        update_metrics = agent.step(obs, action, reward, next_obs, done)
        
        if update_metrics and 'mood' in update_metrics:
            ep_moods.append(update_metrics['mood'])
        
        obs = next_obs
    
    if ep_moods:
        print(f"  Episode {ep}: collected {len(ep_moods)} mood values, "
              f"mean={np.mean(ep_moods):+.4f}, final={ep_moods[-1]:+.4f}")
        all_moods.extend(ep_moods)
    else:
        print(f"  Episode {ep}: NO MOOD VALUES COLLECTED!")

if all_moods:
    print(f"\nTotal mood values: {len(all_moods)}")
    print(f"Overall mean mood: {np.mean(all_moods):+.4f}")
else:
    print("\nERROR: No mood values were returned from agent.step()!")
    print("Check that update() returns 'mood' in its dictionary.")