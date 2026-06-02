"""Debug mood tracking with different parameters."""
import numpy as np
from environments import VisualMazeEnv
from agents import EmotionalDQNAgent

# Test different parameter combinations
params_to_test = [
    {'lambda_mood': 0.95, 'beta': 0.1, 'name': 'Original'},
    {'lambda_mood': 0.8,  'beta': 0.1, 'name': 'Lower lambda'},
    {'lambda_mood': 0.8,  'beta': 1.0, 'name': 'Lower lambda + Higher beta'},
    {'lambda_mood': 0.5,  'beta': 1.0, 'name': 'Fast mood + Higher beta'},
]

env = VisualMazeEnv(maze_name="minimal")

for params in params_to_test:
    print(f"\n{'='*60}")
    print(f"Testing: {params['name']}")
    print(f"  lambda_mood={params['lambda_mood']}, beta={params['beta']}")
    print(f"{'='*60}")
    
    agent = EmotionalDQNAgent(
        observation_shape=(64, 64, 3),
        n_actions=4,
        buffer_size=10000,
        batch_size=32,
        lambda_mood=params['lambda_mood'],
        beta=params['beta'],
        seed=42
    )
    
    # Temporarily disable episode reset
    agent.reset_episode = lambda: None
    
    all_moods = []
    all_biases = []
    
    for episode in range(5):
        obs, info = env.reset()
        done = False
        step = 0
        
        while not done and step < 100:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            metrics = agent.step(obs, action, reward, next_obs, done)
            
            if metrics:
                all_moods.append(metrics['mood'])
                all_biases.append(metrics['mood_bias'])
            
            obs = next_obs
            step += 1
    
    print(f"\n  Results over 5 episodes:")
    print(f"    Mood  - min: {min(all_moods):+.4f}, max: {max(all_moods):+.4f}, final: {all_moods[-1]:+.4f}")
    print(f"    Bias  - min: {min(all_biases):+.4f}, max: {max(all_biases):+.4f}, final: {all_biases[-1]:+.4f}")
    print(f"    Bias magnitude vs step penalty (-0.04): {abs(all_biases[-1]) / 0.04 * 100:.1f}%")