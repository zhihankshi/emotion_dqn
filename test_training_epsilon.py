"""Test with epsilon levels that worked during training."""
import torch
import numpy as np
from environments import VisualMazeEnv
from agents import EmotionalDQNAgent
import glob


def evaluate(agent, env, epsilon: float, n_episodes: int = 20):
    """Evaluate agent."""
    successes = 0
    rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(obs).unsqueeze(0).to(agent.device)
                    q_values = agent.policy_net(state_t)
                    action = q_values.argmax(dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        if terminated and info.get('door_open', False):
            successes += 1
        rewards.append(total_reward)
    
    return successes / n_episodes, np.mean(rewards)


# Load agent
emotional_files = sorted(glob.glob('test_runs/complex_*/emotional_agent.pt'))
if emotional_files:
    env = VisualMazeEnv(maze_name='complex')
    agent = EmotionalDQNAgent(
        observation_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    )
    agent.load(emotional_files[-1])
    
    print("Testing with training-level epsilon values:")
    print("=" * 50)
    print(f"{'Epsilon':<12} {'Success Rate':<15} {'Avg Reward':<12}")
    print("-" * 50)
    
    for eps in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        success_rate, avg_reward = evaluate(agent, env, eps)
        print(f"{eps:<12.1f} {success_rate*100:<15.1f}% {avg_reward:<12.2f}")
    
    # Compare to pure random
    print("-" * 50)
    print("Pure random baseline (epsilon=1.0):")
    success_rate, avg_reward = evaluate(agent, env, 1.0)
    print(f"{1.0:<12.1f} {success_rate*100:<15.1f}% {avg_reward:<12.2f}")