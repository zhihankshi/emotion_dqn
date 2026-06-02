"""Evaluate with small epsilon to break loops."""
import torch
import numpy as np
from environments import VisualMazeEnv
from agents import DQNAgent, EmotionalDQNAgent
import glob


def evaluate_with_epsilon(agent_path: str, agent_type: str, epsilon: float = 0.05, n_episodes: int = 20):
    """Evaluate agent with small epsilon."""
    
    env = VisualMazeEnv(maze_name='complex')
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    if agent_type == 'baseline':
        agent = DQNAgent(observation_shape=observation_shape, n_actions=n_actions)
    else:
        agent = EmotionalDQNAgent(observation_shape=observation_shape, n_actions=n_actions)
    
    agent.load(agent_path)
    agent.epsilon = epsilon
    
    print(f"\n{'='*60}")
    print(f"EVALUATING {agent_type.upper()} (epsilon={epsilon})")
    print(f"{'='*60}")
    
    successes = 0
    total_rewards = []
    total_steps = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() < agent.epsilon:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    state_t = torch.from_numpy(obs).unsqueeze(0).to(agent.device)
                    q_values = agent.policy_net(state_t)
                    action = q_values.argmax(dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        success = terminated and info.get('at_goal', False)
        if success:
            successes += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
    
    print(f"  Success Rate: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"  Avg Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Avg Steps: {np.mean(total_steps):.1f}")
    
    return successes / n_episodes


if __name__ == "__main__":
    # Find agents
    baseline_files = sorted(glob.glob('test_runs/complex_*/baseline_agent.pt'))
    emotional_files = sorted(glob.glob('test_runs/complex_*/emotional_agent.pt'))
    
    print("\nComparing evaluation with different epsilon values:\n")
    
    for eps in [0.0, 0.05, 0.1]:
        print(f"\n{'='*60}")
        print(f"EPSILON = {eps}")
        print(f"{'='*60}")
        
        if baseline_files:
            evaluate_with_epsilon(baseline_files[-1], 'baseline', epsilon=eps)
        
        if emotional_files:
            evaluate_with_epsilon(emotional_files[-1], 'emotional', epsilon=eps)