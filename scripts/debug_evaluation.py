"""Debug why agents fail at epsilon=0."""
import torch
import numpy as np
from environments import VisualMazeEnv
from agents import DQNAgent, EmotionalDQNAgent
from agents.dqn import masked_action_selection


def debug_episode(agent_path: str, agent_type: str, maze_name: str = 'complex'):
    """Watch agent behavior step by step."""
    
    env = VisualMazeEnv(maze_name=maze_name)
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Load agent
    if agent_type == 'baseline':
        agent = DQNAgent(observation_shape=observation_shape, n_actions=n_actions)
    else:
        agent = EmotionalDQNAgent(observation_shape=observation_shape, n_actions=n_actions)
    
    agent.load(agent_path)
    agent.epsilon = 0.0  # No exploration
    
    print(f"\n{'='*60}")
    print(f"DEBUGGING {agent_type.upper()} AGENT")
    print(f"{'='*60}")
    print(f"Epsilon: {agent.epsilon}")
    
    obs, info = env.reset()
    
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    # Track positions to detect loops
    position_history = []
    action_history = []
    
    for step in range(50):  # Just first 50 steps
        with torch.no_grad():
            state_t = torch.from_numpy(obs).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_t).cpu().numpy()[0]
        
        valid_actions = env.get_valid_actions()
        action = masked_action_selection(q_values, valid_actions, epsilon=0.0, training=False)
        
        # Get position (if available)
        pos = info.get('agent_pos', info.get('position', 'unknown'))
        has_key = info.get('has_key', False)
        
        print(f"Step {step:3d} | Pos: {pos} | Key: {has_key} | "
              f"Action: {action_names[action]:5s} | "
              f"Q-values: [{q_values[0]:.2f}, {q_values[1]:.2f}, {q_values[2]:.2f}, {q_values[3]:.2f}]")
        
        position_history.append(str(pos))
        action_history.append(action)
        
        # Check for loops (same position repeated)
        if len(position_history) > 4:
            last_4 = position_history[-4:]
            if len(set(last_4)) <= 2:
                print(f"\n*** LOOP DETECTED! Agent stuck between positions ***")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"\n*** SUCCESS at step {step}! ***")
            break
        if truncated:
            print(f"\n*** TIMEOUT ***")
            break
    
    # Analyze action distribution
    print(f"\n{'='*60}")
    print("ACTION DISTRIBUTION (first 50 steps):")
    for i, name in enumerate(action_names):
        count = action_history.count(i)
        pct = count / len(action_history) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Check for repetitive behavior
    print(f"\nMost common action sequences:")
    for length in [2, 3, 4]:
        sequences = []
        for i in range(len(action_history) - length + 1):
            seq = tuple(action_history[i:i+length])
            sequences.append(seq)
        
        from collections import Counter
        common = Counter(sequences).most_common(3)
        print(f"  Length {length}: {common}")


if __name__ == "__main__":
    import sys
    import glob
    
    # Find most recent runs
    baseline_files = sorted(glob.glob('test_runs/complex_*/baseline_agent.pt'))
    emotional_files = sorted(glob.glob('test_runs/complex_*/emotional_agent.pt'))
    
    if baseline_files:
        print("\nMost recent baseline:", baseline_files[-1])
        debug_episode(baseline_files[-1], 'baseline')
    
    if emotional_files:
        print("\nMost recent emotional:", emotional_files[-1])
        debug_episode(emotional_files[-1], 'emotional')