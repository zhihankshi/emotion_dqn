"""Check reward structure and optimal path reward."""
from environments import VisualMazeEnv
import numpy as np


def analyze_rewards():
    env = VisualMazeEnv('complex')
    
    print("="*60)
    print("REWARD STRUCTURE ANALYSIS")
    print("="*60)
    
    # Get maze config
    if hasattr(env, 'maze_config'):
        config = env.maze_config
    elif hasattr(env, 'config'):
        config = env.config
    else:
        print("Cannot find maze config!")
        return
    
    print(f"\nMaze: complex")
    print(f"Size: {config.get('size', 'unknown')}")
    print(f"Agent start: {config.get('agent_start', 'unknown')}")
    print(f"Key position: {config.get('key_position', 'unknown')}")
    print(f"Door position: {config.get('door_position', 'unknown')}")
    print(f"Goal position: {config.get('goal_position', 'unknown')}")
    
    # Check reward values from config or env
    print(f"\n{'='*60}")
    print("REWARD VALUES")
    print("="*60)
    
    rewards = config.get('rewards', {})
    if rewards:
        print("From config:")
        for key, value in rewards.items():
            print(f"  {key}: {value}")
    
    # Test actual rewards by taking actions
    print(f"\n{'='*60}")
    print("TESTING ACTUAL REWARDS")
    print("="*60)
    
    obs, info = env.reset()
    print(f"Starting position: {info['agent_pos']}")
    
    # Collect reward samples
    reward_samples = {
        'step': [],
        'wall_collision': [],
        'key_pickup': [],
        'door_open': [],
        'goal': []
    }
    
    # Run many random episodes to collect rewards
    for episode in range(50):
        obs, info = env.reset()
        prev_pos = info['agent_pos']
        had_key = False
        door_was_open = False
        
        for step in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            curr_pos = info['agent_pos']
            has_key = info['has_key']
            door_open = info['door_open']
            
            # Categorize reward
            if has_key and not had_key:
                reward_samples['key_pickup'].append(reward)
                print(f"  Key pickup reward: {reward}")
            elif door_open and not door_was_open:
                reward_samples['door_open'].append(reward)
                print(f"  Door open reward: {reward}")
            elif terminated:
                reward_samples['goal'].append(reward)
                print(f"  Goal reward: {reward}")
            elif curr_pos == prev_pos:
                reward_samples['wall_collision'].append(reward)
            else:
                reward_samples['step'].append(reward)
            
            had_key = has_key
            door_was_open = door_open
            prev_pos = curr_pos
            
            if terminated or truncated:
                break
    
    print(f"\n{'='*60}")
    print("REWARD SUMMARY")
    print("="*60)
    
    for reward_type, values in reward_samples.items():
        if values:
            print(f"{reward_type}:")
            print(f"  Count: {len(values)}")
            print(f"  Mean: {np.mean(values):.4f}")
            print(f"  Unique values: {set(values)}")
    
    # Calculate optimal reward
    print(f"\n{'='*60}")
    print("OPTIMAL PATH CALCULATION")
    print("="*60)
    
    # Estimate optimal path length
    start = config.get('agent_start', [0, 0])
    key_pos = config.get('key_position', [0, 6])
    door_pos = config.get('door_position', [3, 3])
    goal_pos = config.get('goal_position', [6, 6])
    
    # Manhattan distances (minimum steps)
    dist_to_key = abs(start[0] - key_pos[0]) + abs(start[1] - key_pos[1])
    dist_key_to_door = abs(key_pos[0] - door_pos[0]) + abs(key_pos[1] - door_pos[1])
    dist_door_to_goal = abs(door_pos[0] - goal_pos[0]) + abs(door_pos[1] - goal_pos[1])
    
    min_steps = dist_to_key + dist_key_to_door + dist_door_to_goal
    
    print(f"Start → Key: {dist_to_key} steps")
    print(f"Key → Door: {dist_key_to_door} steps")
    print(f"Door → Goal: {dist_door_to_goal} steps")
    print(f"Minimum total steps: {min_steps}")
    
    # Calculate optimal reward (assuming values from samples)
    step_penalty = np.mean(reward_samples['step']) if reward_samples['step'] else -0.04
    key_reward = np.mean(reward_samples['key_pickup']) if reward_samples['key_pickup'] else 1.0
    door_reward = np.mean(reward_samples['door_open']) if reward_samples['door_open'] else 2.0
    goal_reward = np.mean(reward_samples['goal']) if reward_samples['goal'] else 10.0
    
    optimal_reward = (min_steps * step_penalty) + key_reward + door_reward + goal_reward
    
    print(f"\nOptimal reward calculation:")
    print(f"  Step penalty: {step_penalty} × {min_steps} steps = {step_penalty * min_steps:.2f}")
    print(f"  Key reward: +{key_reward}")
    print(f"  Door reward: +{door_reward}")
    print(f"  Goal reward: +{goal_reward}")
    print(f"  OPTIMAL TOTAL: {optimal_reward:.2f}")
    
    print(f"\n{'='*60}")
    print("COMPARISON TO TRAINING RESULTS")
    print("="*60)
    print(f"Optimal reward: ~{optimal_reward:.1f}")
    print(f"Best training reward: ~11-12 (from earlier results)")
    print(f"Match: {'Yes' if abs(optimal_reward - 12) < 3 else 'No'}")


if __name__ == "__main__":
    analyze_rewards()