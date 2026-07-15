"""
Analyze how agent's policy and Q-values evolve during training.
Specifically look at episodes around performance peaks and drops.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from environments import VisualMazeEnv
from agents.dqn import masked_action_selection


def load_agent_checkpoint(
    checkpoint_path,
    env,
    agent_type='baseline',
    network_size='standard',
    image_size=64,
):
    """Load agent from a training checkpoint."""
    from agents import DQNAgent, EmotionalDQNAgent, DQNNetwork, SmallDQNNetwork

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if agent_type is None and isinstance(checkpoint, dict):
        agent_type = checkpoint.get("agent_type", "baseline")

    if network_size == 'small' or image_size < 36:
        network_class = SmallDQNNetwork
    else:
        network_class = DQNNetwork

    # Use the env's observation space so frame-stacked agents load correctly
    observation_shape = tuple(env.observation_space.shape)

    if agent_type == 'emotional':
        agent = EmotionalDQNAgent(
            observation_shape=observation_shape,
            n_actions=env.action_space.n,
            network_class=network_class,
        )
    else:
        agent = DQNAgent(
            observation_shape=observation_shape,
            n_actions=env.action_space.n,
            network_class=network_class,
        )

    if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
        agent.load_checkpoint(checkpoint_path)
    elif isinstance(checkpoint, dict):
        agent.policy_net.load_state_dict(checkpoint.get("policy_net", checkpoint))
    else:
        agent.policy_net.load_state_dict(checkpoint)

    agent.epsilon = 0.0  # Greedy for analysis
    return agent


def get_q_values_for_state(agent, obs):
    """Get Q-values for a given observation."""
    with torch.no_grad():
        state_t = torch.from_numpy(obs).unsqueeze(0).float().to(agent.device)
        q_values = agent.policy_net(state_t).cpu().numpy()[0]
    return q_values


def analyze_start_position(agent, env):
    """Analyze Q-values and policy at starting position."""
    obs, info = env.reset()
    q_values = get_q_values_for_state(agent, obs)
    valid_actions = env.get_valid_actions()

    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    best_action_idx = masked_action_selection(
        q_values, valid_actions, epsilon=0.0, training=False
    )
    best_action = action_names[best_action_idx]

    valid_q = q_values[valid_actions]
    
    return {
        'position': info['agent_pos'],
        'q_values': q_values,
        'valid_actions': valid_actions,
        'best_action': best_action,
        'q_max': valid_q.max(),
        'q_min': valid_q.min(),
        'q_spread': valid_q.max() - valid_q.min()
    }


def run_greedy_episode(agent, env, max_steps=100):
    """Run a greedy episode and record trajectory."""
    obs, info = env.reset()
    trajectory = []
    total_reward = 0
    
    for step in range(max_steps):
        q_values = get_q_values_for_state(agent, obs)
        valid_actions = env.get_valid_actions()
        action = masked_action_selection(q_values, valid_actions, epsilon=0.0, training=False)
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        trajectory.append({
            'step': step,
            'position': info['agent_pos'],
            'has_key': info['has_key'],
            'door_open': info.get('door_open', False),
            'q_values': q_values.copy(),
            'action': action_names[action],
            'q_max': q_values.max()
        })
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            trajectory.append({
                'step': step + 1,
                'position': info['agent_pos'],
                'has_key': info['has_key'],
                'door_open': info.get('door_open', False),
                'q_values': None,
                'action': 'DONE' if terminated else 'TIMEOUT',
                'q_max': None
            })
            break
    
    return trajectory, total_reward


def analyze_checkpoints(
    checkpoint_dir,
    maze_name,
    episodes_to_analyze=None,
    agent_type=None,
    network_size='standard',
    image_size=64,
    reward_overrides=None,
    max_steps=None,
    shield_lights_up=None,
    frame_stack=1,
):
    """Analyze policy at different checkpoints."""
    
    env = VisualMazeEnv(
        maze_name=maze_name,
        image_size=image_size,
        reward_overrides=reward_overrides,
        max_steps=max_steps,
        shield_lights_up=shield_lights_up,
        frame_stack=frame_stack,
    )
    
    if episodes_to_analyze is None:
        episodes_to_analyze = [60, 80, 100, 150, 200, 250, 300, 400, 500, 800]
    
    # Find checkpoints
    checkpoint_dir = Path(checkpoint_dir)
    all_checkpoints = sorted(checkpoint_dir.glob("agent_episode_*.pt"))
    if not all_checkpoints:
        all_checkpoints = sorted(checkpoint_dir.glob("*_episode_*.pt"))
    
    if not all_checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return None
    
    print(f"Found {len(all_checkpoints)} checkpoints")
    
    results = []
    
    for target_ep in episodes_to_analyze:
        # Find closest checkpoint
        checkpoint_path = None
        for cp in all_checkpoints:
            ep_num = int(cp.stem.split("_episode_")[-1])
            if ep_num == target_ep:
                checkpoint_path = str(cp)
                break
        
        if checkpoint_path is None:
            print(f"  Episode {target_ep}: No checkpoint found")
            continue
        
        print(f"\n{'='*60}")
        print(f"EPISODE {target_ep}")
        print(f"{'='*60}")
        
        # Load agent
        agent = load_agent_checkpoint(
            checkpoint_path,
            env,
            agent_type=agent_type,
            network_size=network_size,
            image_size=image_size,
        )
        
        # Get mood if emotional agent
        mood = getattr(agent, 'mood_tracker', None)
        mood_value = mood.mood if mood else None
        
        # Analyze start position
        start_analysis = analyze_start_position(agent, env)
        print(f"\nStart Position {start_analysis['position']}:")
        print(f"  Q-values: [{', '.join([f'{q:.3f}' for q in start_analysis['q_values']])}]")
        print(f"  Best Action: {start_analysis['best_action']}")
        print(f"  Q-spread: {start_analysis['q_spread']:.3f}")
        if mood_value is not None:
            print(f"  Mood: {mood_value:.4f}")
        
        # Run greedy episode
        trajectory, total_reward = run_greedy_episode(agent, env)
        
        got_key = any(t['has_key'] for t in trajectory)
        reached_goal = trajectory[-1]['action'] == 'DONE'
        steps = len(trajectory) - 1
        
        print(f"\nGreedy Episode Result:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Got Key: {got_key}")
        print(f"  Reached Goal: {reached_goal}")
        
        # Print trajectory
        print(f"\nTrajectory (first 15 steps):")
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        for t in trajectory[:15]:
            if t['q_values'] is not None:
                q_str = ', '.join([f'{q:6.2f}' for q in t['q_values']])
                print(f"  Step {t['step']:2d}: {str(t['position']):10s} -> {t['action']:5s} | Q: [{q_str}]")
            else:
                print(f"  Step {t['step']:2d}: {str(t['position']):10s} -> {t['action']}")
        
        results.append({
            'episode': target_ep,
            'start_q_values': start_analysis['q_values'],
            'best_start_action': start_analysis['best_action'],
            'q_spread': start_analysis['q_spread'],
            'q_max': start_analysis['q_max'],
            'mood': mood_value,
            'greedy_reward': total_reward,
            'greedy_steps': steps,
            'got_key': got_key,
            'reached_goal': reached_goal
        })
    
    return pd.DataFrame(results)


def plot_policy_evolution(results_df, save_path=None):
    """Plot how policy evolves over training."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Q-values at start over time (only valid actions shown solid)
    ax = axes[0, 0]
    q_vals = np.array(results_df['start_q_values'].tolist())
    valid_set = set(results_df['valid_actions'].iloc[0]) if 'valid_actions' in results_df else {0,1,2,3}
    for i, action in enumerate(['UP', 'DOWN', 'LEFT', 'RIGHT']):
        if i in valid_set:
            ax.plot(results_df['episode'], q_vals[:, i], 'o-', label=action)
        else:
            ax.plot(results_df['episode'], q_vals[:, i], 'x--', alpha=0.3,
                    label=f'{action} (invalid)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-value at Start')
    ax.set_title('Q-values at Starting Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-spread over time
    ax = axes[0, 1]
    ax.plot(results_df['episode'], results_df['q_spread'], 'go-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-spread')
    ax.set_title('Q-value Spread (max - min)')
    ax.grid(True, alpha=0.3)
    
    # Best action over time
    ax = axes[0, 2]
    action_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
    action_nums = [action_map[a] for a in results_df['best_start_action']]
    ax.plot(results_df['episode'], action_nums, 'bo-', markersize=10)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['UP', 'DOWN', 'LEFT', 'RIGHT'])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Best Action')
    ax.set_title('Policy at Starting Position')
    ax.grid(True, alpha=0.3)
    
    # Greedy episode reward
    ax = axes[1, 0]
    ax.plot(results_df['episode'], results_df['greedy_reward'], 'ro-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Greedy Episode Reward')
    ax.set_title('Performance of Greedy Policy')
    ax.grid(True, alpha=0.3)
    
    # Mood over time (if available)
    ax = axes[1, 1]
    if 'mood' in results_df.columns and results_df['mood'].notna().any():
        ax.plot(results_df['episode'], results_df['mood'], 'mo-')
        ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mood')
    ax.set_title('Mood at Checkpoint')
    ax.grid(True, alpha=0.3)
    
    # Key collection and goal reaching
    ax = axes[1, 2]
    ax.plot(results_df['episode'], results_df['got_key'].astype(int), 'go-', label='Got Key')
    ax.plot(results_df['episode'], results_df['reached_goal'].astype(int), 'bo-', label='Reached Goal')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success (0/1)')
    ax.set_title('Key Collection & Goal Reaching')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Policy Evolution During Training', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--maze', type=str, default='complex')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Observation image size (64 for standard, 7 for 1px/cell)')
    parser.add_argument('--network_size', type=str, default='standard',
                        choices=['standard', 'small'],
                        help='Network architecture size')
    parser.add_argument('--agent_type', type=str, default=None,
                        choices=['baseline', 'emotional'],
                        help='Agent type (auto-detected from checkpoint if omitted)')
    
    parser.add_argument('--episodes', type=str, default='60,80,100,150,200,250,300,400,500,800')
    parser.add_argument('--frame_stack', type=int, default=1,
                        help='Frames stacked in the training environment')
    args = parser.parse_args()
    
    episodes = [int(e) for e in args.episodes.split(',')]
    
    results = analyze_checkpoints(
        args.checkpoint_dir,
        args.maze,
        episodes,
        agent_type=args.agent_type,
        network_size=args.network_size,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
    )
    
    if results is not None:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(results.to_string())
        
        plot_policy_evolution(results, 'policy_evolution.png')