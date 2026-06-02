"""
Evaluate trained agent with epsilon=0 (no random exploration).
This shows true learned policy performance.
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import json

from environments import VisualMazeEnv
from agents import DQNAgent, EmotionalDQNAgent


def load_agent(agent_path: str, agent_type: str, env: VisualMazeEnv) -> object:
    """Load a trained agent from checkpoint."""
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    if agent_type == 'baseline':
        agent = DQNAgent(
            observation_shape=observation_shape,
            n_actions=n_actions,
        )
    else:
        agent = EmotionalDQNAgent(
            observation_shape=observation_shape,
            n_actions=n_actions,
        )
    
    agent.load(agent_path)
    return agent


def evaluate_episode(env: VisualMazeEnv, agent: object) -> dict:
    """
    Run single evaluation episode with epsilon=0.
    """
    obs, info = env.reset()
    
    total_reward = 0
    steps = 0
    done = False
    
    trajectory = []
    q_values_history = []
    
    while not done:
        with torch.no_grad():
            state_t = torch.from_numpy(obs).unsqueeze(0).to(agent.device)
            q_values = agent.policy_net(state_t).cpu().numpy()[0]
        
        # NO exploration - pure Q-value action selection
        action = np.argmax(q_values)
        
        q_values_history.append({
            'step': steps,
            'q_values': q_values.tolist(),
            'chosen_action': action,
            'max_q': float(q_values.max()),
        })
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        steps += 1
        
        trajectory.append({
            'step': steps,
            'action': action,
            'reward': reward,
            'has_key': info.get('has_key', False),
            'door_open': info.get('door_open', False),
        })
        
        obs = next_obs
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'success': terminated and info.get('at_goal', False),
        'trajectory': trajectory,
        'q_values_history': q_values_history,
        'key_collected': any(t['has_key'] for t in trajectory),
        'door_opened': any(t['door_open'] for t in trajectory),
    }


def evaluate_agent(
    agent_path: str,
    agent_type: str,
    maze_name: str = 'complex',
    n_episodes: int = 20,
) -> tuple:
    """
    Evaluate agent over multiple episodes with epsilon=0.
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {agent_type.upper()} AGENT (epsilon=0)")
    print(f"{'='*60}")
    print(f"  Agent: {agent_path}")
    print(f"  Maze: {maze_name}")
    print(f"  Episodes: {n_episodes}")
    
    env = VisualMazeEnv(maze_name=maze_name)
    agent = load_agent(agent_path, agent_type, env)
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    print(f"  Original epsilon: {original_epsilon:.4f}")
    print(f"  Evaluation epsilon: {agent.epsilon:.4f}")
    print(f"{'='*60}\n")
    
    results = []
    
    for ep in tqdm(range(n_episodes), desc="Evaluating"):
        result = evaluate_episode(env, agent)
        result['episode'] = ep
        results.append(result)
    
    # Create summary DataFrame
    df = pd.DataFrame([{
        'episode': r['episode'],
        'total_reward': r['total_reward'],
        'steps': r['steps'],
        'success': r['success'],
        'key_collected': r['key_collected'],
        'door_opened': r['door_opened'],
    } for r in results])
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS (epsilon=0)")
    print(f"{'='*60}")
    print(f"  Success Rate: {df['success'].mean()*100:.1f}%")
    print(f"  Avg Reward: {df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}")
    print(f"  Avg Steps: {df['steps'].mean():.1f} ± {df['steps'].std():.1f}")
    print(f"  Key Collection Rate: {df['key_collected'].mean()*100:.1f}%")
    print(f"  Door Open Rate: {df['door_opened'].mean()*100:.1f}%")
    print(f"{'='*60}")
    
    return df, results


def compare_evaluations(
    baseline_path: str,
    emotional_path: str,
    maze_name: str = 'complex',
    n_episodes: int = 20,
    output_dir: str = 'eval_results'
):
    """Compare both agents with epsilon=0 and save results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("EPSILON-FREE EVALUATION COMPARISON")
    print("="*70)
    
    # Evaluate baseline
    baseline_df, baseline_results = evaluate_agent(
        baseline_path, 'baseline', maze_name, n_episodes
    )
    
    # Evaluate emotional
    emotional_df, emotional_results = evaluate_agent(
        emotional_path, 'emotional', maze_name, n_episodes
    )
    
    # Save DataFrames
    baseline_df.to_csv(output_path / 'baseline_eval.csv', index=False)
    emotional_df.to_csv(output_path / 'emotional_eval.csv', index=False)
    
    # Save summary
    summary = {
        'maze': maze_name,
        'n_episodes': n_episodes,
        'baseline': {
            'success_rate': float(baseline_df['success'].mean()),
            'avg_reward': float(baseline_df['total_reward'].mean()),
            'std_reward': float(baseline_df['total_reward'].std()),
            'avg_steps': float(baseline_df['steps'].mean()),
            'key_collection_rate': float(baseline_df['key_collected'].mean()),
            'door_open_rate': float(baseline_df['door_opened'].mean()),
        },
        'emotional': {
            'success_rate': float(emotional_df['success'].mean()),
            'avg_reward': float(emotional_df['total_reward'].mean()),
            'std_reward': float(emotional_df['total_reward'].std()),
            'avg_steps': float(emotional_df['steps'].mean()),
            'key_collection_rate': float(emotional_df['key_collected'].mean()),
            'door_open_rate': float(emotional_df['door_opened'].mean()),
        }
    }
    
    with open(output_path / 'eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Comparison table
    print("\n" + "="*70)
    print("COMPARISON (epsilon=0, No Random Exploration)")
    print("="*70)
    print(f"{'Metric':<25} {'Baseline':<20} {'Emotional':<20}")
    print("-"*70)
    print(f"{'Success Rate':<25} {baseline_df['success'].mean()*100:.1f}%{'':<15} {emotional_df['success'].mean()*100:.1f}%")
    print(f"{'Avg Reward':<25} {baseline_df['total_reward'].mean():.2f} ± {baseline_df['total_reward'].std():.2f}{'':<6} {emotional_df['total_reward'].mean():.2f} ± {emotional_df['total_reward'].std():.2f}")
    print(f"{'Avg Steps':<25} {baseline_df['steps'].mean():.1f}{'':<18} {emotional_df['steps'].mean():.1f}")
    print(f"{'Key Collection':<25} {baseline_df['key_collected'].mean()*100:.1f}%{'':<15} {emotional_df['key_collected'].mean()*100:.1f}%")
    print(f"{'Door Opened':<25} {baseline_df['door_opened'].mean()*100:.1f}%{'':<15} {emotional_df['door_opened'].mean()*100:.1f}%")
    print("="*70)
    
    print(f"\nResults saved to: {output_path}")
    
    return baseline_df, emotional_df, summary


def visualize_evaluation(output_dir: str = 'eval_results'):
    """Visualize epsilon=0 evaluation results."""
    import matplotlib.pyplot as plt
    
    output_path = Path(output_dir)
    
    # Load data
    baseline_df = pd.read_csv(output_path / 'baseline_eval.csv')
    emotional_df = pd.read_csv(output_path / 'emotional_eval.csv')
    
    with open(output_path / 'eval_summary.json') as f:
        summary = json.load(f)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = {'baseline': 'blue', 'emotional': 'red'}
    
    # Plot 1: Success Rate Bar Chart
    ax = axes[0, 0]
    success_rates = [
        summary['baseline']['success_rate'] * 100,
        summary['emotional']['success_rate'] * 100
    ]
    bars = ax.bar(['Baseline', 'Emotional'], success_rates, color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate (epsilon=0)')
    ax.set_ylim(0, 105)
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.1f}%', ha='center', fontsize=12)
    
    # Plot 2: Reward Distribution
    ax = axes[0, 1]
    ax.boxplot([baseline_df['total_reward'], emotional_df['total_reward']], 
               labels=['Baseline', 'Emotional'])
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward Distribution (epsilon=0)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Plot 3: Steps Distribution
    ax = axes[1, 0]
    ax.boxplot([baseline_df['steps'], emotional_df['steps']], 
               labels=['Baseline', 'Emotional'])
    ax.set_ylabel('Episode Steps')
    ax.set_title('Steps to Complete (epsilon=0)')
    
    # Plot 4: Episode-by-Episode Comparison
    ax = axes[1, 1]
    episodes = range(len(baseline_df))
    ax.plot(episodes, baseline_df['total_reward'], 'b-', alpha=0.7, label='Baseline')
    ax.plot(episodes, emotional_df['total_reward'], 'r-', alpha=0.7, label='Emotional')
    ax.set_xlabel('Evaluation Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Rewards (epsilon=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Evaluation Results (No Exploration)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    save_path = output_path / 'evaluation_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate agents with epsilon=0")
    parser.add_argument('--baseline', type=str, help='Path to baseline agent')
    parser.add_argument('--emotional', type=str, help='Path to emotional agent')
    parser.add_argument('--maze', type=str, default='complex', help='Maze name')
    parser.add_argument('--episodes', type=int, default=20, help='Evaluation episodes')
    parser.add_argument('--output', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Only visualize existing results')
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_evaluation(args.output)
    else:
        if not args.baseline or not args.emotional:
            print("Error: --baseline and --emotional paths required")
            print("Or use --visualize to view existing results")
            exit(1)
        
        compare_evaluations(
            args.baseline,
            args.emotional,
            args.maze,
            args.episodes,
            args.output
        )
        
        # Auto-visualize after evaluation
        visualize_evaluation(args.output)