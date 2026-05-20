"""
Training script for DQN agents on visual maze.
Supports both baseline and emotional agents.
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any, Optional

from environments import VisualMazeEnv
from agents import DQNAgent, EmotionalDQNAgent
from utils import EpisodeMetrics, MetricsLogger

def create_agent(
    agent_type: str,
    observation_shape: tuple,
    n_actions: int,
    config: Dict[str, Any],
    device: str,
    seed: int
):
    """Create agent based on type."""
    
    common_params = {
        'observation_shape': observation_shape,
        'n_actions': n_actions,
        'learning_rate': config.get('learning_rate', 1e-4),
        'gamma': config.get('gamma', 0.99),
        'epsilon_start': config.get('epsilon_start', 1.0),
        'epsilon_end': config.get('epsilon_end', 0.05),
        'epsilon_decay_steps': config.get('epsilon_decay_steps', 50000),
        'buffer_size': config.get('buffer_size', 50000),
        'batch_size': config.get('batch_size', 32),
        'target_update_freq': config.get('target_update_freq', 1000),
        'device': device,
        'seed': seed,
    }
    
    if agent_type == 'baseline':
        return DQNAgent(**common_params)
    
    elif agent_type == 'emotional':
        # SIMPLIFIED: Only 2 emotional parameters now
        emotional_params = {
            'lambda_mood': config.get('lambda_mood', 0.95),
            'beta': config.get('beta', 0.1),
        }
        return EmotionalDQNAgent(**common_params, **emotional_params)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_episode(
    env,
    agent,
    episode_num: int,
    training: bool = True
) -> EpisodeMetrics:
    """Run a single training episode."""
    obs, info = env.reset()
    
    metrics = EpisodeMetrics(episode=episode_num)
    
    # Accumulators
    losses = []
    td_errors = []
    q_values = []
    moods = []
    
    done = False
    
    while not done:
        action = agent.select_action(obs, training=training)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        metrics.total_reward += reward
        metrics.steps += 1
        
        # Track causal understanding
        if info['has_key'] and metrics.key_found_step == -1:
            metrics.key_found_step = metrics.steps
        
        if info['door_open'] and metrics.door_opened_step == -1:
            metrics.door_opened_step = metrics.steps
        
        metrics.door_attempts_without_key = info.get('door_attempts_without_key', 0)
        
        # Update agent
        if training:
            update_metrics = agent.step(obs, action, reward, next_obs, done)
            
            if update_metrics:
                losses.append(update_metrics.get('loss', 0))
                td_errors.append(update_metrics.get('td_error', 0))
                q_values.append(update_metrics.get('q_value_mean', 0))
                
                # Track mood if emotional agent
                if 'mood' in update_metrics:
                    moods.append(update_metrics['mood'])
        
        obs = next_obs
    
    # Record success
    metrics.success = terminated and info.get('door_open', False)
    
    # Average metrics
    if losses:
        metrics.mean_loss = np.mean(losses)
        metrics.mean_td_error = np.mean(td_errors)
        metrics.mean_q_value = np.mean(q_values)
    
    # Record mood (this is the key fix!)
    if moods:
        metrics.mean_overall_mood = np.mean(moods)
    else:
        metrics.mean_overall_mood = 0.0
    
    # Get epsilon from agent
    metrics.epsilon = getattr(agent, 'epsilon', 0.0)
    
    # Reset episode (for emotional agent, this should do nothing now)
    if hasattr(agent, 'reset_episode'):
        agent.reset_episode()
    
    return metrics
    """
    Run a single training episode.
    
    Returns:
        EpisodeMetrics for this episode
    """
    obs, info = env.reset()
    
    metrics = EpisodeMetrics(episode=episode_num)
    
    # Accumulators for averaging
    losses = []
    td_errors = []
    q_values = []
    mood_values = []
    mood_actions = []
    overall_moods = []
    mood_biases = []
    
    done = False
    
    while not done:
        # Select action
        action = agent.select_action(obs, training=training)
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        metrics.total_reward += reward
        metrics.steps += 1
        
        # Track causal understanding
        if info['has_key'] and metrics.key_found_step == -1:
            metrics.key_found_step = metrics.steps
        
        if info['door_open'] and metrics.door_opened_step == -1:
            metrics.door_opened_step = metrics.steps
        
        metrics.door_attempts_without_key = info['door_attempts_without_key']
        
        # Update agent
        if training:
            update_metrics = agent.step(obs, action, reward, next_obs, done)
            
            if update_metrics:
                losses.append(update_metrics.get('loss', 0))
                td_errors.append(update_metrics.get('td_error_mean', 0))
                q_values.append(update_metrics.get('q_value_mean', 0))
                
                # Emotional metrics
                if 'mood_value' in update_metrics:
                    mood_values.append(update_metrics['mood_value'])
                    mood_actions.append(update_metrics['mood_action'])
                    overall_moods.append(update_metrics['overall_mood'])
                    mood_biases.append(update_metrics.get('mood_bias', 0))
        
        obs = next_obs
    
    # Record success
    metrics.success = terminated and info['door_open']
    
    # Average metrics
    if losses:
        metrics.mean_loss = np.mean(losses)
        metrics.mean_td_error = np.mean(td_errors)
        metrics.mean_q_value = np.mean(q_values)
    
    if mood_values:
        metrics.mean_mood_value = np.mean(mood_values)
        metrics.mean_mood_action = np.mean(mood_actions)
        metrics.mean_overall_mood = np.mean(overall_moods)
        metrics.mean_mood_bias = np.mean(mood_biases)
    
    # Get final epsilon
    metrics.epsilon = agent.epsilon
    
    # Get effective epsilon and exploration boosts for emotional agent
    if hasattr(agent, 'mood_system'):
        metrics.effective_epsilon = agent.mood_system.get_exploration_boost(
            agent.epsilon,
            agent.exploration_boost_scale
        )
        metrics.exploration_boosts = agent.exploration_boosts
        
        # Reset episode-level mood (partial reset)
        agent.reset_episode()
    else:
        metrics.effective_epsilon = agent.epsilon
    
    return metrics


def train(
    maze_name: str = "minimal",
    agent_type: str = "emotional",
    n_episodes: int = 1000,
    seed: int = 42,
    log_dir: str = "runs",
    run_id: int = 0,
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    progress_every: int = 100
) -> MetricsLogger:
    """
    Train an agent on the maze.
    
    Args:
        maze_name: Name of maze to use
        agent_type: 'baseline' or 'emotional'
        n_episodes: Number of episodes to train
        seed: Random seed
        log_dir: Directory for logs
        run_id: ID for this run
        device: 'cuda' or 'cpu'
        config: Additional configuration
        verbose: Whether to print progress
        progress_every: Print progress every N episodes
    
    Returns:
        MetricsLogger with all episode data
    """
    if config is None:
        config = {}
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {agent_type} agent on {maze_name}")
        print(f"{'='*60}")
        print(f"  Device: {device}")
        print(f"  Episodes: {n_episodes}")
        print(f"  Seed: {seed}")
        print(f"  Run ID: {run_id}")
    
    # Create environment
    env = VisualMazeEnv(maze_name=maze_name, image_size=64)
    
    if verbose:
        print(f"  Observation shape: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.n}")
    
    # Create agent
    agent = create_agent(
        agent_type=agent_type,
        observation_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        config=config,
        device=device,
        seed=seed
    )
    
    if verbose:
        n_params = sum(p.numel() for p in agent.policy_net.parameters())
        print(f"  Network parameters: {n_params:,}")
    
    # Create logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = Path(log_dir) / f"{maze_name}_{timestamp}"
    
    logger = MetricsLogger(
        log_dir=run_log_dir,
        agent_type=agent_type,
        maze_name=maze_name,
        run_id=run_id
    )
    
    if verbose:
        print(f"  Logging to: {run_log_dir}")
        print(f"\nStarting training...")
    
    # Training loop
    pbar = tqdm(range(n_episodes), disable=not verbose)
    
    for episode in pbar:
        # Train one episode
        metrics = train_episode(env, agent, episode, training=True)
        
        # Log
        logger.log_episode(metrics)
        
        # Update progress bar
        if episode >= 100:
            success_rate = logger.get_success_rate(100)
            avg_steps = logger.get_avg_steps(100)
            pbar.set_postfix({
                'success': f'{success_rate:.1%}',
                'steps': f'{avg_steps:.0f}',
                'ε': f'{metrics.epsilon:.3f}'
            })
        
        # Print detailed progress
        if verbose and episode > 0 and episode % progress_every == 0:
            logger.print_progress(episode, every=progress_every)
    
    # Save final summary
    logger.save_summary()
    
    # Save agent
    agent_path = run_log_dir / f"{agent_type}_agent.pt"
    agent.save(str(agent_path))
    
    if verbose:
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        
        summary = logger.get_summary()
        print(f"  Final success rate: {summary.get('success_rate', 0):.1%}")
        print(f"  Final avg steps: {summary.get('avg_steps', 0):.1f}")
        print(f"  First success: episode {summary.get('first_success', -1)}")
        print(f"  Total successes: {summary.get('total_successes', 0)}/{n_episodes}")
        
        if agent_type == 'emotional':
            print(f"  Avg mood: {summary.get('avg_mood', 0):.4f}")
            print(f"  Total exploration boosts: {summary.get('total_exploration_boosts', 0)}")
        
        print(f"\n  Agent saved to: {agent_path}")
        print(f"  Logs saved to: {run_log_dir}")
    
    return logger


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent on visual maze")
    
    # Required arguments
    parser.add_argument('--agent', type=str, default='emotional',
                       choices=['baseline', 'emotional'],
                       help='Type of agent to train')
    
    # Environment
    parser.add_argument('--maze', type=str, default='minimal',
                       help='Name of maze to use')
    
    # Training
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='runs',
                       help='Directory for logs')
    parser.add_argument('--run_id', type=int, default=0,
                       help='Run ID')
    parser.add_argument('--progress_every', type=int, default=100,
                       help='Print progress every N episodes')
    
    # DQN hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--buffer_size', type=int, default=50000,
                       help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epsilon_decay', type=int, default=50000,
                       help='Epsilon decay steps')
    
    # Emotional parameters (SIMPLIFIED - only 2 now)
    parser.add_argument('--lambda_mood', type=float, default=0.95,
                       help='Mood persistence (0-1, higher = slower change)')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='Mood influence on Q-targets')
    
    args = parser.parse_args()
    
    # Build config (SIMPLIFIED)
    config = {
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'buffer_size': args.buffer_size,
        'batch_size': args.batch_size,
        'epsilon_decay_steps': args.epsilon_decay,
        # Emotional params
        'lambda_mood': args.lambda_mood,
        'beta': args.beta,
    }
    
    # Train
    train(
        maze_name=args.maze,
        agent_type=args.agent,
        n_episodes=args.episodes,
        seed=args.seed,
        log_dir=args.log_dir,
        run_id=args.run_id,
        device=args.device,
        config=config,
        verbose=True,
        progress_every=args.progress_every
    )


if __name__ == "__main__":
    main()