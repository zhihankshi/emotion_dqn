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
from typing import Dict, Any, Optional, Set

from environments import VisualMazeEnv
from agents import DQNAgent, EmotionalDQNAgent, DQNNetwork, SmallDQNNetwork
from utils import EpisodeMetrics, MetricsLogger

# Extra analysis checkpoints (1-based episode numbers)
ANALYSIS_CHECKPOINT_EPISODES = {60, 80, 100, 150, 200, 250, 300, 400, 500}


def get_checkpoint_episodes(
    n_episodes: int,
    checkpoint_interval: int,
) -> Set[int]:
    """Return 1-based episode numbers where checkpoints should be saved."""
    episodes = set(ANALYSIS_CHECKPOINT_EPISODES)
    for ep in range(checkpoint_interval, n_episodes + 1, checkpoint_interval):
        episodes.add(ep)
    return {ep for ep in episodes if ep <= n_episodes}


def should_save_checkpoint(
    episode_num: int,
    checkpoint_episodes: Set[int],
) -> bool:
    """Whether to save a checkpoint after completing this 1-based episode."""
    return episode_num in checkpoint_episodes


def create_agent(
    agent_type: str,
    observation_shape: tuple,
    n_actions: int,
    config: Dict[str, Any],
    device: str,
    seed: int,
    network_class=None
):
    """Create agent based on type."""
    
    common_params = {
        'observation_shape': observation_shape,
        'n_actions': n_actions,
        'learning_rate': config.get('learning_rate', 1e-4),
        'gamma': config.get('gamma', 0.99),
        'epsilon_start': config.get('epsilon_start', 1.0),
        'epsilon_end': config.get('epsilon_end', 0.05),
        'buffer_size': config.get('buffer_size', 50000),
        'batch_size': config.get('batch_size', 32),
        'target_update_freq': config.get('target_update_freq', 1000),
        'device': device,
        'seed': seed,
        'network_class': network_class,
    }
    
    if agent_type == 'baseline':
        return DQNAgent(**common_params)
    
    elif agent_type == 'emotional':
        emotional_params = {
            'lambda_mood': config.get('lambda_mood', 0.8),
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
    
    # Record success (terminated = reached goal/exit, truncated = timeout)
    metrics.success = terminated
    
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
    progress_every: int = 100,
    checkpoint_interval: int = 50,
    image_size: int = 64,
    network_class=None,
    pretrained_checkpoint: Optional[str] = None,
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
        checkpoint_interval: Save checkpoints every N episodes (1-based)
        image_size: Size of square observation image (e.g. 64 or 7)
        network_class: Network class override (default: auto-select based on size)
        pretrained_checkpoint: Optional path to load policy weights before training
    
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
    env = VisualMazeEnv(maze_name=maze_name, image_size=image_size)

    epsilon_start = config.get('epsilon_start', 1.0)
    epsilon_end = config.get('epsilon_end', 0.05)

    if verbose:
        print(f"  Observation shape: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.n}")
        print(
            f"  Epsilon decay: {epsilon_start} -> {epsilon_end} "
            f"linearly over {n_episodes} episodes"
        )
    
    # Create agent
    agent = create_agent(
        agent_type=agent_type,
        observation_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        config=config,
        device=device,
        seed=seed,
        network_class=network_class,
    )

    if pretrained_checkpoint:
        saved_episode = agent.load_checkpoint(pretrained_checkpoint)
        if verbose:
            print(f"  Loaded pretrained checkpoint: {pretrained_checkpoint}")
            print(f"    Saved at episode: {saved_episode}")
    
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

    checkpoint_dir = run_log_dir / "checkpoints"
    checkpoint_episodes = get_checkpoint_episodes(n_episodes, checkpoint_interval)
    
    if verbose:
        print(f"  Logging to: {run_log_dir}")
        print(f"  Checkpoints: {checkpoint_dir}")
        print(f"  Checkpoint episodes: {sorted(checkpoint_episodes)}")
        print(f"\nStarting training...")
    
    # Training loop
    pbar = tqdm(range(n_episodes), disable=not verbose)
    
    for episode in pbar:
        agent.update_epsilon_for_episode(episode, n_episodes)

        # Train one episode
        metrics = train_episode(env, agent, episode, training=True)
        
        # Log
        logger.log_episode(metrics)

        episode_num = episode + 1
        if should_save_checkpoint(episode_num, checkpoint_episodes):
            checkpoint_path = checkpoint_dir / f"agent_episode_{episode_num}.pt"
            agent.save_checkpoint(str(checkpoint_path), episode=episode_num)
        
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
        print(f"  Checkpoints saved to: {checkpoint_dir}")
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
    parser.add_argument('--image_size', type=int, default=64,
                       help='Observation image size (64 for standard, 7 for 1px/cell)')
    parser.add_argument('--network_size', type=str, default='standard',
                       choices=['standard', 'small'],
                       help='Network architecture size')
    
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
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                       help='Save checkpoints every N episodes (1-based)')
    
    # DQN hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--buffer_size', type=int, default=50000,
                       help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
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
        # Emotional params
        'lambda_mood': args.lambda_mood,
        'beta': args.beta,
    }
    
    # Resolve network class
    network_class = None
    if args.network_size == 'small':
        network_class = SmallDQNNetwork
    
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
        progress_every=args.progress_every,
        checkpoint_interval=args.checkpoint_interval,
        image_size=args.image_size,
        network_class=network_class,
    )


if __name__ == "__main__":
    main()