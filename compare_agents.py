"""
Compare baseline vs emotional DQN agents.
Runs multiple training runs and compares performance.
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from train import train
from utils import MetricsLogger, ExperimentLogger, RunMetrics


def run_comparison(
    maze_name: str = "minimal",
    n_runs: int = 3,
    n_episodes: int = 1000,
    base_seed: int = 42,
    log_dir: str = "experiments",
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> ExperimentLogger:
    """
    Run comparison experiment between baseline and emotional agents.
    
    Args:
        maze_name: Name of maze to use
        n_runs: Number of runs per agent type
        n_episodes: Episodes per run
        base_seed: Base random seed (incremented for each run)
        log_dir: Directory for experiment logs
        device: Device to use
        config: Training configuration
        verbose: Whether to print progress
    
    Returns:
        ExperimentLogger with all results
    """
    if config is None:
        config = {}
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{maze_name}_comparison_{timestamp}"
    experiment_dir = Path(log_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create experiment logger
    exp_logger = ExperimentLogger(
        experiment_dir=str(experiment_dir),
        experiment_name=experiment_name
    )
    
    if verbose:
        print("\n" + "="*70)
        print(f"EXPERIMENT: {experiment_name}")
        print("="*70)
        print(f"  Maze: {maze_name}")
        print(f"  Runs per agent: {n_runs}")
        print(f"  Episodes per run: {n_episodes}")
        print(f"  Total training runs: {n_runs * 2}")
        print(f"  Device: {device or 'auto'}")
        print(f"  Output: {experiment_dir}")
        print(f"  Config: {config}")
        print("="*70)
    
    # Calculate total runs for overall progress
    total_runs = n_runs * 2
    current_run = 0
    
    # Overall progress bar
    overall_pbar = tqdm(
        total=total_runs,
        desc="Overall Progress",
        position=0,
        leave=True,
        colour='green'
    )
    
    # Run experiments for each agent type
    for agent_type in ['baseline', 'emotional']:
        if verbose:
            print(f"\n{'='*70}")
            print(f"TRAINING {agent_type.upper()} AGENT ({n_runs} runs)")
            print(f"{'='*70}")
        
        for run_id in range(n_runs):
            seed = base_seed + run_id * 100
            current_run += 1
            
            overall_pbar.set_description(
                f"Overall [{agent_type} run {run_id + 1}/{n_runs}]"
            )
            
            if verbose:
                print(f"\n--- {agent_type} Run {run_id + 1}/{n_runs} (seed={seed}) ---")
            
            logger = train(
                maze_name=maze_name,
                agent_type=agent_type,
                n_episodes=n_episodes,
                seed=seed,
                log_dir=str(experiment_dir / agent_type),
                run_id=run_id,
                device=device,
                config=config,
                verbose=True,
                progress_every=n_episodes // 5
            )
            
            run_metrics = logger.get_run_metrics()
            exp_logger.add_run(run_metrics)
            
            overall_pbar.update(1)
            
            if verbose:
                print(f"\n  Run {run_id + 1} complete:")
                print(f"    First success: episode {run_metrics.first_success_episode}")
                print(f"    Final success rate: {run_metrics.final_success_rate:.1%}")
                print(f"    Final avg steps: {run_metrics.final_avg_steps:.1f}")
    
    overall_pbar.close()
    
    if verbose:
        exp_logger.print_comparison()
    
    exp_logger.save_comparison()
    
    if verbose:
        print(f"\nResults saved to: {experiment_dir}")
    
    return exp_logger


def quick_test(
    maze_name: str = "minimal",
    n_episodes: int = 100,
    seed: int = 42,
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Quick test to verify both agents work before full experiment.
    
    Args:
        maze_name: Maze to test on
        n_episodes: Episodes to run (short)
        seed: Random seed
        device: Device to use
        config: Training configuration
    """
    print("\n" + "="*70)
    print("QUICK TEST - Verifying both agents work")
    print("="*70)
    
    # Use provided config or defaults with CORRECT emotional params
    if config is None:
        config = {}
    
    # Ensure emotional params are set correctly
    test_config = {
        'epsilon_decay_steps': config.get('epsilon_decay_steps', 5000),
        'buffer_size': config.get('buffer_size', 10000),
        'lambda_mood': config.get('lambda_mood', 0.8),
        'beta': config.get('beta', 1.0),
        **config  # Include any other config values
    }
    
    print(f"  lambda_mood: {test_config['lambda_mood']}")
    print(f"  beta: {test_config['beta']}")
    print("="*70)
    
    for agent_type in ['baseline', 'emotional']:
        print(f"\n--- Testing {agent_type} agent ---")
        
        try:
            logger = train(
                maze_name=maze_name,
                agent_type=agent_type,
                n_episodes=n_episodes,
                seed=seed,
                log_dir="test_runs",
                run_id=0,
                device=device,
                config=test_config,
                verbose=True,
                progress_every=n_episodes + 1
            )
            
            summary = logger.get_summary(last_n=n_episodes)
            print(f"\n  ✓ {agent_type}: {summary.get('total_successes', 0)} successes, "
                  f"avg reward: {summary.get('avg_reward', 0):.2f}")
            
        except Exception as e:
            print(f"  ✗ {agent_type} FAILED: {e}")
            raise
    
    print("\n" + "="*70)
    print("✓ Quick test passed! Both agents work.")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs emotional DQN agents"
    )
    
    # Experiment settings
    parser.add_argument('--maze', type=str, default='minimal',
                       help='Name of maze to use')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per agent type')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Episodes per run')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--log_dir', type=str, default='experiments',
                       help='Directory for experiment logs')
    parser.add_argument('--eta', type=float, default=0.9,
                   help='Balance between TD and mood (0.9 = 90% TD, 10% mood)')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test to verify agents work')
    
    # Training config
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--buffer_size', type=int, default=50000,
                       help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epsilon_decay', type=int, default=20000,
                       help='Epsilon decay steps')
    
    # Emotional parameters - UPDATED DEFAULTS
    parser.add_argument('--lambda_mood', type=float, default=0.8,
                       help='Mood persistence (0-1, higher = slower change)')
    
    args = parser.parse_args()
    
    # Build config with ALL parameters
    config = {
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'buffer_size': args.buffer_size,
        'batch_size': args.batch_size,
        'epsilon_decay_steps': args.epsilon_decay,
        'lambda_mood': args.lambda_mood,
        'eta': args.eta,
    }
    
    if args.quick_test:
        # Pass config to quick_test
        quick_test(
            maze_name=args.maze,
            n_episodes=100,
            seed=args.seed,
            device=args.device,
            config=config
        )
    else:
        run_comparison(
            maze_name=args.maze,
            n_runs=args.runs,
            n_episodes=args.episodes,
            base_seed=args.seed,
            log_dir=args.log_dir,
            device=args.device,
            config=config,
            verbose=True
        )


if __name__ == "__main__":
    main()