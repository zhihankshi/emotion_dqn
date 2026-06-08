"""
Compare baseline vs emotional DQN agents.
Runs multiple training runs and compares performance.
Saves policy checkpoints during training for later analysis.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from train import train, get_checkpoint_episodes
from agents import SmallDQNNetwork
from utils import ExperimentLogger


def get_run_checkpoint_dir(run_log_dir: Path) -> Path:
    """Return the checkpoints directory for a training run."""
    return Path(run_log_dir) / "checkpoints"


def list_saved_checkpoints(checkpoint_dir: Path) -> List[Path]:
    """List checkpoint files saved during training."""
    if not checkpoint_dir.exists():
        return []
    return sorted(checkpoint_dir.glob("agent_episode_*.pt"))


def build_run_checkpoint_entry(
    agent_type: str,
    run_id: int,
    seed: int,
    run_log_dir: Path,
    n_episodes: int,
    checkpoint_interval: int,
) -> Dict[str, Any]:
    """Build manifest entry for one training run's checkpoints."""
    checkpoint_dir = get_run_checkpoint_dir(run_log_dir)
    checkpoints = list_saved_checkpoints(checkpoint_dir)

    return {
        "agent_type": agent_type,
        "run_id": run_id,
        "seed": seed,
        "run_log_dir": str(run_log_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_count": len(checkpoints),
        "checkpoints": [str(p) for p in checkpoints],
        "planned_checkpoint_episodes": sorted(
            get_checkpoint_episodes(n_episodes, checkpoint_interval)
        ),
    }


def save_checkpoint_manifest(
    experiment_dir: Path,
    manifest: List[Dict[str, Any]],
    maze_name: str,
    n_episodes: int,
    checkpoint_interval: int,
) -> Path:
    """Save checkpoint manifest for the experiment."""
    path = experiment_dir / "checkpoint_manifest.json"
    payload = {
        "maze_name": maze_name,
        "n_episodes": n_episodes,
        "checkpoint_interval": checkpoint_interval,
        "checkpoint_filename_format": "agent_episode_{episode}.pt",
        "runs": manifest,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def analyze_run_checkpoints(
    entry: Dict[str, Any],
    maze_name: str,
    analysis_episodes: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Run policy evolution analysis on checkpoints from one training run."""
    from analyze_policy_evolution import analyze_checkpoints, plot_policy_evolution

    checkpoint_dir = entry["checkpoint_dir"]
    agent_type = entry["agent_type"]
    run_id = entry["run_id"]

    print(f"\nAnalyzing checkpoints: {agent_type} run {run_id}")
    print(f"  Directory: {checkpoint_dir}")

    results = analyze_checkpoints(
        checkpoint_dir=checkpoint_dir,
        maze_name=maze_name,
        episodes_to_analyze=analysis_episodes,
        agent_type=agent_type,
    )

    if results is None or results.empty:
        print(f"  No analysis results for {agent_type} run {run_id}")
        return

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{agent_type}_run{run_id}_policy_evolution.csv"
        plot_path = output_dir / f"{agent_type}_run{run_id}_policy_evolution.png"
        results.to_csv(csv_path, index=False)
        plot_policy_evolution(results, str(plot_path))
        print(f"  Saved analysis to {csv_path}")
        print(f"  Saved plot to {plot_path}")


def run_comparison(
    maze_name: str = "minimal",
    n_runs: int = 3,
    n_episodes: int = 1000,
    base_seed: int = 42,
    log_dir: str = "experiments",
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    checkpoint_interval: int = 50,
    analyze_checkpoints: bool = False,
    analysis_episodes: Optional[List[int]] = None,
    image_size: int = 64,
    network_class=None,
) -> ExperimentLogger:
    """
    Run comparison experiment between baseline and emotional agents.

    Each training run saves checkpoints under:
        {experiment_dir}/{agent_type}/{run_log_dir}/checkpoints/
    """
    if config is None:
        config = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{maze_name}_comparison_{timestamp}"
    experiment_dir = Path(log_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    exp_logger = ExperimentLogger(
        experiment_dir=str(experiment_dir),
        experiment_name=experiment_name,
    )

    planned_episodes = sorted(get_checkpoint_episodes(n_episodes, checkpoint_interval))

    if verbose:
        print("\n" + "=" * 70)
        print(f"EXPERIMENT: {experiment_name}")
        print("=" * 70)
        print(f"  Maze: {maze_name}")
        print(f"  Runs per agent: {n_runs}")
        print(f"  Episodes per run: {n_episodes}")
        print(f"  Checkpoint interval: {checkpoint_interval}")
        print(f"  Planned checkpoint episodes: {planned_episodes}")
        print(f"  Total training runs: {n_runs * 2}")
        print(f"  Image size: {image_size}")
        print(f"  Network: {network_class.__name__ if network_class else 'DQNNetwork'}")
        print(f"  Device: {device or 'auto'}")
        print(f"  Output: {experiment_dir}")
        print(f"  Config: {config}")
        print("=" * 70)

    total_runs = n_runs * 2
    checkpoint_manifest: List[Dict[str, Any]] = []

    overall_pbar = tqdm(
        total=total_runs,
        desc="Overall Progress",
        position=0,
        leave=True,
        colour="green",
    )

    for agent_type in ["baseline", "emotional"]:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"TRAINING {agent_type.upper()} AGENT ({n_runs} runs)")
            print(f"{'=' * 70}")

        for run_id in range(n_runs):
            seed = base_seed + run_id * 100

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
                progress_every=max(1, n_episodes // 5),
                checkpoint_interval=checkpoint_interval,
                image_size=image_size,
                network_class=network_class,
            )

            run_metrics = logger.get_run_metrics()
            exp_logger.add_run(run_metrics)

            entry = build_run_checkpoint_entry(
                agent_type=agent_type,
                run_id=run_id,
                seed=seed,
                run_log_dir=Path(logger.log_dir),
                n_episodes=n_episodes,
                checkpoint_interval=checkpoint_interval,
            )
            checkpoint_manifest.append(entry)

            overall_pbar.update(1)

            if verbose:
                print(f"\n  Run {run_id + 1} complete:")
                print(f"    First success: episode {run_metrics.first_success_episode}")
                print(f"    Final success rate: {run_metrics.final_success_rate:.1%}")
                print(f"    Final avg steps: {run_metrics.final_avg_steps:.1f}")
                print(f"    Checkpoints saved: {entry['checkpoint_count']}")
                print(f"    Checkpoint dir: {entry['checkpoint_dir']}")

    overall_pbar.close()

    manifest_path = save_checkpoint_manifest(
        experiment_dir=experiment_dir,
        manifest=checkpoint_manifest,
        maze_name=maze_name,
        n_episodes=n_episodes,
        checkpoint_interval=checkpoint_interval,
    )

    if verbose:
        exp_logger.print_comparison()

    exp_logger.save_comparison()

    if analyze_checkpoints:
        analysis_dir = experiment_dir / "policy_analysis"
        if verbose:
            print(f"\n{'=' * 70}")
            print("CHECKPOINT POLICY ANALYSIS")
            print("=" * 70)

        for entry in checkpoint_manifest:
            analyze_run_checkpoints(
                entry=entry,
                maze_name=maze_name,
                analysis_episodes=analysis_episodes,
                output_dir=analysis_dir,
            )

    if verbose:
        print(f"\nResults saved to: {experiment_dir}")
        print(f"Checkpoint manifest: {manifest_path}")
        if analyze_checkpoints:
            print(f"Policy analysis: {experiment_dir / 'policy_analysis'}")

    return exp_logger


def quick_test(
    maze_name: str = "minimal",
    n_episodes: int = 100,
    seed: int = 42,
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    checkpoint_interval: int = 50,
    image_size: int = 64,
    network_class=None,
) -> None:
    """Quick test to verify both agents work and save checkpoints."""
    print("\n" + "=" * 70)
    print("QUICK TEST - Verifying both agents work")
    print("=" * 70)

    if config is None:
        config = {}

    test_config = {
        "buffer_size": config.get("buffer_size", 10000),
        "lambda_mood": config.get("lambda_mood", 0.8),
        "beta": config.get("beta", 1.0),
        **config,
    }

    print(f"  lambda_mood: {test_config['lambda_mood']}")
    print(f"  beta: {test_config['beta']}")
    print(f"  image_size: {image_size}")
    print(f"  network: {network_class.__name__ if network_class else 'DQNNetwork'}")
    print(f"  checkpoint_interval: {checkpoint_interval}")
    print("=" * 70)

    for agent_type in ["baseline", "emotional"]:
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
                progress_every=n_episodes + 1,
                checkpoint_interval=checkpoint_interval,
                image_size=image_size,
                network_class=network_class,
            )

            checkpoint_dir = get_run_checkpoint_dir(Path(logger.log_dir))
            checkpoints = list_saved_checkpoints(checkpoint_dir)
            summary = logger.get_summary(last_n=n_episodes)

            print(
                f"\n  OK {agent_type}: {summary.get('total_successes', 0)} successes, "
                f"avg reward: {summary.get('avg_reward', 0):.2f}, "
                f"checkpoints: {len(checkpoints)}"
            )
            print(f"  Checkpoint dir: {checkpoint_dir}")

        except Exception as e:
            print(f"  FAILED {agent_type}: {e}")
            raise

    print("\n" + "=" * 70)
    print("Quick test passed! Both agents work.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs emotional DQN agents"
    )

    parser.add_argument("--maze", type=str, default="minimal",
                        help="Name of maze to use")
    parser.add_argument("--image_size", type=int, default=64,
                        help="Observation image size (64 for standard, 7 for 1px/cell)")
    parser.add_argument("--network_size", type=str, default="standard",
                        choices=["standard", "small"],
                        help="Network architecture size")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per agent type")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Episodes per run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--log_dir", type=str, default="experiments",
                        help="Directory for experiment logs")
    parser.add_argument("--eta", type=float, default=0.9,
                        help="Balance between TD and mood (0.9 = 90% TD, 10% mood)")

    parser.add_argument("--quick_test", action="store_true",
                        help="Run quick test to verify agents work")

    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Save checkpoints every N episodes (1-based)")
    parser.add_argument("--analyze_checkpoints", action="store_true",
                        help="Analyze saved checkpoints after training")
    parser.add_argument(
        "--analysis_episodes", type=str, default=None,
        help="Comma-separated checkpoint episodes to analyze "
             "(default: manifest planned episodes)",
    )

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--buffer_size", type=int, default=50000,
                        help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument(
        "--epsilon_decay", type=int, default=None,
        help="Epsilon decay env steps (default: episodes * maze max_steps)",
    )

    parser.add_argument("--lambda_mood", type=float, default=0.8,
                        help="Mood persistence (0-1, higher = slower change)")

    args = parser.parse_args()

    config = {
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "lambda_mood": args.lambda_mood,
        "eta": args.eta,
    }
    if args.epsilon_decay is not None:
        config["epsilon_decay_steps"] = args.epsilon_decay

    analysis_episodes = None
    if args.analysis_episodes:
        analysis_episodes = [int(e) for e in args.analysis_episodes.split(",")]

    network_class = None
    if args.network_size == "small":
        network_class = SmallDQNNetwork

    if args.quick_test:
        quick_test(
            maze_name=args.maze,
            n_episodes=100,
            seed=args.seed,
            device=args.device,
            config=config,
            checkpoint_interval=args.checkpoint_interval,
            image_size=args.image_size,
            network_class=network_class,
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
            verbose=True,
            checkpoint_interval=args.checkpoint_interval,
            analyze_checkpoints=args.analyze_checkpoints,
            analysis_episodes=analysis_episodes,
            image_size=args.image_size,
            network_class=network_class,
        )


if __name__ == "__main__":
    main()
