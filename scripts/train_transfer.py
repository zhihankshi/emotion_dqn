"""
Two-phase maze transfer training.

Trains an agent on a source maze, then loads the final checkpoint and
continues training on a target maze with epsilon reset for exploration.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import SmallDQNNetwork
from train import train, resolve_network_class


def resolve_transfer_checkpoint(run_log_dir: Path, n_episodes: int) -> Path:
    """Pick the checkpoint to use for transfer (prefer final episode)."""
    checkpoint_dir = run_log_dir / "checkpoints"
    preferred = checkpoint_dir / f"agent_episode_{n_episodes}.pt"
    if preferred.exists():
        return preferred

    checkpoints = sorted(
        checkpoint_dir.glob("agent_episode_*.pt"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(
        f"No checkpoints found in {checkpoint_dir}. "
        f"Ensure checkpoint_interval divides phase-1 episodes."
    )


def run_transfer_training(
    source_maze: str,
    target_maze: str,
    agent_type: str = "emotional",
    phase1_episodes: int = 500,
    phase2_episodes: int = 500,
    transfer_epsilon: float = 0.3,
    epsilon_end: float = 0.05,
    seed: int = 42,
    log_dir: str = "runs",
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    checkpoint_interval: int = 50,
    image_size: int = 64,
    network_class=None,
    frame_stack: int = 1,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train on source maze, then transfer to target maze.

    Phase 2 resets epsilon to ``transfer_epsilon`` and linearly decays it
    to ``epsilon_end`` over ``phase2_episodes``.
    """
    if config is None:
        config = {}

    config = {
        "learning_rate": config.get("learning_rate", 1e-4),
        "gamma": config.get("gamma", 0.99),
        "buffer_size": config.get("buffer_size", 50000),
        "batch_size": config.get("batch_size", 32),
        "lambda_mood": config.get("lambda_mood", 0.8),
        "eta": config.get("eta", 0.9),
        "mood_clip_range": config.get("mood_clip_range", 1.0),
        "mood_delta_source": config.get("mood_delta_source", "batch_sequential"),
        "mood_delta_signed": config.get("mood_delta_signed", True),
        "reward_scale": config.get("reward_scale", 1.0),
        "epsilon_start": 1.0,
        "epsilon_end": epsilon_end,
        "double_dqn": config.get("double_dqn", False),
        "target_update_freq": config.get("target_update_freq", 1000),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = (
        Path(log_dir) / f"transfer_{source_maze}_to_{target_maze}_{timestamp}"
    )
    phase1_dir = experiment_dir / "phase1"
    phase2_dir = experiment_dir / "phase2"
    phase1_dir.mkdir(parents=True, exist_ok=True)
    phase2_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("TWO-PHASE TRANSFER TRAINING")
        print("=" * 70)
        print(f"  Source maze:       {source_maze}")
        print(f"  Target maze:       {target_maze}")
        print(f"  Agent:             {agent_type}")
        print(f"  Phase 1 episodes:  {phase1_episodes}")
        print(f"  Phase 2 episodes:  {phase2_episodes}")
        print(f"  Transfer epsilon:  {transfer_epsilon} -> {epsilon_end}")
        print(f"  Frame stack:       {frame_stack}")
        print(f"  Double DQN:        {config['double_dqn']}")
        print(f"  Experiment dir:    {experiment_dir}")
        print("=" * 70)

    # Phase 1: source maze
    if verbose:
        print("\n>>> PHASE 1: Training on source maze\n")

    phase1_config = {**config, "epsilon_start": 1.0, "epsilon_end": epsilon_end}
    phase1_logger = train(
        maze_name=source_maze,
        agent_type=agent_type,
        n_episodes=phase1_episodes,
        seed=seed,
        log_dir=str(phase1_dir),
        device=device,
        config=phase1_config,
        verbose=verbose,
        checkpoint_interval=checkpoint_interval,
        image_size=image_size,
        network_class=network_class,
        frame_stack=frame_stack,
    )

    checkpoint_path = resolve_transfer_checkpoint(
        Path(phase1_logger.log_dir),
        phase1_episodes,
    )

    if verbose:
        print(f"\n>>> Transfer checkpoint: {checkpoint_path}")

    # Phase 2: target maze with epsilon reset
    if verbose:
        print("\n>>> PHASE 2: Transfer training on target maze\n")

    phase2_config = {**config, "epsilon_start": transfer_epsilon, "epsilon_end": epsilon_end}
    phase2_logger = train(
        maze_name=target_maze,
        agent_type=agent_type,
        n_episodes=phase2_episodes,
        seed=seed,
        log_dir=str(phase2_dir),
        device=device,
        config=phase2_config,
        verbose=verbose,
        checkpoint_interval=checkpoint_interval,
        image_size=image_size,
        network_class=network_class,
        pretrained_checkpoint=str(checkpoint_path),
        reset_epsilon=transfer_epsilon,
        frame_stack=frame_stack,
    )

    manifest = {
        "source_maze": source_maze,
        "target_maze": target_maze,
        "agent_type": agent_type,
        "phase1_episodes": phase1_episodes,
        "phase2_episodes": phase2_episodes,
        "transfer_epsilon": transfer_epsilon,
        "epsilon_end": epsilon_end,
        "frame_stack": frame_stack,
        "double_dqn": config["double_dqn"],
        "seed": seed,
        "checkpoint_used": str(checkpoint_path),
        "phase1_log_dir": str(phase1_logger.log_dir),
        "phase2_log_dir": str(phase2_logger.log_dir),
        "phase1_summary": phase1_logger.get_summary(last_n=phase1_episodes),
        "phase2_summary": phase2_logger.get_summary(last_n=phase2_episodes),
    }

    manifest_path = experiment_dir / "transfer_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print("\n" + "=" * 70)
        print("TRANSFER TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Phase 1 success rate: {manifest['phase1_summary'].get('success_rate', 0):.1%}")
        print(f"  Phase 2 success rate: {manifest['phase2_summary'].get('success_rate', 0):.1%}")
        if agent_type == "emotional":
            print(f"  Phase 1 avg mood:     {manifest['phase1_summary'].get('avg_mood', 0):.4f}")
            print(f"  Phase 2 avg mood:     {manifest['phase2_summary'].get('avg_mood', 0):.4f}")
        print(f"  Manifest:             {manifest_path}")
        print("=" * 70)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Train on one maze, then transfer and train on another"
    )

    parser.add_argument(
        "--source_maze", type=str, default="shield_trap",
        help="Maze for phase 1 training",
    )
    parser.add_argument(
        "--target_maze", type=str, default="shield_trap_v2",
        help="Maze for phase 2 transfer training",
    )
    parser.add_argument(
        "--agent", type=str, default="emotional",
        choices=["baseline", "emotional"],
        help="Agent type",
    )
    parser.add_argument(
        "--phase1_episodes", type=int, default=500,
        help="Episodes on source maze",
    )
    parser.add_argument(
        "--phase2_episodes", type=int, default=500,
        help="Episodes on target maze after transfer",
    )
    parser.add_argument(
        "--transfer_epsilon", type=float, default=0.3,
        help="Epsilon at start of phase 2 (decays to epsilon_end)",
    )
    parser.add_argument(
        "--epsilon_end", type=float, default=0.05,
        help="Final epsilon for both phases",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument(
        "--network_size", type=str, default="standard",
        choices=["standard", "small"],
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lambda_mood", type=float, default=0.8)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--mood_min", type=float, default=-1.0)
    parser.add_argument("--mood_max", type=float, default=1.0)
    parser.add_argument("--mood_delta_source", type=str, default="batch_sequential",
                        choices=["online", "batch_mean", "batch_sequential"],
                        help="Which delta feeds the mood (see EmotionalDQNAgent docstring)")
    parser.add_argument("--mood_delta_unsigned", action="store_true",
                        help="Integrate |delta| (arousal) instead of signed delta (valence)")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Multiply rewards fed to the agent (all agent types)")
    parser.add_argument("--frame_stack", type=int, default=1,
                        help="Number of frames to stack in observations")
    parser.add_argument("--double_dqn", action="store_true",
                        help="Use Double DQN targets to curb Q overestimation")
    parser.add_argument("--target_update_freq", type=int, default=1000,
                        help="Gradient updates between target network syncs")

    args = parser.parse_args()

    network_class = resolve_network_class(args.network_size, args.image_size)

    config = {
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "lambda_mood": args.lambda_mood,
        "eta": args.eta,
        "mood_bounds": (args.mood_min, args.mood_max),
        "mood_delta_source": args.mood_delta_source,
        "mood_delta_signed": not args.mood_delta_unsigned,
        "reward_scale": args.reward_scale,
        "double_dqn": args.double_dqn,
        "target_update_freq": args.target_update_freq,
    }

    run_transfer_training(
        source_maze=args.source_maze,
        target_maze=args.target_maze,
        agent_type=args.agent,
        phase1_episodes=args.phase1_episodes,
        phase2_episodes=args.phase2_episodes,
        transfer_epsilon=args.transfer_epsilon,
        epsilon_end=args.epsilon_end,
        seed=args.seed,
        log_dir=args.log_dir,
        device=args.device,
        config=config,
        checkpoint_interval=args.checkpoint_interval,
        image_size=args.image_size,
        network_class=network_class,
        frame_stack=args.frame_stack,
    )


if __name__ == "__main__":
    main()
