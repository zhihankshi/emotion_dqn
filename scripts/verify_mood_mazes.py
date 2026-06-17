"""Verify mood trends on mirrored positive/negative reward mazes."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import train


def median_mood_last_n(logger, n: int = 20) -> float:
    """Mean episode mood over the last n completed episodes."""
    episodes = logger.episodes[-n:]
    if not episodes:
        return 0.0
    return float(np.mean([ep.mean_overall_mood for ep in episodes]))


def early_mood_mean(logger, n: int = 20) -> float:
    """Mean episode mood over the first n completed episodes."""
    episodes = logger.episodes[:n]
    if not episodes:
        return 0.0
    return float(np.mean([ep.mean_overall_mood for ep in episodes]))


def late_reward_mean(logger, n: int = 20) -> float:
    """Mean total reward over the last n completed episodes."""
    episodes = logger.episodes[-n:]
    if not episodes:
        return 0.0
    return float(np.mean([ep.total_reward for ep in episodes]))


def verify_maze_mood(
    maze_name: str,
    expected_sign: str,
    n_episodes: int = 100,
    last_n: int = 20,
    seed: int = 42,
    log_dir: str = "test_runs",
) -> bool:
    """Train briefly and check mood sign in late training."""
    print(f"\n{'='*60}")
    print(f"Verifying mood on {maze_name} (expect {expected_sign})")
    print(f"{'='*60}")

    logger = train(
        maze_name=maze_name,
        agent_type="emotional",
        n_episodes=n_episodes,
        seed=seed,
        log_dir=log_dir,
        verbose=False,
        progress_every=n_episodes,
        checkpoint_interval=n_episodes + 1,
        config={
            "lambda_mood": 0.95,
            "buffer_size": 10000,
            "batch_size": 32,
        },
    )

    late_mood = median_mood_last_n(logger, last_n)
    early_mood = early_mood_mean(logger, last_n)
    late_reward = late_reward_mean(logger, last_n)
    all_moods = [ep.mean_overall_mood for ep in logger.episodes]
    print(f"  Episodes: {len(logger.episodes)}")
    print(f"  Mood range: [{min(all_moods):+.4f}, {max(all_moods):+.4f}]")
    print(f"  Early mood (first {last_n}): {early_mood:+.4f}")
    print(f"  Late mood (last {last_n}): {late_mood:+.4f}")
    print(f"  Late mean reward: {late_reward:+.2f}")

    if expected_sign == "positive":
        # Approach maze: positive reward learning with positive mood excursions
        passed = max(all_moods) > 0 and late_reward > 0
    elif expected_sign == "negative":
        # Avoidance maze: negative rewards dominate with negative mood excursions
        passed = min(all_moods) < 0 and late_reward < 0
    else:
        raise ValueError(f"Unknown expected_sign: {expected_sign}")

    status = "PASS" if passed else "FAIL"
    print(f"  Result: {status}")
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Smoke-test mood trends on mirrored mood mazes"
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--last_n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="test_runs")
    args = parser.parse_args()

    results = {
        "key_approach": verify_maze_mood(
            "key_approach",
            expected_sign="positive",
            n_episodes=args.episodes,
            last_n=args.last_n,
            seed=args.seed,
            log_dir=args.log_dir,
        ),
        "shield_avoidance": verify_maze_mood(
            "shield_avoidance",
            expected_sign="negative",
            n_episodes=args.episodes,
            last_n=args.last_n,
            seed=args.seed + 1,
            log_dir=args.log_dir,
        ),
    }

    print(f"\n{'='*60}")
    print("MOOD VERIFICATION SUMMARY")
    print(f"{'='*60}")
    for maze_name, passed in results.items():
        print(f"  {maze_name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    if all_passed:
        print("\nAll mood checks passed.")
    else:
        print("\nSome mood checks failed. Review training logs in test_runs/.")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
