"""Visualize two-phase transfer training runs."""
import argparse
import sys
from pathlib import Path

from utils.visualization import find_transfer_experiments, find_latest_transfer, plot_transfer_training


def _resolve_experiment(path: str, runs_dir: str, agent_type: str) -> str:
    if path:
        return str(Path(path))
    chosen = find_latest_transfer(runs_dir, agent_type)
    if chosen is None:
        chosen = find_latest_transfer(runs_dir)
    if chosen is None:
        print(f"No transfer experiments found under {runs_dir}")
        sys.exit(1)
    print(f"Using latest {agent_type} transfer run: {chosen}")
    return str(chosen)


def main():
    parser = argparse.ArgumentParser(
        description="Plot transfer training (steps, reward, mood) with optimal benchmarks"
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default=None,
        help="Path to emotional transfer_* folder (transfer_manifest.json)",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to baseline transfer_* folder for comparison overlay",
    )
    parser.add_argument(
        "--runs_dir", type=str, default="runs",
        help="Search here when using --latest",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Use the most recent emotional transfer run in runs_dir",
    )
    parser.add_argument(
        "--latest_baseline", action="store_true",
        help="Use the most recent baseline transfer run for --baseline",
    )
    parser.add_argument(
        "--window", type=int, default=100,
        help="Rolling average window (default: 100)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plot interactively",
    )
    args = parser.parse_args()

    if args.latest or args.experiment_dir is None:
        emotional_dir = _resolve_experiment(args.experiment_dir, args.runs_dir, "emotional")
    else:
        emotional_dir = str(Path(args.experiment_dir))

    baseline_dir = None
    if args.baseline:
        baseline_dir = args.baseline
    elif args.latest_baseline:
        baseline_dir = find_latest_transfer(args.runs_dir, "baseline")
        if baseline_dir:
            print(f"Using latest baseline transfer run: {baseline_dir}")

    plot_transfer_training(
        emotional_dir,
        baseline_dir=baseline_dir,
        window=args.window,
        save_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
