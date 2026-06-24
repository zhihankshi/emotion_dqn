"""Visualize two-phase transfer training runs."""
import argparse
import sys
from pathlib import Path

from utils.visualization import find_transfer_experiments, plot_transfer_training


def main():
    parser = argparse.ArgumentParser(
        description="Plot transfer training (reward, steps, mood) with optimal benchmarks"
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default=None,
        help="Path to transfer_* experiment folder (with transfer_manifest.json)",
    )
    parser.add_argument(
        "--runs_dir", type=str, default="runs",
        help="Search here for transfer experiments when using --latest",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Use the most recent transfer experiment in runs_dir",
    )
    parser.add_argument(
        "--window", type=int, default=20,
        help="Rolling average window (per phase)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path (default: <experiment_dir>/transfer_training.png)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plot interactively",
    )
    args = parser.parse_args()

    if args.latest or args.experiment_dir is None:
        experiments = find_transfer_experiments(args.runs_dir)
        if not experiments:
            print(f"No transfer experiments found under {args.runs_dir}")
            sys.exit(1)
        experiment_dir = experiments[-1]
        print(f"Using latest transfer run: {experiment_dir}")
    else:
        experiment_dir = Path(args.experiment_dir)

    plot_transfer_training(
        str(experiment_dir),
        window=args.window,
        save_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
