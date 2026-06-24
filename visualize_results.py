"""Visualize results from quick test or full run."""
import pandas as pd
from pathlib import Path
import sys
from typing import Optional

from utils.maze_benchmarks import OPTIMAL_ACTIONS
from utils.visualization import plot_agent_comparison, print_comparison_summary


def find_comparison_experiments(base_dir: str = "experiments") -> list:
    """Find compare_agents experiment dirs (contain baseline/ and/or emotional/)."""
    base = Path(base_dir)
    if not base.exists():
        return []
    if (base / "baseline").is_dir() or (base / "emotional").is_dir():
        return [base]
    experiments = [
        p for p in base.iterdir()
        if p.is_dir() and (
            (p / "baseline").is_dir() or (p / "emotional").is_dir()
        )
    ]
    return sorted(experiments, key=lambda p: p.name)


def find_runs(base_dir: str = "test_runs", maze_name: str = None) -> dict:
    """
    Find baseline and emotional episode CSVs.

    Supports:
      - Flat logs: test_runs/shield_trap_*/{baseline,emotional}_run0_episodes.csv
      - Comparison experiments: experiments/*_comparison_*/{baseline,emotional}/*/
    """
    base_path = Path(base_dir).resolve()
    if not base_path.exists():
        raise FileNotFoundError(
            f"Directory not found: {base_path}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"For transfer plots: python visualize_results.py --transfer --latest\n"
            f"For experiments/: python visualize_results.py experiments --latest"
        )

    runs = {"baseline": [], "emotional": []}
    seen_csvs: set = set()

    def _add_csv(csv_path: Path) -> None:
        csv_path = csv_path.resolve()
        if csv_path in seen_csvs:
            return
        name = csv_path.name
        if not name.endswith("_episodes.csv"):
            return
        if name.startswith("baseline_"):
            agent_type = "baseline"
        elif name.startswith("emotional_"):
            agent_type = "emotional"
        else:
            return

        run_dir = csv_path.parent
        path_for_filter = str(run_dir)
        if maze_name and maze_name not in path_for_filter and maze_name not in name:
            return

        seen_csvs.add(csv_path)
        runs[agent_type].append((run_dir, csv_path))

    # Flat layout: CSVs directly under immediate child folders
    for run_dir in sorted(base_path.iterdir()):
        if not run_dir.is_dir():
            continue
        for pattern in ("baseline_*_episodes.csv", "emotional_*_episodes.csv"):
            for csv_path in run_dir.glob(pattern):
                _add_csv(csv_path)

    # Nested comparison layout (experiments/foo_comparison/baseline/run/csv)
    if not runs["baseline"] and not runs["emotional"]:
        comparisons = find_comparison_experiments(str(base_path))
        if comparisons:
            if maze_name:
                comparisons = [c for c in comparisons if maze_name in c.name]
            if not comparisons:
                return runs
            if (base_path / "baseline").is_dir() or (base_path / "emotional").is_dir():
                search_roots = [base_path]
            else:
                search_roots = [comparisons[-1]]
        else:
            search_roots = [base_path]

        for root in search_roots:
            for csv_path in sorted(root.rglob("*_episodes.csv")):
                _add_csv(csv_path)

    return runs


def _infer_maze_name(run_dir: Path, maze_filter: str = None) -> Optional[str]:
    """Guess maze name from directory name or filter."""
    if maze_filter:
        return maze_filter
    for name in OPTIMAL_ACTIONS:
        if name in run_dir.name:
            return name
    return None


def visualize_comparison(
    base_dir: str = "test_runs",
    maze_name: str = None,
    window: int = 50,
    show: bool = True,
):
    """
    Visualize comparison between baseline and emotional agents.

    Args:
        base_dir: Directory containing run folders
        maze_name: Filter by maze name (e.g., 'shield_trap')
        window: Rolling average window
        show: Display plot interactively
    """
    runs = find_runs(base_dir, maze_name)
    
    print(f"Found runs{f' for {maze_name}' if maze_name else ''}:")
    print(f"  Baseline: {len(runs['baseline'])} runs")
    print(f"  Emotional: {len(runs['emotional'])} runs")
    
    if not runs['baseline'] and not runs['emotional']:
        print("\nNo runs found!")
        print(f"  Searched: {Path(base_dir).resolve()}")
        print(f"  Working directory: {Path.cwd()}")
        comparisons = find_comparison_experiments(base_dir)
        if comparisons:
            print(f"  Found {len(comparisons)} comparison experiment(s) but no matching CSVs.")
            print("  Try: python visualize_results.py experiments --latest")
            print("  Or:  python visualize_results.py experiments/<comparison_folder>")
        else:
            print("  Expected episode CSVs like baseline_run0_episodes.csv")
            print("  under test_runs/ or experiments/*_comparison_*/baseline|emotional/*/")
        return
    
    # Load data from most recent runs
    data = {}
    run_maze = maze_name
    for agent_type, run_list in runs.items():
        if run_list:
            # Sort by directory name (timestamp) and use most recent
            run_dir, csv_file = sorted(run_list, key=lambda x: x[0].name)[-1]
            print(f"\nLoading {agent_type} from: {csv_file}")
            if run_maze is None:
                run_maze = _infer_maze_name(run_dir)
            
            df = pd.read_csv(csv_file)
            data[agent_type] = df
            print(f"  Loaded {len(df)} episodes")
            print(f"  Columns: {list(df.columns)}")
    
    if not data:
        print("\nNo data loaded!")
        return

    save_name = f"comparison_{maze_name or run_maze or 'plot'}.png"
    save_path = Path(base_dir) / save_name

    plot_agent_comparison(
        data,
        maze_name=run_maze,
        window=window,
        save_path=str(save_path),
        show=show,
        title=f"Baseline vs Emotional DQN ({run_maze or maze_name or 'training'})",
    )

    print_comparison_summary(data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize baseline vs emotional training results"
    )
    parser.add_argument(
        "base_dir", nargs="?", default="test_runs",
        help="Directory with run folders (default: test_runs)",
    )
    parser.add_argument(
        "--maze", type=str, default=None,
        help="Filter runs by maze name (e.g. shield_trap)",
    )
    parser.add_argument(
        "--window", type=int, default=None,
        help="Rolling average window (transfer default: 100, comparison default: 50)",
    )
    parser.add_argument(
        "--transfer", action="store_true",
        help="Plot a two-phase transfer experiment instead of comparison",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Use latest comparison in base_dir, or latest transfer in --runs-dir",
    )
    parser.add_argument(
        "--experiment-dir", type=str, default=None,
        help="Path to a comparison or transfer experiment folder",
    )
    parser.add_argument(
        "--runs-dir", type=str, default="runs",
        help="Search here for transfer experiments when using --latest",
    )
    parser.add_argument(
        "--baseline-dir", type=str, default=None,
        help="Baseline transfer_* folder for comparison overlay",
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Auto-pair latest emotional and baseline transfer runs",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show plot interactively",
    )
    args = parser.parse_args()

    window = args.window if args.window is not None else (100 if args.transfer else 50)

    if args.transfer or args.compare_baseline:
        from utils.visualization import (
            plot_transfer_training,
            find_latest_transfer,
        )

        baseline_dir = args.baseline_dir
        if args.compare_baseline and baseline_dir is None:
            baseline_dir = find_latest_transfer(args.runs_dir, "baseline")
            if baseline_dir:
                print(f"Using latest baseline transfer run: {baseline_dir}")
            else:
                print("Warning: no baseline transfer run found; plotting emotional only")

        if args.experiment_dir:
            exp_dir = args.experiment_dir
        elif args.latest or args.compare_baseline:
            emotional = find_latest_transfer(args.runs_dir, "emotional")
            if emotional is None:
                emotional = find_latest_transfer(args.runs_dir)
            if emotional is None:
                print(f"No transfer experiments found in {args.runs_dir}/")
                sys.exit(1)
            exp_dir = str(emotional)
            print(f"Using latest emotional transfer run: {exp_dir}")
        else:
            parser.error(
                "Use --latest, --compare-baseline, or --experiment-dir PATH with --transfer"
            )

        plot_transfer_training(
            exp_dir,
            baseline_dir=str(baseline_dir) if baseline_dir else None,
            window=window,
            show=args.show,
        )
    elif args.latest:
        comparisons = find_comparison_experiments(args.base_dir)
        if not comparisons:
            print(f"No comparison experiments found in {Path(args.base_dir).resolve()}")
            print("For transfer plots: python visualize_results.py --transfer --latest")
            sys.exit(1)
        chosen = comparisons[-1]
        print(f"Using latest comparison experiment: {chosen}")
        visualize_comparison(str(chosen), args.maze, window, show=args.show)
    else:
        if args.base_dir.startswith("-"):
            parser.error(
                f"Unknown argument '{args.base_dir}'. "
                f"Example: python visualize_results.py experiments"
            )
        target = args.experiment_dir or args.base_dir
        visualize_comparison(target, args.maze, window, show=args.show)