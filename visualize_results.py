"""Visualize results from quick test or full run."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Optional

from utils.maze_benchmarks import OPTIMAL_ACTIONS, get_maze_benchmark


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


def _add_optimal_hline(ax, maze_name: str, metric: str) -> None:
    """Add optimal reward or steps reference line if maze is known."""
    if maze_name not in OPTIMAL_ACTIONS:
        return
    bench = get_maze_benchmark(maze_name)
    if metric == "reward":
        ax.axhline(
            bench["reward"], color="#16a34a", linestyle="--", linewidth=2,
            label=f"Optimal reward ({bench['reward']:.2f})",
        )
        ax.scatter(
            [ax.get_xlim()[1]], [bench["reward"]],
            color="#16a34a", marker="*", s=100, zorder=5, edgecolors="white",
        )
    elif metric == "steps":
        ax.axhline(
            bench["steps"], color="#16a34a", linestyle="--", linewidth=2,
            label=f"Optimal steps ({bench['steps']})",
        )
        ax.scatter(
            [ax.get_xlim()[1]], [bench["steps"]],
            color="#16a34a", marker="*", s=100, zorder=5, edgecolors="white",
        )


def visualize_comparison(base_dir: str = "test_runs", maze_name: str = None, window: int = 10):
    """
    Visualize comparison between baseline and emotional agents.
    
    Args:
        base_dir: Directory containing run folders
        maze_name: Filter by maze name (e.g., 'complex', 'minimal')
        window: Window size for rolling average
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
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'baseline': 'blue', 'emotional': 'red'}
    
    # Determine column names (may vary)
    def get_column(df, options):
        for col in options:
            if col in df.columns:
                return col
        return None
    
    # Plot 1: Episode Reward
    ax = axes[0, 0]
    for agent_type, df in data.items():
        reward_col = get_column(df, ['total_reward', 'reward', 'episode_reward'])
        
        if reward_col:
            episodes = range(len(df))
            rewards = df[reward_col]
            
            # Raw data (faint)
            ax.plot(episodes, rewards, alpha=0.2, color=colors[agent_type])
            
            # Rolling average
            rolling = rewards.rolling(window=window, min_periods=1).mean()
            ax.plot(episodes, rolling, label=f'{agent_type}', 
                    color=colors[agent_type], linewidth=2)
    
    if run_maze:
        _add_optimal_hline(ax, run_maze, "reward")
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(f'Episode Reward (rolling avg {window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Success Rate
    ax = axes[0, 1]
    for agent_type, df in data.items():
        success_col = get_column(df, ['success', 'succeeded', 'done'])
        
        if success_col:
            episodes = range(len(df))
            success = df[success_col].astype(float)
            
            # Rolling success rate
            rolling = success.rolling(window=window, min_periods=1).mean() * 100
            ax.plot(episodes, rolling, label=agent_type, 
                    color=colors[agent_type], linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Success Rate (rolling avg {window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # Plot 3: Episode Steps
    ax = axes[1, 0]
    for agent_type, df in data.items():
        steps_col = get_column(df, ['steps', 'length', 'episode_length'])
        
        if steps_col:
            episodes = range(len(df))
            steps = df[steps_col]
            
            # Rolling average
            rolling = steps.rolling(window=window, min_periods=1).mean()
            ax.plot(episodes, rolling, label=agent_type, 
                    color=colors[agent_type], linewidth=2)
    
    if run_maze:
        _add_optimal_hline(ax, run_maze, "steps")
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title(f'Episode Length (rolling avg {window})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mood (emotional only)
    ax = axes[1, 1]
    if 'emotional' in data:
        df = data['emotional']
        mood_col = get_column(df, ['mood', 'mean_mood', 'avg_mood', 'mean_overall_mood'])
        
        if mood_col:
            episodes = range(len(df))
            mood = df[mood_col]
            
            # Raw data (faint)
            ax.plot(episodes, mood, alpha=0.2, color='red')
            
            # Rolling average
            rolling = mood.rolling(window=window, min_periods=1).mean()
            ax.plot(episodes, rolling, label='Mood', color='red', linewidth=2)
            
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.axhline(y=1, color='gray', linestyle=':', alpha=0.4)
            ax.axhline(y=-1, color='gray', linestyle=':', alpha=0.4)
            ax.set_ylim(-1.15, 1.15)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Mood')
            ax.set_title('Emotional Agent Mood')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No mood column found\nAvailable: {list(df.columns)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title('Mood (no data)')
    else:
        ax.text(0.5, 0.5, 'No emotional agent data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Mood')
    
    # Add overall title
    title = f"Comparison: Baseline vs Emotional DQN"
    if maze_name or run_maze:
        title += f" ({maze_name or run_maze} maze)"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    save_name = f"comparison_{maze_name}.png" if maze_name else "comparison_plot.png"
    save_path = Path(base_dir) / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for agent_type, df in data.items():
        print(f"\n{agent_type.upper()}:")
        
        reward_col = get_column(df, ['total_reward', 'reward', 'episode_reward'])
        if reward_col:
            print(f"  Reward - Mean: {df[reward_col].mean():.2f}, "
                  f"Std: {df[reward_col].std():.2f}")
        
        success_col = get_column(df, ['success', 'succeeded', 'done'])
        if success_col:
            success_rate = df[success_col].mean() * 100
            print(f"  Success Rate: {success_rate:.1f}%")
            
            # Find first success
            successes = df[df[success_col] == True]
            if len(successes) > 0:
                first_success = successes.index[0]
                print(f"  First Success: Episode {first_success}")
        
        steps_col = get_column(df, ['steps', 'length', 'episode_length'])
        if steps_col:
            print(f"  Steps - Mean: {df[steps_col].mean():.1f}, "
                  f"Std: {df[steps_col].std():.1f}")
    
    # Print comparison
    if 'baseline' in data and 'emotional' in data:
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        
        b_df = data['baseline']
        e_df = data['emotional']
        
        success_col = get_column(b_df, ['success', 'succeeded', 'done'])
        if success_col:
            b_success = b_df[success_col].mean() * 100
            e_success = e_df[success_col].mean() * 100
            improvement = (e_success - b_success) / b_success * 100 if b_success > 0 else float('inf')
            print(f"  Success Rate: {b_success:.1f}% → {e_success:.1f}% ({improvement:+.1f}% relative)")
        
        reward_col = get_column(b_df, ['total_reward', 'reward', 'episode_reward'])
        if reward_col:
            b_reward = b_df[reward_col].mean()
            e_reward = e_df[reward_col].mean()
            print(f"  Avg Reward: {b_reward:.2f} → {e_reward:.2f} ({e_reward - b_reward:+.2f})")
        
        steps_col = get_column(b_df, ['steps', 'length', 'episode_length'])
        if steps_col:
            b_steps = b_df[steps_col].mean()
            e_steps = e_df[steps_col].mean()
            print(f"  Avg Steps: {b_steps:.1f} → {e_steps:.1f} ({e_steps - b_steps:+.1f})")


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
        help="Rolling average window (transfer default: 100, comparison default: 10)",
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

    window = args.window if args.window is not None else (100 if args.transfer else 10)

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
        visualize_comparison(str(chosen), args.maze, window)
    else:
        if args.base_dir.startswith("-"):
            parser.error(
                f"Unknown argument '{args.base_dir}'. "
                f"Example: python visualize_results.py experiments"
            )
        target = args.experiment_dir or args.base_dir
        visualize_comparison(target, args.maze, window)