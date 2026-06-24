"""
Visualization utilities for experiment results.
Generates plots for comparing agent performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json


def load_episode_data(csv_path: str) -> pd.DataFrame:
    """Load episode data from CSV file."""
    return pd.read_csv(csv_path)


def load_experiment_results(experiment_dir: str) -> Dict[str, Any]:
    """
    Load all results from an experiment directory.
    
    Returns:
        Dictionary with 'baseline' and 'emotional' DataFrames
    """
    experiment_dir = Path(experiment_dir)
    
    results = {
        'baseline': [],
        'emotional': []
    }
    
    for agent_type in ['baseline', 'emotional']:
        agent_dir = experiment_dir / agent_type
        if not agent_dir.exists():
            continue
        
        # Find all run directories
        for run_dir in sorted(agent_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            
            # Find episode CSV
            csv_files = list(run_dir.glob("*_episodes.csv"))
            if csv_files:
                df = pd.read_csv(csv_files[0])
                df['run_id'] = run_dir.name
                results[agent_type].append(df)
    
    # Combine runs
    for agent_type in results:
        if results[agent_type]:
            results[agent_type] = pd.concat(results[agent_type], ignore_index=True)
        else:
            results[agent_type] = pd.DataFrame()
    
    return results


def smooth(data: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_learning_curves(
    results: Dict[str, pd.DataFrame],
    metric: str = 'total_reward',
    title: str = None,
    smooth_window: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot learning curves comparing agents.
    
    Args:
        results: Dictionary with 'baseline' and 'emotional' DataFrames
        metric: Column name to plot
        title: Plot title
        smooth_window: Window size for smoothing
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'baseline': '#1f77b4', 'emotional': '#ff7f0e'}
    labels = {'baseline': 'Baseline DQN', 'emotional': 'Emotional DQN'}
    
    for agent_type, df in results.items():
        if df.empty:
            continue
        
        # Group by episode across runs
        grouped = df.groupby('episode')[metric]
        mean = grouped.mean().values
        std = grouped.std().values
        episodes = df.groupby('episode')[metric].mean().index.values
        
        # Smooth
        if smooth_window > 1 and len(mean) > smooth_window:
            mean_smooth = smooth(mean, smooth_window)
            std_smooth = smooth(std, smooth_window)
            episodes_smooth = episodes[:len(mean_smooth)]
        else:
            mean_smooth = mean
            std_smooth = std
            episodes_smooth = episodes
        
        # Plot
        ax.plot(
            episodes_smooth, 
            mean_smooth, 
            color=colors[agent_type],
            label=labels[agent_type],
            linewidth=2
        )
        
        # Confidence band
        ax.fill_between(
            episodes_smooth,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            color=colors[agent_type],
            alpha=0.2
        )
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title or f'Learning Curve: {metric}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_success_rate(
    results: Dict[str, pd.DataFrame],
    window: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot success rate over time.
    
    Args:
        results: Dictionary with 'baseline' and 'emotional' DataFrames
        window: Window for rolling success rate
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'baseline': '#1f77b4', 'emotional': '#ff7f0e'}
    labels = {'baseline': 'Baseline DQN', 'emotional': 'Emotional DQN'}
    
    for agent_type, df in results.items():
        if df.empty:
            continue
        
        # Calculate rolling success rate per run, then average
        runs = df['run_id'].unique()
        all_success_rates = []
        
        for run_id in runs:
            run_df = df[df['run_id'] == run_id].sort_values('episode')
            success_rate = run_df['success'].rolling(window=window, min_periods=1).mean()
            all_success_rates.append(success_rate.values)
        
        # Align and average
        min_len = min(len(sr) for sr in all_success_rates)
        all_success_rates = [sr[:min_len] for sr in all_success_rates]
        
        mean_sr = np.mean(all_success_rates, axis=0)
        std_sr = np.std(all_success_rates, axis=0)
        episodes = np.arange(min_len)
        
        # Plot
        ax.plot(
            episodes, 
            mean_sr * 100, 
            color=colors[agent_type],
            label=labels[agent_type],
            linewidth=2
        )
        
        ax.fill_between(
            episodes,
            (mean_sr - std_sr) * 100,
            (mean_sr + std_sr) * 100,
            color=colors[agent_type],
            alpha=0.2
        )
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Success Rate (rolling window={window})', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_steps_per_episode(
    results: Dict[str, pd.DataFrame],
    smooth_window: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot average steps per episode (lower is better for successful episodes)."""
    return plot_learning_curves(
        results,
        metric='steps',
        title='Steps per Episode (lower is better)',
        smooth_window=smooth_window,
        save_path=save_path,
        figsize=figsize
    )


def plot_emotional_metrics(
    results: Dict[str, pd.DataFrame],
    smooth_window: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot emotional metrics for the emotional agent.
    
    Shows mood values, exploration boosts, etc.
    """
    emotional_df = results.get('emotional', pd.DataFrame())
    
    if emotional_df.empty:
        print("No emotional agent data found")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Get data grouped by episode
    grouped = emotional_df.groupby('episode')
    episodes = grouped['episode'].first().values
    
    # 1. Overall Mood
    ax = axes[0, 0]
    mood = grouped['mean_overall_mood'].mean().values
    mood_smooth = smooth(mood, smooth_window) if len(mood) > smooth_window else mood
    episodes_smooth = episodes[:len(mood_smooth)]
    
    ax.plot(episodes_smooth, mood_smooth, color='purple', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Overall Mood')
    ax.set_title('Mood Over Training')
    ax.grid(True, alpha=0.3)
    
    # 2. Mood Value vs Mood Action
    ax = axes[0, 1]
    mood_value = grouped['mean_mood_value'].mean().values
    mood_action = grouped['mean_mood_action'].mean().values
    
    mv_smooth = smooth(mood_value, smooth_window) if len(mood_value) > smooth_window else mood_value
    ma_smooth = smooth(mood_action, smooth_window) if len(mood_action) > smooth_window else mood_action
    
    ax.plot(episodes_smooth, mv_smooth, color='blue', label='Mood (Value)', linewidth=2)
    ax.plot(episodes_smooth, ma_smooth, color='red', label='Mood (Action)', linewidth=2)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mood')
    ax.set_title('Value vs Action Mood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Exploration Boosts
    ax = axes[1, 0]
    boosts = grouped['exploration_boosts'].mean().values
    boosts_smooth = smooth(boosts, smooth_window) if len(boosts) > smooth_window else boosts
    
    ax.plot(episodes_smooth, boosts_smooth[:len(episodes_smooth)], color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Exploration Boosts')
    ax.set_title('Exploration Boosts per Episode')
    ax.grid(True, alpha=0.3)
    
    # 4. Epsilon vs Effective Epsilon
    ax = axes[1, 1]
    epsilon = grouped['epsilon'].mean().values
    eff_epsilon = grouped['effective_epsilon'].mean().values
    
    ax.plot(episodes[:len(epsilon)], epsilon, color='gray', label='Base ε', linewidth=2)
    ax.plot(episodes[:len(eff_epsilon)], eff_epsilon, color='orange', label='Effective ε', linewidth=2, linestyle='--')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_causal_understanding(
    results: Dict[str, pd.DataFrame],
    smooth_window: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot metrics related to causal understanding.
    
    Shows door attempts without key over time.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'baseline': '#1f77b4', 'emotional': '#ff7f0e'}
    labels = {'baseline': 'Baseline DQN', 'emotional': 'Emotional DQN'}
    
    for agent_type, df in results.items():
        if df.empty:
            continue
        
        grouped = df.groupby('episode')['door_attempts_without_key']
        mean = grouped.mean().values
        episodes = df.groupby('episode')['door_attempts_without_key'].mean().index.values
        
        # Smooth
        if len(mean) > smooth_window:
            mean_smooth = smooth(mean, smooth_window)
            episodes_smooth = episodes[:len(mean_smooth)]
        else:
            mean_smooth = mean
            episodes_smooth = episodes
        
        ax.plot(
            episodes_smooth,
            mean_smooth,
            color=colors[agent_type],
            label=labels[agent_type],
            linewidth=2
        )
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Door Attempts Without Key', fontsize=12)
    ax.set_title('Causal Understanding: Door Attempts Without Key\n(lower is better)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_comparison_summary(
    results: Dict[str, pd.DataFrame],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Create a comprehensive comparison figure with multiple subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    colors = {'baseline': '#1f77b4', 'emotional': '#ff7f0e'}
    labels = {'baseline': 'Baseline DQN', 'emotional': 'Emotional DQN'}
    smooth_window = 20
    
    # 1. Success Rate (top left)
    ax = axes[0, 0]
    for agent_type, df in results.items():
        if df.empty:
            continue
        
        runs = df['run_id'].unique()
        all_success_rates = []
        
        for run_id in runs:
            run_df = df[df['run_id'] == run_id].sort_values('episode')
            success_rate = run_df['success'].rolling(window=50, min_periods=1).mean()
            all_success_rates.append(success_rate.values)
        
        min_len = min(len(sr) for sr in all_success_rates)
        all_success_rates = [sr[:min_len] for sr in all_success_rates]
        
        mean_sr = np.mean(all_success_rates, axis=0) * 100
        std_sr = np.std(all_success_rates, axis=0) * 100
        episodes = np.arange(min_len)
        
        ax.plot(episodes, mean_sr, color=colors[agent_type], label=labels[agent_type], linewidth=2)
        ax.fill_between(episodes, mean_sr - std_sr, mean_sr + std_sr, color=colors[agent_type], alpha=0.2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 2. Total Reward (top right)
    ax = axes[0, 1]
    for agent_type, df in results.items():
        if df.empty:
            continue
        
        grouped = df.groupby('episode')['total_reward']
        mean = grouped.mean().values
        std = grouped.std().values
        episodes = grouped.mean().index.values
        
        mean_smooth = smooth(mean, smooth_window)
        std_smooth = smooth(std, smooth_window)
        episodes_smooth = episodes[:len(mean_smooth)]
        
        ax.plot(episodes_smooth, mean_smooth, color=colors[agent_type], label=labels[agent_type], linewidth=2)
        ax.fill_between(episodes_smooth, mean_smooth - std_smooth, mean_smooth + std_smooth, color=colors[agent_type], alpha=0.2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Steps per Episode (bottom left)
    ax = axes[1, 0]
    for agent_type, df in results.items():
        if df.empty:
            continue
        
        grouped = df.groupby('episode')['steps']
        mean = grouped.mean().values
        std = grouped.std().values
        episodes = grouped.mean().index.values
        
        mean_smooth = smooth(mean, smooth_window)
        std_smooth = smooth(std, smooth_window)
        episodes_smooth = episodes[:len(mean_smooth)]
        
        ax.plot(episodes_smooth, mean_smooth, color=colors[agent_type], label=labels[agent_type], linewidth=2)
        ax.fill_between(episodes_smooth, mean_smooth - std_smooth, mean_smooth + std_smooth, color=colors[agent_type], alpha=0.2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Steps per Episode (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Door Attempts Without Key (bottom right)
    ax = axes[1, 1]
    for agent_type, df in results.items():
        if df.empty:
            continue
        
        grouped = df.groupby('episode')['door_attempts_without_key']
        mean = grouped.mean().values
        episodes = grouped.mean().index.values
        
        mean_smooth = smooth(mean, smooth_window)
        episodes_smooth = episodes[:len(mean_smooth)]
        
        ax.plot(episodes_smooth, mean_smooth, color=colors[agent_type], label=labels[agent_type], linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Door Attempts')
    ax.set_title('Door Attempts Without Key (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_plots(
    experiment_dir: str,
    output_dir: Optional[str] = None
) -> None:
    """
    Generate all plots for an experiment.
    
    Args:
        experiment_dir: Path to experiment directory
        output_dir: Where to save plots (default: experiment_dir/plots)
    """
    experiment_dir = Path(experiment_dir)
    
    if output_dir is None:
        output_dir = experiment_dir / 'plots'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading results from: {experiment_dir}")
    results = load_experiment_results(str(experiment_dir))
    
    print(f"  Baseline episodes: {len(results['baseline'])}")
    print(f"  Emotional episodes: {len(results['emotional'])}")
    
    print(f"\nGenerating plots...")
    
    # Generate all plots
    plot_comparison_summary(results, save_path=str(output_dir / 'comparison_summary.png'))
    plot_success_rate(results, save_path=str(output_dir / 'success_rate.png'))
    plot_learning_curves(results, metric='total_reward', save_path=str(output_dir / 'reward_curve.png'))
    plot_steps_per_episode(results, save_path=str(output_dir / 'steps_curve.png'))
    plot_causal_understanding(results, save_path=str(output_dir / 'causal_understanding.png'))
    
    if not results['emotional'].empty:
        plot_emotional_metrics(results, save_path=str(output_dir / 'emotional_metrics.png'))
    
    print(f"\nAll plots saved to: {output_dir}")


def find_transfer_experiments(base_dir: str = "runs") -> List[Path]:
    """Find transfer experiment directories containing transfer_manifest.json."""
    base = Path(base_dir)
    if not base.exists():
        return []
    experiments = [
        p.parent for p in base.rglob("transfer_manifest.json")
    ]
    return sorted(experiments, key=lambda p: p.name)


def find_latest_transfer(
    base_dir: str = "runs",
    agent_type: Optional[str] = None,
) -> Optional[Path]:
    """Return latest transfer experiment, optionally filtered by manifest agent_type."""
    experiments = find_transfer_experiments(base_dir)
    if agent_type is None:
        return experiments[-1] if experiments else None

    matches = []
    for exp in experiments:
        manifest_path = exp / "transfer_manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("agent_type") == agent_type:
            matches.append(exp)
    return matches[-1] if matches else None


def load_transfer_experiment(experiment_dir: str) -> Dict[str, Any]:
    """
    Load transfer manifest and episode CSVs for both phases.

    Returns dict with manifest, phase1_df, phase2_df, episode_csv paths.
    """
    experiment_dir = Path(experiment_dir)
    manifest_path = experiment_dir / "transfer_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No transfer_manifest.json in {experiment_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    def _load_phase_csv(log_dir_key: str) -> Tuple[pd.DataFrame, Path]:
        log_dir = Path(manifest[log_dir_key])
        agent_type = manifest.get("agent_type", "emotional")
        df = _load_phase_episode_csv(log_dir, agent_type)
        csv_path = log_dir / f"{agent_type}_run0_episodes.csv"
        if not csv_path.exists():
            csv_files = sorted(log_dir.glob("*_episodes.csv"))
            csv_path = csv_files[0]
        return df, csv_path

    phase1_df, phase1_csv = _load_phase_csv("phase1_log_dir")
    phase2_df, phase2_csv = _load_phase_csv("phase2_log_dir")

    return {
        "manifest": manifest,
        "experiment_dir": experiment_dir,
        "phase1_df": phase1_df,
        "phase2_df": phase2_df,
        "phase1_csv": phase1_csv,
        "phase2_csv": phase2_csv,
    }


def _load_phase_episode_csv(log_dir: Path, agent_type: str) -> pd.DataFrame:
    """Load episode CSV from a phase log directory."""
    preferred = log_dir / f"{agent_type}_run0_episodes.csv"
    if preferred.exists():
        return pd.read_csv(preferred)
    csv_files = sorted(log_dir.glob("*_episodes.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No episode CSV in {log_dir}")
    return pd.read_csv(csv_files[0])


def build_transfer_long_dataframe(
    experiment_dir: str,
    agent_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Combine phase 1 and phase 2 episode logs into one timeline per experiment.

    Returns (dataframe, manifest) with columns including episode_global,
    phase, maze_name, agent_type, steps, total_reward, mean_overall_mood.
    """
    data = load_transfer_experiment(experiment_dir)
    manifest = data["manifest"]
    agent_type = manifest.get("agent_type", "emotional")
    label = agent_label or agent_type.capitalize()

    n1 = manifest.get("phase1_episodes", len(data["phase1_df"]))

    frames = []
    for phase_df, phase_name, maze_name, offset in (
        (data["phase1_df"], "Phase 1", manifest["source_maze"], 0),
        (data["phase2_df"], "Phase 2", manifest["target_maze"], n1),
    ):
        part = phase_df.copy()
        part["agent_type"] = label
        part["phase"] = phase_name
        part["maze_name"] = maze_name
        part["episode_global"] = part["episode"] + offset
        frames.append(part)

    combined = pd.concat(frames, ignore_index=True)
    return combined, manifest


def _add_transfer_reference_lines(
    ax,
    transfer_ep: int,
    n_total: int,
    opt1: float,
    opt2: float,
    metric_label: str,
    phase1_name: str,
    phase2_name: str,
) -> None:
    """Phase transition and per-phase optimal reference lines (scoped to each phase)."""
    ax.axvline(
        transfer_ep, color="#6b7280", linestyle="--", linewidth=1.8,
        label="Phase transition", zorder=1,
    )
    # Optimal benchmarks only in the phase they apply to (clearer than full-width lines).
    ax.hlines(
        opt1, 0, transfer_ep, colors="#16a34a", linestyles=":", linewidth=2.2,
        label=f"Optimal {metric_label} ({phase1_name})", zorder=1,
    )
    ax.hlines(
        opt2, transfer_ep, n_total, colors="#ea580c", linestyles=":", linewidth=2.2,
        label=f"Optimal {metric_label} ({phase2_name})", zorder=1,
    )


def _style_transfer_axes(ax, transfer_ep: int, n_total: int) -> None:
    ax.axvspan(0, transfer_ep, color="#eff6ff", alpha=0.55, zorder=0)
    ax.axvspan(transfer_ep, n_total, color="#fff7ed", alpha=0.55, zorder=0)


def plot_transfer_training(
    experiment_dir: str,
    baseline_dir: Optional[str] = None,
    window: int = 100,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[int, int] = (14, 13),
) -> plt.Figure:
    """
    Plot transfer training with seaborn: steps, reward, and mood.

    If baseline_dir is provided, overlays Baseline vs Emotional on the same axes.
    """
    from .maze_benchmarks import get_benchmarks_for_transfer

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)

    emotional_df, manifest = build_transfer_long_dataframe(
        experiment_dir, agent_label="Emotional"
    )
    frames = [emotional_df]

    if baseline_dir is not None:
        baseline_df, _ = build_transfer_long_dataframe(
            baseline_dir, agent_label="Baseline"
        )
        frames.append(baseline_df)

    all_data = pd.concat(frames, ignore_index=True)
    all_data = all_data.sort_values(["agent_type", "episode_global"]).reset_index(drop=True)

    source = manifest["source_maze"]
    target = manifest["target_maze"]
    n1 = manifest.get("phase1_episodes", len(emotional_df[emotional_df["phase"] == "Phase 1"]))
    n2 = manifest.get("phase2_episodes", len(emotional_df[emotional_df["phase"] == "Phase 2"]))
    transfer_ep = n1
    n_total = n1 + n2

    benchmarks = get_benchmarks_for_transfer(source, target)
    opt1_r = benchmarks["phase1"]["reward"]
    opt2_r = benchmarks["phase2"]["reward"]
    opt1_s = benchmarks["phase1"]["steps"]
    opt2_s = benchmarks["phase2"]["steps"]

    for metric, col in (("steps", "avg_steps"), ("reward", "avg_reward"), ("mood", "avg_mood")):
        if metric == "mood":
            if "mean_overall_mood" not in all_data.columns:
                continue
            source_col = "mean_overall_mood"
        else:
            source_col = "steps" if metric == "steps" else "total_reward"
        all_data[col] = (
            all_data.groupby("agent_type")[source_col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )

    palette = {"Emotional": "#e11d48", "Baseline": "#2563eb"}
    hue_order = [a for a in ("Emotional", "Baseline") if a in all_data["agent_type"].unique()]

    has_mood = "avg_mood" in all_data.columns
    n_rows = 3 if has_mood else 2
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)

    title_agent = "Baseline vs Emotional" if baseline_dir else manifest.get("agent_type", "agent").capitalize()
    fig.suptitle(
        f"Transfer Training: {source} → {target}\n({title_agent})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # --- Steps ---
    ax = axes[0]
    _style_transfer_axes(ax, transfer_ep, n_total)
    sns.lineplot(
        data=all_data, x="episode_global", y="avg_steps",
        hue="agent_type", hue_order=hue_order, palette=palette,
        linewidth=2.5, ax=ax, errorbar=None,
    )
    _add_transfer_reference_lines(
        ax, transfer_ep, n_total, opt1_s, opt2_s, "steps", source, target,
    )
    ax.set_ylabel("Average Steps")
    ax.set_title(f"Average Steps per Episode (rolling window = {window})")
    ax.set_xlabel("")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # --- Reward ---
    ax = axes[1]
    _style_transfer_axes(ax, transfer_ep, n_total)
    sns.lineplot(
        data=all_data, x="episode_global", y="avg_reward",
        hue="agent_type", hue_order=hue_order, palette=palette,
        linewidth=2.5, ax=ax, errorbar=None,
    )
    _add_transfer_reference_lines(
        ax, transfer_ep, n_total, opt1_r, opt2_r, "reward", source, target,
    )
    ax.set_ylabel("Average Reward")
    ax.set_title(f"Average Reward per Episode (rolling window = {window})")
    ax.set_xlabel("")
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # --- Mood (emotional agents; baseline mood is usually ~0) ---
    if has_mood:
        ax = axes[2]
        _style_transfer_axes(ax, transfer_ep, n_total)
        mood_data = all_data.copy()
        sns.lineplot(
            data=mood_data, x="episode_global", y="avg_mood",
            hue="agent_type", hue_order=hue_order, palette=palette,
            linewidth=2.5, ax=ax, errorbar=None,
        )
        ax.axvline(transfer_ep, color="#6b7280", linestyle="--", linewidth=1.8, label="Phase transition")
        ax.axhline(0, color="black", linestyle=":", alpha=0.55)
        ax.axhline(1, color="#9ca3af", linestyle=":", alpha=0.45)
        ax.axhline(-1, color="#9ca3af", linestyle=":", alpha=0.45)
        ax.set_ylim(-1.15, 1.15)
        ax.set_ylabel("Average Mood")
        ax.set_title("Average Mood (clipped to [-1, 1] during training)")
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    axes[-1].set_xlabel("Global Episode Number")
    axes[0].annotate(
        f"Phase 1: {source}",
        xy=(transfer_ep * 0.5, 0.97), xycoords=("data", "axes fraction"),
        ha="center", fontsize=10, color="#1d4ed8", fontweight="medium",
    )
    axes[0].annotate(
        f"Phase 2: {target}",
        xy=(transfer_ep + (n_total - transfer_ep) * 0.5, 0.97),
        xycoords=("data", "axes fraction"),
        ha="center", fontsize=10, color="#c2410c", fontweight="medium",
    )

    # Summary footer
    e_only = all_data[all_data["agent_type"] == "Emotional"]
    if not e_only.empty:
        p1 = e_only[e_only["phase"] == "Phase 1"]
        p2 = e_only[e_only["phase"] == "Phase 2"]
        summary = (
            f"Emotional — Phase 1: reward {p1['total_reward'].mean():.2f}, "
            f"steps {p1['steps'].mean():.1f}  |  "
            f"Phase 2: reward {p2['total_reward'].mean():.2f}, "
            f"steps {p2['steps'].mean():.1f}"
        )
        if has_mood:
            summary += (
                f"  |  mood {p1['mean_overall_mood'].mean():.3f} → "
                f"{p2['mean_overall_mood'].mean():.3f}"
            )
        if baseline_dir:
            b_only = all_data[all_data["agent_type"] == "Baseline"]
            bp1 = b_only[b_only["phase"] == "Phase 1"]
            bp2 = b_only[b_only["phase"] == "Phase 2"]
            summary += (
                f"\nBaseline — Phase 1: reward {bp1['total_reward'].mean():.2f}, "
                f"steps {bp1['steps'].mean():.1f}  |  "
                f"Phase 2: reward {bp2['total_reward'].mean():.2f}, "
                f"steps {bp2['steps'].mean():.1f}"
            )
        fig.text(0.5, 0.01, summary, ha="center", fontsize=9, color="#374151")

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    if save_path is None:
        suffix = "_comparison" if baseline_dir else ""
        save_path = Path(experiment_dir) / f"transfer_training{suffix}.png"
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved transfer plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualization plots")
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--output', type=str, default=None, help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    generate_all_plots(args.experiment_dir, args.output)
    
    if args.show:
        plt.show()