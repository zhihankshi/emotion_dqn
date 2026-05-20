"""
Visualization utilities for experiment results.
Generates plots for comparing agent performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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