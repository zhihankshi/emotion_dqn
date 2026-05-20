"""
Metrics logging and tracking for experiments.
Tracks performance, emotional states, and causal understanding.
"""
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    
    # Episode info
    episode: int = 0
    steps: int = 0
    total_reward: float = 0.0
    success: bool = False
    
    # Causal understanding metrics
    key_found_step: int = -1
    door_attempts_without_key: int = 0
    door_opened_step: int = -1
    
    # Learning metrics
    mean_loss: float = 0.0
    mean_td_error: float = 0.0
    mean_q_value: float = 0.0
    
    # Emotional metrics (for emotional agent)
    mean_mood_value: float = 0.0
    mean_mood_action: float = 0.0
    mean_overall_mood: float = 0.0
    exploration_boosts: int = 0
    mean_mood_bias: float = 0.0
    
    # Exploration
    epsilon: float = 0.0
    effective_epsilon: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass 
class RunMetrics:
    """Aggregated metrics for a training run."""
    
    # Run info
    run_id: int = 0
    agent_type: str = ""
    maze_name: str = ""
    total_episodes: int = 0
    
    # Performance summary
    first_success_episode: int = -1
    total_successes: int = 0
    final_success_rate: float = 0.0
    final_avg_steps: float = 0.0
    final_avg_reward: float = 0.0
    
    # Learning speed
    episodes_to_50_percent_success: int = -1
    episodes_to_90_percent_success: int = -1
    
    # Causal understanding
    avg_door_attempts_before_first_key: float = 0.0
    avg_key_found_step: float = 0.0
    
    # Emotional summary (for emotional agent)
    avg_exploration_boosts_per_episode: float = 0.0
    total_mood_bias_applied: float = 0.0


class MetricsLogger:
    """
    Logs and tracks metrics during training.
    Saves to CSV for easy analysis.
    """
    
    def __init__(
        self,
        log_dir: str,
        agent_type: str = "unknown",
        maze_name: str = "unknown",
        run_id: int = 0
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
            agent_type: Type of agent ('baseline' or 'emotional')
            maze_name: Name of maze being used
            run_id: ID for this run
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_type = agent_type
        self.maze_name = maze_name
        self.run_id = run_id
        
        # Episode history
        self.episodes: List[EpisodeMetrics] = []
        
        # Running stats for success rate calculation
        self.recent_successes: List[bool] = []
        self.window_size = 100
        
        # CSV file for real-time logging
        self.csv_path = self.log_dir / f"{agent_type}_run{run_id}_episodes.csv"
        self._init_csv()
        
        # Timestamp
        self.start_time = datetime.now()
    
    def _init_csv(self) -> None:
        """Initialize CSV file with headers."""
        headers = [
            'episode', 'steps', 'total_reward', 'success',
            'key_found_step', 'door_attempts_without_key', 'door_opened_step',
            'mean_loss', 'mean_td_error', 'mean_q_value',
            'mean_mood_value', 'mean_mood_action', 'mean_overall_mood',
            'exploration_boosts', 'mean_mood_bias',
            'epsilon', 'effective_epsilon',
            'success_rate_100'
        ]
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_episode(self, metrics: EpisodeMetrics) -> None:
        """
        Log metrics for a completed episode.
        
        Args:
            metrics: Episode metrics to log
        """
        self.episodes.append(metrics)
        
        # Update running success rate
        self.recent_successes.append(metrics.success)
        if len(self.recent_successes) > self.window_size:
            self.recent_successes.pop(0)
        
        success_rate = sum(self.recent_successes) / len(self.recent_successes)
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.episode,
                metrics.steps,
                f"{metrics.total_reward:.4f}",
                int(metrics.success),
                metrics.key_found_step,
                metrics.door_attempts_without_key,
                metrics.door_opened_step,
                f"{metrics.mean_loss:.6f}",
                f"{metrics.mean_td_error:.6f}",
                f"{metrics.mean_q_value:.6f}",
                f"{metrics.mean_mood_value:.6f}",
                f"{metrics.mean_mood_action:.6f}",
                f"{metrics.mean_overall_mood:.6f}",
                metrics.exploration_boosts,
                f"{metrics.mean_mood_bias:.6f}",
                f"{metrics.epsilon:.4f}",
                f"{metrics.effective_epsilon:.4f}",
                f"{success_rate:.4f}"
            ])
    
    def get_success_rate(self, last_n: Optional[int] = None) -> float:
        """Get success rate over last N episodes."""
        if not self.episodes:
            return 0.0
        
        if last_n is None:
            last_n = self.window_size
        
        recent = self.episodes[-last_n:]
        return sum(1 for e in recent if e.success) / len(recent)
    
    def get_avg_steps(self, last_n: Optional[int] = None) -> float:
        """Get average steps over last N episodes."""
        if not self.episodes:
            return 0.0
        
        if last_n is None:
            last_n = self.window_size
        
        recent = self.episodes[-last_n:]
        return sum(e.steps for e in recent) / len(recent)
    
    def get_avg_reward(self, last_n: Optional[int] = None) -> float:
        """Get average reward over last N episodes."""
        if not self.episodes:
            return 0.0
        
        if last_n is None:
            last_n = self.window_size
        
        recent = self.episodes[-last_n:]
        return sum(e.total_reward for e in recent) / len(recent)
    
    def get_first_success_episode(self) -> int:
        """Get episode number of first success."""
        for ep in self.episodes:
            if ep.success:
                return ep.episode
        return -1
    
    def get_episodes_to_success_rate(self, target_rate: float) -> int:
        """Get number of episodes to reach target success rate."""
        if len(self.episodes) < self.window_size:
            return -1
        
        successes = []
        for i, ep in enumerate(self.episodes):
            successes.append(ep.success)
            if len(successes) >= self.window_size:
                rate = sum(successes[-self.window_size:]) / self.window_size
                if rate >= target_rate:
                    return i + 1
        
        return -1
    
    def get_summary(self, last_n: int = 100) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.episodes:
            return {}
        
        recent = self.episodes[-last_n:]
        
        summary = {
            'total_episodes': len(self.episodes),
            'success_rate': self.get_success_rate(last_n),
            'avg_steps': self.get_avg_steps(last_n),
            'avg_reward': self.get_avg_reward(last_n),
            'first_success': self.get_first_success_episode(),
            'total_successes': sum(1 for e in self.episodes if e.success),
        }
        
        # Causal metrics
        key_found_steps = [e.key_found_step for e in recent if e.key_found_step > 0]
        if key_found_steps:
            summary['avg_key_found_step'] = np.mean(key_found_steps)
        
        door_attempts = [e.door_attempts_without_key for e in recent]
        summary['avg_door_attempts_without_key'] = np.mean(door_attempts)
        
        # Emotional metrics
        if self.agent_type == 'emotional':
            summary['avg_mood'] = np.mean([e.mean_overall_mood for e in recent])
            summary['total_exploration_boosts'] = sum(e.exploration_boosts for e in recent)
            summary['avg_mood_bias'] = np.mean([e.mean_mood_bias for e in recent])
        
        return summary
    
    def get_run_metrics(self) -> RunMetrics:
        """Get aggregated metrics for the entire run."""
        run = RunMetrics(
            run_id=self.run_id,
            agent_type=self.agent_type,
            maze_name=self.maze_name,
            total_episodes=len(self.episodes)
        )
        
        if not self.episodes:
            return run
        
        # Performance
        run.first_success_episode = self.get_first_success_episode()
        run.total_successes = sum(1 for e in self.episodes if e.success)
        run.final_success_rate = self.get_success_rate(100)
        run.final_avg_steps = self.get_avg_steps(100)
        run.final_avg_reward = self.get_avg_reward(100)
        
        # Learning speed
        run.episodes_to_50_percent_success = self.get_episodes_to_success_rate(0.5)
        run.episodes_to_90_percent_success = self.get_episodes_to_success_rate(0.9)
        
        # Causal understanding
        key_steps = [e.key_found_step for e in self.episodes if e.key_found_step > 0]
        if key_steps:
            run.avg_key_found_step = np.mean(key_steps)
        
        # Door attempts before learning
        early_episodes = self.episodes[:min(50, len(self.episodes))]
        run.avg_door_attempts_before_first_key = np.mean(
            [e.door_attempts_without_key for e in early_episodes]
        )
        
        # Emotional
        if self.agent_type == 'emotional':
            run.avg_exploration_boosts_per_episode = np.mean(
                [e.exploration_boosts for e in self.episodes]
            )
            run.total_mood_bias_applied = sum(
                abs(e.mean_mood_bias) for e in self.episodes
            )
        
        return run
    
    def save_summary(self) -> None:
        """Save run summary to JSON."""
        summary = self.get_summary()
        run_metrics = self.get_run_metrics()
        
        output = {
            'run_info': {
                'agent_type': self.agent_type,
                'maze_name': self.maze_name,
                'run_id': self.run_id,
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
            },
            'summary': summary,
            'run_metrics': asdict(run_metrics)
        }
        
        summary_path = self.log_dir / f"{self.agent_type}_run{self.run_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(output, f, indent=2)
    
    def print_progress(self, episode: int, every: int = 100) -> None:
        """Print progress update."""
        if episode % every != 0:
            return
        
        summary = self.get_summary(last_n=every)
        
        print(f"\n{'='*60}")
        print(f"Episode {episode} | {self.agent_type} agent | {self.maze_name}")
        print(f"{'='*60}")
        print(f"  Success rate (last {every}): {summary.get('success_rate', 0):.1%}")
        print(f"  Avg steps: {summary.get('avg_steps', 0):.1f}")
        print(f"  Avg reward: {summary.get('avg_reward', 0):.2f}")
        print(f"  First success: episode {summary.get('first_success', -1)}")
        print(f"  Total successes: {summary.get('total_successes', 0)}")
        
        if self.agent_type == 'emotional':
            print(f"  Avg mood: {summary.get('avg_mood', 0):.4f}")
            print(f"  Exploration boosts: {summary.get('total_exploration_boosts', 0)}")


class ExperimentLogger:
    """
    Logs metrics across multiple runs for comparison.
    """
    
    def __init__(self, experiment_dir: str, experiment_name: str = "experiment"):
        """
        Initialize experiment logger.
        
        Args:
            experiment_dir: Directory for experiment logs
            experiment_name: Name of the experiment
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.runs: Dict[str, List[RunMetrics]] = {
            'baseline': [],
            'emotional': []
        }
    
    def add_run(self, run_metrics: RunMetrics) -> None:
        """Add completed run metrics."""
        self.runs[run_metrics.agent_type].append(run_metrics)
    
    def get_comparison(self) -> Dict[str, Any]:
        """Get comparison between baseline and emotional agents."""
        comparison = {}
        
        for agent_type in ['baseline', 'emotional']:
            runs = self.runs[agent_type]
            if not runs:
                continue
            
            comparison[agent_type] = {
                'n_runs': len(runs),
                'first_success_mean': np.mean([r.first_success_episode for r in runs if r.first_success_episode > 0]),
                'first_success_std': np.std([r.first_success_episode for r in runs if r.first_success_episode > 0]),
                'final_success_rate_mean': np.mean([r.final_success_rate for r in runs]),
                'final_success_rate_std': np.std([r.final_success_rate for r in runs]),
                'final_avg_steps_mean': np.mean([r.final_avg_steps for r in runs]),
                'final_avg_steps_std': np.std([r.final_avg_steps for r in runs]),
                'episodes_to_50_mean': np.mean([r.episodes_to_50_percent_success for r in runs if r.episodes_to_50_percent_success > 0]),
                'episodes_to_90_mean': np.mean([r.episodes_to_90_percent_success for r in runs if r.episodes_to_90_percent_success > 0]),
            }
        
        return comparison
    
    def print_comparison(self) -> None:
        """Print comparison table."""
        comparison = self.get_comparison()
        
        print("\n" + "="*70)
        print(f"EXPERIMENT COMPARISON: {self.experiment_name}")
        print("="*70)
        
        metrics = [
            ('First Success (episode)', 'first_success_mean', 'first_success_std'),
            ('Final Success Rate', 'final_success_rate_mean', 'final_success_rate_std'),
            ('Final Avg Steps', 'final_avg_steps_mean', 'final_avg_steps_std'),
            ('Episodes to 50% Success', 'episodes_to_50_mean', None),
            ('Episodes to 90% Success', 'episodes_to_90_mean', None),
        ]
        
        print(f"\n{'Metric':<30} {'Baseline':<20} {'Emotional':<20}")
        print("-"*70)
        
        for name, mean_key, std_key in metrics:
            baseline_val = comparison.get('baseline', {}).get(mean_key, float('nan'))
            emotional_val = comparison.get('emotional', {}).get(mean_key, float('nan'))
            
            if std_key:
                baseline_std = comparison.get('baseline', {}).get(std_key, 0)
                emotional_std = comparison.get('emotional', {}).get(std_key, 0)
                baseline_str = f"{baseline_val:.2f} ± {baseline_std:.2f}"
                emotional_str = f"{emotional_val:.2f} ± {emotional_std:.2f}"
            else:
                baseline_str = f"{baseline_val:.2f}"
                emotional_str = f"{emotional_val:.2f}"
            
            print(f"{name:<30} {baseline_str:<20} {emotional_str:<20}")
        
        print("="*70)
    
    def save_comparison(self) -> None:
        """Save comparison to JSON."""
        comparison = self.get_comparison()
        
        output = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'comparison': comparison,
            'all_runs': {
                agent_type: [asdict(r) for r in runs]
                for agent_type, runs in self.runs.items()
            }
        }
        
        path = self.experiment_dir / f"{self.experiment_name}_comparison.json"
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)


# Quick test
if __name__ == "__main__":
    import tempfile
    
    print("Testing MetricsLogger...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = MetricsLogger(
            log_dir=tmpdir,
            agent_type="emotional",
            maze_name="minimal",
            run_id=0
        )
        
        # Simulate some episodes
        for i in range(200):
            # Simulate learning: success rate increases over time
            success = np.random.random() < (i / 300)
            steps = int(200 - i * 0.5 + np.random.randint(-20, 20)) if not success else int(50 + np.random.randint(-10, 10))
            steps = max(10, steps)
            
            metrics = EpisodeMetrics(
                episode=i,
                steps=steps,
                total_reward=-0.04 * steps + (10 if success else 0),
                success=success,
                key_found_step=int(steps * 0.3) if success else -1,
                door_attempts_without_key=np.random.randint(0, 5) if not success else 0,
                mean_mood_value=np.random.randn() * 0.1,
                mean_mood_action=np.random.randn() * 0.1,
                mean_overall_mood=np.random.randn() * 0.1,
                exploration_boosts=np.random.randint(0, 10),
                epsilon=max(0.05, 1.0 - i * 0.005),
            )
            
            logger.log_episode(metrics)
            
            if i % 100 == 0:
                logger.print_progress(i, every=100)
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        summary = logger.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Save
        logger.save_summary()
        print(f"\n  Saved to: {tmpdir}")
        
        # Test run metrics
        run_metrics = logger.get_run_metrics()
        print(f"\n  Run metrics: {asdict(run_metrics)}")
    
    print("\n✓ MetricsLogger works!")
    
    # Test ExperimentLogger
    print("\nTesting ExperimentLogger...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_logger = ExperimentLogger(tmpdir, "test_experiment")
        
        # Add some fake runs
        for agent_type in ['baseline', 'emotional']:
            for run_id in range(3):
                run = RunMetrics(
                    run_id=run_id,
                    agent_type=agent_type,
                    maze_name="minimal",
                    total_episodes=1000,
                    first_success_episode=50 + np.random.randint(-20, 20) - (10 if agent_type == 'emotional' else 0),
                    final_success_rate=0.8 + np.random.random() * 0.15,
                    final_avg_steps=40 + np.random.randint(-5, 5),
                    episodes_to_50_percent_success=200 + np.random.randint(-50, 50),
                    episodes_to_90_percent_success=500 + np.random.randint(-100, 100),
                )
                exp_logger.add_run(run)
        
        exp_logger.print_comparison()
        exp_logger.save_comparison()
    
    print("\n✓ ExperimentLogger works!")