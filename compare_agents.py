"""
Compare baseline vs emotional DQN agents.
Runs multiple training runs and compares performance.
Saves policy checkpoints during training for later analysis.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np

from train import train, get_checkpoint_episodes, resolve_network_class
from agents import SmallDQNNetwork
from utils import ExperimentLogger
from utils.reversal_analysis import bootstrap_ci, cohens_d_between

# Sweep grids from the implementation brief. eta runs low because the theory
# predicts mood matters most at low eta: at eta=0.9 the mood term carries only
# (1 - 0.9) = 10% of the target, the minimum-mood setting of the range.
DEFAULT_ETA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)
DEFAULT_LAMBDA_GRID = (0.5, 0.8, 0.95)
# Baseline learning rates spanning a comparable range of effective step size.
# Without this arm, an emotional win cannot be distinguished from a step-size
# change wearing a costume.
DEFAULT_LR_GRID = (3e-5, 1e-4, 3e-4)


def _parse_reward_overrides(pairs: Optional[List[str]]) -> Dict[str, float]:
    """Parse repeated CLI args like: --reward step=-0.5 --reward timeout=-50."""
    if not pairs:
        return {}
    out: Dict[str, float] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Invalid --reward '{raw}'. Use key=value.")
        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Invalid --reward '{raw}'. Empty key.")
        out[k] = float(v)
    return out


def _parse_sweep_spec(items: Optional[List[str]]) -> Dict[str, List[float]]:
    """Parse repeated --sweep key=v1,v2,v3 into {key: [values]}."""
    out: Dict[str, List[float]] = {}
    for raw in items or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --sweep '{raw}'. Use key=v1,v2,v3.")
        key, values = raw.split("=", 1)
        out[key.strip()] = [float(v) for v in values.split(",") if v.strip()]
    return out


def build_sweep_conditions(
    agent_types: Sequence[str],
    emotional_grid: Dict[str, List[float]],
    baseline_grid: Dict[str, List[float]],
) -> List[Dict[str, Any]]:
    """Cross-product conditions, per agent type.

    Emotional/yoked arms sweep the mood parameters; the baseline arm sweeps
    learning rate. Sweeping only the emotional side would leave any emotional
    advantage indistinguishable from a change in effective step size.
    """
    conditions: List[Dict[str, Any]] = []
    for agent_type in agent_types:
        grid = baseline_grid if agent_type == "baseline" else emotional_grid
        if not grid:
            conditions.append({"agent_type": agent_type, "params": {}})
            continue
        keys = sorted(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            conditions.append({
                "agent_type": agent_type,
                "params": dict(zip(keys, combo)),
            })
    return conditions


def condition_label(condition: Dict[str, Any]) -> str:
    params = condition["params"]
    if not params:
        return condition["agent_type"]
    body = ",".join(f"{k}={v:g}" for k, v in sorted(params.items()))
    return f"{condition['agent_type']}[{body}]"


def _run_metrics_for_seed(logger, last_n: int) -> Dict[str, float]:
    """Per-seed outcome measures, taken over the final `last_n` episodes."""
    episodes = logger.episodes[-last_n:] if logger.episodes else []
    if not episodes:
        return {}
    path_counts: Dict[str, int] = {}
    for e in episodes:
        path_counts[e.path_type] = path_counts.get(e.path_type, 0) + 1
    return {
        "success_rate": float(np.mean([e.success for e in episodes])),
        "mean_return": float(np.mean([e.total_reward for e in episodes])),
        "mean_steps": float(np.mean([e.steps for e in episodes])),
        "shield_route_rate": path_counts.get("shield_route", 0) / len(episodes),
        "mean_mood": float(np.mean([e.mean_overall_mood for e in episodes])),
        "mood_clip_fraction": float(np.mean([e.mood_clip_fraction for e in episodes])),
        "first_success": logger.get_first_success_episode(),
    }


def run_sweep(
    maze_name: str = "shield_trap",
    agent_types: Optional[List[str]] = None,
    emotional_grid: Optional[Dict[str, List[float]]] = None,
    baseline_grid: Optional[Dict[str, List[float]]] = None,
    n_seeds: int = 20,
    base_seed: int = 42,
    n_episodes: int = 800,
    last_n: int = 100,
    reference: Optional[str] = None,
    log_dir: str = "experiments",
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    image_size: int = 64,
    network_class=None,
    frame_stack: int = 1,
    reward_overrides: Optional[Dict[str, float]] = None,
    max_steps: Optional[int] = None,
    shield_lights_up: Optional[bool] = None,
    metric: str = "success_rate",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Hyperparameter sweep with per-seed variance and effect sizes.

    Deep RL seed variance is large enough that 3-5 seeds cannot detect a small
    effect, so the default is 20 seeds per condition and every per-seed value
    is kept in the output rather than only the mean. Comparisons report effect
    size with a bootstrap CI; significance stars are deliberately absent.
    """
    agent_types = agent_types or ["baseline", "emotional"]
    emotional_grid = emotional_grid or {
        "eta": list(DEFAULT_ETA_GRID),
        "lambda_mood": list(DEFAULT_LAMBDA_GRID),
    }
    baseline_grid = baseline_grid or {"learning_rate": list(DEFAULT_LR_GRID)}
    config = dict(config or {})

    conditions = build_sweep_conditions(agent_types, emotional_grid, baseline_grid)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(log_dir) / f"sweep_{maze_name}_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    total = len(conditions) * n_seeds
    if verbose:
        print("=" * 72)
        print(f"SWEEP: {maze_name}")
        print("=" * 72)
        print(f"  Conditions: {len(conditions)}  x  {n_seeds} seeds  =  {total} runs")
        for c in conditions:
            print(f"    {condition_label(c)}")
        print(f"  Episodes per run: {n_episodes} (metrics over last {last_n})")
        print(f"  Output: {sweep_dir}")
        print("=" * 72, flush=True)

    rows: List[Dict[str, Any]] = []
    pbar = tqdm(total=total, desc="sweep", colour="green")

    for condition in conditions:
        label = condition_label(condition)
        for i in range(n_seeds):
            seed = base_seed + i * 100
            run_config = {**config, **condition["params"]}
            logger = train(
                maze_name=maze_name,
                agent_type=condition["agent_type"],
                n_episodes=n_episodes,
                seed=seed,
                log_dir=str(sweep_dir / label.replace("/", "_")),
                run_id=i,
                device=device,
                config=run_config,
                verbose=False,
                show_tqdm=False,
                checkpoint_interval=max(n_episodes, 10 ** 9),
                image_size=image_size,
                network_class=network_class,
                frame_stack=frame_stack,
                reward_overrides=reward_overrides,
                max_steps=max_steps,
                shield_lights_up=shield_lights_up,
            )
            rows.append({
                "condition": label,
                "agent_type": condition["agent_type"],
                **{f"param_{k}": v for k, v in condition["params"].items()},
                "seed": seed,
                **_run_metrics_for_seed(logger, last_n),
            })
            pbar.update(1)
            _write_sweep_outputs(sweep_dir, rows, conditions, metric, reference)
    pbar.close()

    summary = _write_sweep_outputs(sweep_dir, rows, conditions, metric, reference)
    if verbose:
        print_sweep_report(summary, metric)
    return summary


def summarize_sweep(
    rows: Sequence[Dict[str, Any]],
    metric: str = "success_rate",
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """Per-condition mean, per-seed variance, CI, and effect size vs reference."""
    labels = sorted({r["condition"] for r in rows})
    per_condition: Dict[str, Any] = {}
    for label in labels:
        values = [r[metric] for r in rows if r["condition"] == label and metric in r]
        per_condition[label] = {
            "n_seeds": len(values),
            "mean": float(np.mean(values)) if values else None,
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "ci95": bootstrap_ci(values),
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
            "per_seed": values,
        }

    # Default reference: the plain baseline at its default learning rate if
    # present, else the first condition.
    if reference is None:
        default_lr = f"baseline[learning_rate={1e-4:g}]"
        reference = default_lr if default_lr in per_condition else labels[0]

    ref_values = per_condition.get(reference, {}).get("per_seed", [])
    contrasts = {}
    for label in labels:
        if label == reference:
            continue
        values = per_condition[label]["per_seed"]
        diff = (float(np.mean(values) - np.mean(ref_values))
                if values and ref_values else None)
        contrasts[label] = {
            "vs": reference,
            "mean_difference": diff,
            "cohens_d": cohens_d_between(values, ref_values),
        }

    return {
        "metric": metric,
        "reference": reference,
        "per_condition": per_condition,
        "contrasts": contrasts,
        "n_rows": len(rows),
    }


def _write_sweep_outputs(
    sweep_dir: Path,
    rows: List[Dict[str, Any]],
    conditions: List[Dict[str, Any]],
    metric: str,
    reference: Optional[str],
) -> Dict[str, Any]:
    """Write per-seed CSV and summary JSON after every run, so a long sweep is
    readable while it is still going and survives interruption."""
    summary = summarize_sweep(rows, metric, reference)
    summary["conditions"] = [condition_label(c) for c in conditions]
    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    fieldnames = sorted({k for r in rows for k in r})
    with open(sweep_dir / "sweep_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary["sweep_dir"] = str(sweep_dir)
    return summary


def print_sweep_report(summary: Dict[str, Any], metric: str) -> None:
    print("\n" + "=" * 78)
    print(f"SWEEP RESULTS — metric: {metric}   reference: {summary['reference']}")
    print("=" * 78)
    print(f"  {'condition':<34}{'n':>4}{'mean':>9}{'sd':>8}{'95% CI':>20}{'d':>7}")
    print("  " + "-" * 74)
    for label, stats in summary["per_condition"].items():
        ci = stats["ci95"]
        ci_s = (f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                if ci and not np.isnan(ci[0]) else "--")
        if label == summary["reference"]:
            d_s = "ref"
        else:
            d = summary["contrasts"].get(label, {}).get("cohens_d")
            # nan means undefined (too few seeds, or zero variance in both
            # groups) — not "this is the reference".
            d_s = f"{d:+.2f}" if d is not None and not np.isnan(d) else "--"
        print(f"  {label:<34}{stats['n_seeds']:>4}"
              f"{(stats['mean'] if stats['mean'] is not None else float('nan')):>9.3f}"
              f"{(stats['sd'] if stats['sd'] is not None else float('nan')):>8.3f}"
              f"{ci_s:>20}{d_s:>7}")
    print("\n  Effect sizes are Cohen's d over per-seed values against the "
          "reference condition.")
    print("  Per-seed values are in sweep_runs.csv — report the spread, not just "
          "the means.")
    print("=" * 78)


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


def evaluate_run_checkpoints_greedy(
    entry: Dict[str, Any],
    maze_name: str,
    network_size: str = "standard",
    image_size: int = 64,
    reward_overrides: Optional[Dict[str, float]] = None,
    max_steps: Optional[int] = None,
    shield_lights_up: Optional[bool] = None,
    frame_stack: int = 1,
) -> Optional[Dict[str, Any]]:
    """
    Greedy-evaluate (epsilon=0) every saved checkpoint of one training run.

    The maze and greedy policy are deterministic, so a single rollout per
    checkpoint fully characterizes its greedy behavior. Prints a table,
    saves greedy_checkpoint_eval.json in the run dir, and returns the best
    checkpoint entry (success first, then total reward).
    """
    from environments import VisualMazeEnv
    from analyze_policy_evolution import load_agent_checkpoint
    from utils.trajectory_diagnostics import rollout_episode

    checkpoints = [Path(p) for p in entry.get("checkpoints", [])]
    if not checkpoints:
        return None

    agent_type = entry["agent_type"]
    env = VisualMazeEnv(
        maze_name=maze_name,
        image_size=image_size,
        reward_overrides=reward_overrides,
        max_steps=max_steps,
        shield_lights_up=shield_lights_up,
        frame_stack=frame_stack,
    )

    results: List[Dict[str, Any]] = []
    for ckpt in sorted(checkpoints, key=lambda p: int(p.stem.split("_episode_")[-1])):
        episode_num = int(ckpt.stem.split("_episode_")[-1])
        agent = load_agent_checkpoint(
            str(ckpt),
            env,
            agent_type=agent_type,
            network_size=network_size,
            image_size=image_size,
        )
        _, summary = rollout_episode(
            agent, env,
            rollout_id=0,
            stage=f"episode_{episode_num}",
            checkpoint_episode=episode_num,
        )
        results.append({
            "checkpoint_episode": episode_num,
            "checkpoint_path": str(ckpt),
            "success": bool(summary["success"]),
            "total_reward": summary["total_reward"],
            "total_steps": summary["total_steps"],
            "path_type": summary["path_type"],
        })

    best = max(results, key=lambda r: (r["success"], r["total_reward"]))

    print(f"\n  Greedy checkpoint evaluation ({agent_type} run {entry['run_id']}):")
    print(f"    {'episode':>8} {'success':>8} {'reward':>9} {'steps':>6}  path_type")
    for r in results:
        marker = "  <-- best" if r is best else ""
        print(
            f"    {r['checkpoint_episode']:>8} {str(r['success']):>8} "
            f"{r['total_reward']:>9.2f} {r['total_steps']:>6}  {r['path_type']}{marker}"
        )

    eval_path = Path(entry["run_log_dir"]) / "greedy_checkpoint_eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "best": best}, f, indent=2)
    print(f"    Saved: {eval_path}")
    print(f"    Best checkpoint: {best['checkpoint_path']}")

    return best


def analyze_run_checkpoints(
    entry: Dict[str, Any],
    maze_name: str,
    analysis_episodes: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    network_size: str = 'standard',
    image_size: int = 64,
    reward_overrides: Optional[Dict[str, float]] = None,
    max_steps: Optional[int] = None,
    shield_lights_up: Optional[bool] = None,
    frame_stack: int = 1,
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
        network_size=network_size,
        image_size=image_size,
        reward_overrides=reward_overrides,
        max_steps=max_steps,
        shield_lights_up=shield_lights_up,
        frame_stack=frame_stack,
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

        # Q-value policy map from final checkpoint
        from environments import VisualMazeEnv
        from analyze_policy_evolution import load_agent_checkpoint
        from utils.policy_map import plot_policy_map_panels

        checkpoints = list(Path(checkpoint_dir).glob("agent_episode_*.pt"))
        if checkpoints:
            final_ckpt = sorted(
                checkpoints,
                key=lambda p: int(p.stem.split("_episode_")[-1]),
            )[-1]
            env = VisualMazeEnv(
                maze_name=maze_name,
                image_size=image_size,
                reward_overrides=reward_overrides,
                max_steps=max_steps,
                shield_lights_up=shield_lights_up,
                frame_stack=frame_stack,
            )
            agent = load_agent_checkpoint(
                str(final_ckpt),
                env,
                agent_type=agent_type,
                network_size=network_size,
                image_size=image_size,
            )
            map_path = output_dir / f"{agent_type}_run{run_id}_policy_map.png"
            plot_policy_map_panels(agent, env, save_path=str(map_path))
            print(f"  Saved policy map to {map_path}")


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
    network_size: str = 'standard',
    baseline_checkpoint: Optional[str] = None,
    emotional_checkpoint: Optional[str] = None,
    reset_epsilon: Optional[float] = None,
    reward_overrides: Optional[Dict[str, float]] = None,
    max_steps: Optional[int] = None,
    shield_lights_up: Optional[bool] = None,
    show_tqdm: bool = False,
    frame_stack: int = 1,
    agent_types: Optional[List[str]] = None,
) -> ExperimentLogger:
    """
    Run comparison experiment across agent types.

    agent_types defaults to ["baseline", "emotional"]. Pass
    ["baseline", "emotional", "yoked"] for the three-way comparison the yoked
    control is for. "yoked" must come after "emotional" in the list: each
    yoked run consumes the mood trace of an emotional run from a *different*
    seed produced earlier in this same experiment (run_id r takes emotional
    run (r + 1) % n_runs), unless config["yoked_traces"] names donors
    explicitly.

    Each training run saves checkpoints under:
        {experiment_dir}/{agent_type}/{run_log_dir}/checkpoints/
    """
    if agent_types is None:
        agent_types = ["baseline", "emotional"]
    if "yoked" in agent_types and "emotional" not in agent_types \
            and not (config or {}).get("yoked_traces"):
        raise ValueError(
            "A yoked run needs donor mood traces: either include 'emotional' "
            "in agent_types or set config['yoked_traces']"
        )
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
        eps_start = (config or {}).get("epsilon_start", 1.0)
        eps_end = (config or {}).get("epsilon_end", 0.05)
        print(f"  Epsilon decay: {eps_start} -> {eps_end} over {n_episodes} episodes")
        print(f"  Checkpoint interval: {checkpoint_interval}")
        print(f"  Planned checkpoint episodes: {planned_episodes}")
        print(f"  Agent types: {', '.join(agent_types)}")
        print(f"  Total training runs: {n_runs * len(agent_types)}")
        print(f"  Image size: {image_size}")
        print(f"  Network: {network_class.__name__ if network_class else 'DQNNetwork'}")
        if baseline_checkpoint:
            print(f"  Baseline pretrained: {baseline_checkpoint}")
        if emotional_checkpoint:
            print(f"  Emotional pretrained: {emotional_checkpoint}")
        if reset_epsilon is not None:
            print(f"  Reset epsilon: {reset_epsilon}")
        print(f"  Device: {device or 'auto'}")
        print(f"  Output: {experiment_dir}")
        print(f"  Config: {config}")
        print("=" * 70)

    total_runs = n_runs * len(agent_types)
    checkpoint_manifest: List[Dict[str, Any]] = []
    # run_id -> mood_trace.csv, filled in as emotional runs finish, so yoked
    # runs can be yoked to a real mood trace from a different seed.
    emotional_traces: Dict[int, str] = {}

    overall_pbar = tqdm(
        total=total_runs,
        desc="Overall Progress",
        position=0,
        leave=True,
        colour="green",
    )

    for agent_type in agent_types:
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

            pretrained_checkpoint = None
            if agent_type == "baseline" and baseline_checkpoint:
                pretrained_checkpoint = baseline_checkpoint
            elif agent_type == "emotional" and emotional_checkpoint:
                pretrained_checkpoint = emotional_checkpoint

            run_config = config
            if agent_type == "yoked" and not config.get("yoked_traces"):
                # Yoke to an emotional run from a different seed: shifting by
                # one keeps each donor distinct from the run it feeds, and with
                # n_runs == 1 there is no different-seed donor to be had.
                donor_id = (run_id + 1) % n_runs
                if donor_id == run_id or donor_id not in emotional_traces:
                    donor_id = next(
                        (rid for rid in sorted(emotional_traces) if rid != run_id),
                        None,
                    )
                if donor_id is None:
                    raise RuntimeError(
                        "No emotional mood trace from a different seed is "
                        "available to yoke run {} to. Use --runs >= 2 or pass "
                        "--yoked_trace explicitly.".format(run_id)
                    )
                run_config = {**config, "yoked_traces": [emotional_traces[donor_id]]}
                if verbose:
                    print(f"  Yoked to emotional run {donor_id}: "
                          f"{emotional_traces[donor_id]}")

            logger = train(
                maze_name=maze_name,
                agent_type=agent_type,
                n_episodes=n_episodes,
                seed=seed,
                log_dir=str(experiment_dir / agent_type),
                run_id=run_id,
                device=device,
                config=run_config,
                verbose=True,
                progress_every=max(1, n_episodes // 5),
                checkpoint_interval=checkpoint_interval,
                image_size=image_size,
                network_class=network_class,
                pretrained_checkpoint=pretrained_checkpoint,
                reset_epsilon=reset_epsilon,
                reward_overrides=reward_overrides,
                max_steps=max_steps,
                shield_lights_up=shield_lights_up,
                show_tqdm=show_tqdm,
                frame_stack=frame_stack,
            )

            run_metrics = logger.get_run_metrics()
            exp_logger.add_run(run_metrics)

            if agent_type == "emotional":
                trace_path = Path(logger.log_dir) / "mood_trace.csv"
                if trace_path.exists():
                    emotional_traces[run_id] = str(trace_path)

            entry = build_run_checkpoint_entry(
                agent_type=agent_type,
                run_id=run_id,
                seed=seed,
                run_log_dir=Path(logger.log_dir),
                n_episodes=n_episodes,
                checkpoint_interval=checkpoint_interval,
            )
            checkpoint_manifest.append(entry)

            # Greedy-evaluate every checkpoint so a late-training collapse
            # doesn't hide the best policy found during the run
            try:
                evaluate_run_checkpoints_greedy(
                    entry=entry,
                    maze_name=maze_name,
                    network_size=network_size,
                    image_size=image_size,
                    reward_overrides=reward_overrides,
                    max_steps=max_steps,
                    shield_lights_up=shield_lights_up,
                    frame_stack=frame_stack,
                )
            except Exception as e:
                print(f"  (greedy checkpoint eval failed: {e})")

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
                network_size=network_size,
                image_size=image_size,
                reward_overrides=reward_overrides,
                max_steps=max_steps,
                shield_lights_up=shield_lights_up,
                frame_stack=frame_stack,
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
    reward_overrides: Optional[Dict[str, float]] = None,
    max_steps: Optional[int] = None,
    shield_lights_up: Optional[bool] = None,
    show_tqdm: bool = False,
    frame_stack: int = 1,
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
        "eta": config.get("eta", 0.9),
        "mood_clip_range": config.get("mood_clip_range", 1.0),
        **config,
    }

    print(f"  lambda_mood: {test_config['lambda_mood']}")
    print(f"  eta: {test_config['eta']}")
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
                reward_overrides=reward_overrides,
                max_steps=max_steps,
                shield_lights_up=shield_lights_up,
                show_tqdm=show_tqdm,
                frame_stack=frame_stack,
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
                        help="η in Q_target = Q + ηδ; shared by baseline and emotional agents")

    parser.add_argument(
        "--reward",
        action="append",
        default=None,
        help="Override maze rewards at runtime (repeatable), e.g. --reward step=-0.5 --reward timeout=-50",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override maze max_steps at runtime",
    )
    parser.add_argument(
        "--shield_lights_up",
        action="store_true",
        help="Brighten the entire observation when the agent holds the shield",
    )
    parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Show per-episode tqdm bars (off by default to reduce console clutter)",
    )
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=1,
        help="Stack the m most recent frames along channels (Nature DQN m=4)",
    )

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
    parser.add_argument("--double_dqn", action="store_true",
                        help="Use Double DQN targets to curb Q overestimation")
    parser.add_argument("--epsilon_end", type=float, default=0.05,
                        help="Final exploration rate (epsilon floor)")
    parser.add_argument("--target_update_freq", type=int, default=1000,
                        help="Gradient updates between target network syncs")
    parser.add_argument("--buffer_size", type=int, default=50000,
                        help="Replay buffer size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")

    parser.add_argument("--lambda_mood", type=float, default=0.8,
                        help="Mood retention per update (0-1, higher = slower "
                             "change). With --mood_delta_source batch_sequential "
                             "this is applied batch_size times per gradient step")
    parser.add_argument("--mood_min", type=float, default=None,
                        help="Lower bound for clipped mood (overrides --mood_clip_range)")
    parser.add_argument("--mood_max", type=float, default=None,
                        help="Upper bound for clipped mood (overrides --mood_clip_range)")
    parser.add_argument("--mood_clip_range", type=float, default=1.0,
                        help="Symmetric mood clip +/-C. Only meaningful relative "
                             "to the reward scale: raw maze rewards span ~+/-55")
    parser.add_argument("--mood_delta_source", type=str, default="batch_sequential",
                        choices=["online", "batch_mean", "batch_sequential"],
                        help="Which delta feeds the mood (see EmotionalDQNAgent docstring)")
    parser.add_argument("--mood_delta_unsigned", action="store_true",
                        help="Integrate |delta| (arousal) instead of signed delta (valence)")
    parser.add_argument("--sweep_mode", action="store_true",
                        help="Run a hyperparameter sweep instead of a comparison")
    parser.add_argument("--sweep", action="append", default=None,
                        help="Emotional-arm grid, repeatable: --sweep eta=0.1,0.3,0.5,0.7,0.9 "
                             "--sweep lambda_mood=0.5,0.8,0.95 (default: both)")
    parser.add_argument("--sweep_baseline", action="append", default=None,
                        help="Baseline-arm grid, repeatable: "
                             "--sweep_baseline learning_rate=3e-5,1e-4,3e-4")
    parser.add_argument("--sweep_seeds", type=int, default=20,
                        help="Seeds per condition (>=20; deep RL seed variance is "
                             "too large for 3-5 to detect a small effect)")
    parser.add_argument("--sweep_metric", type=str, default="success_rate",
                        help="Per-seed outcome to summarize")
    parser.add_argument("--sweep_last_n", type=int, default=100,
                        help="Episodes at the end of each run used for the metric")
    parser.add_argument("--sweep_reference", type=str, default=None,
                        help="Condition label used as the contrast reference")
    parser.add_argument("--agents", type=str, default="baseline,emotional",
                        help="Comma-separated agent types to run, in order. Use "
                             "'baseline,emotional,yoked' for the three-way "
                             "comparison; yoked must follow emotional")
    parser.add_argument("--yoked_mode", type=str, default="replay_trace",
                        choices=["replay_trace", "ou_process"],
                        help="How yoked agents get their mood")
    parser.add_argument("--yoked_trace", type=str, nargs="+", default=None,
                        help="Explicit donor mood_trace.csv file(s)/run dir(s). "
                             "Omit to yoke each run to an emotional run from a "
                             "different seed in this same experiment")
    parser.add_argument("--bootstrap_on_truncation", action="store_true",
                        help="Treat timeouts as non-terminal in the value target "
                             "(all agent types); default off preserves old results")
    parser.add_argument("--reward_scale", type=float, default=1.0,
                        help="Multiply rewards fed to the agent, identically for "
                             "all agent types; logged returns stay unscaled")

    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                        help="Path to pretrained baseline checkpoint (.pt)")
    parser.add_argument("--emotional_checkpoint", type=str, default=None,
                        help="Path to pretrained emotional checkpoint (.pt)")
    parser.add_argument("--reset_epsilon", type=float, default=None,
                        help="Reset epsilon to this value when loading checkpoint")

    args = parser.parse_args()

    reward_overrides = _parse_reward_overrides(args.reward)

    config = {
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
        "lambda_mood": args.lambda_mood,
        "eta": args.eta,
        "mood_clip_range": args.mood_clip_range,
        "mood_delta_source": args.mood_delta_source,
        "mood_delta_signed": not args.mood_delta_unsigned,
        "reward_scale": args.reward_scale,
        "bootstrap_on_truncation": args.bootstrap_on_truncation,
        "yoked_mode": args.yoked_mode,
        "yoked_traces": args.yoked_trace,
        "double_dqn": args.double_dqn,
        "epsilon_end": args.epsilon_end,
        "target_update_freq": args.target_update_freq,
    }

    agent_types = [a.strip() for a in args.agents.split(",") if a.strip()]
    unknown = [a for a in agent_types if a not in ("baseline", "emotional", "yoked")]
    if unknown:
        parser.error(f"Unknown agent type(s): {', '.join(unknown)}")
    if "yoked" in agent_types and "emotional" in agent_types \
            and agent_types.index("yoked") < agent_types.index("emotional"):
        parser.error("'yoked' must come after 'emotional': it consumes the mood "
                     "traces those runs produce")

    if args.mood_min is not None or args.mood_max is not None:
        config["mood_bounds"] = (
            args.mood_min if args.mood_min is not None else -args.mood_clip_range,
            args.mood_max if args.mood_max is not None else args.mood_clip_range,
        )

    analysis_episodes = None
    if args.analysis_episodes:
        analysis_episodes = [int(e) for e in args.analysis_episodes.split(",")]

    network_class = resolve_network_class(args.network_size, args.image_size)

    if args.sweep_mode:
        run_sweep(
            maze_name=args.maze,
            agent_types=agent_types,
            emotional_grid=_parse_sweep_spec(args.sweep) or None,
            baseline_grid=_parse_sweep_spec(args.sweep_baseline) or None,
            n_seeds=args.sweep_seeds,
            base_seed=args.seed,
            n_episodes=args.episodes,
            last_n=args.sweep_last_n,
            reference=args.sweep_reference,
            log_dir=args.log_dir,
            device=args.device,
            config=config,
            image_size=args.image_size,
            network_class=network_class,
            frame_stack=args.frame_stack,
            reward_overrides=reward_overrides,
            max_steps=args.max_steps,
            shield_lights_up=args.shield_lights_up,
            metric=args.sweep_metric,
        )
    elif args.quick_test:
        quick_test(
            maze_name=args.maze,
            n_episodes=args.episodes,
            seed=args.seed,
            device=args.device,
            config=config,
            checkpoint_interval=args.checkpoint_interval,
            image_size=args.image_size,
            network_class=network_class,
            reward_overrides=reward_overrides,
            max_steps=args.max_steps,
            shield_lights_up=args.shield_lights_up,
            show_tqdm=args.tqdm,
            frame_stack=args.frame_stack,
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
            network_size=args.network_size,
            baseline_checkpoint=args.baseline_checkpoint,
            emotional_checkpoint=args.emotional_checkpoint,
            reset_epsilon=args.reset_epsilon,
            reward_overrides=reward_overrides,
            max_steps=args.max_steps,
            shield_lights_up=args.shield_lights_up,
            show_tqdm=args.tqdm,
            frame_stack=args.frame_stack,
            agent_types=agent_types,
        )


if __name__ == "__main__":
    main()
