"""
Pilot: calibrate the reversal period K by measuring episodes-to-recovery.

Runs acquisition to criterion, executes a **single** flip, and measures how
long the agent takes to adapt. K for the full study is then set to roughly
1.5-2x the median recovery, so each reversal block is long enough for
adaptation to complete but short enough to fit R of them in a run.

Recovery metric
---------------
Performance is **adherence to the currently-optimal route**, not raw return:
the two contingencies have very different optimal returns (+12 vs -27 on
shield_trap_easy), so "80% of the pre-flip return" would be meaningless.

    pre_level  = fraction of the last `window` acquisition episodes that took
                 the pre-flip optimal route (shield_route)
    post(i)    = fraction of episodes in the `window` ending at post-flip
                 episode i that took the post-flip optimal route (direct /
                 trap_rush)
    recovery   = first i with post(i) >= recovery_fraction * pre_level

Runs where post-flip adherence never reaches that level within the cap are
**right-censored** and reported as such, never silently dropped or clipped —
censoring is itself a result (it means K must exceed the cap, or the maze is
too hard for the reversal design).
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import resolve_network_class
from scripts.train_reversal import (
    run_reversal_training,
    _parse_reward_overrides,
    PROTECTIVE,
    NON_PROTECTIVE,
)

# Path type that is optimal under each contingency.
OPTIMAL_PATH_TYPE = {PROTECTIVE: "shield_route", NON_PROTECTIVE: "trap_rush"}


def rolling_fraction(flags: List[int], window: int) -> List[Optional[float]]:
    """Rolling mean over `window`, None until the window is full."""
    out: List[Optional[float]] = []
    total = 0
    for i, f in enumerate(flags):
        total += f
        if i >= window:
            total -= flags[i - window]
        out.append(total / window if i >= window - 1 else None)
    return out


def measure_recovery(
    schedule_csv: str,
    window: int = 20,
    recovery_fraction: float = 0.80,
) -> Dict[str, Any]:
    """Episodes from the flip until optimal-route adherence recovers."""
    rows = list(csv.DictReader(open(schedule_csv)))
    acq = [r for r in rows if r["phase"] == "acquisition"]
    post = [r for r in rows if r["phase"] == "reversal"]
    if not post:
        raise ValueError(f"{schedule_csv} has no post-flip episodes")

    post_contingency = post[0]["contingency"]
    pre_target = OPTIMAL_PATH_TYPE[PROTECTIVE]
    post_target = OPTIMAL_PATH_TYPE[post_contingency]

    pre_window = acq[-window:] if len(acq) >= window else acq
    pre_level = (
        sum(r["path_type"] == pre_target for r in pre_window) / len(pre_window)
        if pre_window else 0.0
    )
    threshold = recovery_fraction * pre_level

    flags = [int(r["path_type"] == post_target) for r in post]
    series = rolling_fraction(flags, window)

    # A pre-flip level of ~0 makes the threshold trivially satisfiable, so
    # "recovery" would be an artifact. Such runs are marked invalid rather
    # than contributing a spurious recovery time of 1 window.
    recovery_valid = pre_level > 0.0

    recovery_episode = None
    if recovery_valid:
        for i, value in enumerate(series):
            if value is not None and value >= threshold:
                recovery_episode = i + 1  # episodes after the flip
                break

    # Perseveration: post-flip episodes still taking the old-optimal route
    # before the first episode that takes the new-optimal one.
    perseveration = 0
    for r in post:
        if r["path_type"] == post_target:
            break
        perseveration += 1 if r["path_type"] == pre_target else 0

    return {
        "pre_level": round(pre_level, 4),
        "threshold": round(threshold, 4),
        "post_contingency": post_contingency,
        "post_target_path": post_target,
        "recovery_episode": recovery_episode,
        "recovery_valid": recovery_valid,
        "censored": recovery_valid and recovery_episode is None,
        "post_flip_episodes": len(post),
        "final_adherence": round(series[-1], 4) if series and series[-1] is not None else None,
        "perseveration_episodes": perseveration,
        "mean_return_pre": round(
            float(np.mean([float(r["total_reward"]) for r in pre_window])), 3
        ) if pre_window else None,
        "mean_return_post_last_window": round(
            float(np.mean([float(r["total_reward"]) for r in post[-window:]])), 3
        ),
    }


def summarize(values: List[Optional[int]], cap: int) -> Dict[str, Any]:
    """Median/IQR over recovery times, counting censored runs explicitly.

    Censored runs enter the quantiles at the cap (a lower bound on their true
    value), so the reported median is conservative — it can only understate
    how long recovery takes. The censored count is reported alongside so this
    is never mistaken for a clean estimate.
    """
    observed = [v for v in values if v is not None]
    filled = [v if v is not None else cap for v in values]
    if not filled:
        return {"n": 0}
    return {
        "n": len(filled),
        "n_censored": len(filled) - len(observed),
        "median": float(np.median(filled)),
        "q25": float(np.percentile(filled, 25)),
        "q75": float(np.percentile(filled, 75)),
        "iqr": float(np.percentile(filled, 75) - np.percentile(filled, 25)),
        "min": int(np.min(filled)),
        "max": int(np.max(filled)),
        "observed": observed,
        "all_with_censored_at_cap": filled,
    }


def run_pilot(
    maze_name: str = "shield_trap_easy",
    agent_types: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    post_flip_episodes: int = 200,
    max_acquisition_episodes: int = 400,
    criterion_rate: float = 0.80,
    criterion_window: int = 50,
    recovery_window: int = 20,
    recovery_fraction: float = 0.80,
    epsilon_floor: float = 0.05,
    epsilon_decay_episodes: int = 200,
    buffer_size: int = 12000,
    non_protective_trap: Optional[float] = None,
    reward_overrides: Optional[Dict[str, float]] = None,
    image_size: int = 64,
    network_size: str = "standard",
    log_dir: str = "runs",
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    reversals_for_budget: int = 8,
    n_seeds_for_budget: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    agent_types = agent_types or ["baseline", "emotional"]
    seeds = seeds or [1, 2, 3, 4, 5]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pilot_dir = Path(log_dir) / f"pilot_{maze_name}_{timestamp}"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"PILOT: episodes-to-recovery on {maze_name}")
    print("=" * 72)
    print(f"  Agents: {', '.join(agent_types)}   Seeds: {seeds}")
    print(f"  Acquisition cap {max_acquisition_episodes}, post-flip cap {post_flip_episodes}")
    print(f"  Recovery: rolling {recovery_window}-episode optimal-route adherence "
          f">= {recovery_fraction:.0%} of pre-flip level")
    print(f"  Output: {pilot_dir}")
    print("=" * 72)

    results: List[Dict[str, Any]] = []
    t_start = time.time()
    total_episodes = 0

    for agent_type in agent_types:
        for seed in seeds:
            t0 = time.time()
            print(f"\n--- {agent_type} seed {seed} ---")
            manifest = run_reversal_training(
                maze_name=maze_name,
                agent_type=agent_type,
                reversals=1,
                reversal_period=post_flip_episodes,
                criterion_rate=criterion_rate,
                criterion_window=criterion_window,
                max_acquisition_episodes=max_acquisition_episodes,
                epsilon_floor=epsilon_floor,
                epsilon_decay_episodes=epsilon_decay_episodes,
                buffer_size=buffer_size,
                seed=seed,
                log_dir=str(pilot_dir / agent_type),
                device=device,
                config=config,
                image_size=image_size,
                network_class=resolve_network_class(network_size, image_size),
                non_protective_trap=non_protective_trap,
                reward_overrides=reward_overrides,
                verbose=False,
            )
            elapsed = time.time() - t0
            total_episodes += manifest["total_episodes"]

            recovery = measure_recovery(
                manifest["schedule_csv"],
                window=recovery_window,
                recovery_fraction=recovery_fraction,
            )
            row = {
                "agent_type": agent_type,
                "seed": seed,
                "reached_criterion": manifest["reached_criterion"],
                "acquisition_episodes": manifest["acquisition_episodes"],
                "total_episodes": manifest["total_episodes"],
                "seconds": round(elapsed, 1),
                "seconds_per_episode": round(elapsed / max(manifest["total_episodes"], 1), 4),
                "log_dir": manifest["log_dir"],
                **recovery,
            }
            results.append(row)

            print(f"    criterion {'reached' if row['reached_criterion'] else 'MISSED'} "
                  f"at {row['acquisition_episodes']} eps | pre-level "
                  f"{row['pre_level']:.0%} -> threshold {row['threshold']:.0%} | "
                  f"recovery {row['recovery_episode'] if not row['censored'] else 'CENSORED'} "
                  f"| perseveration {row['perseveration_episodes']} "
                  f"| {elapsed:.0f}s")

    # ---- summary ---------------------------------------------------------
    per_agent: Dict[str, Any] = {}
    for agent_type in agent_types:
        rows = [r for r in results if r["agent_type"] == agent_type]
        # Only runs that acquired the pre-flip policy *and* have a meaningful
        # pre-flip level can inform recovery time.
        acquired = [r for r in rows if r["reached_criterion"] and r["recovery_valid"]]
        per_agent[agent_type] = {
            "n_runs": len(rows),
            "n_reached_criterion": len(acquired),
            "acquisition_episodes_median": float(
                np.median([r["acquisition_episodes"] for r in rows])
            ),
            # Only runs that actually acquired can inform recovery time.
            "recovery": summarize(
                [r["recovery_episode"] for r in acquired], cap=post_flip_episodes
            ),
            "perseveration_median": float(
                np.median([r["perseveration_episodes"] for r in acquired])
            ) if acquired else None,
        }

    seconds_per_episode = float(np.mean([r["seconds_per_episode"] for r in results]))
    medians = [
        per_agent[a]["recovery"].get("median")
        for a in agent_types
        if per_agent[a]["recovery"].get("median") is not None
    ]
    pooled_median = float(np.median(medians)) if medians else None

    recommendation: Dict[str, Any] = {"pooled_median_recovery": pooled_median}
    if pooled_median:
        k_low, k_high = int(round(1.5 * pooled_median)), int(round(2.0 * pooled_median))
        median_acq = float(np.median([r["acquisition_episodes"] for r in results]))
        episodes_per_run = median_acq + reversals_for_budget * k_high
        total_study_episodes = 3 * n_seeds_for_budget * episodes_per_run
        recommendation.update({
            "K_range": [k_low, k_high],
            "K_recommended": k_high,
            "median_acquisition_episodes": median_acq,
            "episodes_per_run_at_K": episodes_per_run,
            "full_study_episodes": total_study_episodes,
            "full_study_hours_serial": round(
                total_study_episodes * seconds_per_episode / 3600, 2
            ),
            "seconds_per_episode": round(seconds_per_episode, 4),
            "assumes": (
                f"3 agent types x {n_seeds_for_budget} seeds x (acquisition + "
                f"{reversals_for_budget} x K) episodes, serial execution"
            ),
        })
        if pooled_median > 300:
            recommendation["maze_warning"] = (
                "Median recovery exceeds ~300 episodes. The reversal study tests "
                "adaptation, not maze difficulty, so switching to an easier maze "
                "is legitimate and should be stated as such."
            )

    summary = {
        "maze_name": maze_name,
        "agent_types": agent_types,
        "seeds": seeds,
        "non_protective_trap": non_protective_trap,
        "recovery_window": recovery_window,
        "recovery_fraction": recovery_fraction,
        "post_flip_cap": post_flip_episodes,
        "acquisition_cap": max_acquisition_episodes,
        "per_agent": per_agent,
        "recommendation": recommendation,
        "wall_clock_seconds": round(time.time() - t_start, 1),
        "total_episodes_run": total_episodes,
        "runs": results,
    }

    with open(pilot_dir / "pilot_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(pilot_dir / "pilot_runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    if verbose:
        print("\n" + "=" * 72)
        print("PILOT SUMMARY")
        print("=" * 72)
        for agent_type in agent_types:
            info = per_agent[agent_type]
            rec = info["recovery"]
            print(f"  {agent_type}:")
            print(f"    reached criterion: {info['n_reached_criterion']}/{info['n_runs']}"
                  f"  (median acquisition {info['acquisition_episodes_median']:.0f} eps)")
            if rec.get("n"):
                print(f"    recovery median {rec['median']:.0f}  "
                      f"IQR [{rec['q25']:.0f}, {rec['q75']:.0f}]  "
                      f"range [{rec['min']}, {rec['max']}]  "
                      f"censored {rec['n_censored']}/{rec['n']}")
                print(f"    perseveration median: {info['perseveration_median']}")
            else:
                print(f"    no runs reached criterion — recovery not estimable")
        print(f"\n  Pooled median recovery: {pooled_median}")
        if recommendation.get("K_range"):
            print(f"  Recommended K: {recommendation['K_range'][0]}-"
                  f"{recommendation['K_range'][1]} (use {recommendation['K_recommended']})")
            print(f"  Full study: {recommendation['full_study_episodes']:.0f} episodes, "
                  f"~{recommendation['full_study_hours_serial']} h serial "
                  f"({recommendation['seconds_per_episode']:.3f} s/episode)")
            if recommendation.get("maze_warning"):
                print(f"  ! {recommendation['maze_warning']}")
        print(f"\n  Wall clock: {summary['wall_clock_seconds'] / 60:.1f} min")
        print(f"  Summary: {pilot_dir / 'pilot_summary.json'}")
        print("=" * 72)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Pilot to calibrate reversal period K")
    parser.add_argument("--maze", type=str, default="shield_trap_easy")
    parser.add_argument("--agents", type=str, default="baseline,emotional")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--post_flip_episodes", type=int, default=200)
    parser.add_argument("--max_acquisition_episodes", type=int, default=400)
    parser.add_argument("--criterion_rate", type=float, default=0.80)
    parser.add_argument("--criterion_window", type=int, default=50)
    parser.add_argument("--recovery_window", type=int, default=20)
    parser.add_argument("--recovery_fraction", type=float, default=0.80)
    parser.add_argument("--epsilon_floor", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_episodes", type=int, default=200)
    parser.add_argument("--buffer_size", type=int, default=12000)
    parser.add_argument("--non_protective_trap", type=float, default=None)
    parser.add_argument("--reward", action="append", default=None,
                        help="Override a base reward key (repeatable), applied to "
                             "BOTH contingencies, e.g. --reward trap_no_shield=-20")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--network_size", type=str, default="standard",
                        choices=["standard", "small"])
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--lambda_mood", type=float, default=0.8)
    parser.add_argument("--mood_delta_source", type=str, default="batch_sequential",
                        choices=["online", "batch_mean", "batch_sequential"])
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--bootstrap_on_truncation", action="store_true",
                        help="Treat timeouts as non-terminal in the value target")
    parser.add_argument("--reversals_for_budget", type=int, default=8)
    parser.add_argument("--seeds_for_budget", type=int, default=20)
    args = parser.parse_args()

    run_pilot(
        maze_name=args.maze,
        agent_types=[a.strip() for a in args.agents.split(",") if a.strip()],
        seeds=args.seeds,
        post_flip_episodes=args.post_flip_episodes,
        max_acquisition_episodes=args.max_acquisition_episodes,
        criterion_rate=args.criterion_rate,
        criterion_window=args.criterion_window,
        recovery_window=args.recovery_window,
        recovery_fraction=args.recovery_fraction,
        epsilon_floor=args.epsilon_floor,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        buffer_size=args.buffer_size,
        non_protective_trap=args.non_protective_trap,
        reward_overrides=_parse_reward_overrides(args.reward),
        image_size=args.image_size,
        network_size=args.network_size,
        log_dir=args.log_dir,
        device=args.device,
        config={
            "eta": args.eta,
            "lambda_mood": args.lambda_mood,
            "mood_delta_source": args.mood_delta_source,
            "reward_scale": args.reward_scale,
            "bootstrap_on_truncation": args.bootstrap_on_truncation,
        },
        reversals_for_budget=args.reversals_for_budget,
        n_seeds_for_budget=args.seeds_for_budget,
    )


if __name__ == "__main__":
    main()
