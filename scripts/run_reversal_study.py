"""
Full three-way reversal study: baseline vs emotional vs yoked, N seeds each.

Runs ``train_reversal`` for every (agent_type, seed) cell and writes a study
manifest tying them together. Agent order matters: emotional runs come before
yoked ones because each yoked run consumes a **mood trace from an emotional
run with a different seed** (seed i is yoked to the emotional run at
(i + 1) % n_seeds), which is what makes it a control rather than a second
emotional agent.

Resumable: a cell whose ``reversal_manifest.json`` already exists is skipped,
so an interrupted overnight study can be restarted with the same command
without redoing finished work.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from train import resolve_network_class
from scripts.train_reversal import run_reversal_training, _parse_reward_overrides

AGENT_ORDER = ["baseline", "emotional", "yoked"]


def cell_dir(study_dir: Path, agent_type: str, seed: int) -> Path:
    return study_dir / agent_type / f"seed_{seed:03d}"


def find_existing_manifest(cell: Path) -> Optional[Path]:
    hits = sorted(cell.glob("*/reversal_manifest.json"))
    return hits[0] if hits else None


def find_mood_trace(cell: Path) -> Optional[Path]:
    hits = sorted(cell.glob("*/mood_trace.csv"))
    return hits[0] if hits else None


def run_study(
    maze_name: str = "shield_trap_easy",
    agent_types: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    reversals: int = 8,
    reversal_period: int = 400,
    criterion_rate: float = 0.80,
    criterion_window: int = 50,
    max_acquisition_episodes: int = 1000,
    epsilon_floor: float = 0.05,
    epsilon_decay_episodes: int = 120,
    buffer_size: int = 12000,
    non_protective_trap: Optional[float] = -60.0,
    reward_overrides: Optional[Dict[str, float]] = None,
    yoked_mode: str = "replay_trace",
    image_size: int = 64,
    network_size: str = "standard",
    log_dir: str = "experiments",
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    study_name: Optional[str] = None,
) -> Dict[str, Any]:
    agent_types = agent_types or list(AGENT_ORDER)
    seeds = seeds or list(range(1, 21))
    agent_types = [a for a in AGENT_ORDER if a in agent_types]  # enforce order

    if "yoked" in agent_types and "emotional" not in agent_types:
        raise ValueError(
            "A yoked arm needs emotional runs to donate mood traces; include "
            "'emotional' in --agents"
        )
    if "yoked" in agent_types and len(seeds) < 2:
        raise ValueError("Yoking needs >= 2 seeds so a donor from a different seed exists")

    study_dir = Path(log_dir) / (
        study_name or f"reversal_study_{maze_name}_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    study_dir.mkdir(parents=True, exist_ok=True)

    total_cells = len(agent_types) * len(seeds)
    print("=" * 72)
    print(f"REVERSAL STUDY: {maze_name}")
    print("=" * 72)
    print(f"  Agents: {', '.join(agent_types)}   Seeds: {len(seeds)} "
          f"({seeds[0]}..{seeds[-1]})")
    print(f"  R={reversals} reversals, K={reversal_period} episodes")
    print(f"  Acquisition cap {max_acquisition_episodes}, criterion "
          f"{criterion_rate:.0%}/{criterion_window}")
    print(f"  Buffer {buffer_size}, epsilon floor {epsilon_floor} from episode "
          f"{epsilon_decay_episodes}")
    print(f"  non_protective_trap={non_protective_trap}, "
          f"reward overrides={reward_overrides}")
    print(f"  bootstrap_on_truncation="
          f"{bool((config or {}).get('bootstrap_on_truncation', False))}")
    print(f"  Cells: {total_cells}   Output: {study_dir}")
    print("=" * 72, flush=True)

    rows: List[Dict[str, Any]] = []
    t_study = time.time()
    done_cells = 0

    for agent_type in agent_types:
        for seed in seeds:
            done_cells += 1
            cell = cell_dir(study_dir, agent_type, seed)
            cell.mkdir(parents=True, exist_ok=True)
            tag = f"[{done_cells}/{total_cells}] {agent_type} seed {seed}"

            existing = find_existing_manifest(cell)
            if existing:
                manifest = json.loads(existing.read_text())
                print(f"{tag}: already complete, skipping", flush=True)
                rows.append({"agent_type": agent_type, "seed": seed,
                             "skipped": True, **_row_from(manifest)})
                continue

            run_config = dict(config or {})
            donor_trace = None
            if agent_type == "yoked":
                donor_seed = seeds[(seeds.index(seed) + 1) % len(seeds)]
                donor_trace = find_mood_trace(cell_dir(study_dir, "emotional", donor_seed))
                if donor_trace is None:
                    print(f"{tag}: SKIPPED — no mood trace from emotional seed "
                          f"{donor_seed}", flush=True)
                    rows.append({"agent_type": agent_type, "seed": seed,
                                 "skipped": True, "error": "missing donor trace"})
                    continue
                run_config["yoked_mode"] = yoked_mode
                run_config["yoked_traces"] = [str(donor_trace)]

            t0 = time.time()
            print(f"{tag}: starting"
                  + (f" (yoked to {donor_trace.parent.name})" if donor_trace else ""),
                  flush=True)
            manifest = run_reversal_training(
                maze_name=maze_name,
                agent_type=agent_type,
                reversals=reversals,
                reversal_period=reversal_period,
                criterion_rate=criterion_rate,
                criterion_window=criterion_window,
                max_acquisition_episodes=max_acquisition_episodes,
                epsilon_floor=epsilon_floor,
                epsilon_decay_episodes=epsilon_decay_episodes,
                buffer_size=buffer_size,
                seed=seed,
                run_id=seed,
                log_dir=str(cell),
                device=device,
                config=run_config,
                image_size=image_size,
                network_class=resolve_network_class(network_size, image_size),
                reward_overrides=reward_overrides,
                non_protective_trap=non_protective_trap,
                verbose=False,
            )
            elapsed = time.time() - t0
            row = {"agent_type": agent_type, "seed": seed, "skipped": False,
                   "donor_trace": str(donor_trace) if donor_trace else None,
                   "seconds": round(elapsed, 1), **_row_from(manifest)}
            rows.append(row)

            eta = (time.time() - t_study) / done_cells * (total_cells - done_cells)
            print(f"{tag}: done in {elapsed / 60:.1f} min | criterion "
                  f"{'OK' if row['reached_criterion'] else 'MISSED'} | "
                  f"{row['total_episodes']} eps | ETA {eta / 3600:.1f} h", flush=True)

            _write_study_manifest(study_dir, locals())

    summary = _write_study_manifest(study_dir, locals())
    print("\n" + "=" * 72)
    print("STUDY COMPLETE")
    print("=" * 72)
    for agent_type in agent_types:
        cells = [r for r in rows if r["agent_type"] == agent_type]
        acquired = [r for r in cells if r.get("reached_criterion")]
        print(f"  {agent_type}: {len(cells)} runs, "
              f"{len(acquired)} reached criterion")
    print(f"  Wall clock: {(time.time() - t_study) / 3600:.2f} h")
    print(f"  Manifest: {study_dir / 'study_manifest.json'}")
    print("=" * 72)
    return summary


def _row_from(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "reached_criterion": manifest.get("reached_criterion"),
        "acquisition_episodes": manifest.get("acquisition_episodes"),
        "total_episodes": manifest.get("total_episodes"),
        "log_dir": manifest.get("log_dir"),
        "schedule_csv": manifest.get("schedule_csv"),
    }


def _write_study_manifest(study_dir: Path, scope: Dict[str, Any]) -> Dict[str, Any]:
    """Write the manifest after every cell so an interrupted study stays readable."""
    summary = {
        "maze_name": scope["maze_name"],
        "agent_types": scope["agent_types"],
        "seeds": scope["seeds"],
        "reversals": scope["reversals"],
        "reversal_period": scope["reversal_period"],
        "criterion_rate": scope["criterion_rate"],
        "criterion_window": scope["criterion_window"],
        "max_acquisition_episodes": scope["max_acquisition_episodes"],
        "epsilon_floor": scope["epsilon_floor"],
        "epsilon_decay_episodes": scope["epsilon_decay_episodes"],
        "buffer_size": scope["buffer_size"],
        "non_protective_trap": scope["non_protective_trap"],
        "reward_overrides": scope["reward_overrides"],
        "yoked_mode": scope["yoked_mode"],
        "config": scope["config"],
        "study_dir": str(scope["study_dir"]),
        "runs": scope["rows"],
    }
    with open(Path(scope["study_dir"]) / "study_manifest.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Three-way reversal study driver")
    parser.add_argument("--maze", type=str, default="shield_trap_easy")
    parser.add_argument("--agents", type=str, default="baseline,emotional,yoked")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--reversals", type=int, default=8)
    parser.add_argument("--reversal_period", type=int, default=400)
    parser.add_argument("--criterion_rate", type=float, default=0.80)
    parser.add_argument("--criterion_window", type=int, default=50)
    parser.add_argument("--max_acquisition_episodes", type=int, default=1000)
    parser.add_argument("--epsilon_floor", type=float, default=0.05)
    parser.add_argument("--epsilon_decay_episodes", type=int, default=120)
    parser.add_argument("--buffer_size", type=int, default=12000)
    parser.add_argument("--non_protective_trap", type=float, default=-60.0)
    parser.add_argument("--reward", action="append", default=None)
    parser.add_argument("--yoked_mode", type=str, default="replay_trace",
                        choices=["replay_trace", "ou_process"])
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--network_size", type=str, default="standard",
                        choices=["standard", "small"])
    parser.add_argument("--log_dir", type=str, default="experiments")
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--lambda_mood", type=float, default=0.8)
    parser.add_argument("--mood_delta_source", type=str, default="batch_sequential",
                        choices=["online", "batch_mean", "batch_sequential"])
    parser.add_argument("--mood_clip_range", type=float, default=1.0)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--bootstrap_on_truncation", action="store_true")
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--double_dqn", action="store_true")
    args = parser.parse_args()

    run_study(
        maze_name=args.maze,
        agent_types=[a.strip() for a in args.agents.split(",") if a.strip()],
        seeds=args.seeds,
        reversals=args.reversals,
        reversal_period=args.reversal_period,
        criterion_rate=args.criterion_rate,
        criterion_window=args.criterion_window,
        max_acquisition_episodes=args.max_acquisition_episodes,
        epsilon_floor=args.epsilon_floor,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        buffer_size=args.buffer_size,
        non_protective_trap=args.non_protective_trap,
        reward_overrides=_parse_reward_overrides(args.reward),
        yoked_mode=args.yoked_mode,
        image_size=args.image_size,
        network_size=args.network_size,
        log_dir=args.log_dir,
        study_name=args.study_name,
        device=args.device,
        config={
            "eta": args.eta,
            "lambda_mood": args.lambda_mood,
            "mood_delta_source": args.mood_delta_source,
            "mood_clip_range": args.mood_clip_range,
            "reward_scale": args.reward_scale,
            "bootstrap_on_truncation": args.bootstrap_on_truncation,
            "target_update_freq": args.target_update_freq,
            "double_dqn": args.double_dqn,
        },
    )


if __name__ == "__main__":
    main()
