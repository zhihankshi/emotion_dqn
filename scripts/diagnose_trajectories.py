"""
Diagnose greedy trajectories from saved training checkpoints.

Rolls out N episodes at early/mid/late checkpoints and reports reward by
path type (shield_route vs trap_rush vs timeout).

Example:
  python scripts/diagnose_trajectories.py \\
    --checkpoint_dir test_runs/.../checkpoints \\
    --maze shield_trap_easy \\
    --agent emotional \\
    --network_size small \\
    --image_size 28 \\
    --n_episodes 50 \\
    --output_dir diagnostics/shield_trap_easy_emotional
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Allow running as `python scripts/diagnose_trajectories.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from environments import VisualMazeEnv
from utils.trajectory_diagnostics import (
    diagnose_checkpoints,
    print_stage_report,
    write_csv,
)


def _parse_reward_overrides(pairs: Optional[List[str]]) -> Dict[str, float]:
    if not pairs:
        return {}
    out: Dict[str, float] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Invalid --reward '{raw}'. Use key=value.")
        k, v = raw.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid --reward '{raw}'. Empty key.")
        out[k] = float(v.strip())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log greedy trajectories from checkpoints and compare path-type rewards"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Directory containing agent_episode_*.pt checkpoints",
    )
    parser.add_argument("--maze", type=str, default="shield_trap_easy")
    parser.add_argument(
        "--agent", type=str, default="baseline",
        choices=["baseline", "emotional"],
    )
    parser.add_argument(
        "--network_size", type=str, default="standard",
        choices=["standard", "small"],
    )
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument(
        "--n_episodes", type=int, default=50,
        help="Greedy rollouts per checkpoint/stage",
    )
    parser.add_argument(
        "--stages", type=str, default="early,mid,late",
        help="Comma-separated stages from available checkpoints (ignored if --episodes set)",
    )
    parser.add_argument(
        "--episodes", type=str, default=None,
        help="Comma-separated checkpoint episode numbers, e.g. 50,100,200",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=str, default="diagnostics/trajectory_diag",
        help="Directory for steps.csv / episodes.csv / summary.csv",
    )
    parser.add_argument(
        "--reward", action="append", default=None,
        help="Runtime reward override (repeatable), e.g. --reward step=-0.5",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Override env max_steps for rollouts",
    )
    parser.add_argument(
        "--shield_lights_up", action="store_true",
        help="Brighten observation when agent holds the shield",
    )
    parser.add_argument(
        "--frame_stack", type=int, default=1,
        help="Frame stack size used when the checkpoints were trained",
    )

    args = parser.parse_args()

    episode_list = None
    if args.episodes:
        episode_list = [int(x.strip()) for x in args.episodes.split(",") if x.strip()]

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    reward_overrides = _parse_reward_overrides(args.reward)

    env = VisualMazeEnv(
        maze_name=args.maze,
        image_size=args.image_size,
        reward_overrides=reward_overrides or None,
        max_steps=args.max_steps,
        shield_lights_up=args.shield_lights_up if args.shield_lights_up else None,
        frame_stack=args.frame_stack,
    )

    print("=" * 70)
    print("TRAJECTORY DIAGNOSTIC")
    print("=" * 70)
    print(f"  Maze:            {args.maze}")
    print(f"  Agent:           {args.agent}")
    print(f"  Checkpoint dir:  {args.checkpoint_dir}")
    print(f"  Rollouts/stage:  {args.n_episodes}")
    print(f"  Image size:      {args.image_size}")
    print(f"  Network:         {args.network_size}")
    print(f"  Env max_steps:   {env.max_steps}")
    if reward_overrides:
        print(f"  Reward overrides:{reward_overrides}")
    print(f"  Output:          {args.output_dir}")
    print("=" * 70)

    steps, episodes, summary = diagnose_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        env=env,
        agent_type=args.agent,
        network_size=args.network_size,
        image_size=args.image_size,
        n_episodes=args.n_episodes,
        stages=None if episode_list else stages,
        episodes=episode_list,
        seed=args.seed,
    )

    # Console report
    stages_seen = []
    for row in episodes:
        key = (row["stage"], row["checkpoint_episode"])
        if key not in stages_seen:
            stages_seen.append(key)
    for stage, ckpt_ep in stages_seen:
        print_stage_report(stage, ckpt_ep, summary)

    # Frequency trend across stages (helps spot unlearning)
    print("\n=== Path-type frequency across stages ===")
    path_types = sorted({r["path_type"] for r in summary})
    for stage, ckpt_ep in stages_seen:
        stage_rows = [r for r in summary if r["stage"] == stage]
        total = sum(int(r["count"]) for r in stage_rows) or 1
        parts = []
        for pt in path_types:
            count = next((int(r["count"]) for r in stage_rows if r["path_type"] == pt), 0)
            parts.append(f"{pt}={count}/{total} ({100.0 * count / total:.0f}%)")
        print(f"  {stage} (ep {ckpt_ep}): " + ", ".join(parts))

    out = Path(args.output_dir)
    write_csv(out / "steps.csv", steps)
    write_csv(out / "episodes.csv", episodes)
    write_csv(out / "summary.csv", summary)

    print("\n" + "=" * 70)
    print(f"Saved: {out / 'steps.csv'}")
    print(f"Saved: {out / 'episodes.csv'}")
    print(f"Saved: {out / 'summary.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
