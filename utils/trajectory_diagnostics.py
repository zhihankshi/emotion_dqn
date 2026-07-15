"""
Trajectory diagnostics for shield-trap mazes.

Roll out greedy episodes from saved checkpoints, log per-step trajectories,
classify path types (including failed trap rushes), and aggregate reward by
path type at early / mid / late training stages.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from agents.dqn import masked_action_selection

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
CHECKPOINT_EPISODE_RE = re.compile(r"agent_episode_(\d+)\.pt$")


def classify_diagnostic_path(info: Mapping[str, Any], success: bool) -> str:
    """
    Classify an episode for reward-by-path diagnostics.

    Unlike training's classify_episode_path, failed trap rushes are labeled
    trap_rush (not swallowed into timeout).

    Priority:
      1. shield_route — shield picked up before trap contact (or shield with no trap hit)
      2. trap_rush — trap hit without prior shield pickup
      3. timeout — no goal and neither of the above
      4. other — reached goal some other way / unmatched pattern
    """
    shield_step = int(info.get("shield_pickup_step", -1) or -1)
    trap_step = int(info.get("trap_hit_step", -1) or -1)

    # Classic shield path: collected shield before hitting the trap
    if shield_step > 0 and trap_step > 0 and shield_step < trap_step:
        return "shield_route"

    # Hit trap without ever picking up shield
    if trap_step > 0 and shield_step <= 0:
        return "trap_rush"

    if not success:
        return "timeout"

    # Goal without the trap sequences above (e.g. shield then goal)
    if shield_step > 0:
        return "shield_route"

    return "other"


def build_event_sequence(
    step_rows: Sequence[Mapping[str, Any]],
    success: bool,
) -> List[str]:
    """Build a high-level event timeline from per-step event tags."""
    events: List[str] = ["start"]
    seen = set()

    for row in step_rows:
        tag = row.get("event") or ""
        if not tag or tag in seen:
            continue
        if tag == "shield_pickup":
            events.append("shield_pickup")
            seen.add(tag)
        elif tag == "trap_hit":
            # Prefer "trap_crossed" once shield was already held this episode
            if "shield_pickup" in seen or row.get("has_shield"):
                events.append("trap_crossed")
            else:
                events.append("trap_hit")
            seen.add(tag)
        elif tag == "goal":
            events.append("goal")
            seen.add(tag)
        elif tag == "timeout":
            events.append("timeout")
            seen.add(tag)

    if success and "goal" not in events:
        events.append("goal")
    if not success and "timeout" not in events and "goal" not in events:
        # Truncation / failure without an explicit final timeout tag
        events.append("timeout")

    return events


def _get_q_values(agent, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        state_t = torch.from_numpy(obs).unsqueeze(0).float().to(agent.device)
        return agent.policy_net(state_t).cpu().numpy()[0]


def _step_event(
    prev_info: Mapping[str, Any],
    info: Mapping[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
) -> str:
    """Detect the primary special event that occurred on this transition."""
    if terminated:
        return "goal"
    if truncated:
        # Prefer more specific mid-step event tags when they also apply
        pass

    prev_shield_step = int(prev_info.get("shield_pickup_step", -1) or -1)
    shield_step = int(info.get("shield_pickup_step", -1) or -1)
    if prev_shield_step <= 0 and shield_step > 0:
        return "shield_pickup"

    prev_trap_step = int(prev_info.get("trap_hit_step", -1) or -1)
    trap_step = int(info.get("trap_hit_step", -1) or -1)
    if prev_trap_step <= 0 and trap_step > 0:
        return "trap_hit"

    if truncated:
        return "timeout"
    return ""


def rollout_episode(
    agent,
    env,
    rollout_id: int = 0,
    stage: str = "",
    checkpoint_episode: int = -1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Run one masked-greedy episode and return (step_rows, episode_summary).
    """
    obs, info = env.reset()
    agent.epsilon = 0.0

    step_rows: List[Dict[str, Any]] = []
    cumulative = 0.0
    success = False
    truncated = False

    max_steps = int(getattr(env, "max_steps", 200))

    for step_idx in range(max_steps):
        pos = info.get("agent_pos", (None, None))
        row, col = (pos[0], pos[1]) if pos is not None else (None, None)
        has_shield = bool(info.get("has_shield", False))

        q_values = _get_q_values(agent, obs)
        valid_actions = env.get_valid_actions()
        action = masked_action_selection(
            q_values, valid_actions, epsilon=0.0, training=False
        )

        prev_info = dict(info)
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative += float(reward)
        event = _step_event(prev_info, info, reward, terminated, truncated)

        step_rows.append({
            "stage": stage,
            "checkpoint_episode": checkpoint_episode,
            "rollout_id": rollout_id,
            "step": step_idx + 1,
            "row": row,
            "col": col,
            "action": ACTION_NAMES[action],
            "reward": float(reward),
            "cumulative_reward": float(cumulative),
            "has_shield": has_shield,
            "event": event,
        })

        if terminated or truncated:
            success = bool(terminated)
            break

    path_type = classify_diagnostic_path(info, success)
    events = build_event_sequence(step_rows, success)

    summary = {
        "stage": stage,
        "checkpoint_episode": checkpoint_episode,
        "rollout_id": rollout_id,
        "total_steps": len(step_rows),
        "total_reward": float(cumulative),
        "success": int(success),
        "path_type": path_type,
        "events": "|".join(events),
        "shield_pickup_step": int(info.get("shield_pickup_step", -1) or -1),
        "trap_hit_step": int(info.get("trap_hit_step", -1) or -1),
        "final_row": info.get("agent_pos", (None, None))[0],
        "final_col": info.get("agent_pos", (None, None))[1],
        "has_shield_final": int(bool(info.get("has_shield", False))),
        "shield_consumed": int(bool(info.get("shield_consumed", False))),
    }
    return step_rows, summary


def list_checkpoint_files(checkpoint_dir: Union[str, Path]) -> List[Tuple[int, Path]]:
    """Return sorted (episode, path) pairs for agent_episode_*.pt files."""
    checkpoint_dir = Path(checkpoint_dir)
    found: List[Tuple[int, Path]] = []
    if not checkpoint_dir.exists():
        return found
    for path in checkpoint_dir.glob("agent_episode_*.pt"):
        match = CHECKPOINT_EPISODE_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda x: x[0])
    return found


def select_stage_checkpoints(
    checkpoint_dir: Union[str, Path],
    stages: Optional[Sequence[str]] = None,
    episodes: Optional[Sequence[int]] = None,
) -> Dict[str, Tuple[int, Path]]:
    """
    Select checkpoints for analysis.

    If `episodes` is provided, map each episode to a stage label
    (early/mid/late by order, or episode_N if more than 3).
    Otherwise pick first / middle / last from available checkpoints for
    stages in `stages` (default: early, mid, late).
    """
    available = list_checkpoint_files(checkpoint_dir)
    if not available:
        raise FileNotFoundError(f"No agent_episode_*.pt files found in {checkpoint_dir}")

    by_episode = {ep: path for ep, path in available}

    if episodes:
        selected: Dict[str, Tuple[int, Path]] = {}
        default_labels = ["early", "mid", "late"]
        for i, ep in enumerate(episodes):
            if ep not in by_episode:
                raise FileNotFoundError(
                    f"Requested checkpoint episode {ep} not found in {checkpoint_dir}. "
                    f"Available: {sorted(by_episode)}"
                )
            if len(episodes) <= 3 and i < len(default_labels):
                label = default_labels[i]
            else:
                label = f"episode_{ep}"
            selected[label] = (ep, by_episode[ep])
        return selected

    stages = list(stages) if stages is not None else ["early", "mid", "late"]
    n = len(available)
    index_map = {
        "early": 0,
        "mid": n // 2,
        "late": n - 1,
    }

    selected = {}
    for stage in stages:
        stage = stage.strip()
        if stage not in index_map:
            raise ValueError(
                f"Unknown stage '{stage}'. Use early/mid/late or pass --episodes."
            )
        ep, path = available[index_map[stage]]
        selected[stage] = (ep, path)
    return selected


def aggregate_path_stats(
    episode_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate count / mean / std of reward and steps by stage × path_type."""
    groups: Dict[Tuple[str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        key = (
            str(row.get("stage", "")),
            int(row.get("checkpoint_episode", -1)),
            str(row.get("path_type", "other")),
        )
        groups[key].append(row)

    summaries: List[Dict[str, Any]] = []
    for (stage, ckpt_ep, path_type), rows in sorted(groups.items()):
        rewards = np.array([float(r["total_reward"]) for r in rows], dtype=np.float64)
        steps = np.array([float(r["total_steps"]) for r in rows], dtype=np.float64)
        successes = np.array([int(r.get("success", 0)) for r in rows], dtype=np.float64)
        summaries.append({
            "stage": stage,
            "checkpoint_episode": ckpt_ep,
            "path_type": path_type,
            "count": len(rows),
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std(ddof=0)) if len(rows) else 0.0,
            "mean_steps": float(steps.mean()),
            "std_steps": float(steps.std(ddof=0)) if len(rows) else 0.0,
            "success_rate": float(successes.mean()) if len(rows) else 0.0,
        })
    return summaries


def reward_gap(summary_rows: Sequence[Mapping[str, Any]], stage: str) -> Optional[float]:
    """mean_reward(shield_route) - mean_reward(trap_rush) for a stage, if both present."""
    means = {
        r["path_type"]: float(r["mean_reward"])
        for r in summary_rows
        if r.get("stage") == stage
    }
    if "shield_route" in means and "trap_rush" in means:
        return means["shield_route"] - means["trap_rush"]
    return None


def write_csv(path: Union[str, Path], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a list of dict rows to CSV (no-op if empty)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_stage_report(
    stage: str,
    checkpoint_episode: int,
    summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Print a path-type breakdown table for one stage."""
    rows = [r for r in summary_rows if r.get("stage") == stage]
    print(f"\n=== Stage {stage} (episode {checkpoint_episode}) ===")
    if not rows:
        print("  (no episodes)")
        return

    header = f"{'path_type':<14} {'count':>6} {'mean_reward':>12} {'mean_steps':>11} {'success%':>9}"
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda x: (-int(x["count"]), x["path_type"])):
        print(
            f"{r['path_type']:<14} {int(r['count']):>6} "
            f"{float(r['mean_reward']):>12.2f} {float(r['mean_steps']):>11.1f} "
            f"{100.0 * float(r['success_rate']):>8.1f}%"
        )

    gap = reward_gap(summary_rows, stage)
    if gap is not None:
        print(f"\nReward gap (shield_route - trap_rush): {gap:+.2f}")
        print(
            "  -> shield_route better than trap_rush"
            if gap > 0
            else "  -> trap_rush better than (or equal to) shield_route"
        )
    else:
        print("\nReward gap: n/a (need both shield_route and trap_rush counts > 0)")


def diagnose_checkpoints(
    checkpoint_dir: Union[str, Path],
    env,
    agent_type: str,
    network_size: str,
    image_size: int,
    n_episodes: int = 50,
    stages: Optional[Sequence[str]] = None,
    episodes: Optional[Sequence[int]] = None,
    seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Run diagnostic rollouts on selected checkpoints.

    Returns (all_step_rows, all_episode_rows, summary_rows).
    """
    from analyze_policy_evolution import load_agent_checkpoint

    selected = select_stage_checkpoints(
        checkpoint_dir, stages=stages, episodes=episodes
    )

    all_steps: List[Dict[str, Any]] = []
    all_episodes: List[Dict[str, Any]] = []

    for stage, (ckpt_ep, ckpt_path) in selected.items():
        print(f"\nLoading {stage}: {ckpt_path} (episode {ckpt_ep})")
        agent = load_agent_checkpoint(
            str(ckpt_path),
            env,
            agent_type=agent_type,
            network_size=network_size,
            image_size=image_size,
        )

        for rollout_id in range(n_episodes):
            # Seed each rollout for reproducibility while keeping diversity
            np.random.seed(seed + ckpt_ep * 1000 + rollout_id)
            torch.manual_seed(seed + ckpt_ep * 1000 + rollout_id)
            step_rows, summary = rollout_episode(
                agent,
                env,
                rollout_id=rollout_id,
                stage=stage,
                checkpoint_episode=ckpt_ep,
            )
            all_steps.extend(step_rows)
            all_episodes.append(summary)

    summary_rows = aggregate_path_stats(all_episodes)
    return all_steps, all_episodes, summary_rows
