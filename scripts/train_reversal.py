"""
Repeated contingency reversal training.

Replaces the single source->target flip in ``train_transfer.py``. One flip gives
one adaptation event per run; this gives R of them, so each run contributes
repeated measures and the primary metric is a within-run average.

Protocol
--------
**Phase A - acquisition.** Train on the base maze under the *protective*
contingency until a competence criterion is met: at least ``criterion_rate``
(default 0.80) ``shield_route`` episodes over a rolling ``criterion_window``
(default 50) window, capped at ``max_acquisition_episodes`` (default 1000).
Runs that never reach criterion are flagged in the manifest and printed as a
warning; they are not silently pooled with the rest.

**Phase B - reversal block.** Flip the contingency every ``K`` episodes, ``R``
times (default 8). The two contingencies are built from the *same maze file*
with a single reward key changed:

    protective      trap_with_shield = <as authored>   (shield reduces trap cost)
    non_protective  trap_with_shield = trap_no_shield  (shield buys nothing)

Nothing else differs - not geometry, sprites, start/goal, step cost, pickup
value, timeout, or max_steps - so the CNN cannot see the switch. This is
checked at startup by rendering every reachable state under both contingencies
and asserting pixel equality (``verify_visual_identity``); the run aborts if
they differ. Note this supersedes the ``shield_trap`` / ``shield_avoidance``
pair, which also differ in ``trap_no_shield`` (-50 vs -5) and
``shield_pickup`` (+5 vs 0) and so confound the contingency flip with a change
in trap severity and pickup value.

Constraints - these are the experiment, not implementation details
------------------------------------------------------------------
* **Epsilon is never reset or bumped at a reversal.** It decays linearly to
  ``epsilon_floor`` over ``epsilon_decay_episodes`` and is then held constant
  for the rest of the run, across every reversal. Bumping it would signal the
  change externally and destroy the hypothesis: the claim under test is that
  mood is the *internal* signal that detects the change.
* **The replay buffer is never flushed.** Same information leak.
* **The buffer is small** (default 12,000) so it turns over within roughly one
  reversal period. At 50k it spans the whole run and every agent is dominated
  by stale, wrong-contingency transitions. Applied identically to all three
  agent types.
* **The target network is never reset**, and the agent object is never
  rebuilt - so optimizer state, buffer, target net and mood all persist
  continuously across every phase and reversal boundary.
* **Mood persists.** M is never reset.

Outputs (per run directory)
---------------------------
``<agent>_run<N>_episodes.csv``   standard per-episode metrics
``reversal_schedule.csv``         episode -> phase / block / contingency / reversal index
``reversal_manifest.json``        full config, criterion outcome, block boundaries
``mood_trace.csv``                per-step M (mood-carrying agents)
``mood_source.json``              yoked donor description
"""
import argparse
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments import VisualMazeEnv
from environments.maze_loader import load_maze
from train import create_agent, resolve_network_class, train_episode
from utils import MetricsLogger

PROTECTIVE = "protective"
NON_PROTECTIVE = "non_protective"
CONTINGENCIES = (PROTECTIVE, NON_PROTECTIVE)


# Canonical routes per maze, used only to validate that a reversal actually
# reverses which route is optimal. Actions: 0=up 1=down 2=left 3=right.
CANONICAL_ROUTES: Dict[str, Dict[str, List[int]]] = {
    "shield_trap_easy": {"shield_route": [1, 0] + [3] * 4, "direct": [3] * 4},
    "shield_trap": {"shield_route": [1] * 4 + [0] * 4 + [3] * 6, "direct": [3] * 6},
    "shield_avoidance": {"shield_route": [1] * 4 + [0] * 4 + [3] * 6, "direct": [3] * 6},
}


def contingency_overrides(
    maze_name: str,
    extra_overrides: Optional[Dict[str, float]] = None,
    non_protective_trap: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """Reward overrides for each contingency, from one maze file.

    Only ``trap_with_shield`` differs between the two.

    Under ``non_protective`` it defaults to ``trap_no_shield`` — the shield
    stops protecting. Beware: that alone does **not** guarantee the optimal
    *route* reverses. If the shield pickup bonus exceeds the detour's step
    cost, the detour stays weakly optimal even when the shield is useless, and
    there is no behavioural change to measure. Pass ``non_protective_trap``
    (a worse-than-unshielded trap cost, i.e. carrying the shield now hurts) to
    get a genuine reversal. ``validate_contingency_reversal`` checks this.
    """
    rewards = dict(load_maze(maze_name)["rewards"])
    base = dict(extra_overrides or {})

    protective = {**base, "trap_with_shield": float(rewards["trap_with_shield"])}
    if non_protective_trap is None:
        non_protective_trap = base.get("trap_no_shield", rewards["trap_no_shield"])
    non_protective = {**base, "trap_with_shield": float(non_protective_trap)}

    if protective["trap_with_shield"] == non_protective["trap_with_shield"]:
        raise ValueError(
            f"Maze '{maze_name}' has trap_with_shield == trap_no_shield "
            f"({protective['trap_with_shield']}), so the two contingencies are "
            f"identical and there is nothing to reverse."
        )
    return {PROTECTIVE: protective, NON_PROTECTIVE: non_protective}


def route_returns(env, routes: Dict[str, List[int]]) -> Dict[str, float]:
    """Undiscounted return of each scripted route in this env."""
    out: Dict[str, float] = {}
    for name, actions in routes.items():
        env.reset()
        total = 0.0
        for action in actions:
            _, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        out[name] = round(total, 4)
    return out


def validate_contingency_reversal(
    maze_name: str,
    envs: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Check the flip actually reverses which route is optimal.

    A contingency change that leaves the same route optimal produces no
    behavioural adaptation, so episodes-to-recovery is undefined and the whole
    reversal readout is vacuous. This is easy to get wrong: making the shield
    merely useless is not enough when the pickup bonus already pays for the
    detour.
    """
    routes = CANONICAL_ROUTES.get(maze_name)
    if not routes:
        if verbose:
            print(f"  (no canonical routes registered for '{maze_name}' — "
                  f"reversal strength not validated)")
        return {"validated": False, "reversed": None}

    returns = {c: route_returns(envs[c], routes) for c in CONTINGENCIES}
    best = {c: max(returns[c], key=returns[c].get) for c in CONTINGENCIES}
    margins = {
        c: abs(returns[c]["shield_route"] - returns[c]["direct"]) for c in CONTINGENCIES
    }
    flipped = best[PROTECTIVE] != best[NON_PROTECTIVE]

    if verbose:
        for c in CONTINGENCIES:
            print(f"  {c:15s} shield_route {returns[c]['shield_route']:+8.1f} | "
                  f"direct {returns[c]['direct']:+8.1f} | best: {best[c]} "
                  f"(margin {margins[c]:.1f})")
        if flipped:
            print(f"  Reversal is genuine: optimal route flips "
                  f"{best[PROTECTIVE]} -> {best[NON_PROTECTIVE]}")
        else:
            print(f"  !! NO REVERSAL: '{best[PROTECTIVE]}' stays optimal under both "
                  f"contingencies. There is no behavioural change to adapt to.")

    return {
        "validated": True,
        "reversed": flipped,
        "returns": returns,
        "best_route": best,
        "margins": margins,
    }


def verify_visual_identity(env_a, env_b, verbose: bool = True) -> Tuple[bool, List[str]]:
    """Assert the two contingencies are pixel-identical in every reachable state.

    A visible cue would let the CNN detect the switch directly and confound the
    whole experiment, so this runs before training and aborts on mismatch.
    """
    differences: List[str] = []
    cells = env_a.iter_walkable_cells()
    n_checked = 0

    # Only enumerate state dimensions the maze actually has, so mazes without
    # a key or door don't pad the check with unreachable states.
    key_states = (False, True) if getattr(env_a, "key_pos", None) is not None else (False,)
    door_states = (False, True) if getattr(env_a, "door_pos", None) is not None else (False,)

    for pos in cells:
        for has_shield in (False, True):
            for shield_consumed in (False, True):
                for has_key in key_states:
                    for door_open in door_states:
                        state = dict(
                            has_key=has_key,
                            door_open=door_open,
                            has_shield=has_shield,
                            shield_consumed=shield_consumed,
                        )
                        obs_a = env_a.set_state_for_observation(pos, **state)
                        obs_b = env_b.set_state_for_observation(pos, **state)
                        n_checked += 1
                        if not np.array_equal(obs_a, obs_b):
                            n_diff = int((obs_a != obs_b).sum())
                            differences.append(
                                f"pos={pos} {state}: {n_diff} pixels differ"
                            )

    if verbose:
        if differences:
            print(f"  !! VISUAL CUE DETECTED in {len(differences)}/{n_checked} states:")
            for d in differences[:10]:
                print(f"     {d}")
        else:
            print(f"  Visual identity verified: {n_checked} states, 0 pixel differences")
    return (not differences), differences


def epsilon_for_episode(
    episode: int,
    epsilon_start: float,
    epsilon_floor: float,
    decay_episodes: int,
) -> float:
    """Linear decay to the floor, then constant forever.

    Deliberately a function of the *global* episode index, so a reversal cannot
    change it. There is no per-phase schedule to reset.
    """
    if decay_episodes <= 0:
        return epsilon_floor
    progress = min(1.0, episode / decay_episodes)
    return max(epsilon_floor, epsilon_start - (epsilon_start - epsilon_floor) * progress)


def run_reversal_training(
    maze_name: str = "shield_trap",
    agent_type: str = "emotional",
    reversals: int = 8,
    reversal_period: int = 150,
    criterion_rate: float = 0.80,
    criterion_window: int = 50,
    criterion_path_type: str = "shield_route",
    max_acquisition_episodes: int = 1000,
    min_acquisition_episodes: Optional[int] = None,
    epsilon_start: float = 1.0,
    epsilon_floor: float = 0.05,
    epsilon_decay_episodes: int = 300,
    buffer_size: int = 12000,
    seed: int = 42,
    run_id: int = 0,
    log_dir: str = "runs",
    device: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    image_size: int = 64,
    network_class=None,
    frame_stack: int = 1,
    max_steps: Optional[int] = None,
    reward_overrides: Optional[Dict[str, float]] = None,
    non_protective_trap: Optional[float] = None,
    allow_weak_reversal: bool = False,
    shield_lights_up: Optional[bool] = None,
    verbose: bool = True,
    progress_every: int = 50,
) -> Dict[str, Any]:
    """Acquisition to criterion, then R reversals every K episodes. One agent throughout."""
    config = dict(config or {})
    # The buffer must turn over within about one reversal period; this is a
    # constraint of the design, applied identically to every agent type.
    config["buffer_size"] = buffer_size

    # Competence measured under a still-annealing epsilon is not competence:
    # by default the criterion cannot be met until exploration has reached its
    # floor, so phase B always starts from a genuinely acquired policy.
    if min_acquisition_episodes is None:
        min_acquisition_episodes = epsilon_decay_episodes
    config["epsilon_start"] = epsilon_start
    config["epsilon_end"] = epsilon_floor

    np.random.seed(seed)
    torch.manual_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    overrides = contingency_overrides(maze_name, reward_overrides, non_protective_trap)

    envs = {
        name: VisualMazeEnv(
            maze_name=maze_name,
            image_size=image_size,
            reward_overrides=overrides[name],
            max_steps=max_steps,
            shield_lights_up=shield_lights_up,
            frame_stack=frame_stack,
        )
        for name in CONTINGENCIES
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = Path(log_dir) / f"reversal_{maze_name}_{agent_type}_{timestamp}"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("REPEATED CONTINGENCY REVERSAL")
        print("=" * 70)
        print(f"  Maze:              {maze_name}")
        print(f"  Agent:             {agent_type}")
        print(f"  Contingencies:     "
              f"protective trap_with_shield={overrides[PROTECTIVE]['trap_with_shield']}, "
              f"non_protective trap_with_shield={overrides[NON_PROTECTIVE]['trap_with_shield']}")
        print(f"  Criterion:         >={criterion_rate:.0%} {criterion_path_type} "
              f"over {criterion_window} episodes (cap {max_acquisition_episodes})")
        print(f"  Reversals:         R={reversals} every K={reversal_period} episodes")
        print(f"  Epsilon:           {epsilon_start} -> {epsilon_floor} over "
              f"{epsilon_decay_episodes} episodes, then CONSTANT (never bumped)")
        print(f"  Buffer:            {buffer_size} (never flushed)")
        print(f"  Seed:              {seed}")
        print(f"  Output:            {run_log_dir}")

    reversal_check = validate_contingency_reversal(maze_name, envs, verbose=verbose)
    if reversal_check.get("validated") and not reversal_check.get("reversed") \
            and not allow_weak_reversal:
        raise RuntimeError(
            f"The contingency flip does not change which route is optimal on "
            f"'{maze_name}' ({reversal_check['best_route'][PROTECTIVE]} wins under "
            f"both), so there is no adaptation to measure. Pass "
            f"--non_protective_trap <worse-than-unshielded cost> to make the "
            f"shield actively harmful, or --allow_weak_reversal to override."
        )

    identical, differences = verify_visual_identity(
        envs[PROTECTIVE], envs[NON_PROTECTIVE], verbose=verbose
    )
    if not identical:
        raise RuntimeError(
            "The two contingencies are visually distinguishable, so the CNN "
            "could detect the reversal directly. Refusing to run. "
            f"First difference: {differences[0]}"
        )

    if network_class is None and image_size < 36:
        from agents import SmallDQNNetwork
        network_class = SmallDQNNetwork

    # One agent for the entire run: buffer, optimizer, target net and mood all
    # persist across every phase and reversal boundary.
    agent = create_agent(
        agent_type=agent_type,
        observation_shape=envs[PROTECTIVE].observation_space.shape,
        n_actions=envs[PROTECTIVE].action_space.n,
        config=config,
        device=device,
        seed=seed,
        network_class=network_class,
    )

    logger = MetricsLogger(
        log_dir=run_log_dir,
        agent_type=agent_type,
        maze_name=maze_name,
        run_id=run_id,
    )

    reward_scale = float(config.get("reward_scale", 1.0))
    bootstrap_on_truncation = bool(config.get("bootstrap_on_truncation", False))
    schedule_rows: List[Dict[str, Any]] = []
    episode_end_steps: List[int] = []
    recent_paths: deque = deque(maxlen=criterion_window)

    episode = 0
    reached_criterion = False
    acquisition_episodes = 0

    def run_one(contingency: str, phase: str, block: int, reversal_index: int) -> str:
        nonlocal episode
        agent.epsilon = epsilon_for_episode(
            episode, epsilon_start, epsilon_floor, epsilon_decay_episodes
        )
        metrics = train_episode(
            envs[contingency], agent, episode, training=True,
            reward_scale=reward_scale,
            bootstrap_on_truncation=bootstrap_on_truncation,
        )
        logger.log_episode(metrics)
        episode_end_steps.append(getattr(agent, "steps", 0))
        schedule_rows.append({
            "episode": episode,
            "phase": phase,
            "block": block,
            "contingency": contingency,
            "reversal_index": reversal_index,
            "epsilon": round(agent.epsilon, 6),
            "path_type": metrics.path_type,
            "success": int(bool(metrics.success)),
            "total_reward": round(metrics.total_reward, 4),
            "steps": metrics.steps,
            "mean_mood": round(metrics.mean_overall_mood, 6),
        })
        episode += 1
        return metrics.path_type

    # ---- Phase A: acquisition to criterion -------------------------------
    if verbose:
        print(f"\n>>> PHASE A: acquisition under '{PROTECTIVE}'\n")

    while episode < max_acquisition_episodes:
        path_type = run_one(PROTECTIVE, "acquisition", 0, -1)
        recent_paths.append(path_type)

        if (
            len(recent_paths) == criterion_window
            and episode >= min_acquisition_episodes
            and sum(p == criterion_path_type for p in recent_paths) / criterion_window
            >= criterion_rate
        ):
            reached_criterion = True
            break

        if verbose and episode % progress_every == 0 and recent_paths:
            rate = sum(p == criterion_path_type for p in recent_paths) / len(recent_paths)
            print(f"  ep {episode:4d}  {criterion_path_type} rate {rate:.0%}  "
                  f"eps {agent.epsilon:.3f}")

    acquisition_episodes = episode
    if verbose:
        rate = (sum(p == criterion_path_type for p in recent_paths) / len(recent_paths)
                if recent_paths else 0.0)
        if reached_criterion:
            print(f"  Criterion reached at episode {acquisition_episodes} "
                  f"({criterion_path_type} rate {rate:.0%})")
        else:
            print(f"  !! CRITERION NOT REACHED in {acquisition_episodes} episodes "
                  f"(final {criterion_path_type} rate {rate:.0%}). This run is "
                  f"flagged; do not pool it with runs that acquired.")

    # ---- Phase B: reversal block -----------------------------------------
    # Check the buffer actually turns over within a reversal period, using the
    # observed step rate rather than an assumed one. A buffer that spans
    # several periods keeps every agent training on wrong-contingency data and
    # would mask exactly the adaptation this experiment measures.
    steps_per_episode = (
        episode_end_steps[-1] / len(episode_end_steps) if episode_end_steps else 0.0
    )
    steps_per_period = steps_per_episode * reversal_period
    buffer_periods = buffer_size / steps_per_period if steps_per_period else float("inf")
    if verbose:
        print(f"\n  Buffer turnover: ~{steps_per_episode:.0f} steps/episode x K={reversal_period}"
              f" = {steps_per_period:.0f} steps/period; buffer {buffer_size} spans "
              f"{buffer_periods:.2f} periods")
        if buffer_periods > 1.5:
            print(f"  WARNING: buffer spans {buffer_periods:.1f} reversal periods. "
                  f"Consider --buffer_size {int(steps_per_period)} so it turns over "
                  f"within roughly one period.")

    if verbose:
        print(f"\n>>> PHASE B: {reversals} reversals every {reversal_period} episodes\n")

    block_boundaries: List[Dict[str, Any]] = []
    contingency = PROTECTIVE
    for r in range(reversals):
        # Flip. Nothing else changes: no epsilon bump, no buffer flush, no
        # target-net reset, no mood reset.
        contingency = NON_PROTECTIVE if contingency == PROTECTIVE else PROTECTIVE
        block_start = episode
        block_boundaries.append({
            "reversal_index": r,
            "contingency": contingency,
            "start_episode": block_start,
            "end_episode": block_start + reversal_period - 1,
        })
        if verbose:
            print(f"  --- reversal {r + 1}/{reversals} -> '{contingency}' "
                  f"at episode {block_start} (eps {agent.epsilon:.3f})")

        for _ in range(reversal_period):
            run_one(contingency, "reversal", r + 1, r)

        if verbose:
            block_rows = schedule_rows[-reversal_period:]
            hits = sum(row["path_type"] == criterion_path_type for row in block_rows)
            print(f"      {criterion_path_type} {hits}/{reversal_period} "
                  f"({hits / reversal_period:.0%})  "
                  f"mean reward {np.mean([r_['total_reward'] for r_ in block_rows]):+.2f}")

    # ---- Outputs ----------------------------------------------------------
    logger.save_summary()
    agent.save(str(run_log_dir / f"{agent_type}_agent.pt"))

    if hasattr(agent, "save_mood_trace"):
        agent.save_mood_trace(str(run_log_dir / "mood_trace.csv"), episode_end_steps)
    if hasattr(agent, "describe_mood_source"):
        with open(run_log_dir / "mood_source.json", "w") as f:
            json.dump(agent.describe_mood_source(), f, indent=2)

    schedule_path = run_log_dir / "reversal_schedule.csv"
    with open(schedule_path, "w", newline="") as f:
        cols = list(schedule_rows[0].keys())
        f.write(",".join(cols) + "\n")
        for row in schedule_rows:
            f.write(",".join(str(row[c]) for c in cols) + "\n")

    manifest = {
        "maze_name": maze_name,
        "agent_type": agent_type,
        "seed": seed,
        "run_id": run_id,
        "reached_criterion": reached_criterion,
        "acquisition_episodes": acquisition_episodes,
        "criterion": {
            "rate": criterion_rate,
            "window": criterion_window,
            "path_type": criterion_path_type,
            "max_episodes": max_acquisition_episodes,
        },
        "reversals": reversals,
        "reversal_period": reversal_period,
        "total_episodes": episode,
        "epsilon": {
            "start": epsilon_start,
            "floor": epsilon_floor,
            "decay_episodes": epsilon_decay_episodes,
            "reset_at_reversals": False,
            "final": agent.epsilon,
        },
        "buffer_size": buffer_size,
        "buffer_periods_spanned": round(buffer_periods, 3),
        "acquisition_steps_per_episode": round(steps_per_episode, 2),
        "buffer_flushed_at_reversals": False,
        "target_net_reset_at_reversals": False,
        "mood_reset_at_reversals": False,
        "contingencies": overrides,
        "reversal_check": reversal_check,
        "visual_identity_verified": identical,
        "block_boundaries": block_boundaries,
        "config": {k: v for k, v in config.items() if k != "yoked_traces"},
        "yoked_traces": config.get("yoked_traces"),
        "log_dir": str(run_log_dir),
        "schedule_csv": str(schedule_path),
    }
    with open(run_log_dir / "reversal_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        print("\n" + "=" * 70)
        print("REVERSAL TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Reached criterion:   {reached_criterion} "
              f"(after {acquisition_episodes} episodes)")
        print(f"  Total episodes:      {episode}")
        print(f"  Final epsilon:       {agent.epsilon:.4f} (floor {epsilon_floor})")
        print(f"  Manifest:            {run_log_dir / 'reversal_manifest.json'}")
        print("=" * 70)

    return manifest


def _parse_reward_overrides(items: Optional[List[str]]) -> Optional[Dict[str, float]]:
    if not items:
        return None
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--reward expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Repeated contingency reversal training"
    )
    parser.add_argument("--maze", type=str, default="shield_trap",
                        help="Base maze; both contingencies are built from this one file")
    parser.add_argument("--agent", type=str, default="emotional",
                        choices=["baseline", "emotional", "yoked"])
    parser.add_argument("--reversals", type=int, default=8, help="R")
    parser.add_argument("--reversal_period", type=int, default=150,
                        help="K, episodes between reversals (calibrate with pilot_recovery)")
    parser.add_argument("--criterion_rate", type=float, default=0.80)
    parser.add_argument("--criterion_window", type=int, default=50)
    parser.add_argument("--criterion_path_type", type=str, default="shield_route")
    parser.add_argument("--max_acquisition_episodes", type=int, default=1000)
    parser.add_argument("--min_acquisition_episodes", type=int, default=None,
                        help="Earliest episode the criterion may be met "
                             "(default: --epsilon_decay_episodes, so competence "
                             "is only judged at the epsilon floor)")

    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_floor", type=float, default=0.05,
                        help="Constant epsilon through all of phase B; never bumped")
    parser.add_argument("--epsilon_decay_episodes", type=int, default=300)
    parser.add_argument("--buffer_size", type=int, default=12000,
                        help="Small on purpose: must turn over within ~one reversal period")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--network_size", type=str, default="standard",
                        choices=["standard", "small"])
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--reward", action="append", default=None,
                        help="Override a reward key (repeatable), e.g. --reward step=-0.25")
    parser.add_argument("--non_protective_trap", type=float, default=None,
                        help="trap_with_shield under the non_protective contingency. "
                             "Default = trap_no_shield (shield merely useless), which "
                             "often does NOT reverse the optimal route; set a worse "
                             "value so carrying the shield actively hurts")
    parser.add_argument("--allow_weak_reversal", action="store_true",
                        help="Run even if the flip leaves the same route optimal")
    parser.add_argument("--shield_lights_up", action="store_true", default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--double_dqn", action="store_true")
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--lambda_mood", type=float, default=0.8)
    parser.add_argument("--mood_clip_range", type=float, default=1.0)
    parser.add_argument("--mood_delta_source", type=str, default="batch_sequential",
                        choices=["online", "batch_mean", "batch_sequential"])
    parser.add_argument("--mood_delta_unsigned", action="store_true")
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--bootstrap_on_truncation", action="store_true",
                        help="Treat timeouts as non-terminal in the value target "
                             "(all agent types); default off preserves old behavior")
    parser.add_argument("--yoked_exhaustion", type=str, default="reflect",
                        choices=["reflect", "hold"],
                        help="Padding once a yoked run outlives its donor trace.")
    parser.add_argument("--yoked_mode", type=str, default="replay_trace",
                        choices=["replay_trace", "ou_process"])
    parser.add_argument("--yoked_trace", type=str, nargs="+", default=None)

    args = parser.parse_args()

    config = {
        "learning_rate": args.lr,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "target_update_freq": args.target_update_freq,
        "double_dqn": args.double_dqn,
        "eta": args.eta,
        "lambda_mood": args.lambda_mood,
        "mood_clip_range": args.mood_clip_range,
        "mood_delta_source": args.mood_delta_source,
        "mood_delta_signed": not args.mood_delta_unsigned,
        "reward_scale": args.reward_scale,
        "bootstrap_on_truncation": args.bootstrap_on_truncation,
        "yoked_mode": args.yoked_mode,
        "yoked_exhaustion": args.yoked_exhaustion,
        "yoked_traces": args.yoked_trace,
    }

    run_reversal_training(
        maze_name=args.maze,
        agent_type=args.agent,
        reversals=args.reversals,
        reversal_period=args.reversal_period,
        criterion_rate=args.criterion_rate,
        criterion_window=args.criterion_window,
        criterion_path_type=args.criterion_path_type,
        max_acquisition_episodes=args.max_acquisition_episodes,
        min_acquisition_episodes=args.min_acquisition_episodes,
        epsilon_start=args.epsilon_start,
        epsilon_floor=args.epsilon_floor,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        buffer_size=args.buffer_size,
        seed=args.seed,
        run_id=args.run_id,
        log_dir=args.log_dir,
        device=args.device,
        config=config,
        image_size=args.image_size,
        network_class=resolve_network_class(args.network_size, args.image_size),
        frame_stack=args.frame_stack,
        max_steps=args.max_steps,
        reward_overrides=_parse_reward_overrides(args.reward),
        non_protective_trap=args.non_protective_trap,
        allow_weak_reversal=args.allow_weak_reversal,
        shield_lights_up=args.shield_lights_up,
    )


if __name__ == "__main__":
    main()
