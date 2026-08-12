# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research codebase testing whether a computational mood signal (Emanuel & Eldar, "Emotions as
Computations") improves DQN learning in small visual mazes, against a *matched* baseline DQN.

The two agents differ in exactly one term:

- Baseline (`agents/dqn.py`): `Q_target = Q + η·δ`
- Emotional (`agents/emotional_dqn.py`): `Q_target = Q + η·δ + (1-η)·M`
- Mood update: `M ← M + (1-λ)·(η·δ - M)`, clipped to `mood_bounds`, **persisting across episodes**
  (`reset_episode()` is intentionally a no-op)

`η` (`--eta`) is shared by both agents, so any change to it must apply to both or the comparison
stops being matched. The same holds for network init, seeding, optimizer, buffer, target-net sync,
and the ε schedule — keeping baseline/emotional symmetric is the core invariant of this repo.

`agents/mood_system.py` (`MoodSystem`, value+action mood, exploration boost) is an older, richer
formulation that the current `EmotionalDQNAgent` does **not** use — it uses the simpler `MoodTracker`
in `emotional_dqn.py`. Don't assume `MoodSystem` is live.

## Environment

Windows, virtualenv at `.venv`. Run scripts with `.venv\Scripts\python.exe` (or activate first).

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`configs/training.yaml` is empty — all configuration is CLI flags. There is no test framework;
`tests/sanity_checks.py` is a hand-rolled assert script.

## Commands

```powershell
# Sanity checks (env mechanics + per-maze reward accounting). Run before training.
.venv\Scripts\python.exe tests/sanity_checks.py

# Single agent training run
.venv\Scripts\python.exe train.py --agent emotional --maze shield_trap --episodes 1000

# Main experiment driver: N runs each of baseline + emotional, side by side
.venv\Scripts\python.exe compare_agents.py --maze shield_trap --runs 3 --episodes 800 --analyze_checkpoints

# Fast smoke test of both agents (short run, verifies checkpoints save)
.venv\Scripts\python.exe compare_agents.py --quick_test --maze shield_trap_easy

# Two-phase transfer: train on source maze, reload final checkpoint, continue on target
.venv\Scripts\python.exe scripts/train_transfer.py --source_maze shield_avoidance --target_maze shield_trap

# Plots
.venv\Scripts\python.exe visualize_results.py
.venv\Scripts\python.exe visualize_transfer.py
.venv\Scripts\python.exe analyze_policy_evolution.py --checkpoint_dir <run>/checkpoints --maze shield_trap
```

Running a single sanity check: import it from `tests/sanity_checks.py` (each `test_*` is a plain
function returning `True`); `run_all_tests()` drives the whole suite.

Several modules have `if __name__ == "__main__"` demo blocks (`agents/emotional_dqn.py`,
`agents/mood_system.py`, `environments/visual_maze.py`) that are useful quick probes.

## Architecture

**`environments/`** — `VisualMazeEnv` (Gymnasium) is the single env class; all maze variation lives in
YAML under `mazes/`, loaded by `maze_loader.py` and drawn by `renderer.py`.

- Observations are **channel-first uint8** `(3 * frame_stack, image_size, image_size)`.
- Mechanics are opt-in per maze by presence of a key: `key_position`, `door_position`,
  `shield_position`/`trap_position` (conditional trap), `traps` (flat penalty list). A `null`
  `door_position` means no door; `key_required: false` allows reaching the goal without the key.
- The conditional trap fires **once per episode** (`trap_hit_step`), so dithering on it can't stack
  penalties. Shield pickup is once per episode too (`shield_consumed`).
- `rewards.repeat_cell` / `repeat_free_visits` is anti-oscillation shaping applied per revisit.
- `get_valid_actions()` / action masking is load-bearing: invalid actions are excluded both when
  acting *and* in the bootstrap target (`next_valid_masks` stored in the replay buffer), because
  Q-values for never-executed actions otherwise inflate the max and overestimate.
- `set_state_for_observation()` lets analysis code render arbitrary states without stepping.

**`agents/`** — `DQNNetwork` (Nature-style, 8×8 first kernel) vs `SmallDQNNetwork` (3×3 convs).
`image_size < 36` auto-selects the small net (`resolve_network_class` in `train.py`); the standard
net cannot consume tiny images. `--double_dqn` is available on both agents.

**`train.py`** — `train()` is the reusable entry point (`compare_agents.py` and
`scripts/train_transfer.py` both call it, not the CLI). Notable behaviors:

- ε decays **linearly across `n_episodes`**, set once per episode via `update_epsilon_for_episode`.
  Changing episode count changes the exploration schedule — short runs are not prefixes of long runs.
- Checkpoints are saved at `ANALYSIS_CHECKPOINT_EPISODES` (a fixed set: 60, 80, 100, …, 500) *plus*
  every `checkpoint_interval` episodes, as `checkpoints/agent_episode_{N}.pt`.
- Transfer relies on the phase-1 final-episode checkpoint existing, so `checkpoint_interval` must
  divide phase-1 episodes.
- Each episode is classified by `utils/path_analysis.classify_episode_path` into
  `shield_route` / `trap_rush` / `key_route` / `direct` / `timeout` / `other` — this is the main
  behavioral readout for the shield-trap mazes, more informative than success rate alone.

**`compare_agents.py`** — the real experiment driver. Writes
`experiments/{maze}_comparison_{timestamp}/` containing per-agent run dirs, a
`checkpoint_manifest.json`, greedy per-checkpoint evaluations, and comparison JSON/PNG.
`--reward key=value` (repeatable), `--max_steps`, and `--shield_lights_up` override maze YAML at
runtime, which is the intended way to sweep reward shaping without editing the YAML.

**`utils/metrics.py`** — `EpisodeMetrics` → `MetricsLogger` (per-run CSV, streamed per episode) →
`ExperimentLogger` (cross-run aggregation). Downstream plotting in `utils/visualization.py` reads
those CSV/JSON artifacts, so new metrics must be added to `EpisodeMetrics` *and* the CSV header.

**Output dirs**: `experiments/` (git-ignored, comparison runs), `runs/` and `test_runs/` (single/
transfer runs), `diagnostics/`, `eval_results/`.

## Working notes

- The mazes are small, deterministic, and stationary — the least favorable regime for the mood
  mechanism. A null result is a plausible finding, not automatically a bug.
- Mood is currently fed δ from **sampled replay batches** (`MoodTracker.update_batch` in
  `EmotionalDQNAgent.update`), not the online experienced transition — that's temporally scrambled
  relative to the theory. Be precise about which δ any change touches.
- Prefer config-gated changes with defaults that preserve current behavior, so existing experiment
  results stay reproducible.
- New mazes need a matching `test_*_rewards` sanity check in `tests/sanity_checks.py`; every existing
  maze has one that walks a scripted path and asserts the exact reward total.
