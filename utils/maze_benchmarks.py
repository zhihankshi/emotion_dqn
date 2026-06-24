"""
Optimal-path benchmarks for maze environments.

Used by visualization scripts to draw convergence reference lines.
"""
from typing import Dict, List, Any

from environments import VisualMazeEnv

# Scripted optimal actions per maze (verified in sanity_checks)
OPTIMAL_ACTIONS: Dict[str, List[int]] = {
    "key_approach": [1] * 4 + [0] * 4 + [3] * 6,
    "shield_avoidance": [1] * 4 + [0] * 4 + [3] * 6,
    "shield_trap": [1] * 4 + [0] * 4 + [3] * 6,
    "shield_trap_v2": [3] * 6,
}


def run_scripted_episode(maze_name: str, actions: List[int]) -> Dict[str, Any]:
    """Run a fixed action sequence and return total reward, steps, success."""
    env = VisualMazeEnv(maze_name=maze_name)
    env.reset()

    total_reward = 0.0
    steps = 0
    success = False

    for action in actions:
        _, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated:
            success = True
        if terminated or truncated:
            break

    return {
        "maze": maze_name,
        "reward": total_reward,
        "steps": steps,
        "success": success,
    }


def get_maze_benchmark(maze_name: str) -> Dict[str, Any]:
    """
    Return optimal reward and step count for a maze.

    Returns dict with keys: reward, steps, success, label
    """
    actions = OPTIMAL_ACTIONS.get(maze_name)
    if actions is None:
        raise ValueError(
            f"No optimal path defined for maze '{maze_name}'. "
            f"Add it to OPTIMAL_ACTIONS in utils/maze_benchmarks.py"
        )

    result = run_scripted_episode(maze_name, actions)
    result["label"] = f"optimal ({result['reward']:.2f}, {result['steps']} steps)"
    return result


def get_benchmarks_for_transfer(
    source_maze: str,
    target_maze: str,
) -> Dict[str, Dict[str, Any]]:
    """Benchmarks for both phases of a transfer experiment."""
    return {
        "phase1": get_maze_benchmark(source_maze),
        "phase2": get_maze_benchmark(target_maze),
    }
