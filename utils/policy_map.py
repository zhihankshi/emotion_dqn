"""
Q-value policy map: greedy action direction at each maze cell.

Renders a grid overlay with arrows showing which direction the agent would
move from each walkable cell under a greedy (epsilon=0) policy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from agents.dqn import masked_action_selection

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_VECTORS = {
    0: (0.0, -0.35),
    1: (0.0, 0.35),
    2: (-0.35, 0.0),
    3: (0.35, 0.0),
}


def get_q_values(agent, obs: np.ndarray) -> np.ndarray:
    """Return Q-values for a single observation."""
    with torch.no_grad():
        state_t = torch.from_numpy(obs).unsqueeze(0).to(agent.device)
        return agent.policy_net(state_t).cpu().numpy()[0]


def greedy_action(
    agent,
    obs: np.ndarray,
    valid_actions: Sequence[int],
) -> Tuple[int, np.ndarray]:
    """Greedy action and full Q-vector for one observation."""
    q_values = get_q_values(agent, obs)
    action = masked_action_selection(
        q_values, valid_actions, epsilon=0.0, training=False
    )
    return action, q_values


def build_policy_grid(
    agent,
    env,
    has_key: bool = False,
    door_open: bool = False,
    has_shield: bool = False,
    shield_consumed: bool = False,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Compute greedy policy and Q-values at every walkable cell.

    Uses a fixed inventory state so the map reflects navigation intent
    from each location under consistent conditions.
    """
    grid: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for cell in env.iter_walkable_cells():
        obs = env.set_state_for_observation(
            agent_pos=cell,
            has_key=has_key,
            door_open=door_open,
            has_shield=has_shield,
            shield_consumed=shield_consumed,
        )
        valid_actions = env.get_valid_actions(
            agent_pos=cell,
            has_key=has_key,
            door_open=door_open,
        )
        action, q_values = greedy_action(agent, obs, valid_actions)
        grid[cell] = {
            "action": action,
            "action_name": ACTION_NAMES[action],
            "q_values": q_values,
            "q_max": float(q_values[action]),
            "q_spread": float(q_values[valid_actions].max() - q_values[valid_actions].min()),
            "valid_actions": list(valid_actions),
        }

    return grid


def _draw_maze_base(ax, env, rows: int, cols: int) -> None:
    """Draw walls and special cells on the axes."""
    colors = env.config.get("colors", {})
    floor = np.array(colors.get("floor", [240, 240, 240])) / 255.0
    wall = np.array(colors.get("wall", [64, 64, 64])) / 255.0

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True, color="white", linewidth=0.5, alpha=0.4)

    for row in range(rows):
        for col in range(cols):
            color = wall if (row, col) in env.wall_set else floor
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.5,
                )
            )

    specials = [
        ("agent_start", "A", "cyan"),
        ("key_position", "K", "gold"),
        ("shield_position", "S", "cyan"),
        ("door_position", "D", "saddlebrown"),
        ("trap_position", "T", "red"),
        ("goal_position", "G", "limegreen"),
    ]
    for key, label, color in specials:
        pos = env.config.get(key)
        if pos is not None:
            ax.text(
                pos[1], pos[0], label,
                ha="center", va="center", fontsize=11, fontweight="bold", color=color,
            )


def plot_policy_map(
    agent,
    env,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    has_key: bool = False,
    door_open: bool = False,
    has_shield: bool = False,
    shield_consumed: bool = False,
    show_q_values: bool = False,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """
    Plot greedy policy arrows on the maze grid.

    Args:
        agent: Trained DQN or EmotionalDQN agent
        env: VisualMazeEnv instance
        save_path: Optional output image path
        title: Plot title
        has_key/door_open/has_shield/shield_consumed: Fixed state for all cells
        show_q_values: Annotate cells with max Q instead of only arrows
    """
    rows, cols = env.rows, env.cols
    policy = build_policy_grid(
        agent, env,
        has_key=has_key,
        door_open=door_open,
        has_shield=has_shield,
        shield_consumed=shield_consumed,
    )

    fig, ax = plt.subplots(figsize=figsize)
    _draw_maze_base(ax, env, rows, cols)

    for (row, col), data in policy.items():
        action = data["action"]
        dx, dy = ACTION_VECTORS[action]
        ax.annotate(
            "",
            xy=(col + dx, row + dy),
            xytext=(col, row),
            arrowprops=dict(arrowstyle="->", color="blue", lw=2.0),
        )
        if show_q_values:
            ax.text(
                col, row + 0.22,
                f"{data['q_max']:.1f}",
                ha="center", va="center", fontsize=7, color="navy",
            )

    state_bits = []
    if has_key:
        state_bits.append("key")
    if door_open:
        state_bits.append("door_open")
    if has_shield:
        state_bits.append("shield")
    if shield_consumed:
        state_bits.append("shield_used")
    state_label = ", ".join(state_bits) if state_bits else "default"

    if title is None:
        title = f"Policy map — {env.maze_name} ({state_label})"

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("column")
    ax.set_ylabel("row")

    legend = (
        "Arrows: greedy action (UP/DOWN/LEFT/RIGHT)\n"
        "Invalid moves (walls, blocked door) are masked"
    )
    ax.text(
        0.02, 0.02, legend,
        transform=ax.transAxes, fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_policy_map_panels(
    agent,
    env,
    save_path: Optional[Union[str, Path]] = None,
    panels: Optional[List[Dict[str, Any]]] = None,
) -> plt.Figure:
    """
    Plot multiple policy maps for different inventory states side by side.

    Default panels depend on maze mechanics (shield vs key).
    """
    if panels is None:
        panels = [{"title": "default"}]
        if env.has_shield_mechanic:
            panels = [
                {"title": "no shield", "has_shield": False},
                {"title": "with shield", "has_shield": True},
            ]
        elif env.has_key_mechanic and env.key_required:
            panels = [
                {"title": "no key", "has_key": False, "door_open": False},
                {"title": "has key", "has_key": True, "door_open": False},
                {"title": "door open", "has_key": True, "door_open": True},
            ]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    rows, cols = env.rows, env.cols

    for ax, panel in zip(axes, panels):
        panel = dict(panel)
        panel_title = panel.pop("title", "policy")
        policy = build_policy_grid(
            agent, env,
            has_key=panel.get("has_key", False),
            door_open=panel.get("door_open", False),
            has_shield=panel.get("has_shield", False),
            shield_consumed=panel.get("shield_consumed", False),
        )
        _draw_maze_base(ax, env, rows, cols)
        for (row, col), data in policy.items():
            action = data["action"]
            dx, dy = ACTION_VECTORS[action]
            ax.annotate(
                "",
                xy=(col + dx, row + dy),
                xytext=(col, row),
                arrowprops=dict(arrowstyle="->", color="blue", lw=2.0),
            )
        ax.set_title(panel_title)

    fig.suptitle(f"Policy maps — {env.maze_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def save_policy_map_csv(
    agent,
    env,
    save_path: Union[str, Path],
    **state_kwargs,
) -> None:
    """Export per-cell policy and Q-values to CSV."""
    import csv

    policy = build_policy_grid(agent, env, **state_kwargs)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row", "col", "action", "action_name",
            "q_up", "q_down", "q_left", "q_right", "q_max", "q_spread",
        ])
        for (row, col) in sorted(policy.keys()):
            data = policy[(row, col)]
            q = data["q_values"]
            writer.writerow([
                row, col, data["action"], data["action_name"],
                f"{q[0]:.4f}", f"{q[1]:.4f}", f"{q[2]:.4f}", f"{q[3]:.4f}",
                f"{data['q_max']:.4f}", f"{data['q_spread']:.4f}",
            ])
