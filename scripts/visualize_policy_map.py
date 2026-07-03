"""Visualize greedy policy / Q-value map for a trained agent."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt

from agents import DQNAgent, EmotionalDQNAgent, DQNNetwork, SmallDQNNetwork
from analyze_policy_evolution import load_agent_checkpoint
from environments import VisualMazeEnv
from utils.policy_map import plot_policy_map, plot_policy_map_panels, save_policy_map_csv


def main():
    parser = argparse.ArgumentParser(description="Plot Q-value policy map on maze grid")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to agent checkpoint (.pt)")
    parser.add_argument("--maze", type=str, default="shield_trap")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--network_size", type=str, default="standard",
                        choices=["standard", "small"])
    parser.add_argument("--agent_type", type=str, default=None,
                        choices=["baseline", "emotional"])
    parser.add_argument("--output", type=str, default="policy_map.png")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV export of per-cell Q-values")
    parser.add_argument("--panels", action="store_true",
                        help="Plot multiple panels for key/shield states")
    parser.add_argument("--show_q", action="store_true",
                        help="Annotate cells with max Q-value")
    args = parser.parse_args()

    env = VisualMazeEnv(maze_name=args.maze, image_size=args.image_size)
    agent = load_agent_checkpoint(
        args.checkpoint,
        env,
        agent_type=args.agent_type,
        network_size=args.network_size,
        image_size=args.image_size,
    )
    agent.epsilon = 0.0

    if args.panels:
        fig = plot_policy_map_panels(agent, env, save_path=args.output)
    else:
        fig = plot_policy_map(
            agent, env,
            save_path=args.output,
            show_q_values=args.show_q,
        )

    if args.csv:
        save_policy_map_csv(agent, env, args.csv)

    print(f"Saved policy map to {args.output}")
    if args.csv:
        print(f"Saved Q-value table to {args.csv}")

    plt.show()


if __name__ == "__main__":
    main()
