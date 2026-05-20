"""Visualize a maze to verify it looks correct."""
import matplotlib.pyplot as plt
import numpy as np
from environments import VisualMazeEnv


def visualize_maze(maze_name: str, save_path: str = None):
    """
    Render and display a maze.
    
    Args:
        maze_name: Name of maze to visualize
        save_path: Optional path to save image
    """
    print(f"Loading maze: {maze_name}")
    env = VisualMazeEnv(maze_name=maze_name)
    obs, info = env.reset()
    
    # Try different ways to access config
    config = None
    if hasattr(env, 'maze_config'):
        config = env.maze_config
    elif hasattr(env, 'config'):
        config = env.config
    elif hasattr(env, 'maze'):
        config = env.maze
    
    if config:
        print(f"Maze size: {config.get('size', 'unknown')}")
        print(f"Agent start: {config.get('agent_start', 'unknown')}")
        print(f"Key position: {config.get('key_position', 'unknown')}")
        print(f"Door position: {config.get('door_position', 'unknown')}")
        print(f"Goal position: {config.get('goal_position', 'unknown')}")
        print(f"Number of walls: {len(config.get('walls', []))}")
    
    print(f"Observation shape: {obs.shape}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Show observation
    ax.imshow(obs)
    ax.set_title(f"Maze: {maze_name}")
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()
    
    # Print ASCII if config available
    if config and 'size' in config:
        print("\nASCII representation:")
        size = config['size']
        
        grid = [['.' for _ in range(size[1])] for _ in range(size[0])]
        
        # Place elements
        for wall in config.get('walls', []):
            grid[wall[0]][wall[1]] = '#'
        
        for trap in config.get('traps', []):
            grid[trap[0]][trap[1]] = 'X'
        
        if 'key_position' in config:
            k = config['key_position']
            grid[k[0]][k[1]] = 'K'
        
        if 'door_position' in config:
            d = config['door_position']
            grid[d[0]][d[1]] = 'D'
        
        if 'goal_position' in config:
            g = config['goal_position']
            grid[g[0]][g[1]] = 'G'
        
        if 'agent_start' in config:
            a = config['agent_start']
            grid[a[0]][a[1]] = 'A'
        
        print("   " + " ".join(str(i) for i in range(size[1])))
        print("   " + "-" * (size[1] * 2 - 1))
        for i, row in enumerate(grid):
            print(f"{i} |" + " ".join(row))
        
        print("\nLegend: A=Agent, K=Key, D=Door, G=Goal, #=Wall, X=Trap")
    
    return env


if __name__ == "__main__":
    import sys
    
    maze_name = sys.argv[1] if len(sys.argv) > 1 else "minimal"
    visualize_maze(maze_name)