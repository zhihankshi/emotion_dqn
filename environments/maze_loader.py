"""
Maze configuration loader.
Loads maze definitions from YAML files.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List


def load_maze(maze_name: str, mazes_dir: str = None) -> Dict[str, Any]:
    """
    Load maze configuration from YAML file.
    
    Args:
        maze_name: Name of the maze (without .yaml extension)
        mazes_dir: Directory containing maze files (default: mazes/)
    
    Returns:
        Dictionary containing maze configuration
    """
    if mazes_dir is None:
        # Default to mazes/ directory relative to this file
        mazes_dir = Path(__file__).parent.parent / "mazes"
    else:
        mazes_dir = Path(mazes_dir)
    
    maze_path = mazes_dir / f"{maze_name}.yaml"
    
    if not maze_path.exists():
        available = list_available_mazes(mazes_dir)
        raise FileNotFoundError(
            f"Maze '{maze_name}' not found at {maze_path}. "
            f"Available mazes: {available}"
        )
    
    with open(maze_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    _validate_maze_config(config, maze_name)
    
    return config


def list_available_mazes(mazes_dir: str = None) -> List[str]:
    """
    List all available maze configurations.
    
    Args:
        mazes_dir: Directory containing maze files
    
    Returns:
        List of maze names (without .yaml extension)
    """
    if mazes_dir is None:
        mazes_dir = Path(__file__).parent.parent / "mazes"
    else:
        mazes_dir = Path(mazes_dir)
    
    if not mazes_dir.exists():
        return []
    
    return [p.stem for p in mazes_dir.glob("*.yaml")]


def _validate_maze_config(config: Dict[str, Any], maze_name: str) -> None:
    """Validate that maze config has all required fields."""
    required_fields = [
        'name', 'size', 'agent_start',
        'goal_position', 'walls', 'rewards', 'colors'
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(
                f"Maze '{maze_name}' missing required field: '{field}'"
            )
    
    # Validate size
    if len(config['size']) != 2:
        raise ValueError(f"Maze size must be [rows, cols], got: {config['size']}")
    
    rows, cols = config['size']
    
    # Validate positions are within bounds
    positions = {
        'agent_start': config['agent_start'],
        'goal_position': config['goal_position'],
    }
    for optional_pos in ('key_position', 'door_position',
                         'shield_position', 'trap_position'):
        val = config.get(optional_pos)
        if val is not None:
            positions[optional_pos] = val

    for name, pos in positions.items():
        if pos is None or len(pos) != 2:
            raise ValueError(f"{name} must be [row, col], got: {pos}")
        if not (0 <= pos[0] < rows and 0 <= pos[1] < cols):
            raise ValueError(
                f"{name} {pos} is out of bounds for {rows}x{cols} maze"
            )
    
    # Validate walls
    for wall in config['walls']:
        if len(wall) != 2:
            raise ValueError(f"Wall position must be [row, col], got: {wall}")
        if not (0 <= wall[0] < rows and 0 <= wall[1] < cols):
            raise ValueError(
                f"Wall {wall} is out of bounds for {rows}x{cols} maze"
            )


def create_maze_grid(config: Dict[str, Any]) -> List[List[str]]:
    """
    Create a 2D grid representation of the maze for debugging.
    
    Returns:
        2D list where each cell contains a character:
        '.' = floor, '#' = wall, 'A' = agent start, 
        'K' = key, 'D' = door, 'G' = goal, 'T' = trap
    """
    rows, cols = config['size']
    grid = [['.' for _ in range(cols)] for _ in range(rows)]
    
    # Place walls
    for wall in config['walls']:
        grid[wall[0]][wall[1]] = '#'
    
    # Place traps
    for trap in config.get('traps', []):
        grid[trap[0]][trap[1]] = 'T'
    
    # Place special locations
    if config.get('key_position') is not None:
        r, c = config['key_position']
        grid[r][c] = 'K'
    
    if config.get('door_position') is not None:
        r, c = config['door_position']
        grid[r][c] = 'D'

    if config.get('shield_position') is not None:
        r, c = config['shield_position']
        grid[r][c] = 'S'

    if config.get('trap_position') is not None:
        r, c = config['trap_position']
        grid[r][c] = 'T'

    r, c = config['goal_position']
    grid[r][c] = 'G'
    
    r, c = config['agent_start']
    grid[r][c] = 'A'
    
    return grid


def print_maze(config: Dict[str, Any]) -> None:
    """Print ASCII representation of maze."""
    grid = create_maze_grid(config)
    
    print(f"\nMaze: {config['name']}")
    print(f"Size: {config['size'][0]}x{config['size'][1]}")
    print("-" * (config['size'][1] * 2 + 1))
    
    for row in grid:
        print("|" + " ".join(row) + "|")
    
    print("-" * (config['size'][1] * 2 + 1))
    print("Legend: A=agent, K=key, D=door, G=goal, #=wall, T=trap, S=shield")


# Quick test
if __name__ == "__main__":
    print("Available mazes:", list_available_mazes())
    
    config = load_maze("minimal")
    print_maze(config)