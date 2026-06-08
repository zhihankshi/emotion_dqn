"""
Maze renderer - converts maze state to pixel observations.
"""
import numpy as np
from typing import Dict, Any, Tuple, List


class MazeRenderer:
    """Renders maze state to RGB pixel array."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        image_size: int = 64,
        cell_padding: int = 1
    ):
        """
        Initialize renderer.
        
        Args:
            config: Maze configuration dictionary
            image_size: Output image size (square)
            cell_padding: Padding between cells in pixels
        """
        self.config = config
        self.image_size = image_size
        self.cell_padding = cell_padding
        
        self.rows, self.cols = config['size']
        
        # Calculate cell size to fit grid in image
        self.cell_size = image_size // max(self.rows, self.cols)
        
        # Parse colors from config
        self.colors = {
            key: np.array(value, dtype=np.uint8)
            for key, value in config['colors'].items()
        }
        
        # Precompute wall positions as set for fast lookup
        self.wall_set = set(tuple(w) for w in config['walls'])
        self.trap_set = set(tuple(t) for t in config.get('traps', []))
        
        # Shield/trap mechanic positions (singular, conditional penalty)
        shield_raw = config.get('shield_position')
        self.shield_pos = tuple(shield_raw) if shield_raw is not None else None
        trap_raw = config.get('trap_position')
        self.trap_pos = tuple(trap_raw) if trap_raw is not None else None
    
    def render(
        self,
        agent_pos: List[int],
        has_key: bool,
        door_open: bool,
        has_shield: bool = False,
        shield_consumed: bool = False
    ) -> np.ndarray:
        """
        Render current maze state to RGB image.
        
        Args:
            agent_pos: Current agent position [row, col]
            has_key: Whether agent has picked up the key
            door_open: Whether door is open
            has_shield: Whether agent currently holds the shield
            shield_consumed: Whether the shield has been used up
        
        Returns:
            RGB image as numpy array of shape (image_size, image_size, 3)
        """
        # Create blank image with floor color
        img = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8)
        img[:] = self.colors['floor']
        
        # Draw each cell
        for row in range(self.rows):
            for col in range(self.cols):
                self._draw_cell(img, row, col, agent_pos, has_key, door_open,
                                has_shield, shield_consumed)
        
        return img
    
    def _draw_cell(
        self,
        img: np.ndarray,
        row: int,
        col: int,
        agent_pos: List[int],
        has_key: bool,
        door_open: bool,
        has_shield: bool = False,
        shield_consumed: bool = False
    ) -> None:
        """Draw a single cell on the image."""
        # Calculate pixel coordinates
        y1 = row * self.cell_size + self.cell_padding
        y2 = (row + 1) * self.cell_size - self.cell_padding
        x1 = col * self.cell_size + self.cell_padding
        x2 = (col + 1) * self.cell_size - self.cell_padding
        
        # Ensure we don't go out of bounds
        y2 = min(y2, self.image_size)
        x2 = min(x2, self.image_size)
        
        pos = (row, col)
        
        # Determine cell color based on what's there
        if pos in self.wall_set:
            color = self.colors['wall']
        elif pos in self.trap_set:
            color = self.colors['trap']
        elif self.trap_pos is not None and pos == self.trap_pos:
            color = self.colors['trap']
        elif list(pos) == self.config['goal_position']:
            color = self.colors['goal']
        elif (
            self.config.get('door_position') is not None
            and list(pos) == self.config['door_position']
        ):
            color = self.colors['door_open'] if door_open else self.colors['door_locked']
        elif (
            self.shield_pos is not None
            and pos == self.shield_pos
            and not has_shield
            and not shield_consumed
        ):
            color = self.colors['shield']
        elif (
            self.config.get('key_position') is not None
            and list(pos) == self.config['key_position']
            and not has_key
        ):
            color = self.colors['key']
        else:
            color = self.colors['floor']
        
        # Fill cell with color
        img[y1:y2, x1:x2] = color
        
        # Draw agent on top if present
        if list(pos) == list(agent_pos):
            margin = self.cell_size // 4
            ay1 = y1 + margin
            ay2 = y2 - margin
            ax1 = x1 + margin
            ax2 = x2 - margin
            img[ay1:ay2, ax1:ax2] = self.colors['agent']
    
    def render_with_info(
        self,
        agent_pos: List[int],
        has_key: bool,
        door_open: bool,
        has_shield: bool = False,
        shield_consumed: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Render and return additional debug info.
        
        Returns:
            Tuple of (image, info_dict)
        """
        img = self.render(agent_pos, has_key, door_open,
                          has_shield, shield_consumed)
        
        info = {
            'agent_pos': agent_pos,
            'has_key': has_key,
            'door_open': door_open,
            'has_shield': has_shield,
            'shield_consumed': shield_consumed,
            'cell_size': self.cell_size,
            'image_size': self.image_size
        }
        
        return img, info


# Quick test
if __name__ == "__main__":
    from maze_loader import load_maze
    import matplotlib.pyplot as plt
    
    # Load maze
    config = load_maze("minimal")
    
    # Create renderer
    renderer = MazeRenderer(config, image_size=64)
    
    # Render initial state
    img = renderer.render(
        agent_pos=config['agent_start'],
        has_key=False,
        door_open=False
    )
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    
    # Display
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.title("Minimal Maze - Initial State")
    plt.axis('off')
    plt.savefig("test_render.png")
    plt.show()
    
    # Test with key picked up
    img2 = renderer.render(
        agent_pos=[2, 4],  # At key position
        has_key=True,
        door_open=False
    )
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img2)
    plt.title("Minimal Maze - Key Picked Up")
    plt.axis('off')
    plt.show()