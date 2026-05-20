"""
Visual Maze Environment.
Gymnasium-compatible environment with pixel observations.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from .maze_loader import load_maze
from .renderer import MazeRenderer


class VisualMazeEnv(gym.Env):
    """
    Visual maze environment where agent must find key to open door to reach goal.
    
    Observations are RGB pixel images.
    Actions are discrete: 0=up, 1=down, 2=left, 3=right
    
    The agent must learn the causal relationship:
    key -> door opens -> goal accessible
    """
    
    metadata = {"render_modes": ["rgb_array", "human"]}
    
    # Action mappings
    ACTIONS = {
        0: (-1, 0),   # Up
        1: (1, 0),    # Down
        2: (0, -1),   # Left
        3: (0, 1)     # Right
    }
    ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}
    
    def __init__(
        self,
        maze_name: str = "minimal",
        render_mode: str = "rgb_array",
        image_size: int = 64,
        mazes_dir: Optional[str] = None
    ):
        """
        Initialize the visual maze environment.
        
        Args:
            maze_name: Name of maze config file (without .yaml)
            render_mode: "rgb_array" or "human"
            image_size: Size of square observation image
            mazes_dir: Directory containing maze YAML files
        """
        super().__init__()
        
        # Load maze configuration
        self.config = load_maze(maze_name, mazes_dir)
        self.maze_name = maze_name
        self.render_mode = render_mode
        self.image_size = image_size
        
        # Grid dimensions
        self.rows, self.cols = self.config['size']
        
        # Create renderer
        self.renderer = MazeRenderer(self.config, image_size=image_size)
        
        # Precompute sets for fast collision detection
        self.wall_set = set(tuple(w) for w in self.config['walls'])
        self.trap_set = set(tuple(t) for t in self.config.get('traps', []))
        
        # Special positions
        self.key_pos = tuple(self.config['key_position'])
        self.door_pos = tuple(self.config['door_position'])
        self.goal_pos = tuple(self.config['goal_position'])
        
        # Rewards
        self.rewards = self.config['rewards']
        self.max_steps = self.config.get('max_steps', 200)
        
        # State variables (initialized in reset)
        self.agent_pos = None
        self.has_key = False
        self.door_open = False
        self.steps = 0
        
        # Metrics for causal understanding
        self.door_attempts_without_key = 0
        self.key_pickup_step = -1
        
        # Define spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(image_size, image_size, 3),
            dtype=np.uint8
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset state
        self.agent_pos = list(self.config['agent_start'])
        self.has_key = False
        self.door_open = False
        self.steps = 0
        
        # Reset metrics
        self.door_attempts_without_key = 0
        self.key_pickup_step = -1
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action (0=up, 1=down, 2=left, 3=right)
        
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.steps += 1
        reward = self.rewards['step']  # Base step cost
        terminated = False
        
        # Calculate new position
        dr, dc = self.ACTIONS[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        new_pos = (new_row, new_col)
        
        # Check boundaries
        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            # Hit boundary - stay in place, small penalty
            reward += self.rewards['wall_bump']
        
        # Check wall collision
        elif new_pos in self.wall_set:
            # Hit wall - stay in place, small penalty
            reward += self.rewards['wall_bump']
        
        # Check trap
        elif new_pos in self.trap_set:
            # Hit trap - move there but take damage
            self.agent_pos = [new_row, new_col]
            reward += self.rewards['trap']
        
        # Check door (special case)
        elif new_pos == self.door_pos:
            if self.has_key:
                # Have key - can pass through door
                self.agent_pos = [new_row, new_col]
                if not self.door_open:
                    # First time opening door
                    self.door_open = True
                    reward += self.rewards['door_open']
            else:
                # No key - blocked by door
                self.door_attempts_without_key += 1
                reward += self.rewards['door_without_key']
                # Stay in place
        
        # Normal move
        else:
            self.agent_pos = [new_row, new_col]
        
        # Check key pickup
        if tuple(self.agent_pos) == self.key_pos and not self.has_key:
            self.has_key = True
            self.key_pickup_step = self.steps
            reward += self.rewards['key']
        
        # Check goal
        if tuple(self.agent_pos) == self.goal_pos:
            if self.door_open:
                # Success! Reached goal after opening door
                reward += self.rewards['goal']
                terminated = True
            # If door not open, agent shouldn't be able to reach goal
            # (maze design should prevent this)
        
        # Check truncation (timeout)
        truncated = self.steps >= self.max_steps
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Render current state to pixel observation."""
        return self.renderer.render(
            agent_pos=self.agent_pos,
            has_key=self.has_key,
            door_open=self.door_open
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current state info for debugging and metrics."""
        return {
            'agent_pos': tuple(self.agent_pos),
            'has_key': self.has_key,
            'door_open': self.door_open,
            'steps': self.steps,
            'door_attempts_without_key': self.door_attempts_without_key,
            'key_pickup_step': self.key_pickup_step,
            'maze_name': self.maze_name
        }
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array if render_mode is "rgb_array", else None
        """
        if self.render_mode == "rgb_array":
            return self._get_observation()
        elif self.render_mode == "human":
            # Could add pygame window here for human viewing
            # For now, just return the array
            return self._get_observation()
        return None
    
    def get_state_summary(self) -> str:
        """Get human-readable state summary."""
        return (
            f"Step {self.steps}: "
            f"pos={tuple(self.agent_pos)}, "
            f"has_key={self.has_key}, "
            f"door_open={self.door_open}"
        )
    
    def close(self):
        """Clean up resources."""
        pass


# Quick test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Creating environment...")
    env = VisualMazeEnv(maze_name="minimal", image_size=64)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset
    obs, info = env.reset()
    print(f"\nInitial state: {info}")
    print(f"Observation shape: {obs.shape}")
    
    # Take some random actions
    print("\nTaking random actions...")
    total_reward = 0
    
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"  Action: {env.ACTION_NAMES[action]:5s} | "
              f"Reward: {reward:+.2f} | "
              f"{env.get_state_summary()}")
        
        if terminated:
            print("  >>> GOAL REACHED! <<<")
            break
        if truncated:
            print("  >>> TIMEOUT <<<")
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    print(f"Door attempts without key: {info['door_attempts_without_key']}")
    
    # Visualize final state
    plt.figure(figsize=(4, 4))
    plt.imshow(obs)
    plt.title(f"Final State\n{env.get_state_summary()}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("test_env.png")
    print("\nSaved visualization to test_env.png")
    plt.show()