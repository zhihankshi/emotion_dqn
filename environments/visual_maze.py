"""
Visual Maze Environment.
Gymnasium-compatible environment with pixel observations.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List, Sequence

from .maze_loader import load_maze
from .renderer import MazeRenderer


class VisualMazeEnv(gym.Env):
    """
    Visual maze environment with key, optional door, and goal.
    
    Observations are RGB pixel images.
    Actions are discrete: 0=up, 1=down, 2=left, 3=right
    
    Classic mazes: key -> door opens -> goal (key_required default True).
    Optional-key mazes: goal reachable without key; holding key may boost
    goal reward (goal_with_key). door_position may be null (no door).
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
        key_raw = self.config.get('key_position')
        self.has_key_mechanic = key_raw is not None
        self.key_pos = tuple(key_raw) if self.has_key_mechanic else None
        self.goal_pos = tuple(self.config['goal_position'])

        door_raw = self.config.get('door_position')
        self.has_door = door_raw is not None
        self.door_pos = tuple(door_raw) if self.has_door else None

        self.key_required = self.config.get('key_required', True)

        # Shield/trap mechanic
        shield_raw = self.config.get('shield_position')
        self.has_shield_mechanic = shield_raw is not None
        self.shield_pos = tuple(shield_raw) if self.has_shield_mechanic else None

        trap_raw = self.config.get('trap_position')
        self.has_trap_mechanic = trap_raw is not None
        self.trap_pos = tuple(trap_raw) if self.has_trap_mechanic else None

        # Rewards
        self.rewards = self.config['rewards']
        self.max_steps = self.config.get('max_steps', 200)
        
        # State variables (initialized in reset)
        self.agent_pos = None
        self.has_key = False
        self.door_open = False
        self.has_shield = False
        self.shield_consumed = False
        self.steps = 0
        self._visit_counts: Dict[Tuple[int, int], int] = {}
        self._max_visit_count: int = 0
        
        # Metrics for causal understanding
        self.door_attempts_without_key = 0
        self.key_pickup_step = -1
        self.shield_pickup_step = -1
        self.trap_hit_step = -1
        
        # Define spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, image_size, image_size),
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
        self.has_shield = False
        self.shield_consumed = False
        self.steps = 0
        self._visit_counts = {}
        start_pos = tuple(self.agent_pos)
        self._visit_counts[start_pos] = 1
        self._max_visit_count = 1
        
        # Reset metrics
        self.door_attempts_without_key = 0
        self.key_pickup_step = -1
        self.shield_pickup_step = -1
        self.trap_hit_step = -1
        
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
            reward += self.rewards['wall_bump']
        
        # Check wall collision
        elif new_pos in self.wall_set:
            reward += self.rewards['wall_bump']
        
        # Check conditional trap (shield/trap mechanic)
        elif self.has_trap_mechanic and new_pos == self.trap_pos:
            self.agent_pos = [new_row, new_col]
            if self.has_shield:
                reward += self.rewards['trap_with_shield']
                self.has_shield = False
                self.shield_consumed = True
            else:
                reward += self.rewards['trap_no_shield']
            if self.trap_hit_step == -1:
                self.trap_hit_step = self.steps
        
        # Check simple traps (flat penalty list)
        elif new_pos in self.trap_set:
            self.agent_pos = [new_row, new_col]
            reward += self.rewards['trap']
        
        # Check door (special case)
        elif self.has_door and new_pos == self.door_pos:
            if self.has_key:
                self.agent_pos = [new_row, new_col]
                if not self.door_open:
                    self.door_open = True
                    reward += self.rewards['door_open']
            else:
                self.door_attempts_without_key += 1
                reward += self.rewards['door_without_key']
        
        # Normal move
        else:
            self.agent_pos = [new_row, new_col]
        
        # Check shield pickup
        if (self.has_shield_mechanic
                and tuple(self.agent_pos) == self.shield_pos
                and not self.has_shield
                and not self.shield_consumed):
            self.has_shield = True
            self.shield_pickup_step = self.steps
            reward += self.rewards.get('shield_pickup', 0)
        
        # Check key pickup
        if (self.has_key_mechanic
                and tuple(self.agent_pos) == self.key_pos
                and not self.has_key):
            self.has_key = True
            self.key_pickup_step = self.steps
            reward += self.rewards['key']
        
        # Check goal
        if tuple(self.agent_pos) == self.goal_pos and self._can_complete_goal():
            if self.has_key and 'goal_with_key' in self.rewards:
                reward += self.rewards['goal_with_key']
            else:
                reward += self.rewards['goal']
            terminated = True
        
        # Check truncation (timeout)
        truncated = self.steps >= self.max_steps
        if truncated and not terminated:
            reward += self.rewards.get('timeout', 0)

        # Anti-loop shaping: penalize excessive revisits (e.g., right-left oscillations)
        # This is immediate (unlike timeout) and maze-configurable via:
        #   rewards.repeat_cell: negative penalty applied when revisiting too often
        #   rewards.repeat_free_visits: number of visits per cell allowed without penalty (default 2)
        pos_t = tuple(self.agent_pos)
        self._visit_counts[pos_t] = self._visit_counts.get(pos_t, 0) + 1
        self._max_visit_count = max(self._max_visit_count, self._visit_counts[pos_t])

        repeat_penalty = float(self.rewards.get('repeat_cell', 0.0))
        free_visits = int(self.rewards.get('repeat_free_visits', 2))
        if repeat_penalty != 0.0 and self._visit_counts[pos_t] > free_visits:
            reward += repeat_penalty
        
        # Get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _can_complete_goal(self) -> bool:
        """Whether standing on the goal cell ends the episode successfully."""
        if self.key_required and not self.has_key:
            return False
        if self.has_door and not self.door_open:
            return False
        return True

    def is_action_valid(
        self,
        action: int,
        agent_pos: Optional[Sequence[int]] = None,
        has_key: Optional[bool] = None,
        door_open: Optional[bool] = None,
    ) -> bool:
        """Whether an action moves the agent (not wall, boundary, or blocked door)."""
        pos = list(agent_pos if agent_pos is not None else self.agent_pos)
        hk = has_key if has_key is not None else self.has_key

        dr, dc = self.ACTIONS[action]
        new_row = pos[0] + dr
        new_col = pos[1] + dc
        new_pos = (new_row, new_col)

        if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
            return False
        if new_pos in self.wall_set:
            return False
        if self.has_door and new_pos == self.door_pos and not hk:
            return False
        return True

    def get_valid_actions(
        self,
        agent_pos: Optional[Sequence[int]] = None,
        has_key: Optional[bool] = None,
        door_open: Optional[bool] = None,
    ) -> List[int]:
        """Return action indices that do not bump walls or blocked doors."""
        valid = [
            action for action in self.ACTIONS
            if self.is_action_valid(action, agent_pos, has_key, door_open)
        ]
        return valid or list(self.ACTIONS.keys())

    def set_state_for_observation(
        self,
        agent_pos: Sequence[int],
        has_key: bool = False,
        door_open: bool = False,
        has_shield: bool = False,
        shield_consumed: bool = False,
    ) -> np.ndarray:
        """Place the agent in a specific state and return the rendered observation."""
        self.agent_pos = list(agent_pos)
        self.has_key = has_key
        self.door_open = door_open
        self.has_shield = has_shield
        self.shield_consumed = shield_consumed
        return self._get_observation()

    def iter_walkable_cells(self) -> List[Tuple[int, int]]:
        """All non-wall grid cells."""
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) not in self.wall_set
        ]

    def _get_observation(self) -> np.ndarray:
        """Render current state to pixel observation (C, H, W)."""
        observation = self.renderer.render(
            agent_pos=self.agent_pos,
            has_key=self.has_key,
            door_open=self.door_open,
            has_shield=self.has_shield,
            shield_consumed=self.shield_consumed
        )
        observation = np.transpose(observation, (2, 0, 1))
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current state info for debugging and metrics."""
        return {
            'agent_pos': tuple(self.agent_pos),
            'has_key': self.has_key,
            'door_open': self.door_open,
            'has_door': self.has_door,
            'key_required': self.key_required,
            'has_shield': self.has_shield,
            'shield_consumed': self.shield_consumed,
            'steps': self.steps,
            'door_attempts_without_key': self.door_attempts_without_key,
            'key_pickup_step': self.key_pickup_step,
            'shield_pickup_step': self.shield_pickup_step,
            'trap_hit_step': self.trap_hit_step,
            'maze_name': self.maze_name,
            'cell_visit_count': self._visit_counts.get(tuple(self.agent_pos), 0),
            'max_cell_visit_count': self._max_visit_count,
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
        parts = [
            f"Step {self.steps}:",
            f"pos={tuple(self.agent_pos)}",
        ]
        if self.has_key_mechanic:
            parts.append(f"has_key={self.has_key}")
        if self.has_door:
            parts.append(f"door_open={self.door_open}")
        if self.has_shield_mechanic:
            parts.append(f"has_shield={self.has_shield}")
            parts.append(f"shield_consumed={self.shield_consumed}")
        return " ".join(parts)
    
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