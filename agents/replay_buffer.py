"""
Experience Replay Buffer for DQN.
Stores transitions and samples mini-batches for training.
"""
import numpy as np
from collections import deque
from typing import Tuple, Dict, Any, Optional
import random


class ReplayBuffer:
    """
    Standard experience replay buffer.
    Stores (state, action, reward, next_state, done) transitions.
    """
    
    def __init__(self, capacity: int = 50000, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state (observation)
            action: Action taken
            reward: Reward received
            next_state: Next state (observation)
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            Each is a numpy array with batch_size elements
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Samples transitions based on TD-error priority.
    
    This is where emotions can influence learning by modifying priorities!
    """
    
    def __init__(
        self,
        capacity: int = 50000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        seed: Optional[int] = None
    ):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (annealed to 1)
            beta_increment: How much to increase beta each sample
            epsilon: Small constant to ensure non-zero priorities
            seed: Random seed
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        if seed is not None:
            np.random.seed(seed)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None
    ) -> None:
        """
        Add a transition with priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            priority: Initial priority (if None, uses max priority)
        """
        # Default to max priority for new transitions
        if priority is None:
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        transition = (state, action, reward, next_state, done)
        
        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch based on priorities.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
            - weights: Importance sampling weights
            - indices: Buffer indices (for updating priorities)
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # Get transitions
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        weights = weights.astype(np.float32)
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Buffer indices
            priorities: New priorities (typically |TD-error| + epsilon)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def update_priorities_with_mood(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray,
        mood: float,
        mood_influence: float = 0.5
    ) -> None:
        """
        Update priorities with mood influence.
        
        Emotional experiences (high |TD-error|) get boosted priority.
        Mood can additionally bias which experiences are prioritized.
        
        Args:
            indices: Buffer indices
            td_errors: TD errors for each transition
            mood: Current mood value
            mood_influence: How much mood affects priority (0 to 1)
        """
        base_priorities = np.abs(td_errors)
        
        # Mood-based adjustment:
        # - Negative mood + negative TD-error = boost priority (learn from mistakes)
        # - Positive mood + positive TD-error = boost priority (reinforce success)
        mood_alignment = np.sign(mood) * np.sign(td_errors)
        mood_boost = 1.0 + mood_influence * mood_alignment * abs(mood)
        
        adjusted_priorities = base_priorities * mood_boost
        
        self.update_priorities(indices, adjusted_priorities)
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


# Quick test
if __name__ == "__main__":
    print("Testing ReplayBuffer...")
    
    buffer = ReplayBuffer(capacity=1000, seed=42)
    
    # Add some fake transitions
    for i in range(100):
        state = np.random.rand(64, 64, 3).astype(np.uint8)
        action = np.random.randint(4)
        reward = np.random.randn()
        next_state = np.random.rand(64, 64, 3).astype(np.uint8)
        done = i % 20 == 0
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"  Buffer size: {len(buffer)}")
    print(f"  Ready for batch of 32: {buffer.is_ready(32)}")
    
    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"  Sampled batch shapes:")
    print(f"    states: {states.shape}")
    print(f"    actions: {actions.shape}")
    print(f"    rewards: {rewards.shape}")
    print(f"    next_states: {next_states.shape}")
    print(f"    dones: {dones.shape}")
    
    print("✓ ReplayBuffer works!")
    
    print("\nTesting PrioritizedReplayBuffer...")
    
    pbuffer = PrioritizedReplayBuffer(capacity=1000, seed=42)
    
    for i in range(100):
        state = np.random.rand(64, 64, 3).astype(np.uint8)
        action = np.random.randint(4)
        reward = np.random.randn()
        next_state = np.random.rand(64, 64, 3).astype(np.uint8)
        done = i % 20 == 0
        
        pbuffer.push(state, action, reward, next_state, done)
    
    states, actions, rewards, next_states, dones, weights, indices = pbuffer.sample(32)
    print(f"  Sampled with weights shape: {weights.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # Test priority update
    td_errors = np.random.randn(32)
    pbuffer.update_priorities(indices, np.abs(td_errors))
    print(f"  Updated priorities: ✓")
    
    # Test mood-based priority update
    pbuffer.update_priorities_with_mood(indices, td_errors, mood=-0.3, mood_influence=0.5)
    print(f"  Updated priorities with mood: ✓")
    
    print("✓ PrioritizedReplayBuffer works!")