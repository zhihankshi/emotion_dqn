"""
Emotional Deep Q-Network Agent (Simplified).

ONLY differs from baseline in Q-value calculation.
Based on "Emotions as Computations" paper (Section 3.4.1).

Key equation:
    Q_target = r + γ * max Q(s', a') + β * M
    
Where M is the mood (running average of TD errors).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .dqn import DQNNetwork
from .replay_buffer import ReplayBuffer  # Same as baseline - no prioritized replay

class MoodTracker:
    def __init__(self, lambda_mood: float = 0.8):  # Changed from 0.95
        self.lambda_mood = lambda_mood
        self.mood = 0.0
        self.mood_history = []
        self.td_error_history = []
    
    def update(self, td_error: float) -> float:
        # M = M + (1 - λ)(δ - M)
        self.mood = self.mood + (1 - self.lambda_mood) * (td_error - self.mood)
        
        # Record history
        self.mood_history.append(self.mood)
        self.td_error_history.append(td_error)
        
        if len(self.mood_history) > 10000:
            self.mood_history = self.mood_history[-10000:]
            self.td_error_history = self.td_error_history[-10000:]
        
        return self.mood
    
    def get_mood(self) -> float:
        return self.mood
    
    def reset(self, full: bool = False) -> None:
        """Reset mood - only on full reset, not between episodes."""
        if full:
            self.mood = 0.0
            self.mood_history = []
            self.td_error_history = []
        # else: do nothing - mood persists!


class EmotionalDQNAgent:
    """
    DQN agent with emotion-biased Q-value updates.
    
    ONLY DIFFERENCE FROM BASELINE:
        Q_target = r + γ * max Q(s', a') + β * mood
    
    Everything else (network, replay, exploration) is identical to baseline.
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_actions: int,
        # Standard DQN params
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        # Emotion params - UPDATED DEFAULTS
        lambda_mood: float = 0.8,   # Was 0.95
        beta: float = 1.0,          # Was 0.1
        # Other
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize Emotional DQN agent.
        
        Args:
            observation_shape: Shape of observations (H, W, C)
            n_actions: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            lambda_mood: Mood persistence (0-1, higher = more persistent)
            beta: How much mood biases Q-targets (generalization strength)
            device: 'cuda' or 'cpu'
            seed: Random seed
        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration parameters (SAME as baseline)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        
        # Emotion parameter (ONLY new parameter)
        self.beta = beta
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Networks (SAME architecture as baseline)
        self.policy_net = DQNNetwork(observation_shape, n_actions).to(self.device)
        self.target_net = DQNNetwork(observation_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer (SAME as baseline)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer (SAME as baseline - standard replay, not prioritized)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        
        # Mood tracker (NEW - but simple)
        self.mood_tracker = MoodTracker(lambda_mood=lambda_mood)
        
        # Counters
        self.steps = 0
        self.updates = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        IDENTICAL to baseline - no emotion-based exploration changes.
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store transition in replay buffer (IDENTICAL to baseline)."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one gradient update step.
        
        THIS IS WHERE THE ONLY DIFFERENCE IS:
            Q_target = r + γ * max Q(s', a') + β * mood
                                               ^^^^^^^^^^
                                               This term is added
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch (SAME as baseline)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors (SAME as baseline)
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        
        # Current Q values (SAME as baseline)
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Target Q values (SAME as baseline)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            standard_target = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        # TD error (computed BEFORE adding mood bias)
        td_error = (standard_target - current_q).mean().item()
        
        # Update mood based on TD error
        current_mood = self.mood_tracker.update(td_error)

        # DEBUG - print every 500 updates
        if self.updates % 500 == 0:
            print(f"    [DEBUG] Update {self.updates}: TD={td_error:+.4f}, "
                f"Mood={current_mood:+.4f}, Beta={self.beta}, Bias={self.beta * current_mood:+.4f}")
        
        # ============================================================
        # THIS IS THE ONLY DIFFERENCE FROM BASELINE
        # ============================================================
        # Add mood bias to target (emotion-biased value update)
        # From paper: V̂(s) = V̂(s) + η*δ + (1-η)*M
        # We implement as: Q_target = standard_target + β * mood
        # ============================================================
        mood_bias = self.beta * current_mood
        biased_target = standard_target + mood_bias
        # ============================================================
        
        # Loss (using biased target)
        loss = F.mse_loss(current_q, biased_target)
        
        # Optimize (SAME as baseline)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.updates += 1
        
        # Update target network (SAME as baseline)
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return {
            'loss': loss.item(),
            'td_error': td_error,
            'q_value_mean': current_q.mean().item(),
            'mood': current_mood,
            'mood_bias': mood_bias,
        }
    
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[Dict[str, float]]:
        """
        Complete step: store transition, update, decay epsilon.
        """
        self.steps += 1
        
        # Store transition
        self.store_transition(state, action, reward, next_state, done)
        
        # Update
        metrics = self.update()
        
        # Decay epsilon (SAME as baseline)
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        if metrics:
            metrics['epsilon'] = self.epsilon
            metrics['buffer_size'] = len(self.replay_buffer)
        
        return metrics
    
    def reset_episode(self) -> None:
        # Don't reset mood - it should persist across episodes
        pass
    
    def get_mood(self) -> float:
        """Get current mood value."""
        return self.mood_tracker.get_mood()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics."""
        return {
            'steps': self.steps,
            'updates': self.updates,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'mood': self.mood_tracker.get_mood(),
        }
    
    def save(self, path: str) -> None:
        """Save agent to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'updates': self.updates,
            'epsilon': self.epsilon,
            'mood': self.mood_tracker.get_mood(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']
        self.epsilon = checkpoint['epsilon']
        self.mood_tracker.mood = checkpoint.get('mood', 0.0)


# Quick test
if __name__ == "__main__":
    print("Testing Simplified Emotional DQN Agent...")
    print("=" * 60)
    print("ONLY DIFFERENCE FROM BASELINE:")
    print("  Q_target = r + γ * max Q(s', a') + β * mood")
    print("=" * 60)
    
    # Create agent
    agent = EmotionalDQNAgent(
        observation_shape=(64, 64, 3),
        n_actions=4,
        buffer_size=1000,
        batch_size=32,
        lambda_mood=0.95,
        beta=0.1,
        seed=42
    )
    
    print(f"\nDevice: {agent.device}")
    print(f"Beta (mood influence): {agent.beta}")
    
    # Fill buffer
    print("\nFilling replay buffer...")
    for i in range(100):
        state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        action = np.random.randint(4)
        reward = np.random.randn()
        next_state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        done = i % 20 == 0
        
        agent.store_transition(state, action, reward, next_state, done)
    
    # Test updates
    print("\nRunning updates...")
    for i in range(10):
        metrics = agent.update()
        if metrics and i % 3 == 0:
            print(f"  Update {i}: TD={metrics['td_error']:+.4f}, "
                  f"Mood={metrics['mood']:+.4f}, Bias={metrics['mood_bias']:+.4f}")
    
    print(f"\nFinal mood: {agent.get_mood():.4f}")
    print("\n✓ Simplified Emotional DQN works!")