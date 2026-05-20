"""
Emotional Deep Q-Network Agent (Paper-Faithful).

Based on "Emotions as Computations" paper.

Paper equation (Section 3.3.2):
    V̂(s) = V(s) + M
    
Where M is mood (running average of TD errors).

In Q-learning terms:
    Q_target = r + γ * max Q(s', a') + γ * M

This is the ONLY difference from baseline DQN.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .dqn import DQNNetwork
from .replay_buffer import ReplayBuffer


class MoodTracker:
    """
    Simple mood tracker - running average of TD errors.
    
    From paper: "Mood integrates recent reward prediction errors"
    
    Update rule:
        M_t+1 = λ * M_t + (1 - λ) * δ_t
    
    Where:
        M = mood
        λ = persistence (how slowly mood changes)
        δ = TD error
    """
    
    def __init__(self, lambda_mood: float = 0.8):
        """
        Args:
            lambda_mood: Persistence of mood.
                         0.8 = moderate persistence
                         0.95 = very slow change
                         0.5 = fast change
        """
        self.lambda_mood = lambda_mood
        self.mood = 0.0
    
    def update(self, td_error: float) -> float:
        """
        Update mood based on TD error.
        
        Args:
            td_error: Current TD error (δ)
        
        Returns:
            Updated mood value
        """
        self.mood = self.lambda_mood * self.mood + (1 - self.lambda_mood) * td_error
        return self.mood
    
    def get_mood(self) -> float:
        """Get current mood value."""
        return self.mood
    
    def reset(self) -> None:
        """Full reset of mood (only for new training runs)."""
        self.mood = 0.0


class EmotionalDQNAgent:
    """
    DQN agent with emotion-biased Q-value updates.
    
    ONLY DIFFERENCE FROM BASELINE:
    
        Baseline:  Q_target = r + γ * max Q(s', a')
        Emotional: Q_target = r + γ * max Q(s', a') + γ * mood
                                                      ^^^^^^^^^
                                                      This term added
    
    Everything else (network, replay, exploration) is IDENTICAL to baseline.
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_actions: int,
        # Standard DQN params (IDENTICAL to baseline)
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        # Emotion param (ONLY new parameter)
        lambda_mood: float = 0.8,
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
        
        # Replay buffer (SAME as baseline)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        
        # Mood tracker (NEW - the only addition)
        self.mood_tracker = MoodTracker(lambda_mood=lambda_mood)
        
        # Counters
        self.steps = 0
        self.updates = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        IDENTICAL to baseline - no emotion-based changes.
        
        Args:
            state: Current observation
            training: Whether in training mode (uses epsilon-greedy)
        
        Returns:
            Selected action
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
        """
        Store transition in replay buffer.
        
        IDENTICAL to baseline.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one gradient update step.
        
        THIS IS WHERE THE ONLY DIFFERENCE IS:
        
            Baseline:  Q_target = r + γ * max Q(s', a')
            Emotional: Q_target = r + γ * max Q(s', a') + γ * mood
        
        Returns:
            Dictionary of metrics, or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # ============================================================
        # EVERYTHING BELOW IS SAME AS BASELINE UNTIL MARKED
        # ============================================================
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        
        # Current Q values: Q(s, a)
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Next Q values: max_a' Q(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            
            # Standard target: r + γ * max Q(s', a')
            standard_target = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        # TD error (computed before mood adjustment)
        td_error = (standard_target - current_q).mean().item()
        
        # Update mood based on TD error
        current_mood = self.mood_tracker.update(td_error)
        
        # ============================================================
        # THIS IS THE ONLY DIFFERENCE FROM BASELINE
        # ============================================================
        # Paper: V̂(s) = V(s) + M
        # Therefore: Q_target = r + γ * max Q(s', a') + γ * M
        # ============================================================
        
        mood_bias = self.gamma * current_mood
        emotional_target = standard_target + mood_bias
        
        # ============================================================
        # EVERYTHING BELOW IS SAME AS BASELINE
        # ============================================================
        
        # Loss
        loss = F.mse_loss(current_q, emotional_target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.updates += 1
        
        # Update target network
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
        Complete step: store transition, update network, decay epsilon.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        
        Returns:
            Update metrics or None
        """
        self.steps += 1
        
        # Store transition (SAME as baseline)
        self.store_transition(state, action, reward, next_state, done)
        
        # Update network
        metrics = self.update()
        
        # Decay epsilon (SAME as baseline)
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        if metrics:
            metrics['epsilon'] = self.epsilon
        
        return metrics
    
    def reset_episode(self) -> None:
        """
        Reset for new episode.
        
        Mood PERSISTS across episodes - this is intentional.
        From paper: mood is a persistent state that generalizes.
        """
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
            'lambda_mood': self.mood_tracker.lambda_mood,
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


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PAPER-FAITHFUL EMOTIONAL DQN")
    print("=" * 60)
    print()
    print("From 'Emotions as Computations' paper:")
    print("  V̂(s) = V(s) + M")
    print()
    print("In Q-learning:")
    print("  Q_target = r + γ * max Q(s', a') + γ * mood")
    print()
    print("This is the ONLY difference from baseline.")
    print("=" * 60)
    
    # Create agent
    agent = EmotionalDQNAgent(
        observation_shape=(64, 64, 3),
        n_actions=4,
        gamma=0.99,
        lambda_mood=0.8,
        buffer_size=1000,
        batch_size=32,
        seed=42
    )
    
    print(f"\nAgent created:")
    print(f"  Device: {agent.device}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Lambda mood: {agent.mood_tracker.lambda_mood}")
    print(f"  Mood bias formula: {agent.gamma} * mood")
    
    # Fill buffer with random transitions
    print("\nFilling replay buffer...")
    for i in range(100):
        state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        action = np.random.randint(4)
        reward = np.random.choice([-0.5, -0.04, 1.0, 2.0, 10.0], p=[0.1, 0.7, 0.1, 0.05, 0.05])
        next_state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        done = np.random.random() < 0.05
        
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    
    # Run some updates
    print("\nRunning updates...")
    print("-" * 60)
    print(f"{'Update':<8} {'TD Error':<12} {'Mood':<12} {'Mood Bias':<12}")
    print("-" * 60)
    
    for i in range(20):
        metrics = agent.update()
        if metrics and i % 4 == 0:
            print(f"{i:<8} {metrics['td_error']:+.6f}    {metrics['mood']:+.6f}    {metrics['mood_bias']:+.6f}")
    
    print("-" * 60)
    print(f"\nFinal mood: {agent.get_mood():+.6f}")
    print(f"Final mood bias: {agent.gamma * agent.get_mood():+.6f}")
    print("\n✓ Paper-faithful emotional DQN working!")