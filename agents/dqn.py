"""
Baseline Deep Q-Network (DQN) Agent.
Based on the Nature DQN paper (Mnih et al., 2015).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .replay_buffer import ReplayBuffer


class DQNNetwork(nn.Module):
    """
    Convolutional neural network for Q-value estimation.
    Architecture based on Nature DQN paper, scaled down for smaller inputs.
    """
    
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        """
        Initialize DQN network.
        
        Args:
            input_shape: Shape of input observations (H, W, C)
            n_actions: Number of possible actions
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Input is (H, W, C), but PyTorch expects (C, H, W)
        # We'll transpose in forward()
        h, w, c = input_shape
        
        # Convolutional layers (scaled down from Nature DQN)
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_output_size(c, h, w)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output_size(self, c: int, h: int, w: int) -> int:
        """Calculate the output size of conv layers."""
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            return int(np.prod(out.shape[1:]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, H, W, C) with values in [0, 255]
        
        Returns:
            Q-values for each action, shape (batch, n_actions)
        """
        # Normalize to [0, 1]
        x = x.float() / 255.0
        
        # Transpose from (batch, H, W, C) to (batch, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Conv layers
        x = self.conv(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # FC layers
        x = self.fc(x)
        
        return x


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    """
    
    def __init__(
        self,
        observation_shape: Tuple[int, ...],
        n_actions: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        network_class=None
    ):
        """
        Initialize DQN agent.
        
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
            device: 'cuda' or 'cpu' (auto-detect if None)
            seed: Random seed
            network_class: Network class to use (default: DQNNetwork)
        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Exploration parameters
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
        
        # Networks
        if network_class is None:
            network_class = DQNNetwork
        self.policy_net = network_class(observation_shape, n_actions).to(self.device)
        self.target_net = network_class(observation_shape, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)
        
        # Counters
        self.steps = 0
        self.updates = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current observation
            training: Whether in training mode (enables exploration)
        
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
        """Store a transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one gradient update step.
        
        Returns:
            Dictionary of metrics, or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
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
        
        # Current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)
        
        # TD error
        td_error = target_q - current_q
        
        # Loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.updates += 1
        
        # Update target network
        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return {
            'loss': loss.item(),
            'td_error_mean': td_error.mean().item(),
            'td_error_std': td_error.std().item(),
            'q_value_mean': current_q.mean().item(),
            'q_value_max': current_q.max().item(),
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
        
        Returns:
            Update metrics or None
        """
        self.steps += 1
        
        # Store transition
        self.store_transition(state, action, reward, next_state, done)
        
        # Update
        metrics = self.update()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
        if metrics:
            metrics['epsilon'] = self.epsilon
            metrics['buffer_size'] = len(self.replay_buffer)
        
        return metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics."""
        return {
            'steps': self.steps,
            'updates': self.updates,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
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
        }, path)

    def save_checkpoint(self, path: str, episode: int) -> None:
        """Save lightweight checkpoint for policy analysis at a training stage."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'agent_type': 'baseline',
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load policy checkpoint. Returns the saved episode number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.epsilon = checkpoint['epsilon']
        return int(checkpoint['episode'])
    
    def load(self, path: str) -> None:
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']
        self.epsilon = checkpoint['epsilon']


# Quick test
if __name__ == "__main__":
    print("Testing DQN Agent...")
    
    # Create agent
    agent = DQNAgent(
        observation_shape=(64, 64, 3),
        n_actions=4,
        buffer_size=1000,
        batch_size=32,
        seed=42
    )
    
    print(f"  Device: {agent.device}")
    print(f"  Network parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    
    # Test action selection
    dummy_state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    action = agent.select_action(dummy_state)
    print(f"  Selected action: {action}")
    
    # Fill buffer with dummy transitions
    print("  Filling replay buffer...")
    for i in range(100):
        state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        action = np.random.randint(4)
        reward = np.random.randn()
        next_state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        done = i % 20 == 0
        
        agent.store_transition(state, action, reward, next_state, done)
    
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    
    # Test update
    print("  Running update...")
    metrics = agent.update()
    if metrics:
        print(f"    Loss: {metrics['loss']:.4f}")
        print(f"    TD error mean: {metrics['td_error_mean']:.4f}")
        print(f"    Q value mean: {metrics['q_value_mean']:.4f}")
    
    # Test save/load
    agent.save("test_agent.pt")
    agent.load("test_agent.pt")
    print("  Save/load: ✓")
    
    # Cleanup
    Path("test_agent.pt").unlink()
    
    print("✓ DQN Agent works!")