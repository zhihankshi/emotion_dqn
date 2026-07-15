"""
Baseline Deep Q-Network (DQN) Agent.

Value learning follows paper eq. 3.1.3 (Emanuel & Eldar, 2023):
    Q_target = Q + η * δ

With η=1 this reduces to standard full-TD DQN. Use the same η as the
emotional agent so comparisons isolate the mood term (eq. 3.4.1).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, Sequence
from pathlib import Path

from .replay_buffer import ReplayBuffer


def masked_action_selection(
    q_values: np.ndarray,
    valid_actions: Sequence[int],
    epsilon: float = 0.0,
    training: bool = True,
) -> int:
    """Epsilon-greedy action selection restricted to valid actions."""
    valid_actions = list(valid_actions)
    if not valid_actions:
        valid_actions = list(range(len(q_values)))

    if training and np.random.random() < epsilon:
        return int(np.random.choice(valid_actions))

    masked = np.full(len(q_values), -np.inf)
    for action in valid_actions:
        masked[action] = q_values[action]
    return int(np.argmax(masked))


class DQNNetwork(nn.Module):
    """
    Convolutional neural network for Q-value estimation.
    Architecture based on Nature DQN paper, scaled down for smaller inputs.
    """
    
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        """
        Initialize DQN network.
        
        Args:
            input_shape: Shape of input observations (C, H, W)
            n_actions: Number of possible actions
        """
        super().__init__()
        
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        c, h, w = input_shape
        
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
            x: Input tensor of shape (batch, C, H, W) with values in [0, 255]
        
        Returns:
            Q-values for each action, shape (batch, n_actions)
        """
        # Normalize to [0, 1]
        x = x.float() / 255.0
        
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
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        eta: float = 0.9,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        network_class=None,
        double_dqn: bool = False,
    ):
        """
        Initialize DQN agent.
        
        Args:
            observation_shape: Shape of observations (C, H, W)
            n_actions: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate (reached at last episode)
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            eta: TD learning rate in Q_target = Q + ηδ (paper eq. 3.1.3).
                 Use the same value as the emotional agent (default 0.9).
            device: 'cuda' or 'cpu' (auto-detect if None)
            seed: Random seed
            network_class: Network class to use (default: DQNNetwork)
            double_dqn: Use Double DQN targets (policy net selects the next
                action, target net evaluates it) to curb overestimation
        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eta = eta
        self.double_dqn = double_dqn
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        
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
    
    def update_epsilon_for_episode(self, episode: int, n_episodes: int) -> None:
        """Linear epsilon decay over training episodes."""
        if n_episodes <= 1:
            self.epsilon = self.epsilon_end
            return
        progress = episode / (n_episodes - 1)
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        self.epsilon = max(self.epsilon_end, self.epsilon)
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True,
        valid_actions: Optional[Sequence[int]] = None,
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current observation
            training: Whether in training mode (enables exploration)
            valid_actions: Optional list of legal actions (e.g. excludes walls)

        Returns:
            Selected action
        """
        if valid_actions is None:
            valid_actions = list(range(self.n_actions))

        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).cpu().numpy()[0]
            return masked_action_selection(
                q_values, valid_actions, epsilon=0.0, training=False
            )
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: Optional[Sequence[int]] = None,
    ) -> None:
        """Store a transition in replay buffer.

        next_valid_actions: actions legal in next_state. Invalid actions are
        excluded from the bootstrap target, so Q-values that never receive
        real experience cannot inflate the targets. None = all actions valid.
        """
        mask = np.zeros(self.n_actions, dtype=np.float32)
        if next_valid_actions is None:
            mask[:] = 1.0
        else:
            mask[list(next_valid_actions)] = 1.0
            if mask.sum() == 0:
                mask[:] = 1.0
        self.replay_buffer.push(state, action, reward, next_state, done, mask)
    
    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one gradient update step.
        
        Returns:
            Dictionary of metrics, or None if buffer not ready
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones, next_valid_masks = (
            self.replay_buffer.sample(self.batch_size)
        )
        
        # Convert to tensors
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        masks_t = torch.from_numpy(next_valid_masks).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        # Target Q values (paper eq. 3.1.3: Q + ηδ)
        # Invalid next-state actions are excluded from the max/argmax: they are
        # never executed (rollouts mask them), so their Q-values get no gradient
        # signal from real experience and would otherwise inflate the bootstrap.
        with torch.no_grad():
            invalid = masks_t == 0
            if self.double_dqn:
                # Double DQN: policy net picks the action, target net scores it
                next_policy_q = self.policy_net(next_states_t).masked_fill(invalid, -1e9)
                next_actions = next_policy_q.argmax(dim=1)
                next_q = self.target_net(next_states_t).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_target_q = self.target_net(next_states_t).masked_fill(invalid, -1e9)
                next_q = next_target_q.max(dim=1)[0]
            standard_target = rewards_t + self.gamma * next_q * (1 - dones_t)
            td_error = standard_target - current_q
            target_q = current_q.detach() + self.eta * td_error.detach()
        
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
            'td_error': td_error.mean().item(),
            'td_error_mean': td_error.mean().item(),
            'td_error_std': td_error.std().item(),
            'q_value_mean': current_q.mean().item(),
            'q_value_max': current_q.max().item(),
            'eta': self.eta,
        }
    
    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_actions: Optional[Sequence[int]] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Complete step: store transition and update network.
        
        Returns:
            Update metrics or None
        """
        self.steps += 1
        
        # Store transition
        self.store_transition(state, action, reward, next_state, done, next_valid_actions)
        
        # Update
        metrics = self.update()
        
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
            'eta': self.eta,
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
        self.eta = checkpoint.get('eta', self.eta)
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
        self.eta = checkpoint.get('eta', self.eta)


# Quick test
if __name__ == "__main__":
    print("Testing DQN Agent...")
    
    # Create agent
    agent = DQNAgent(
        observation_shape=(3, 64, 64),
        n_actions=4,
        buffer_size=1000,
        batch_size=32,
        seed=42
    )
    
    print(f"  Device: {agent.device}")
    print(f"  Network parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    
    # Test action selection
    dummy_state = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
    action = agent.select_action(dummy_state)
    print(f"  Selected action: {action}")
    
    # Fill buffer with dummy transitions
    print("  Filling replay buffer...")
    for i in range(100):
        state = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
        action = np.random.randint(4)
        reward = np.random.randn()
        next_state = np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8)
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