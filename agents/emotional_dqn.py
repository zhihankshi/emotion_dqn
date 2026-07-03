"""
Emotional Deep Q-Network Agent.

Paper mood (eq. 3.3.2):
    M_{t+1} = M_t + (1 - λ) * (η * δ_t - M_t)

Q-target (grad student / paper learning rule):
    Q_target = Q + η * δ + (1 - η) * M

Where δ = r + γ * max Q(s',a') - Q(s,a).
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, Sequence
from pathlib import Path

from .dqn import DQNNetwork, masked_action_selection
from .replay_buffer import ReplayBuffer


class MoodTracker:
    """
    Cumulative emotional state as discounted sum of value changes (paper §3.3).

    Recursive form (eq. 3.3.2):
        M_{t+1} = M_t + (1 - λ) * (η * δ_t - M_t)
    """

    def __init__(
        self,
        lambda_mood: float = 0.8,
        mood_bounds: Tuple[float, float] = (-1.0, 1.0),
    ):
        self.lambda_mood = lambda_mood
        self.mood_bounds = mood_bounds
        self.mood = 0.0

    def update(self, td_error: float, eta: float) -> float:
        """One timestep of mood accumulation from TD error δ_t."""
        value_change = eta * td_error
        self.mood = self.mood + (1 - self.lambda_mood) * (value_change - self.mood)
        self.mood = float(np.clip(self.mood, self.mood_bounds[0], self.mood_bounds[1]))
        return self.mood

    def update_batch(self, td_errors: np.ndarray, eta: float) -> float:
        """Apply sequential per-timestep updates for each δ in a replay batch."""
        for td_error in td_errors:
            self.update(float(td_error), eta)
        return self.mood

    def get_mood(self) -> float:
        return self.mood

    def reset(self) -> None:
        self.mood = 0.0


class EmotionalDQNAgent:
    """
    DQN agent with emotion-biased Q-value updates.

    Q-target (grad student / paper learning rule):
        Q_target = Q + η * δ + (1 - η) * M    (eq. 3.4.1)

    Baseline uses the same η with M = 0 (eq. 3.1.3).
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
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        # Emotion params
        lambda_mood: float = 0.8,
        eta: float = 0.9,
        mood_bounds: Tuple[float, float] = (-1.0, 1.0),
        # Other
        device: Optional[str] = None,
        seed: Optional[int] = None,
        network_class=None
    ):
        """
        Initialize Emotional DQN agent.

        Args:
            eta: Balance between TD learning and mood (0-1)
                 0.9 = 90% TD error, 10% mood
                 1.0 = pure TD learning (standard DQN)
                 0.5 = equal weight
            network_class: Network class to use (default: DQNNetwork)
        """
        self.observation_shape = observation_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eta = eta

        # Exploration
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

        # Mood tracker
        self.mood_tracker = MoodTracker(
            lambda_mood=lambda_mood,
            mood_bounds=mood_bounds,
        )
        self.mood_bounds = mood_bounds

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
        """Select action using epsilon-greedy (same schedule as baseline)."""
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
        done: bool
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[Dict[str, float]]:
        """
        Perform one gradient update step.

        CORRECTED EQUATION:
            Q_target = Q_current + η * δ + (1-η) * M

        Where δ = r + γ * max Q'(s',a') - Q(s,a)
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

        # Current Q values: Q(s, a)
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Next Q values from TARGET network (frozen): max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]

            # Standard TD target: r + γ * max Q'
            standard_target = rewards_t + self.gamma * next_q * (1 - dones_t)

            # TD error: δ = r + γ * max Q' - Q
            td_error_batch = standard_target - current_q
            td_error = td_error_batch.mean().item()

        # Use M_t for the value update; accumulate mood after (paper §3.3 → §3.4)
        mood_for_target = self.mood_tracker.get_mood()

        with torch.no_grad():
            td_component = self.eta * (standard_target - current_q.detach())
            mood_component = (1 - self.eta) * mood_for_target
            emotional_target = current_q.detach() + td_component + mood_component

        loss = F.mse_loss(current_q, emotional_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        current_mood = self.mood_tracker.update_batch(
            td_error_batch.detach().cpu().numpy(),
            self.eta,
        )

        self.updates += 1

        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return {
            'loss': loss.item(),
            'td_error': td_error,
            'q_value_mean': current_q.mean().item(),
            'mood': current_mood,
            'mood_component': mood_component,
            'td_component': td_component.mean().item(),
            'eta': self.eta,
        }

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[Dict[str, float]]:
        """Complete step: store transition and update network."""
        self.steps += 1
        self.store_transition(state, action, reward, next_state, done)
        metrics = self.update()

        if metrics:
            metrics['epsilon'] = self.epsilon

        return metrics

    def reset_episode(self) -> None:
        """Reset for new episode - mood persists."""
        pass

    def get_mood(self) -> float:
        """Get current mood."""
        return self.mood_tracker.get_mood()

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        return {
            'steps': self.steps,
            'updates': self.updates,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'mood': self.mood_tracker.get_mood(),
            'eta': self.eta,
        }

    def save(self, path: str) -> None:
        """Save agent."""
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
            'mood_bounds': self.mood_bounds,
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
            'mood': self.mood_tracker.mood,
            'agent_type': 'emotional',
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load policy checkpoint. Returns the saved episode number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.epsilon = checkpoint['epsilon']
        self.mood_tracker.mood = checkpoint.get('mood', 0.0)
        if 'mood_bounds' in checkpoint:
            self.mood_bounds = tuple(checkpoint['mood_bounds'])
            self.mood_tracker.mood_bounds = self.mood_bounds
        self.mood_tracker.mood = float(np.clip(
            self.mood_tracker.mood,
            self.mood_bounds[0],
            self.mood_bounds[1],
        ))
        return int(checkpoint['episode'])

    def load(self, path: str) -> None:
        """Load agent."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']
        self.epsilon = checkpoint['epsilon']
        self.mood_tracker.mood = checkpoint.get('mood', 0.0)
        if 'mood_bounds' in checkpoint:
            self.mood_bounds = tuple(checkpoint['mood_bounds'])
            self.mood_tracker.mood_bounds = self.mood_bounds
        self.mood_tracker.mood = float(np.clip(
            self.mood_tracker.mood,
            self.mood_bounds[0],
            self.mood_bounds[1],
        ))
        self.eta = checkpoint.get('eta', 0.9)


if __name__ == "__main__":
    print("=" * 60)
    print("CORRECTED EMOTIONAL DQN")
    print("=" * 60)
    print()
    print("Paper equation:")
    print("  Q_target = Q_current + η * δ + (1-η) * M")
    print()
    print("Where:")
    print("  δ = r + γ * max Q'(s',a') - Q(s,a)  [TD error]")
    print("  η = balance parameter (0.9 = 90% TD, 10% mood)")
    print("  M = mood (running average of TD errors)")
    print("=" * 60)

    agent = EmotionalDQNAgent(
        observation_shape=(64, 64, 3),
        n_actions=4,
        gamma=0.99,
        lambda_mood=0.8,
        eta=0.9,
        buffer_size=1000,
        batch_size=32,
        seed=42
    )

    print(f"\nAgent created:")
    print(f"  Device: {agent.device}")
    print(f"  Gamma (γ): {agent.gamma}")
    print(f"  Eta (η): {agent.eta}")
    print(f"  Lambda mood (λ): {agent.mood_tracker.lambda_mood}")
    print(f"  Target update freq: {agent.target_update_freq}")
    print(f"  Has target network: {agent.target_net is not None}")

    print("\nFilling replay buffer...")
    for i in range(100):
        state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        action = np.random.randint(4)
        reward = np.random.choice([-0.5, -0.04, 1.0, 2.0, 10.0])
        next_state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        done = np.random.random() < 0.05
        agent.store_transition(state, action, reward, next_state, done)

    print("\nRunning updates...")
    print("-" * 70)
    print(f"{'Update':<8} {'TD Error':<12} {'Mood':<12} {'TD Comp':<12} {'Mood Comp':<12}")
    print("-" * 70)

    for i in range(10):
        metrics = agent.update()
        if metrics:
            print(f"{i:<8} {metrics['td_error']:+.4f}     {metrics['mood']:+.4f}     "
                  f"{metrics['td_component']:+.4f}     {metrics['mood_component']:+.4f}")

    print("-" * 70)
    print("\n✓ Corrected emotional DQN working!")
