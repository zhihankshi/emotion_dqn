"""
Emotional Deep Q-Network Agent (Corrected Implementation).

Based on grad student feedback and paper's actual equation:
    Q_new = Q_old + η * δ + (1-η) * M

Where:
    δ = r + γ * max Q(s',a') - Q(s,a)  [TD error]
    η = learning balance (0.9 = 90% TD, 10% mood)
    M = mood (running average of TD errors)
    γ = discount factor (0.99)
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
    Mood tracker - running average of TD errors.
    
    M_t+1 = λ * M_t + (1 - λ) * δ_t
    """
    
    def __init__(self, lambda_mood: float = 0.8):
        self.lambda_mood = lambda_mood
        self.mood = 0.0
    
    def update(self, td_error: float) -> float:
        self.mood = self.lambda_mood * self.mood + (1 - self.lambda_mood) * td_error
        return self.mood
    
    def get_mood(self) -> float:
        return self.mood
    
    def reset(self) -> None:
        self.mood = 0.0


class EmotionalDQNAgent:
    """
    DQN agent with emotion-biased Q-value updates.
    
    CORRECTED EQUATION:
        Q_target = Q_current + η * δ + (1-η) * M
        
    Where:
        δ = r + γ * max Q' - Q_current (TD error)
        η = learning balance parameter
        M = mood
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
        # Emotion params
        lambda_mood: float = 0.8,
        eta: float = 0.9,
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
        
        # Mood tracker
        self.mood_tracker = MoodTracker(lambda_mood=lambda_mood)
        
        # Counters
        self.steps = 0
        self.updates = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy."""
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
        
        # Update mood with average TD error
        current_mood = self.mood_tracker.update(td_error)
        
        # ============================================================
        # CORRECTED EQUATION (from grad student feedback):
        # Q_target = Q_current + η * δ + (1-η) * M
        #          = Q_current + η * (standard_target - Q_current) + (1-η) * M
        # ============================================================
        
        with torch.no_grad():
            # η * δ = η * (standard_target - current_q)
            td_component = self.eta * (standard_target - current_q.detach())
            
            # (1-η) * M
            mood_component = (1 - self.eta) * current_mood
            
            # Q_target = Q_current + η*δ + (1-η)*M
            emotional_target = current_q.detach() + td_component + mood_component
        
        # ============================================================
        
        # Loss
        loss = F.mse_loss(current_q, emotional_target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.updates += 1
        
        # Update target network periodically (addressing feedback #2)
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
        """Complete step: store, update, decay epsilon."""
        self.steps += 1
        self.store_transition(state, action, reward, next_state, done)
        metrics = self.update()
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        
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
        self.eta = checkpoint.get('eta', 0.9)


# ============================================================
# TEST
# ============================================================
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
    
    # Fill buffer
    print("\nFilling replay buffer...")
    for i in range(100):
        state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        action = np.random.randint(4)
        reward = np.random.choice([-0.5, -0.04, 1.0, 2.0, 10.0])
        next_state = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        done = np.random.random() < 0.05
        agent.store_transition(state, action, reward, next_state, done)
    
    # Run updates
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