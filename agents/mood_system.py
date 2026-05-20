"""
Mood System for Emotional DQN.

Based on "Emotions as Computations" paper:
- Happiness/Sadness: Changes in expected value (TD error)
- Content/Anger: Action effectiveness (Advantage)
- Mood: Running average that persists and generalizes

Key equations from paper:
- Value emotion: M_V = M_V + (1 - λ)(η * δ - M_V)
- Action emotion: M_A = M_A + (1 - λ)(A(s,a) - M_A)
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EmotionState:
    """Current emotional state with all components."""
    
    # Core emotions (instantaneous)
    happiness: float = 0.0      # Positive TD error
    sadness: float = 0.0        # Negative TD error
    content: float = 0.0        # Positive advantage
    anger: float = 0.0          # Negative advantage (from own actions)
    
    # Moods (persistent, running averages)
    mood_value: float = 0.0     # From happiness/sadness
    mood_action: float = 0.0    # From content/anger
    
    # Combined
    overall_mood: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'happiness': self.happiness,
            'sadness': self.sadness,
            'content': self.content,
            'anger': self.anger,
            'mood_value': self.mood_value,
            'mood_action': self.mood_action,
            'overall_mood': self.overall_mood,
        }


class MoodSystem:
    """
    Computes and tracks emotional states based on RL signals.
    
    Implements the computational theory of emotion where:
    - Emotions arise from changes in value estimates
    - Moods are persistent states that generalize across situations
    - Different emotions serve different computational functions
    """
    
    def __init__(
        self,
        lambda_mood: float = 0.95,
        eta: float = 0.1,
        value_weight: float = 0.6,
        action_weight: float = 0.4,
        mood_bounds: Tuple[float, float] = (-1.0, 1.0)
    ):
        """
        Initialize mood system.
        
        Args:
            lambda_mood: Persistence of mood (0 = instant, 1 = permanent)
                        Higher values = longer lasting mood
            eta: Learning rate for value updates (scales TD error impact)
            value_weight: Weight of value-based mood in overall mood
            action_weight: Weight of action-based mood in overall mood
            mood_bounds: Min and max bounds for mood values
        """
        self.lambda_mood = lambda_mood
        self.eta = eta
        self.value_weight = value_weight
        self.action_weight = action_weight
        self.mood_bounds = mood_bounds
        
        # Current emotional state
        self.state = EmotionState()
        
        # History for analysis
        self.history = {
            'td_errors': [],
            'advantages': [],
            'mood_values': [],
            'mood_actions': [],
            'overall_moods': [],
        }
        self.max_history = 10000
    
    def update(
        self,
        td_error: float,
        advantage: Optional[float] = None,
        record_history: bool = True
    ) -> EmotionState:
        """
        Update emotional state based on new RL signals.
        
        Args:
            td_error: Temporal difference error (reward prediction error)
                     Positive = better than expected (happiness)
                     Negative = worse than expected (sadness)
            advantage: Action advantage A(s,a) = Q(s,a) - V(s)
                      Positive = good action choice (content)
                      Negative = bad action choice (anger/frustration)
            record_history: Whether to record in history
        
        Returns:
            Updated EmotionState
        """
        # Compute instantaneous emotions from TD error
        if td_error > 0:
            self.state.happiness = td_error
            self.state.sadness = 0.0
        else:
            self.state.happiness = 0.0
            self.state.sadness = abs(td_error)
        
        # Compute instantaneous emotions from advantage
        if advantage is not None:
            if advantage > 0:
                self.state.content = advantage
                self.state.anger = 0.0
            else:
                self.state.content = 0.0
                self.state.anger = abs(advantage)
        
        # Update mood (running average with persistence)
        # M_V = M_V + (1 - λ)(η * δ - M_V)
        value_change = self.eta * td_error
        self.state.mood_value = (
            self.state.mood_value + 
            (1 - self.lambda_mood) * (value_change - self.state.mood_value)
        )
        
        # Update action mood if advantage provided
        if advantage is not None:
            self.state.mood_action = (
                self.state.mood_action +
                (1 - self.lambda_mood) * (advantage - self.state.mood_action)
            )
        
        # Clip moods to bounds
        self.state.mood_value = np.clip(
            self.state.mood_value, 
            self.mood_bounds[0], 
            self.mood_bounds[1]
        )
        self.state.mood_action = np.clip(
            self.state.mood_action,
            self.mood_bounds[0],
            self.mood_bounds[1]
        )
        
        # Compute overall mood (weighted combination)
        self.state.overall_mood = (
            self.value_weight * self.state.mood_value +
            self.action_weight * self.state.mood_action
        )
        
        # Record history
        if record_history:
            self._record_history(td_error, advantage)
        
        return self.state
    
    def _record_history(self, td_error: float, advantage: Optional[float]) -> None:
        """Record emotional signals in history."""
        self.history['td_errors'].append(td_error)
        self.history['advantages'].append(advantage if advantage else 0.0)
        self.history['mood_values'].append(self.state.mood_value)
        self.history['mood_actions'].append(self.state.mood_action)
        self.history['overall_moods'].append(self.state.overall_mood)
        
        # Trim history if too long
        if len(self.history['td_errors']) > self.max_history:
            for key in self.history:
                self.history[key] = self.history[key][-self.max_history:]
    
    def get_mood(self) -> float:
        """Get current overall mood."""
        return self.state.overall_mood
    
    def get_mood_value(self) -> float:
        """Get value-based mood (happiness/sadness track)."""
        return self.state.mood_value
    
    def get_mood_action(self) -> float:
        """Get action-based mood (content/anger track)."""
        return self.state.mood_action
    
    def get_state(self) -> EmotionState:
        """Get full emotional state."""
        return self.state
    
    def get_exploration_boost(
        self,
        base_epsilon: float,
        boost_scale: float = 2.0,
        threshold: float = -0.05
    ) -> float:
        """
        Calculate exploration boost based on mood.
        
        Negative mood (frustration/anger) increases exploration.
        From paper: negative emotions signal we should change course.
        
        Args:
            base_epsilon: Base exploration rate
            boost_scale: How much to scale the boost
            threshold: Mood threshold below which to boost exploration
        
        Returns:
            Adjusted epsilon value
        """
        if self.state.mood_action < threshold:
            # Frustrated/angry - boost exploration
            boost = min(0.4, abs(self.state.mood_action) * boost_scale)
            return min(0.95, base_epsilon + boost)
        elif self.state.mood_action > -threshold:
            # Content - slightly reduce exploration
            reduction = min(0.03, self.state.mood_action * 0.5)
            return max(0.01, base_epsilon - reduction)
        else:
            return base_epsilon
    
    def get_value_bias(self, generalization_strength: float = 0.5) -> float:
        """
        Get mood-based bias for Q-value updates.
        
        Positive mood → optimistic value estimates
        Negative mood → pessimistic value estimates
        
        This implements "generalization" from the emotions paper:
        mood biases subsequent value computations.
        
        Args:
            generalization_strength: How much mood affects values
        
        Returns:
            Bias to add to Q-target
        """
        return generalization_strength * self.state.overall_mood
    
    def reset(self) -> None:
        """Reset emotional state (e.g., at start of new episode)."""
        # Don't fully reset - mood should persist somewhat
        # Just decay towards neutral
        decay = 0.5
        self.state.mood_value *= decay
        self.state.mood_action *= decay
        self.state.overall_mood *= decay
        
        # Reset instantaneous emotions
        self.state.happiness = 0.0
        self.state.sadness = 0.0
        self.state.content = 0.0
        self.state.anger = 0.0
    
    def reset_full(self) -> None:
        """Fully reset emotional state (for new training run)."""
        self.state = EmotionState()
        self.history = {key: [] for key in self.history}
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of recent emotional history."""
        if len(self.history['td_errors']) == 0:
            return {}
        
        recent_n = min(100, len(self.history['td_errors']))
        
        return {
            'recent_td_error_mean': np.mean(self.history['td_errors'][-recent_n:]),
            'recent_td_error_std': np.std(self.history['td_errors'][-recent_n:]),
            'recent_mood_value_mean': np.mean(self.history['mood_values'][-recent_n:]),
            'recent_mood_action_mean': np.mean(self.history['mood_actions'][-recent_n:]),
            'recent_overall_mood_mean': np.mean(self.history['overall_moods'][-recent_n:]),
            'mood_value_current': self.state.mood_value,
            'mood_action_current': self.state.mood_action,
            'overall_mood_current': self.state.overall_mood,
        }


# Quick test
if __name__ == "__main__":
    print("Testing MoodSystem...")
    
    mood_system = MoodSystem(
        lambda_mood=0.95,
        eta=0.1
    )
    
    print(f"\nInitial state:")
    print(f"  Overall mood: {mood_system.get_mood():.4f}")
    print(f"  Mood value: {mood_system.get_mood_value():.4f}")
    print(f"  Mood action: {mood_system.get_mood_action():.4f}")
    
    # Simulate positive surprise (good reward)
    print(f"\n--- Positive surprise (TD error = +1.0) ---")
    state = mood_system.update(td_error=1.0, advantage=0.5)
    print(f"  Happiness: {state.happiness:.4f}")
    print(f"  Sadness: {state.sadness:.4f}")
    print(f"  Content: {state.content:.4f}")
    print(f"  Overall mood: {state.overall_mood:.4f}")
    
    # Simulate several positive experiences
    print(f"\n--- After 10 more positive experiences ---")
    for _ in range(10):
        mood_system.update(td_error=0.5, advantage=0.3)
    print(f"  Overall mood: {mood_system.get_mood():.4f}")
    print(f"  Mood value: {mood_system.get_mood_value():.4f}")
    
    # Simulate negative surprise (bad outcome)
    print(f"\n--- Negative surprise (TD error = -2.0) ---")
    state = mood_system.update(td_error=-2.0, advantage=-0.8)
    print(f"  Happiness: {state.happiness:.4f}")
    print(f"  Sadness: {state.sadness:.4f}")
    print(f"  Anger: {state.anger:.4f}")
    print(f"  Overall mood: {state.overall_mood:.4f}")
    
    # Test exploration boost
    print(f"\n--- Exploration boost ---")
    base_epsilon = 0.1
    
    # With negative mood
    mood_system.state.mood_action = -0.3
    boosted = mood_system.get_exploration_boost(base_epsilon)
    print(f"  Base epsilon: {base_epsilon}")
    print(f"  Mood action: {mood_system.state.mood_action}")
    print(f"  Boosted epsilon: {boosted:.4f}")
    
    # With positive mood
    mood_system.state.mood_action = 0.3
    reduced = mood_system.get_exploration_boost(base_epsilon)
    print(f"  Mood action: {mood_system.state.mood_action}")
    print(f"  Reduced epsilon: {reduced:.4f}")
    
    # Test value bias
    print(f"\n--- Value bias ---")
    mood_system.state.overall_mood = 0.2
    bias = mood_system.get_value_bias(generalization_strength=0.5)
    print(f"  Overall mood: {mood_system.state.overall_mood}")
    print(f"  Value bias: {bias:.4f}")
    
    # Test episode reset
    print(f"\n--- Episode reset ---")
    print(f"  Mood before reset: {mood_system.get_mood():.4f}")
    mood_system.reset()
    print(f"  Mood after reset: {mood_system.get_mood():.4f}")
    
    # Summary
    print(f"\n--- Summary stats ---")
    summary = mood_system.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✓ MoodSystem works!")