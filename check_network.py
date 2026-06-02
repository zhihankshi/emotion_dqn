"""Verify target network implementation per DQN paper."""
import torch
from agents import DQNAgent, EmotionalDQNAgent


def check_agent(agent, name):
    print(f"\n{'='*60}")
    print(f"CHECKING {name}")
    print(f"{'='*60}")
    
    # Check 1: Does target network exist?
    has_target = hasattr(agent, 'target_net') and agent.target_net is not None
    print(f"1. Has target_net: {has_target}")
    
    if not has_target:
        print("   ❌ MISSING TARGET NETWORK!")
        return False
    
    # Check 2: Are policy_net and target_net different objects?
    same_object = agent.policy_net is agent.target_net
    print(f"2. policy_net and target_net are different objects: {not same_object}")
    
    if same_object:
        print("   ❌ SAME OBJECT - should be separate!")
        return False
    
    # Check 3: Do they have the same architecture?
    policy_params = sum(p.numel() for p in agent.policy_net.parameters())
    target_params = sum(p.numel() for p in agent.target_net.parameters())
    same_arch = policy_params == target_params
    print(f"3. Same architecture: {same_arch} ({policy_params} params)")
    
    # Check 4: Is target network in eval mode?
    target_training = agent.target_net.training
    print(f"4. Target net in eval mode: {not target_training}")
    
    if target_training:
        print("   ⚠️ Target net should be in eval mode!")
    
    # Check 5: Target update frequency
    has_update_freq = hasattr(agent, 'target_update_freq')
    print(f"5. Has target_update_freq: {has_update_freq}")
    
    if has_update_freq:
        print(f"   Update frequency: {agent.target_update_freq} steps")
    else:
        print("   ❌ MISSING target_update_freq!")
    
    # Check 6: Are weights currently the same or different?
    policy_weight = list(agent.policy_net.parameters())[0].data.clone()
    target_weight = list(agent.target_net.parameters())[0].data.clone()
    weights_same = torch.allclose(policy_weight, target_weight)
    print(f"6. Weights currently identical: {weights_same}")
    print("   (Should be same at init, may differ after training)")
    
    # Check 7: Verify update method uses target_net for next Q values
    print(f"7. Checking update method...")
    
    import inspect
    if hasattr(agent, 'update'):
        source = inspect.getsource(agent.update)
        uses_target = 'target_net' in source
        print(f"   Uses target_net in update: {uses_target}")
        
        if not uses_target:
            print("   ❌ Update method should use target_net for next Q values!")
    
    print(f"\n{'='*60}")
    
    return True


# Create agents and check
print("Creating agents...")

baseline = DQNAgent(
    observation_shape=(64, 64, 3),
    n_actions=4
)
check_agent(baseline, "BASELINE DQN")

emotional = EmotionalDQNAgent(
    observation_shape=(64, 64, 3),
    n_actions=4
)
check_agent(emotional, "EMOTIONAL DQN")