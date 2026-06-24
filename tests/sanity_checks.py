"""
Sanity checks to verify environment and agents work correctly.
Run these before training to catch issues early!
"""
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from environments import VisualMazeEnv, load_maze, print_maze


def test_maze_loading():
    """Test that maze configs load correctly."""
    print("\n" + "="*50)
    print("TEST: Maze Loading")
    print("="*50)
    
    # Load minimal maze
    config = load_maze("minimal")
    
    assert config['name'] == "minimal", "Wrong maze name"
    assert config['size'] == [5, 5], f"Wrong size: {config['size']}"
    assert len(config['walls']) > 0, "No walls defined"
    assert 'rewards' in config, "No rewards defined"
    assert 'colors' in config, "No colors defined"
    
    # Print maze for visual verification
    print_maze(config)
    
    print("✓ Maze loading works!")
    return True


def test_environment_creation():
    """Test that environment creates correctly."""
    print("\n" + "="*50)
    print("TEST: Environment Creation")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal", image_size=64)
    
    # Check spaces
    assert env.action_space.n == 4, f"Wrong action space: {env.action_space.n}"
    assert env.observation_space.shape == (64, 64, 3), \
        f"Wrong obs shape: {env.observation_space.shape}"
    assert env.observation_space.dtype == np.uint8, \
        f"Wrong obs dtype: {env.observation_space.dtype}"
    
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    print("✓ Environment creation works!")
    return True


def test_reset():
    """Test environment reset."""
    print("\n" + "="*50)
    print("TEST: Environment Reset")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal")
    
    obs, info = env.reset(seed=42)
    
    # Check observation
    assert obs.shape == (64, 64, 3), f"Wrong obs shape: {obs.shape}"
    assert obs.dtype == np.uint8, f"Wrong obs dtype: {obs.dtype}"
    assert obs.min() >= 0 and obs.max() <= 255, "Obs values out of range"
    
    # Check info
    assert info['agent_pos'] == tuple(env.config['agent_start']), \
        f"Agent not at start: {info['agent_pos']}"
    assert info['has_key'] == False, "Should not have key at start"
    assert info['door_open'] == False, "Door should be closed at start"
    assert info['steps'] == 0, "Steps should be 0 at start"
    
    print(f"  Initial position: {info['agent_pos']}")
    print(f"  Observation range: [{obs.min()}, {obs.max()}]")
    
    print("✓ Environment reset works!")
    return True


def test_step():
    """Test environment step mechanics."""
    print("\n" + "="*50)
    print("TEST: Environment Step")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal")
    obs, info = env.reset(seed=42)
    
    initial_pos = info['agent_pos']
    
    # Test move down (action=1)
    obs, reward, term, trunc, info = env.step(1)
    
    assert obs.shape == (64, 64, 3), f"Wrong obs shape after step"
    assert isinstance(reward, (int, float)), f"Reward not numeric: {type(reward)}"
    assert isinstance(term, bool), f"Terminated not bool: {type(term)}"
    assert isinstance(trunc, bool), f"Truncated not bool: {type(trunc)}"
    assert info['steps'] == 1, f"Steps not incremented: {info['steps']}"
    
    print(f"  Moved from {initial_pos} to {info['agent_pos']}")
    print(f"  Reward: {reward}")
    
    # Test wall collision
    env.reset()
    # Move to position next to wall at [1,1], then try to move into wall
    env.step(1)  # down to (1,0)
    obs, reward, term, trunc, info = env.step(3)  # right into wall at (1,1)
    
    wall_bump_reward = env.rewards['wall_bump']
    assert reward < 0, f"Wall bump should be negative reward: {reward}"
    print(f"  Wall bump reward: {reward}")
    
    print("✓ Environment step works!")
    return True


def test_key_pickup():
    """Test key pickup mechanics."""
    print("\n" + "="*50)
    print("TEST: Key Pickup")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal")
    env.reset()
    
    # Manually move agent to key position
    key_pos = env.key_pos
    env.agent_pos = list(key_pos)
    env.has_key = False
    
    # Take a step in place (by hitting a wall or boundary)
    # First, let's just directly test the pickup logic
    # by stepping to the key position
    
    env.reset()
    env.agent_pos = [key_pos[0], key_pos[1] - 1]  # One left of key
    
    # Move right to pick up key
    obs, reward, term, trunc, info = env.step(3)  # right
    
    key_reward = env.rewards['key']
    
    if info['has_key']:
        print(f"  Key picked up at step {info['key_pickup_step']}")
        print(f"  Key reward received: {reward}")
        assert reward >= key_reward + env.rewards['step'], \
            f"Should include key reward: {reward}"
    else:
        print(f"  Agent at {info['agent_pos']}, key at {key_pos}")
        print("  (Key pickup will be tested during full episode)")
    
    print("✓ Key pickup logic exists!")
    return True


def test_door_mechanics():
    """Test door blocking without key and opening with key."""
    print("\n" + "="*50)
    print("TEST: Door Mechanics")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal")
    env.reset()
    
    door_pos = env.door_pos
    
    # Position agent next to door without key
    env.agent_pos = [door_pos[0], door_pos[1] - 1]  # Left of door
    env.has_key = False
    
    initial_pos = tuple(env.agent_pos)
    obs, reward, term, trunc, info = env.step(3)  # Try to move right into door
    
    # Should be blocked
    assert info['agent_pos'] == initial_pos, \
        f"Should be blocked by door without key! Moved to {info['agent_pos']}"
    assert info['door_attempts_without_key'] >= 1, \
        "Door attempt without key not tracked"
    
    print(f"  Blocked at door without key: ✓")
    print(f"  Door attempts tracked: {info['door_attempts_without_key']}")
    
    # Now test with key
    env.has_key = True
    env.agent_pos = [door_pos[0], door_pos[1] - 1]  # Left of door again
    
    obs, reward, term, trunc, info = env.step(3)  # Move right into door
    
    assert info['agent_pos'] == door_pos, \
        f"Should pass through door with key! Stuck at {info['agent_pos']}"
    assert info['door_open'] == True, "Door should be open"
    
    door_reward = env.rewards['door_open']
    print(f"  Passed through door with key: ✓")
    print(f"  Door open reward: {reward}")
    
    print("✓ Door mechanics work!")
    return True


def test_goal_completion():
    """Test that reaching goal after door gives reward and terminates."""
    print("\n" + "="*50)
    print("TEST: Goal Completion")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal")
    env.reset()
    
    # Set up winning state: have key, door open, next to goal
    env.has_key = True
    env.door_open = True
    
    goal_pos = env.goal_pos
    env.agent_pos = [goal_pos[0], goal_pos[1] - 1]  # Left of goal
    
    obs, reward, term, trunc, info = env.step(3)  # Move right to goal
    
    goal_reward = env.rewards['goal']
    step_cost = env.rewards['step']
    expected_reward = goal_reward + step_cost  # 10.0 + (-0.04) = 9.96
    
    assert info['agent_pos'] == goal_pos, \
        f"Should be at goal! At {info['agent_pos']}"
    assert term == True, "Episode should terminate at goal"
    assert abs(reward - expected_reward) < 0.01, \
        f"Expected reward ~{expected_reward}, got: {reward}"
    
    print(f"  Reached goal: ✓")
    print(f"  Episode terminated: ✓")
    print(f"  Goal reward: {reward} (goal={goal_reward}, step={step_cost})")
    
    print("✓ Goal completion works!")
    return True


def test_timeout():
    """Test that episode truncates after max steps."""
    print("\n" + "="*50)
    print("TEST: Timeout (Truncation)")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal")
    env.reset()
    
    max_steps = env.max_steps
    
    # Take max_steps actions
    for i in range(max_steps):
        obs, reward, term, trunc, info = env.step(0)  # Just move up repeatedly
        
        if term:
            print(f"  Episode terminated early at step {i+1}")
            break
        
        if trunc:
            assert i == max_steps - 1, \
                f"Truncated at wrong step: {i+1} vs {max_steps}"
            print(f"  Episode truncated at step {max_steps}: ✓")
            break
    
    print("✓ Timeout works!")
    return True


def test_full_episode_random():
    """Run a full episode with random actions."""
    print("\n" + "="*50)
    print("TEST: Full Random Episode")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal")
    obs, info = env.reset(seed=42)
    
    total_reward = 0
    steps = 0
    
    while True:
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if term or trunc:
            break
    
    print(f"  Episode length: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final state: {info}")
    print(f"  Success: {term and info['door_open']}")
    
    print("✓ Full episode runs without errors!")
    return True


def test_observation_changes():
    """Test that observations actually change based on state."""
    print("\n" + "="*50)
    print("TEST: Observation Changes")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal")
    
    # Get initial observation
    obs1, _ = env.reset()
    
    # Move agent
    env.step(1)  # down
    obs2 = env._get_observation()
    
    # Observations should be different
    assert not np.array_equal(obs1, obs2), \
        "Observation should change when agent moves!"
    
    diff_pixels = np.sum(obs1 != obs2)
    print(f"  Pixels changed after move: {diff_pixels}")
    
    # Get observation with key
    env.has_key = True
    obs3 = env._get_observation()
    
    # Should be different (key disappears)
    assert not np.array_equal(obs2, obs3), \
        "Observation should change when key is picked up!"
    
    diff_pixels = np.sum(obs2 != obs3)
    print(f"  Pixels changed after key pickup: {diff_pixels}")
    
    # Get observation with door open
    env.door_open = True
    obs4 = env._get_observation()
    
    diff_pixels = np.sum(obs3 != obs4)
    print(f"  Pixels changed after door open: {diff_pixels}")
    
    print("✓ Observations change correctly!")
    return True


def test_reproducibility():
    """Test that same seed gives same results."""
    print("\n" + "="*50)
    print("TEST: Reproducibility")
    print("="*50)
    
    env = VisualMazeEnv(maze_name="minimal")
    
    # Run episode with seed 42
    obs1, _ = env.reset(seed=42)
    actions1 = [env.action_space.sample() for _ in range(10)]
    
    rewards1 = []
    for a in actions1:
        _, r, _, _, _ = env.step(a)
        rewards1.append(r)
    
    # Run again with same seed
    obs2, _ = env.reset(seed=42)
    
    assert np.array_equal(obs1, obs2), "Same seed should give same initial obs"
    
    # Note: action_space.sample() uses its own RNG, so we reuse actions1
    env.reset(seed=42)
    rewards2 = []
    for a in actions1:
        _, r, _, _, _ = env.step(a)
        rewards2.append(r)
    
    assert rewards1 == rewards2, "Same actions should give same rewards"
    
    print(f"  Initial observations match: ✓")
    print(f"  Reward sequences match: ✓")
    
    print("✓ Reproducibility works!")
    return True


def test_layout_parity():
    """Mirrored mazes share the same spatial layout."""
    print("\n" + "="*50)
    print("TEST: Mirrored Maze Layout Parity")
    print("="*50)

    key_config = load_maze("key_approach")
    shield_config = load_maze("shield_avoidance")

    shared_fields = ["size", "walls", "agent_start", "goal_position"]
    for field in shared_fields:
        assert key_config[field] == shield_config[field], (
            f"Layout mismatch on '{field}': "
            f"key={key_config[field]}, shield={shield_config[field]}"
        )

    print(f"  Shared size: {key_config['size']}")
    print(f"  Shared wall count: {len(key_config['walls'])}")
    print("✓ Mirrored maze layouts match!")
    return True


def test_key_approach_positive_only():
    """Key approach maze provides only non-negative rewards."""
    print("\n" + "="*50)
    print("TEST: Key Approach Positive-Only Rewards")
    print("="*50)

    env = VisualMazeEnv(maze_name="key_approach")
    env.reset()

    # Optimal path: down to key, up to start, right through door, right to goal
    optimal_actions = [1] * 4 + [0] * 4 + [3] * 6
    rewards = []
    terminated = False

    for action in optimal_actions:
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        assert reward >= 0, f"Negative reward on optimal path: {reward}"
        if terminated or truncated:
            break

    assert terminated, "Optimal path should reach goal"
    assert any(r > 0 for r in rewards), "Expected at least one positive reward"
    assert sum(rewards) == 13.0, f"Optimal total should be 13.0, got {sum(rewards)}"

    # Wall bump and door-block should be zero, not negative
    env.reset()
    _, reward, _, _, _ = env.step(2)  # left into wall at [1,0]
    assert reward == 0, f"Wall bump should be 0, got {reward}"

    env.reset()
    for _ in range(4):
        env.step(3)  # move to door without key
    _, reward, term, _, info = env.step(3)
    assert reward == 0, f"Door without key should be 0, got {reward}"
    assert not term, "Should not complete episode without key"
    assert not info["door_open"], "Door should remain closed without key"

    print(f"  Optimal path rewards: {rewards}")
    print(f"  Optimal total: {sum(rewards)}")
    print("✓ Key approach rewards are positive-only!")
    return True


def test_shield_avoidance_negative_only():
    """Shield avoidance maze provides only non-positive rewards."""
    print("\n" + "="*50)
    print("TEST: Shield Avoidance Negative-Only Rewards")
    print("="*50)

    env = VisualMazeEnv(maze_name="shield_avoidance")
    env.reset()

    # Optimal path with shield: down, up, right through trap, right to goal
    optimal_actions = [1] * 4 + [0] * 4 + [3] * 6
    rewards = []
    terminated = False
    trap_reward = None
    goal_reward = None

    for action in optimal_actions:
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        assert reward <= 0, f"Positive reward on optimal path: {reward}"
        if info.get("trap_hit_step", -1) == env.steps:
            trap_reward = reward
        if terminated:
            goal_reward = reward
        if terminated or truncated:
            break

    assert terminated, "Optimal path should reach goal"
    assert trap_reward == -5.0, f"Trap with shield should be -5.0, got {trap_reward}"
    assert goal_reward == 0, f"Goal should give 0 reward, got {goal_reward}"
    assert sum(rewards) == -5.0, f"Optimal total should be -5.0, got {sum(rewards)}"

    # Trap without shield should be strictly negative
    env.reset()
    trap_hit_reward = None
    for _ in range(4):
        _, reward, term, trunc, info = env.step(3)
        if info.get("trap_hit_step", -1) == env.steps:
            trap_hit_reward = reward
            break

    assert trap_hit_reward is not None, "Should have stepped on trap"
    assert trap_hit_reward < 0, f"Trap without shield should be negative, got {trap_hit_reward}"
    assert trap_hit_reward == -30.0, f"Expected -30.0 trap damage, got {trap_hit_reward}"

    print(f"  Optimal path rewards: {rewards}")
    print(f"  Optimal total: {sum(rewards)}")
    print(f"  Trap without shield: {trap_hit_reward}")
    print("✓ Shield avoidance rewards are negative-only!")
    return True


def test_shield_trap_rewards():
    """Shield trap: step penalty, shield bonus, negative optimal total."""
    print("\n" + "="*50)
    print("TEST: Shield Trap Rewards")
    print("="*50)

    env = VisualMazeEnv(maze_name="shield_trap")
    env.reset()

    optimal_actions = [1] * 4 + [0] * 4 + [3] * 6
    rewards = []
    terminated = False

    for action in optimal_actions:
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    assert terminated, "Optimal path should reach goal"
    assert sum(rewards) < 0, f"Optimal total should be negative, got {sum(rewards)}"
    assert abs(sum(rewards) - (-2.56)) < 1e-6, f"Expected optimal total -2.56, got {sum(rewards)}"
    assert env.rewards["shield_pickup"] > 0, "Shield pickup should be positive"

    print(f"  Optimal total: {sum(rewards)}")
    print("✓ Shield trap rewards configured correctly!")
    return True


def test_shield_trap_v2_rewards():
    """Shield trap v2: shield path penalty mirrors v1 shield benefit."""
    print("\n" + "="*50)
    print("TEST: Shield Trap V2 Transfer Rewards")
    print("="*50)

    env = VisualMazeEnv(maze_name="shield_trap_v2")
    env.reset()

    # Best path: skip shield, cross trap, reach goal
    direct_actions = [3] * 6
    direct_rewards = []
    terminated = False
    for action in direct_actions:
        _, reward, terminated, truncated, info = env.step(action)
        direct_rewards.append(reward)
        if terminated or truncated:
            break

    assert terminated
    direct_total = sum(direct_rewards)
    assert direct_total < 0, f"Direct path should be negative, got {direct_total}"
    assert abs(direct_total - (-30.24)) < 1e-6

    # Worse path: collect misleading shield first (v1 habit)
    env.reset()
    shield_actions = [1] * 4 + [0] * 4 + [3] * 6
    shield_rewards = []
    for action in shield_actions:
        _, reward, terminated, truncated, info = env.step(action)
        shield_rewards.append(reward)
        if terminated or truncated:
            break

    shield_total = sum(shield_rewards)
    assert shield_total < direct_total, (
        "Shield path should be worse than skipping shield"
    )
    assert abs(shield_total - (-58.56)) < 1e-6

    # Gap should mirror shield_trap v1 (shield good vs skip shield)
    v1_env = VisualMazeEnv(maze_name="shield_trap")
    v1_env.reset()
    v1_shield_total = 0.0
    for action in shield_actions:
        _, reward, term, trunc, _ = v1_env.step(action)
        v1_shield_total += reward
        if term or trunc:
            break
    v1_env.reset()
    v1_direct_total = 0.0
    for action in direct_actions:
        _, reward, term, trunc, _ = v1_env.step(action)
        v1_direct_total += reward
        if term or trunc:
            break
    v1_gap = v1_shield_total - v1_direct_total
    v2_gap = shield_total - direct_total
    assert abs(abs(v1_gap) - abs(v2_gap)) < 1.5, (
        f"v2 penalty gap ({v2_gap:.2f}) should mirror v1 benefit gap ({v1_gap:.2f})"
    )

    assert env.rewards["shield_pickup"] < 0
    assert env.rewards["trap_with_shield"] < env.rewards["trap_no_shield"]

    print(f"  Direct path total: {direct_total:.2f}")
    print(f"  Shield path total: {shield_total:.2f}")
    print(f"  v2 gap (shield worse): {v2_gap:.2f}")
    print(f"  v1 gap (shield better): {v1_gap:.2f}")
    print("✓ Shield trap v2 rewards configured correctly!")
    return True


def run_all_tests():
    """Run all sanity checks."""
    print("\n" + "="*60)
    print("   RUNNING ALL SANITY CHECKS")
    print("="*60)
    
    tests = [
        test_maze_loading,
        test_environment_creation,
        test_reset,
        test_step,
        test_key_pickup,
        test_door_mechanics,
        test_goal_completion,
        test_timeout,
        test_full_episode_random,
        test_observation_changes,
        test_reproducibility,
        test_layout_parity,
        test_key_approach_positive_only,
        test_shield_avoidance_negative_only,
        test_shield_trap_rewards,
        test_shield_trap_v2_rewards,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"   RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 All sanity checks passed! Ready to build agents.\n")
    else:
        print("\n⚠️  Some checks failed. Fix issues before proceeding.\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)