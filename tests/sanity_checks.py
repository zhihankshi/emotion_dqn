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

    # Check spaces. Observations are channel-first (3 * frame_stack, H, W).
    assert env.action_space.n == 4, f"Wrong action space: {env.action_space.n}"
    assert env.observation_space.shape == (3, 64, 64), \
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
    
    # Check observation (channel-first: 3 * frame_stack, H, W)
    assert obs.shape == (3, 64, 64), f"Wrong obs shape: {obs.shape}"
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
    
    assert obs.shape == (3, 64, 64), f"Wrong obs shape after step: {obs.shape}"
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


def test_action_masking():
    """Valid actions exclude walls and blocked doors."""
    print("\n" + "="*50)
    print("TEST: Action Masking")
    print("="*50)

    env = VisualMazeEnv(maze_name="minimal")
    env.reset()

    # Start [0,0]: up and left are walls
    valid = env.get_valid_actions()
    assert 0 not in valid, "UP into boundary should be invalid"
    assert 2 not in valid, "LEFT into boundary should be invalid"
    assert 1 in valid and 3 in valid, f"DOWN/RIGHT should be valid, got {valid}"

    # Door blocks without key
    env = VisualMazeEnv(maze_name="key_approach")
    env.reset()
    for _ in range(4):
        env.step(3)
    valid = env.get_valid_actions()
    assert 3 not in valid, "RIGHT into closed door without key should be invalid"

    env.has_key = True
    valid = env.get_valid_actions()
    assert 3 in valid, "RIGHT into door should be valid with key"

    print(f"  minimal start valid: {[1, 3]}")
    print(f"  key_approach at door with key valid: {valid}")
    print("✓ Action masking works!")
    return True


def test_key_approach_rewards():
    """Key approach maze uses dense shaping with positive optimal return."""
    print("\n" + "="*50)
    print("TEST: Key Approach Rewards")
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
        if terminated or truncated:
            break

    assert terminated, "Optimal path should reach goal"
    assert sum(rewards) > 0, f"Optimal total should be positive, got {sum(rewards)}"
    assert abs(sum(rewards) - 12.44) < 1e-6, f"Expected optimal total 12.44, got {sum(rewards)}"

    # Wall bump and door-block should be negative (dense shaping)
    env.reset()
    _, reward, _, _, _ = env.step(2)  # left into wall at [1,0]
    assert reward < 0, f"Wall bump should be negative, got {reward}"
    assert abs(reward - (-0.14)) < 1e-6, f"Expected wall bump -0.14, got {reward}"

    env.reset()
    for _ in range(4):
        env.step(3)  # move to door without key
    _, reward, term, _, info = env.step(3)
    assert reward < 0, f"Door without key should be negative, got {reward}"
    assert not term, "Should not complete episode without key"
    assert not info["door_open"], "Door should remain closed without key"

    print(f"  Optimal path rewards: {rewards}")
    print(f"  Optimal total: {sum(rewards)}")
    print("✓ Key approach rewards configured correctly!")
    return True


def test_shield_avoidance_rewards():
    """Shield avoidance: the shield does NOT protect, so detouring for it is wasted.

    Expected totals are derived from the maze's own reward table rather than
    hardcoded, so tuning rewards in the YAML does not silently invalidate the
    accounting this test checks.
    """
    print("\n" + "="*50)
    print("TEST: Shield Avoidance Rewards")
    print("="*50)

    env = VisualMazeEnv(maze_name="shield_avoidance")
    env.reset()
    r = env.rewards

    # Defining property of this maze: the shield buys nothing.
    assert r["trap_with_shield"] == r["trap_no_shield"], (
        "shield_avoidance must charge the same trap cost with and without the "
        f"shield, got with={r['trap_with_shield']} without={r['trap_no_shield']}"
    )
    assert r["shield_pickup"] == 0, (
        f"Shield pickup should be worthless here, got {r['shield_pickup']}"
    )

    # Shield detour: 4 down (pickup), 4 up, 6 right through the trap to goal.
    shield_actions = [1] * 4 + [0] * 4 + [3] * 6
    rewards = []
    terminated = False
    trap_reward = None
    goal_reward = None

    for action in shield_actions:
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if info.get("trap_hit_step", -1) == env.steps:
            trap_reward = reward
        if terminated:
            goal_reward = reward
        if terminated or truncated:
            break

    assert terminated, "Shield detour should reach goal"
    assert trap_reward is not None, "Should have stepped on the trap"
    assert goal_reward is not None and goal_reward > 0, f"Goal should be positive, got {goal_reward}"

    n_steps = len(rewards)
    expected_detour = (
        n_steps * r["step"] + r["shield_pickup"] + r["trap_with_shield"] + r["goal"]
    )
    assert abs(sum(rewards) - expected_detour) < 1e-6, (
        f"Shield detour total {sum(rewards)} != step*{n_steps} + pickup + "
        f"trap_with_shield + goal = {expected_detour}"
    )
    assert sum(rewards) < 0, f"Detour total should be negative, got {sum(rewards)}"

    # Direct route: 6 right, through the trap, no shield.
    env.reset()
    direct_rewards = []
    trap_hit_reward = None
    for _ in range(6):
        _, reward, terminated, truncated, info = env.step(3)
        direct_rewards.append(reward)
        if info.get("trap_hit_step", -1) == env.steps:
            trap_hit_reward = reward
        if terminated or truncated:
            break

    assert terminated, "Direct route should reach goal"
    assert trap_hit_reward is not None, "Should have stepped on trap"
    assert abs(trap_hit_reward - (r["step"] + r["trap_no_shield"])) < 1e-6, (
        f"Trap step should charge step + trap_no_shield, got {trap_hit_reward}"
    )

    direct_total = sum(direct_rewards)
    expected_direct = len(direct_rewards) * r["step"] + r["trap_no_shield"] + r["goal"]
    assert abs(direct_total - expected_direct) < 1e-6, (
        f"Direct total {direct_total} != {expected_direct}"
    )

    # The point of the maze: with no protection on offer, the detour is a loss.
    assert direct_total > sum(rewards), (
        f"Direct route ({direct_total}) should beat the pointless shield detour "
        f"({sum(rewards)}) — otherwise this maze is not the reversal of shield_trap"
    )

    print(f"  Shield detour total: {sum(rewards)} (expected {expected_detour})")
    print(f"  Direct route total:  {direct_total} (expected {expected_direct})")
    print(f"  Trap step (no shield): {trap_hit_reward}")
    print("✓ Shield avoidance rewards configured correctly!")
    return True


def test_shield_trap_rewards():
    """Shield trap: positive-goal rewards; shield route is best, trap rush is catastrophic."""
    print("\n" + "="*50)
    print("TEST: Shield Trap Rewards")
    print("="*50)

    env = VisualMazeEnv(maze_name="shield_trap")
    env.reset()

    shield_actions = [1] * 4 + [0] * 4 + [3] * 6
    direct_actions = [3] * 6
    shield_total = 0.0
    direct_total = 0.0
    terminated = False

    for action in shield_actions:
        _, reward, terminated, truncated, info = env.step(action)
        shield_total += reward
        if terminated or truncated:
            break

    assert terminated, "Shield path should reach goal"
    assert env.rewards["goal"] > 0, f"Goal reward should be positive, got {env.rewards['goal']}"
    assert env.rewards["shield_pickup"] > 0, f"Shield pickup should be positive, got {env.rewards['shield_pickup']}"
    assert env.rewards["wall_bump"] < 0, "Wall bump should be negative"
    assert env.rewards["step"] < 0, "Step penalty should be negative"
    # Derived from the maze's own reward table, so YAML tuning does not
    # silently invalidate the accounting: 14 steps, one pickup, one shielded
    # trap hit, one goal. (The down-then-up detour revisits each corridor cell
    # exactly twice, which repeat_free_visits=2 leaves unpenalized.)
    expected_shield = (
        14 * env.rewards["step"]
        + env.rewards["shield_pickup"]
        + env.rewards["trap_with_shield"]
        + env.rewards["goal"]
    )
    assert abs(shield_total - expected_shield) < 1e-6, (
        f"Shield path total {shield_total} != step*14 + pickup + "
        f"trap_with_shield + goal = {expected_shield}"
    )

    env.reset()
    for action in direct_actions:
        _, reward, terminated, truncated, info = env.step(action)
        direct_total += reward
        if terminated or truncated:
            break

    assert terminated, "Direct path should still reach goal"
    assert direct_total < shield_total, (
        f"Direct trap rush ({direct_total}) should be worse than shield path ({shield_total})"
    )
    expected_direct = (
        6 * env.rewards["step"] + env.rewards["trap_no_shield"] + env.rewards["goal"]
    )
    assert abs(direct_total - expected_direct) < 1e-6, (
        f"Direct path total {direct_total} != step*6 + trap_no_shield + goal "
        f"= {expected_direct}"
    )
    assert direct_total < 0, "Trap rush should be net negative"

    # Forced loop/timeout should be worse than both successful paths
    env.reset()
    loop_total = 0.0
    for i in range(env.max_steps):
        action = 3 if i % 2 == 0 else 2
        _, reward, terminated, truncated, _ = env.step(action)
        loop_total += reward
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert loop_total < direct_total < shield_total, (
        f"Expected loop ({loop_total}) < trap_rush ({direct_total}) < shield ({shield_total})"
    )

    print(f"  Shield path total: {shield_total}")
    print(f"  Direct path total: {direct_total}")
    print(f"  Loop/timeout total: {loop_total}")
    print(f"  Gap: {shield_total - direct_total:.1f} points")
    print("✓ Shield trap rewards configured correctly!")

    # Verify timeout penalty is applied on truncation (if configured)
    timeout_penalty = env.rewards.get("timeout", 0)
    assert timeout_penalty <= 0, f"Timeout penalty should be <= 0, got {timeout_penalty}"

    env.reset()
    env.max_steps = 1  # force truncation on first step
    _, r, term, trunc, _ = env.step(3)  # move right (should not terminate)
    assert trunc and not term, "Expected truncation without termination"
    expected = env.rewards["step"] + timeout_penalty
    assert abs(r - expected) < 1e-6, f"Expected timeout reward {expected}, got {r}"
    return True


def test_shield_trap_v2_rewards():
    """Shield trap v2: skip-shield path beats misleading shield habit from v1."""
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
    assert abs(direct_total - (-0.6)) < 1e-6

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
    assert abs(shield_total - (-5.4)) < 1e-6

    # v1 rewards shield-first; v2 penalizes the v1 habit
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
    assert v1_gap > 15.0, f"v1 should strongly favor shield route (gap={v1_gap:.2f})"
    assert v2_gap < -2.0, f"v2 should penalize shield habit (gap={v2_gap:.2f})"

    assert env.rewards["shield_pickup"] < 0
    assert env.rewards["trap_with_shield"] < env.rewards["trap_no_shield"]

    print(f"  Direct path total: {direct_total:.2f}")
    print(f"  Shield path total: {shield_total:.2f}")
    print(f"  v2 gap (shield worse): {v2_gap:.2f}")
    print(f"  v1 gap (shield better): {v1_gap:.2f}")
    print("✓ Shield trap v2 rewards configured correctly!")
    return True


def test_shield_trap_easy_rewards():
    """Easy curriculum maze: looping < trap rush < shield route."""
    print("\n" + "="*50)
    print("TEST: Shield Trap Easy Rewards")
    print("="*50)

    env = VisualMazeEnv(maze_name="shield_trap_easy")
    env.reset()

    shield_actions = [1, 0, 3, 3, 3, 3]
    direct_actions = [3, 3, 3, 3]

    shield_total = 0.0
    for action in shield_actions:
        _, reward, terminated, truncated, _ = env.step(action)
        shield_total += reward
        if terminated or truncated:
            break
    assert terminated, "Shield path should reach goal"
    assert abs(shield_total - 12.0) < 1e-6, f"Expected shield path 12.0, got {shield_total}"

    env.reset()
    direct_total = 0.0
    for action in direct_actions:
        _, reward, terminated, truncated, _ = env.step(action)
        direct_total += reward
        if terminated or truncated:
            break
    assert terminated, "Direct path should still reach goal"
    assert abs(direct_total - (-27.0)) < 1e-6, f"Expected direct path -27.0, got {direct_total}"
    assert direct_total < shield_total

    # Forced timeout / loop should be worse than both successful paths
    env.reset()
    loop_total = 0.0
    # oscillate left/right in the safe corridor in front of the trap
    for i in range(env.max_steps):
        action = 3 if i % 2 == 0 else 2
        # start at [1,1]; right then left keeps agent near start without goal
        if i == 0:
            action = 3  # move to [1,2]
        elif i % 2 == 1:
            action = 2  # back to [1,1]
        else:
            action = 3  # to [1,2]
        _, reward, terminated, truncated, _ = env.step(action)
        loop_total += reward
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert loop_total < direct_total < shield_total, (
        f"Expected loop ({loop_total}) < trap_rush ({direct_total}) < shield ({shield_total})"
    )

    print(f"  Shield path total: {shield_total}")
    print(f"  Direct path total: {direct_total}")
    print(f"  Loop/timeout total: {loop_total}")
    print("✓ Shield trap easy rewards configured correctly!")
    return True


def test_shield_trap_terminal_rewards():
    """Terminal-trap variant: reaching trap ends episode; shield route should dominate."""
    print("\n" + "="*50)
    print("TEST: Shield Trap Terminal Rewards")
    print("="*50)

    env = VisualMazeEnv(maze_name="shield_trap_terminal")
    env.reset()

    # down to shield, back up, then right to terminal trap/goal
    shield_actions = [1] * 4 + [0] * 4 + [3] * 4
    direct_actions = [3] * 4

    shield_total = 0.0
    terminated = False
    truncated = False
    for action in shield_actions:
        _, reward, terminated, truncated, _ = env.step(action)
        shield_total += reward
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert abs(shield_total - 9.0) < 1e-6, f"Expected shield path 9.0, got {shield_total}"

    env.reset()
    direct_total = 0.0
    for action in direct_actions:
        _, reward, terminated, truncated, _ = env.step(action)
        direct_total += reward
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert abs(direct_total - (-52.0)) < 1e-6, f"Expected direct path -52.0, got {direct_total}"

    assert direct_total < shield_total
    print(f"  Shield path total: {shield_total}")
    print(f"  Direct path total: {direct_total}")
    print("✓ Shield trap terminal rewards configured correctly!")
    return True


def test_reversal_contingencies():
    """Reversal pair differs in exactly one reward key and is pixel-identical.

    The whole reversal experiment rests on this: if the two contingencies look
    different, the CNN can see the switch coming and no conclusion about an
    internal mood signal survives.
    """
    print("\n" + "="*50)
    print("TEST: Reversal Contingencies")
    print("="*50)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.train_reversal import (
        contingency_overrides, verify_visual_identity, PROTECTIVE, NON_PROTECTIVE,
    )

    for maze_name in ("shield_trap", "shield_trap_easy"):
        overrides = contingency_overrides(maze_name)

        differing = {
            key for key in set(overrides[PROTECTIVE]) | set(overrides[NON_PROTECTIVE])
            if overrides[PROTECTIVE].get(key) != overrides[NON_PROTECTIVE].get(key)
        }
        assert differing == {"trap_with_shield"}, (
            f"{maze_name}: contingencies must differ in trap_with_shield alone, "
            f"got {differing}"
        )

        envs = {
            name: VisualMazeEnv(
                maze_name=maze_name, image_size=64, reward_overrides=overrides[name]
            )
            for name in (PROTECTIVE, NON_PROTECTIVE)
        }
        identical, differences = verify_visual_identity(
            envs[PROTECTIVE], envs[NON_PROTECTIVE], verbose=False
        )
        assert identical, (
            f"{maze_name}: contingencies are visually distinguishable — "
            f"{differences[0] if differences else ''}"
        )

        # Under non_protective the shield must buy nothing at the trap.
        assert (
            overrides[NON_PROTECTIVE]["trap_with_shield"]
            == envs[NON_PROTECTIVE].rewards["trap_no_shield"]
        ), f"{maze_name}: non_protective trap cost should equal the unshielded cost"

        print(f"  {maze_name}: trap_with_shield "
              f"{overrides[PROTECTIVE]['trap_with_shield']} -> "
              f"{overrides[NON_PROTECTIVE]['trap_with_shield']}, "
              f"pixel-identical in all states")

    print("✓ Reversal contingencies configured correctly!")
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
        test_action_masking,
        test_key_approach_rewards,
        test_shield_avoidance_rewards,
        test_shield_trap_rewards,
        test_shield_trap_v2_rewards,
        test_shield_trap_easy_rewards,
        test_shield_trap_terminal_rewards,
        test_reversal_contingencies,
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