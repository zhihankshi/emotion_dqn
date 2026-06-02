import sys
sys.path.insert(0, '.')

from environments import VisualMazeEnv, print_maze, load_maze

# Print maze
print_maze(load_maze('minimal'))

env = VisualMazeEnv(maze_name="minimal")
obs, info = env.reset()

print(f"\nStarting: {info['agent_pos']}")
print(f"Key at: {env.key_pos}")
print(f"Door at: {env.door_pos}")
print(f"Goal at: {env.goal_pos}")

actions = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

# Optimal path:
# 1. Go right to key: right, right, right, right (4 moves)
# 2. Go down to door: down, down, down (3 moves) + left, left to align
# 3. Go through door: down through door
# 4. Go to goal: already there (door and goal same column)

path = [
    # Get key (go right)
    3, 3, 3, 3,
    # Go to door (down, then to center)
    1, 1,
    2, 2,  # left to column 2
    1,     # down to door
    # Through door to goal
    1,     # down to goal
]

print("\n--- Executing optimal path ---")
total_reward = 0

for i, action in enumerate(path):
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    
    status = []
    if info['has_key']:
        status.append("HAS_KEY")
    if info['door_open']:
        status.append("DOOR_OPEN")
    
    print(f"Step {i+1}: {actions[action]:5s} -> {info['agent_pos']} | reward={reward:+.2f} | {' '.join(status)}")
    
    if term:
        print("\n*** SUCCESS! ***")
        break

print(f"\nTotal reward: {total_reward:.2f}")
print(f"Final info: {info}")