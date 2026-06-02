# test_random_success.py
import numpy as np
from environments import VisualMazeEnv

# Test how often a RANDOM agent succeeds
env = VisualMazeEnv(maze_name="minimal")
successes = 0
n_tests = 100

for i in range(n_tests):
    obs, info = env.reset()
    done = False
    
    while not done:
        action = np.random.randint(4)  # Random action
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
    
    if term and info['door_open']:
        successes += 1

print(f"Random agent success rate: {successes}/{n_tests} = {successes/n_tests:.1%}")