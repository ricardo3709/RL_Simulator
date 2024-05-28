from multiprocessing import Pool
import cProfile
from main import run_sim
import itertools
from RL.train import warm_up_train

# Define variable values
REWARD_THETA = [0.25, 0.75, 1.25, 1.75]
REWARD_TYPE = ['GEN', 'REJ']
NODE_LAYERS = [1, 2, 3]
MOVING_AVG_WINDOW = [20, 60, 120]


# Generate all combinations of the variables
combinations = list(itertools.product(REWARD_THETA, REWARD_TYPE, NODE_LAYERS, MOVING_AVG_WINDOW))

# Create a list of dictionaries for each test case
test_cases = [{'REWARD_THETA': t, 'REWARD_TYPE': r, 'NODE_LAYERS': n, 'MOVING_AVG_WINDOW': m}
              for t, r, n, m in combinations]

def process_function():
    warm_up_train()  # Adjust if run_sim expects different parameters
    # Optionally use cProfile here if profiling is necessary

def multi_process_test():
    # Using a pool of workers equal to the number of cores, 4 cores
    with Pool(4) as pool:
        pool.map(process_function)

if __name__ == "__main__":
    multi_process_test()
