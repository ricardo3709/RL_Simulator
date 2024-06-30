import gym
from gym import spaces
import numpy as np
from collections import deque
from src.simulator.Simulator_platform import Simulator_Platform
from src.simulator.config import *

class ManhattanTrafficEnv(gym.Env):
    """定义曼哈顿路网模拟环境"""
    
    def __init__(self):
        super(ManhattanTrafficEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(1,), dtype=np.float32)
        
        # define the observation space, shape is number of features
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
        
        # initial status
        self.state = None
        self.config = ConfigManager()
        self.simulator = Simulator_Platform(0, self.config)  # Sim start time, ConfigManager

        # self.n_steps_delay = (self.config.get("RL_DURATION")/(RL_STEP_LENGTH*TIME_STEP))*2 # delay for two whole simulation
        # self.past_actions = [deque(maxlen=self.n_steps_delay)]
        self.past_actions = []
        # self.past_rejections = deque(maxlen=self.n_steps_delay)
        self.past_rejections = []

        self.decay_factor = self.config.get('DECAY_FACTOR')

        self.old_avg_rej = 0.0

        self.Rmax = 0.5 # maximum reward
        self.Rmin = -0.5 # minimum reward

        # initialize the config
        self.init_config({'REWARD_THETA': 5.0, 'REWARD_TYPE': 'REJ', 'NODE_LAYERS': 2, 'MOVING_AVG_WINDOW': 40, 'DECAY_FACTOR': 1.0})

    def init_config(self, args: dict):
        self.change_config(self.config, args) # change the config based on the args 

        REWARD_THETA = self.config.get('REWARD_THETA')
        REWARD_TYPE = self.config.get('REWARD_TYPE')
        NODE_LAYERS = self.config.get('NODE_LAYERS')
        MOVING_AVG_WINDOW = self.config.get('MOVING_AVG_WINDOW')
        DECAY_FACTOR = self.config.get('DECAY_FACTOR')
        SIMULATION_DURATION = self.config.get('RL_DURATION')

        print(f"[INFO] Running simulation with reward theta: {REWARD_THETA}")
        print(f"[INFO] Running simulation with reward type: {REWARD_TYPE}")
        print(f"[INFO] Running simulation with node layers: {NODE_LAYERS}")
        print(f"[INFO] Running simulation with moving average window: {MOVING_AVG_WINDOW}")
        print(f"[INFO] Running simulation with decay factor: {DECAY_FACTOR}")
        print(f"[INFO] Running simulation with duration: {SIMULATION_DURATION}")
    
    def change_config(self, config: ConfigManager, args:list):
        for variable in args.keys():
            value = args[variable]
            config.set(variable, value)

    def warm_up_step(self):
        current_rejection_rate = np.mean(self.simulator.current_cycle_rej_rate)
        self.past_rejections.append(current_rejection_rate)
        done = self.simulator.is_warm_up_done()
        if done:
            return done, self.past_rejections
        return done, None

    def step(self, action):
        # calculate reward
        # record the past actions and rejections
        self.past_actions.append(action)
        current_rejection_rate = np.mean(self.simulator.current_cycle_rej_rate)
        self.past_rejections.append(current_rejection_rate)

        # if len(self.past_rejections) >= self.n_steps_delay: # delay is over
        #     reward = self.calculate_reward(self.past_rejections)

        # else:
        #     reward = 0.0000001  # minimum reward until the delay is over

        reward = self.calculate_reward(self.past_rejections)
        print(f"Reward: {reward}")
        print(f"current rejection rate: {current_rejection_rate}")
        print(f"current action: {action}")

        new_theta = self.simulator.update_theta(action)
        
        # get new state
        # self.state = self.simulator.get_simulator_state_by_areas()
        
        done = self.simulator.is_done()

        return reward, done, new_theta

    def reset(self):
        # random reset simulator
        # self.simulator.random_reset_simulator()
        self.simulator.uniform_reset_simulator()
        self.state, network = self.simulator.get_simulator_state_by_areas()
        return self.state, network
    
    def calculate_reward(self, past_rejections):
        if len(past_rejections) < 2:
            return 0.0
        return past_rejections[-2] - past_rejections[-1]
    
    def calculate_reward_ori(self, past_rejections):
        LEARNING_WINDOW = self.config.get('LEARNING_WINDOW')
        CONSIDER_NUM_CYCLES = self.config.get('CONSIDER_NUM_CYCLES')
        CYCLE_WINDOW = int(LEARNING_WINDOW/(RL_STEP_LENGTH*TIME_STEP))

        cycle_reward = 0.0
        old_avg_rej = np.mean(past_rejections[-CYCLE_WINDOW*2 : -CYCLE_WINDOW]) # last 60mins to last 30mins
        current_avg_rej = np.mean(past_rejections[-CYCLE_WINDOW:]) # last 30mins
        current_cycle_rej = past_rejections[-1]

        if current_cycle_rej > 1: #bug handle
            return 0.0

        cycle_weight = 0.9
        long_weight = 0.1

        for cycle in range(1, CONSIDER_NUM_CYCLES+1):
            last_cycle_rej = past_rejections[-(cycle+1)]
            cycle_reward += (last_cycle_rej - current_cycle_rej) * (self.decay_factor ** cycle)
        # # Calculate reward based on past rejections
        # for i in range(len(past_rejections)-1): 
        #     old_avg_rej += past_rejections[i] * (self.decay_factor ** i) 
        # reward = old_avg_rej - current_rej # reward is positive if the current rejection rate is lower than the past average
        long_reward = (old_avg_rej - current_avg_rej) 
        combined_reward = cycle_weight * cycle_reward + long_weight * long_reward
        normalized_reward = 2 * ((combined_reward - self.Rmin) / (self.Rmax - self.Rmin)) - 1

        return normalized_reward

    # def render(self, mode='console'):
    #     if mode == 'console':
    #         print(f"Current state: {self.state}")
    #     else:
    #         raise NotImplementedError("Only console mode is supported")
