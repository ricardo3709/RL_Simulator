"""
constants are found here
"""
import pickle
import os
from dateutil.parser import parse

##################################################################################
# Changable Attributes
##################################################################################

class ConfigManager:
    def __init__(self):
        self.settings = {
            "REWARD_THETA": 1.0,
            "REWARD_TYPE": 'GEN',# or 'REJ'
            "NODE_LAYERS": 1, # number of layers of rejected rate to consider
            "MOVING_AVG_WINDOW": 12, # 3mins
            "DECAY_FACTOR": 0.9,
            "RL_DURATION": 1800, # 30 minutes = 1800 seconds
        }
    def get(self, key):
        return self.settings[key]
    def set(self, key, value):
        self.settings[key] = value

##################################################################################
# Reinforcement Learning Config
##################################################################################


##################################################################################
# Data File Path
##################################################################################
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# MAP_NAME = "SmallGrid" 
MAP_NAME = "Manhattan"

# small-grid-data
PATH_SMALLGRID_ARCS = f"{ROOT_PATH}/SmallGridData/SmallGrid_Arcs.csv"
PATH_SMALLGRID_REQUESTS = f"{ROOT_PATH}/SmallGridData/SmallGrid_Requests.csv"
PATH_SMALLGRID_TIMECOST = f"{ROOT_PATH}/SmallGridData/SmallGrid_TimeCost.csv"
PATH_SMALLGRID_ALL_PATH_TABLE = f"{ROOT_PATH}/SmallGridData/SmallGrid_AllPathTable.pickle"

NUM_NODES_SMALLGRID = 100

# Manhattan-data
PATH_MANHATTAN_ALL_PATH_MATRIX = f"{ROOT_PATH}/NYC/NYC_Manhattan_AllPathMatrix.pickle"
PATH_MANHATTAN_ALL_PATH_TIME_MATRIX = f"{ROOT_PATH}/NYC/NYC_Manhattan_AllPathTimeMatrix.pickle"
PATH_MANHATTAN_CITYARC = f"{ROOT_PATH}/NYC/NYC_Manhattan_CityArc.pickle"
PATH_MANHATTAN_REQUESTS = f"{ROOT_PATH}/NYC/NYC_Manhattan_Requests.csv"

NUM_NODES_MANHATTAN = 4091

##################################################################################
# Mod System Config
##################################################################################
# dispatch_config
DISPATCHER = "SBA"        # 3 options: SBA, OSP-NR, OSP
REBALANCER = "NJO"        # 3 options: NONE, NPO, NJO

HEURISTIC_ENABLE = True

# for small-grid-data
# FLEET_SIZE = [100]
# VEH_CAPACITY = [4]
# MAX_PICKUP_WAIT_TIME = 5 # 5 min
# MAX_DETOUR_TIME = 10 # 10 min

# for Manhattan-data
FLEET_SIZE = [1000]
VEH_CAPACITY = [6]

MAX_PICKUP_WAIT_TIME = 5*60 # 5 min
MAX_DETOUR_TIME = 10*60 # 10 min

MAX_NUM_VEHICLES_TO_CONSIDER = 20
MAX_SCHEDULE_LENGTH = 30

MAX_DELAY_REBALANCE = 10*60 # 10 min

PENALTY = 800.0 #penalty for ignoring a request
REBALANCER_PENALTY = 800.0 #penalty for ignoring a request in rebalancer

# MOVING_AVG_WINDOW = 12 # 3mins

##################################################################################
# Anticipatory ILP Config
##################################################################################
# REWARD_THETA = 0
PW = 4.64/3600 # usd/s User's Cost of waiting
PV = 2.32/3600 # usd/s User's Cost of travelling in vehicle
PO = 3.48/3600 # usd/s Operator's Cost of operating a vehicle

# REWARD_TYPE = 'GEN' # or 'REJ'
# NODE_LAYERS = 1 # number of layers of rejected rate to consider
PSI = 1 #𝜓 is a tuning parameter (the higher this parameter, the more uniform the resulting rates).
##################################################################################
# Simulation Config
##################################################################################
DEBUG_PRINT = False

# for small-grid-data
# SIMULATION_DURATION = 60.0 #SmallGridData has 60 minutes of data
# TIME_STEP = 0.25 # 15 seconds
# COOL_DOWN_DURATION = 60.0 # 60 minutes
# PENALTY = 5.0 #penalty for ignoring a request

# for Manhattan-data
SIMULATION_DURATION = 3600 # 60 minutes = 3600 seconds
TIME_STEP = 15 # 15 seconds
COOL_DOWN_DURATION = 3600 # 20 minutes = 1200 seconds

