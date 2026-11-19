# Traffic Simulation Configuration - Indian Traffic Context

# Window settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (60, 60, 60)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
LIGHT_BLUE = (173, 216, 230)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
LIGHT_GREEN = (144, 238, 144)
NAVY = (0, 0, 128)
AMBER = (255, 191, 0)

# Intersection geometry
INTERSECTION_X = 600
INTERSECTION_Y = 400
ROAD_WIDTH = 90
INTERSECTION_SIZE = 150

# Vehicle categories (Indian traffic)
VEHICLE_TYPES = {
    'car': {'length': 20, 'width': 12, 'speed': 2.8, 'color': (100, 150, 255)},
    'motorcycle': {'length': 12, 'width': 8, 'speed': 3.2, 'color': (200, 100, 255)},
    'scooter': {'length': 14, 'width': 9, 'speed': 3.0, 'color': (255, 100, 100)},
    'auto_rickshaw': {'length': 16, 'width': 10, 'speed': 2.6, 'color': (255, 200, 0)},
    'bus': {'length': 32, 'width': 14, 'speed': 2.2, 'color': (255, 0, 0)},
    'truck': {'length': 36, 'width': 14, 'speed': 2.0, 'color': (100, 100, 100)},
    'ambulance': {'length': 24, 'width': 12, 'speed': 3.5, 'color': (255, 255, 255)},
    'police': {'length': 22, 'width': 12, 'speed': 3.4, 'color': (0, 0, 200)},
}

EMERGENCY_TYPES = {'ambulance', 'police'}

# Directions
DIRECTIONS = ['N', 'S', 'E', 'W']
DIRECTION_NAMES = {'N': 'North', 'S': 'South', 'E': 'East', 'W': 'West'}

# Traffic light timing
LIGHT_RADIUS = 14
MIN_GREEN_TIME = 30      # frames (~1 second at 30fps)
MAX_GREEN_TIME = 180     # frames (~6 seconds)
YELLOW_TIME = 15         # frames
ALL_RED_TIME = 10        # frames
PEDESTRIAN_CROSS_TIME = 90  # frames (~3 seconds)
GREEN_TIME = 60          # frames for fixed-time baseline

# Spawn rates
BASE_SPAWN_PROB = 0.03
PEDESTRIAN_SPAWN_PROB = 0.02
EMERGENCY_SPAWN_PROB = 0.005

# Q-Learning parameters
LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.95
EPSILON_START = 0.4
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

# State discretization bins
VEHICLE_BINS = 3   # low, medium, high
PEDESTRIAN_BINS = 3  # low, medium, high

# Action space
ACTIONS = [
    'NS_GREEN',      # Green for North-South
    'EW_GREEN',      # Green for East-West
    'PEDESTRIAN_N',  # Pedestrian crossing North side
    'PEDESTRIAN_S',  # Pedestrian crossing South side
    'PEDESTRIAN_E',  # Pedestrian crossing East side
    'PEDESTRIAN_W',  # Pedestrian crossing West side
    'EXTEND_GREEN',  # Extend current green phase
]

NUM_ACTIONS = len(ACTIONS)

# Reward function weights
REWARD_WAITING_VEHICLE = -0.5
REWARD_WAITING_PEDESTRIAN = -0.3
REWARD_VEHICLE_CROSSED = +1.0
REWARD_PEDESTRIAN_CROSSED = +0.8
REWARD_EMERGENCY_CROSSED = +5.0
REWARD_SIGNAL_CHANGE = -0.1
REWARD_EMERGENCY_BLOCKED = -10.0
REWARD_CONFLICT = -20.0

# File paths
Q_TABLE_PATH = 'models/q_table.npy'
METRICS_PATH = 'data/metrics.csv'
COMPARISON_PATH = 'data/comparison.csv'