# Traffic Simulation Configuration - Indian Traffic Context

# Window settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (60, 60, 60)
ROAD_GRAY = (90, 90, 90)
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
ZEBRA_WHITE = (255, 255, 255)
STOP_LINE_COLOR = (255, 255, 255)

# Intersection geometry
INTERSECTION_X = 700
INTERSECTION_Y = 450
ROAD_WIDTH = 100
INTERSECTION_SIZE = 180

# Lane positions within each road (offset from center for multiple lanes)
LANE_OFFSETS = {
    'N': [-25, 0, 25],   # Three lanes for North-bound
    'S': [-25, 0, 25],   # Three lanes for South-bound
    'E': [-25, 0, 25],   # Three lanes for East-bound
    'W': [-25, 0, 25],   # Three lanes for West-bound
}

# Vehicle categories (Indian traffic)
VEHICLE_TYPES = {
    'car': {'length': 24, 'width': 14, 'base_speed': 2.5, 'color': (100, 150, 255), 'accel': 0.15, 'decel': 0.25},
    'motorcycle': {'length': 14, 'width': 9, 'base_speed': 3.0, 'color': (200, 100, 255), 'accel': 0.2, 'decel': 0.3},
    'scooter': {'length': 16, 'width': 10, 'base_speed': 2.8, 'color': (255, 100, 100), 'accel': 0.18, 'decel': 0.28},
    'auto_rickshaw': {'length': 18, 'width': 11, 'base_speed': 2.4, 'color': (255, 200, 0), 'accel': 0.14, 'decel': 0.22},
    'bus': {'length': 36, 'width': 16, 'base_speed': 2.0, 'color': (220, 20, 60), 'accel': 0.1, 'decel': 0.18},
    'truck': {'length': 40, 'width': 16, 'base_speed': 1.8, 'color': (100, 100, 100), 'accel': 0.08, 'decel': 0.15},
    'ambulance': {'length': 26, 'width': 14, 'base_speed': 3.2, 'color': (255, 255, 255), 'accel': 0.25, 'decel': 0.35},
    'police': {'length': 24, 'width': 14, 'base_speed': 3.1, 'color': (0, 0, 200), 'accel': 0.22, 'decel': 0.32},
}

EMERGENCY_TYPES = {'ambulance', 'police'}

# Directions
DIRECTIONS = ['N', 'S', 'E', 'W']
DIRECTION_NAMES = {'N': 'North', 'S': 'South', 'E': 'East', 'W': 'West'}
DIRECTION_OPPOSITE = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

# Traffic light timing (configurable)
LIGHT_RADIUS = 16
MIN_GREEN_TIME = 45      # frames (~1.5 seconds at 30fps)
MAX_GREEN_TIME = 300     # frames (~10 seconds)
YELLOW_TIME = 20         # frames
ALL_RED_TIME = 15        # frames
PEDESTRIAN_CROSS_TIME = 120  # frames (~4 seconds)
GREEN_TIME = 120         # frames for fixed-time baseline

# Simulation speed presets (multiplier applied to base speed)
SPEED_PRESETS = {
    '0.25x': 0.25,
    '0.5x': 0.5,
    '1x': 1.0,
    '2x': 2.0,
}
DEFAULT_SPEED_PRESET = '1x'

# Spawn rates
BASE_SPAWN_PROB = 0.04
PEDESTRIAN_SPAWN_PROB = 0.03
EMERGENCY_SPAWN_PROB = 0.008

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

# Safe following distance (in pixels)
SAFE_FOLLOWING_DISTANCE = 40

# Stop line buffer (distance before stop line where vehicle should begin slowing)
SLOW_DOWN_DISTANCE = 60