# config.py - India-specific Adaptive Traffic Signal Control System

# Window / Display
WINDOW_W = 900
WINDOW_H = 640
FPS = 30

# Intersection geometry
INTER_X = 450
INTER_Y = 320
INTER_SIZE = 130
ROAD_W = 85

# Vehicle types: speed (float), length (int), width (int), weight (int), is_emergency (bool)
VEHICLE_TYPES = {
    'car':        {'speed': 2.5, 'length': 18, 'width': 10, 'weight': 35, 'is_emergency': False},
    'motorcycle': {'speed': 3.5, 'length': 12, 'width':  7, 'weight': 30, 'is_emergency': False},
    'auto':       {'speed': 2.0, 'length': 15, 'width': 10, 'weight': 15, 'is_emergency': False},
    'bus':        {'speed': 1.5, 'length': 28, 'width': 14, 'weight':  8, 'is_emergency': False},
    'truck':      {'speed': 1.2, 'length': 32, 'width': 15, 'weight':  7, 'is_emergency': False},
    'ambulance':  {'speed': 4.0, 'length': 20, 'width': 11, 'weight':  2, 'is_emergency': True},
    'police':     {'speed': 3.5, 'length': 20, 'width': 11, 'weight':  1, 'is_emergency': True},
}

# Vehicle colors (hex)
VEHICLE_COLORS = {
    'car':        '#4fc3f7',
    'motorcycle': '#ff8a65',
    'auto':       '#ffeb3b',
    'bus':        '#66bb6a',
    'truck':      '#a1887f',
    'ambulance':  '#ef5350',
    'police':     '#5c6bc0',
}

# Signal timing (frames)
GREEN_MIN = 25
GREEN_MAX = 90
YELLOW_TIME = 6
ALL_RED_TIME = 4

# Simulation
SIMULATION_FRAMES = 2500

# Spawn probabilities
SPAWN_PROB_BASE = 0.025
EMERGENCY_SPAWN_PROB = 0.003

# Pedestrian parameters
PED_SPAWN_PROB = 0.02
PED_CROSSING_TIME = 50
PED_HIGH_DEMAND = 4
PED_MAX_WAIT = 200

# Emergency vehicle priority distance (pixels)
EMERGENCY_PRIORITY_DIST = 180

# Q-Learning hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 0.5
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9985
