# Adaptive Traffic Signal Control Using Q-Learning

## India-Specific Intelligent Traffic Management System

A complete Reinforcement Learning-based adaptive traffic signal control system
designed for Indian road conditions with mixed vehicle types, pedestrian priority,
and emergency vehicle handling.

## Project Structure

```
demo_omni/
├── main.py                      # Entry point
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
├── traffic_sim/
│   ├── __init__.py
│   ├── config.py                # Configuration constants
│   ├── engine.py                # Traffic simulation engine (vehicles, pedestrians, lights)
│   └── simulation.py            # Main simulation runner (integrates all modules)
├── rl_agent/
│   ├── __init__.py
│   ├── q_learning.py            # Q-Learning algorithm from scratch
│   ├── reward_function.py       # Reward function design
│   ├── state_manager.py         # Traffic state management
│   └── safety_controller.py     # Safety constraints (yellow, all-red, conflicts)
├── dashboard/
│   ├── __init__.py
│   └── app.py                   # Streamlit interactive dashboard
├── models/
│   └── q_table.npy              # Saved Q-Learning model
└── data/
    └── metrics.csv              # Training metrics
```

## Features

### 1. Indian Traffic Simulation
- **Vehicle Types:** Cars, motorcycles, scooters, auto-rickshaws, buses, trucks
- **Emergency Vehicles:** Ambulances and police vehicles with priority
- **Pedestrians:** Crowd-based crossing with zebra crossings
- **Realistic Patterns:** Mixed density, irregular arrivals, peak/non-peak scenarios

### 2. Q-Learning Algorithm (From Scratch)
- Q-Table implementation using NumPy
- State: Discretized vehicle counts, pedestrian counts, emergency presence, signal phase
- Actions: NS Green, EW Green, Pedestrian crossings (4 sides), Extend Green
- Reward: Waiting time reduction, crossing rewards, emergency priority, safety penalties
- Epsilon-greedy exploration with decay
- Model save/load support

### 3. Safety Controller
- Minimum/maximum green duration enforcement
- Yellow signal transitions
- All-red clearance between conflicting phases
- No conflicting green signals
- Pedestrian clearance time
- Emergency vehicle priority override

### 4. Streamlit Dashboard
- Live simulation monitoring
- Q-Learning agent statistics
- Performance comparison (AI vs Fixed-Time)
- Training controls
- Real-time metrics

## Installation

```bash
# Clone or navigate to project directory
cd demo_omni

# Install dependencies
pip install -r requirements.txt

# Install Streamlit (if not included)
pip install streamlit plotly
```

## Usage

### Train the Agent
```bash
python main.py --train --episodes 100 --frames 2000
```

### Evaluate Trained Agent
```bash
python main.py --eval --frames 3000
```

### Run Fixed-Time Baseline
```bash
python main.py --baseline --frames 3000
```

### Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

## Q-Learning Design

### State Space
```
State = (vehicle_bins_N, vehicle_bins_S, vehicle_bins_E, vehicle_bins_W,
         pedestrian_bins_N, pedestrian_bins_S, pedestrian_bins_E, pedestrian_bins_W,
         emergency_present, current_phase)
```
- Vehicle counts discretized into 3 bins (low/medium/high)
- Pedestrian counts discretized into 3 bins
- Emergency presence: binary
- Current phase: NS_GREEN, EW_GREEN, ALL_RED, PEDESTRIAN

### Action Space
1. NS_GREEN - Green for North-South
2. EW_GREEN - Green for East-West
3. PEDESTRIAN_N - Pedestrian crossing North
4. PEDESTRIAN_S - Pedestrian crossing South
5. PEDESTRIAN_E - Pedestrian crossing East
6. PEDESTRIAN_W - Pedestrian crossing West
7. EXTEND_GREEN - Extend current phase

### Reward Function
- Vehicle crossed: +1.0
- Pedestrian crossed: +0.8
- Emergency crossed: +5.0
- Waiting vehicle: -0.5 per vehicle
- Waiting pedestrian: -0.3 per pedestrian
- Signal change: -0.1
- Emergency blocked: -10.0
- Conflict: -20.0

## Safety Constraints

The SafetyController enforces:
- Minimum green time: 30 frames
- Maximum green time: 180 frames
- Yellow transition: 15 frames
- All-red clearance: 10 frames
- Pedestrian crossing time: 90 frames
- No conflicting green signals
- Emergency override priority

## License

MIT License - For educational and research purposes.