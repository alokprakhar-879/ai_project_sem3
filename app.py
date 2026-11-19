# app.py - Flask server for India-specific Adaptive Traffic Signal Control
import json
import random
import os

import numpy as np
from flask import Flask, render_template, request, jsonify

from simulation import Intersection
from q_learning import QLearningAgent
from config import (
    SIMULATION_FRAMES, VEHICLE_COLORS, VEHICLE_TYPES,
    GREEN_MIN, GREEN_MAX,
    WINDOW_W, WINDOW_H, INTER_X, INTER_Y, INTER_SIZE, ROAD_W,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
intersection = None
agent = None
sim_running = False
use_ai_mode = True
frame_count = 0
ai_results = None
fixed_results = None
history = []  # list of {frame, crossed, waiting, reward, peak_hour}


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def reset_simulation(use_ai):
    global intersection, agent, sim_running, use_ai_mode, frame_count, history
    intersection = Intersection()
    if use_ai:
        agent = QLearningAgent()  # loads qtable.npy if exists
    else:
        agent = None
    sim_running = True
    use_ai_mode = use_ai
    frame_count = 0
    history = []


def step_frames(n_steps):
    global sim_running, frame_count, ai_results, fixed_results, history, agent

    last_reward = 0.0

    for _ in range(n_steps):
        if not sim_running or frame_count >= SIMULATION_FRAMES:
            sim_running = False
            break

        # Spawn
        intersection.spawn_vehicles()
        intersection.spawn_pedestrians()

        state_info = intersection.get_state_info()

        if use_ai_mode and agent:
            state_idx = agent.encode_state(state_info)
            action = agent.choose_action(state_idx, state_info)
            phase_map = {
                0: 'NS', 1: 'EW',
                2: 'PED_N', 3: 'PED_S', 4: 'PED_E', 5: 'PED_W',
                6: 'EXTEND',
            }
            if action != 6:
                intersection.signal.request_phase(phase_map[action])
            # action == 6: extend, do nothing
        else:
            # Fixed-time: cycle NS -> EW every GREEN_MIN+10 frames
            fixed_phase = 'NS' if (frame_count // (GREEN_MIN + 10)) % 2 == 0 else 'EW'
            intersection.signal.request_phase(fixed_phase)

        intersection.update()

        if use_ai_mode and agent:
            next_state_info = intersection.get_state_info()
            next_state_idx = agent.encode_state(next_state_info)
            reward = agent.calculate_reward(state_info, action, next_state_info)
            agent.update(state_idx, action, reward, next_state_idx)
            agent.decay_epsilon()
            last_reward = reward
        else:
            last_reward = 0.0

        frame_count += 1

        if frame_count % 50 == 0:
            history.append({
                'frame': frame_count,
                'crossed': intersection.crossed_count,
                'waiting': intersection.get_metrics()['queue'],
                'reward': last_reward if use_ai_mode else 0,
                'peak_hour': intersection.peak_hour,
            })

    if not sim_running:
        m = intersection.get_metrics()
        result = {
            'crossed': m['crossed_count'],
            'delay': m['total_delay'],
            'avg_wait': m['avg_wait'],
            'queue': m['queue'],
            'ped_crossed': m['ped_crossed'],
            'emergency_delay': m['emergency_delay'],
            'signal_changes': m['signal_changes'],
        }
        if use_ai_mode:
            global ai_results
            ai_results = result
            if agent:
                agent.save()
        else:
            global fixed_results
            fixed_results = result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template(
        'index.html',
        vehicle_colors=VEHICLE_COLORS,
        total_frames=SIMULATION_FRAMES,
        window_w=WINDOW_W,
        window_h=WINDOW_H,
        inter_x=INTER_X,
        inter_y=INTER_Y,
        inter_size=INTER_SIZE,
        road_w=ROAD_W,
    )


@app.route('/api/start', methods=['POST'])
def api_start():
    data = request.get_json(force=True, silent=True) or {}
    use_ai = bool(data.get('use_ai', True))
    reset_simulation(use_ai)
    return jsonify({'ok': True})


@app.route('/api/step', methods=['POST'])
def api_step():
    data = request.get_json(force=True, silent=True) or {}
    n_steps = int(data.get('steps', 5))
    step_frames(n_steps)

    sim_dict = intersection.to_dict()
    lights = sim_dict['lights']
    ped_lights = sim_dict['ped_lights']

    rl_info = None
    if use_ai_mode and agent:
        rl_info = agent.get_info()

    response = {
        'frame': frame_count,
        'total_frames': SIMULATION_FRAMES,
        'running': sim_running,
        'use_ai': use_ai_mode,
        'vehicles': sim_dict['vehicles'],
        'pedestrians': sim_dict['pedestrians'],
        'lights': lights,
        'ped_lights': ped_lights,
        'phase': sim_dict['phase'],
        'sig_state': sim_dict['sig_state'],
        'phase_timer': sim_dict['phase_timer'],
        'emergency_active': sim_dict['emergency_active'],
        'emergency_dir': sim_dict['emergency_dir'],
        'emergency_vtype': sim_dict['emergency_vtype'],
        'peak_hour': sim_dict['peak_hour'],
        'metrics': sim_dict['metrics'],
        'rl_info': rl_info,
        'ai_results': ai_results,
        'fixed_results': fixed_results,
        'history': history,
        'vehicle_colors': VEHICLE_COLORS,
    }
    return jsonify(response)


@app.route('/api/results', methods=['GET'])
def api_results():
    return jsonify({
        'ai_results': ai_results,
        'fixed_results': fixed_results,
        'history': history,
    })


if __name__ == '__main__':
    app.run(debug=False, port=5000)
