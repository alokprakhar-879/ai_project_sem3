# q_learning.py - Q-Learning agent for India-specific Adaptive Traffic Signal Control
import os
import random
import numpy as np

from config import (
    LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    PED_HIGH_DEMAND, PED_MAX_WAIT,
)

DIRECTIONS = ['N', 'S', 'E', 'W']

# State dimensions
# q_N, q_S, q_E, q_W: 3 bins each
# phase_idx: 6 options
# has_emergency: 2
# ped_demand: 3 bins
# Total: 3*3*3*3*6*2*3 = 2916

STATE_SIZE = 3 * 3 * 3 * 3 * 6 * 2 * 3  # 2916
ACTION_SIZE = 7  # NS=0, EW=1, PED_N=2, PED_S=3, PED_E=4, PED_W=5, EXTEND=6

PHASE_LIST = ['NS', 'EW', 'PED_N', 'PED_S', 'PED_E', 'PED_W']

NS_DIRS = {'N', 'S'}
EW_DIRS = {'E', 'W'}


class QLearningAgent:
    def __init__(self, qtable_path='qtable.npy'):
        self.qtable_path = qtable_path
        if os.path.exists(qtable_path):
            self.q_table = np.load(qtable_path)
            # Validate shape
            if self.q_table.shape != (STATE_SIZE, ACTION_SIZE):
                self.q_table = np.zeros((STATE_SIZE, ACTION_SIZE), dtype=np.float32)
        else:
            self.q_table = np.zeros((STATE_SIZE, ACTION_SIZE), dtype=np.float32)

        self.lr = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.last_reward = 0.0
        self.total_reward = 0.0

    # ------------------------------------------------------------------
    # State encoding
    # ------------------------------------------------------------------

    def _discretize_queue(self, n):
        if n < 3:
            return 0
        elif n < 8:
            return 1
        else:
            return 2

    def _discretize_ped(self, n):
        if n < 3:
            return 0
        elif n < 7:
            return 1
        else:
            return 2

    def _phase_to_idx(self, phase_str):
        try:
            return PHASE_LIST.index(phase_str)
        except ValueError:
            return 0

    def encode_state(self, state_info):
        qc = state_info['queue_counts']  # [N, S, E, W]
        q_N = self._discretize_queue(qc[0])
        q_S = self._discretize_queue(qc[1])
        q_E = self._discretize_queue(qc[2])
        q_W = self._discretize_queue(qc[3])

        phase_idx = self._phase_to_idx(state_info['phase'])
        em = 1 if state_info['has_emergency'] else 0

        pc = state_info['ped_counts']
        ped_max = max(pc)
        ped_demand = self._discretize_ped(ped_max)

        # Flat index: q_N * (3*3*3*6*2*3) + q_S * (3*3*6*2*3) + q_E * (3*6*2*3) + q_W * (6*2*3) + phase_idx * (2*3) + em * 3 + ped_demand
        idx = (
            q_N * (3 * 3 * 3 * 6 * 2 * 3)
            + q_S * (3 * 3 * 6 * 2 * 3)
            + q_E * (3 * 6 * 2 * 3)
            + q_W * (6 * 2 * 3)
            + phase_idx * (2 * 3)
            + em * 3
            + ped_demand
        )
        return int(idx)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state_idx, state_info):
        # Safety: emergency override
        if state_info['has_emergency'] and state_info['emergency_dir']:
            ed = state_info['emergency_dir']
            if ed in NS_DIRS:
                return 0  # NS
            elif ed in EW_DIRS:
                return 1  # EW

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_SIZE - 1)
        else:
            return int(np.argmax(self.q_table[state_idx]))

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def calculate_reward(self, state_info, action, next_state_info):
        r = 0.0

        qc = state_info['queue_counts']
        wt = state_info['waiting_times']
        pc = state_info['ped_counts']
        pmw = state_info['ped_max_wait']

        nqc = next_state_info['queue_counts']

        r -= 0.15 * sum(qc)
        r -= 0.03 * sum(wt)

        queue_reduction = sum(qc) - sum(nqc)
        r += 0.4 * queue_reduction

        r -= 0.2 * sum(pc)
        r -= 0.08 * sum(pmw)

        # Pedestrian phase rewards/penalties
        if action in [2, 3, 4, 5]:
            d_idx = action - 2  # 0=N, 1=S, 2=E, 3=W
            ped_count_at_dir = pc[d_idx]
            if ped_count_at_dir >= PED_HIGH_DEMAND:
                r += 5.0 * ped_count_at_dir
            elif ped_count_at_dir == 0:
                r -= 3.0  # wasted pedestrian phase

        # Mandatory crossing violated
        if any(w > PED_MAX_WAIT for w in pmw):
            r -= 10.0

        # Emergency handling
        if state_info['has_emergency'] and state_info['emergency_dir']:
            ed = state_info['emergency_dir']
            eidx = DIRECTIONS.index(ed)
            r -= 0.3 * wt[eidx]
            if ed in NS_DIRS and action == 0:
                r += 15.0
            elif ed in EW_DIRS and action == 1:
                r += 15.0
            else:
                r -= 5.0

        # Penalize lazy extending
        if action == 6:
            r -= 0.5

        # Penalize phase jitter
        if state_info['phase'] != next_state_info['phase']:
            r -= 0.2

        return round(r, 4)

    # ------------------------------------------------------------------
    # Q-Learning update
    # ------------------------------------------------------------------

    def update(self, state_idx, action, reward, next_state_idx):
        current_q = self.q_table[state_idx, action]
        max_next_q = float(np.max(self.q_table[next_state_idx]))
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_idx, action] = new_q
        self.last_reward = reward
        self.total_reward += reward

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path=None):
        if path is None:
            path = self.qtable_path
        np.save(path, self.q_table)

    def get_info(self):
        nonzero = int(np.count_nonzero(self.q_table))
        return {
            'epsilon': round(float(self.epsilon), 6),
            'last_reward': float(self.last_reward),
            'total_reward': round(float(self.total_reward), 4),
            'q_nonzero': nonzero,
            'q_max': round(float(np.max(self.q_table)), 4),
            'q_min': round(float(np.min(self.q_table)), 4),
        }
