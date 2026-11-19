"""Q-Learning Agent for Adaptive Traffic Signal Control"""

import numpy as np
import random
import os
import pickle
from traffic_sim.config import *


class QLearningAgent:
    """
    Q-Learning Agent for traffic signal control.

    Implements the Q-Learning algorithm from scratch:
    Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
    """

    def __init__(self, learning_rate=LEARNING_RATE, discount_factor=DISCOUNT_FACTOR,
                 epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # State space: vehicle bins (3^4) * pedestrian bins (3^4) * emergency (2) * phase (3)
        # To keep Q-table manageable, we use a compressed state representation
        self.num_actions = NUM_ACTIONS
        self.q_table = {}
        self.action_names = ACTIONS

        # Training metrics
        self.training_rewards = []
        self.training_episodes = 0
        self.total_updates = 0

    def _discretize_vehicles(self, counts):
        """Discretize vehicle counts into bins."""
        bins = []
        for c in counts:
            if c < 3:
                bins.append(0)  # low
            elif c < 7:
                bins.append(1)  # medium
            else:
                bins.append(2)  # high
        return tuple(bins)

    def _discretize_pedestrians(self, counts):
        """Discretize pedestrian counts into bins."""
        bins = []
        for c in counts:
            if c < 3:
                bins.append(0)  # low
            elif c < 6:
                bins.append(1)  # medium
            else:
                bins.append(2)  # high
        return tuple(bins)

    def get_state(self, vehicle_counts, pedestrian_counts, emergency_present, current_phase):
        """
        Create a discrete state representation.

        State = (vehicle_bins_N, vehicle_bins_S, vehicle_bins_E, vehicle_bins_W,
                 pedestrian_bins_N, pedestrian_bins_S, pedestrian_bins_E, pedestrian_bins_W,
                 emergency_present, current_phase_index)
        """
        v_bins = self._discretize_vehicles(vehicle_counts)
        p_bins = self._discretize_pedestrians(pedestrian_counts)
        emergency_bit = 1 if emergency_present else 0

        # Map phase to index
        phase_map = {'NS_GREEN': 0, 'EW_GREEN': 1, 'ALL_RED': 2, 'PEDESTRIAN': 3}
        phase_idx = phase_map.get(current_phase, 0)

        # Compress state: use string key for sparse Q-table
        state_key = (v_bins, p_bins, emergency_bit, phase_idx)
        return state_key

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def set_q_value(self, state, action, value):
        """Set Q-value for state-action pair."""
        self.q_table[(state, action)] = value

    def choose_action(self, state, training=True):
        """
        Epsilon-greedy action selection.

        During training: explore with probability epsilon
        During evaluation: always exploit (epsilon=0)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)

        # Exploit: choose best action
        q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done=False):
        """
        Q-Learning update rule:
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        """
        current_q = self.get_q_value(state, action)

        if done:
            max_next_q = 0.0
        else:
            next_q_values = [self.get_q_value(next_state, a) for a in range(self.num_actions)]
            max_next_q = max(next_q_values)

        td_target = reward + self.discount_factor * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error

        self.set_q_value(state, action, new_q)
        self.total_updates += 1

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath=Q_TABLE_PATH):
        """Save Q-table to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'training_episodes': self.training_episodes,
            'total_updates': self.total_updates,
            'training_rewards': self.training_rewards,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath=Q_TABLE_PATH):
        """Load Q-table from file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.q_table = data.get('q_table', {})
            self.learning_rate = data.get('learning_rate', LEARNING_RATE)
            self.discount_factor = data.get('discount_factor', DISCOUNT_FACTOR)
            self.epsilon = data.get('epsilon', EPSILON_MIN)
            self.training_episodes = data.get('training_episodes', 0)
            self.total_updates = data.get('total_updates', 0)
            self.training_rewards = data.get('training_rewards', [])
            return True
        return False

    def get_stats(self):
        """Get agent statistics."""
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'training_episodes': self.training_episodes,
            'total_updates': self.total_updates,
            'avg_reward_last_100': np.mean(self.training_rewards[-100:]) if self.training_rewards else 0,
        }