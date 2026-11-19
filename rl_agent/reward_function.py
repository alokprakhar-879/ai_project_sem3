"""Reward Function for Q-Learning Traffic Signal Control"""

from traffic_sim.config import *


class RewardFunction:
    """
    Computes rewards for Q-Learning agent based on traffic conditions.

    Reward design encourages:
    - Reduced vehicle waiting time
    - Reduced pedestrian waiting time
    - Reduced queue length
    - Successful vehicle/pedestrian crossings
    - Emergency vehicle priority
    - Fewer unnecessary signal changes

    Penalizes:
    - Excessive waiting
    - Unnecessary signal switching
    - Blocking emergency vehicles
    - Unsafe signal combinations
    """

    def __init__(self):
        self.weights = {
            'waiting_vehicle': REWARD_WAITING_VEHICLE,
            'waiting_pedestrian': REWARD_WAITING_PEDESTRIAN,
            'vehicle_crossed': REWARD_VEHICLE_CROSSED,
            'pedestrian_crossed': REWARD_PEDESTRIAN_CROSSED,
            'emergency_crossed': REWARD_EMERGENCY_CROSSED,
            'signal_change': REWARD_SIGNAL_CHANGE,
            'emergency_blocked': REWARD_EMERGENCY_BLOCKED,
            'conflict': REWARD_CONFLICT,
        }

    def compute_reward(self, prev_info, curr_info, action, prev_action,
                       emergency_handled=False, emergency_blocked=False,
                       conflict=False, signal_changed=False):
        """
        Compute total reward for the current step.

        Args:
            prev_info: Dictionary with previous step metrics
            curr_info: Dictionary with current step metrics
            action: Current action taken
            prev_action: Previous action taken
            emergency_handled: Whether emergency vehicle was given priority
            emergency_blocked: Whether emergency vehicle was blocked
            conflict: Whether unsafe conflict occurred
            signal_changed: Whether signal phase changed

        Returns:
            Total reward value
        """
        reward = 0.0

        # Vehicle waiting penalty (per waiting vehicle)
        curr_vehicle_waiting = curr_info.get('current_vehicle_waiting', 0)
        prev_vehicle_waiting = prev_info.get('current_vehicle_waiting', 0)
        waiting_diff = curr_vehicle_waiting - prev_vehicle_waiting
        reward += self.weights['waiting_vehicle'] * waiting_diff

        # Pedestrian waiting penalty
        curr_ped_waiting = curr_info.get('current_pedestrian_waiting', 0)
        prev_ped_waiting = prev_info.get('current_pedestrian_waiting', 0)
        ped_waiting_diff = curr_ped_waiting - prev_ped_waiting
        reward += self.weights['waiting_pedestrian'] * ped_waiting_diff

        # Vehicle crossed reward
        vehicles_crossed = curr_info.get('vehicles_crossed_this_step', 0)
        reward += self.weights['vehicle_crossed'] * vehicles_crossed

        # Pedestrian crossed reward
        pedestrians_crossed = curr_info.get('pedestrians_crossed_this_step', 0)
        reward += self.weights['pedestrian_crossed'] * pedestrians_crossed

        # Emergency vehicle crossed reward
        emergency_crossed = curr_info.get('emergency_crossed_this_step', 0)
        reward += self.weights['emergency_crossed'] * emergency_crossed

        # Emergency blocked penalty
        if emergency_blocked:
            reward += self.weights['emergency_blocked']

        # Conflict penalty (unsafe)
        if conflict:
            reward += self.weights['conflict']

        # Signal change penalty (discourage unnecessary switching)
        if signal_changed and prev_action is not None and action != prev_action:
            reward += self.weights['signal_change']

        return reward

    def compute_episode_reward(self, metrics):
        """
        Compute aggregate reward for an entire training episode.

        Args:
            metrics: Dictionary with episode-level metrics

        Returns:
            Total episode reward
        """
        reward = 0.0

        # Normalize by number of vehicles/pedestrians
        vehicles_crossed = metrics.get('vehicles_crossed', 1)
        pedestrians_crossed = metrics.get('pedestrians_crossed', 1)
        total_waiting = metrics.get('total_vehicle_waiting', 0)
        total_ped_waiting = metrics.get('total_pedestrian_waiting', 0)
        signal_changes = metrics.get('signal_changes', 0)
        emergency_crossed = metrics.get('emergency_crossed', 0)

        # Average waiting per vehicle/pedestrian
        avg_vehicle_wait = total_waiting / max(vehicles_crossed, 1)
        avg_ped_wait = total_ped_waiting / max(pedestrians_crossed, 1)

        reward += -0.1 * avg_vehicle_wait
        reward += -0.05 * avg_ped_wait
        reward += 0.5 * vehicles_crossed
        reward += 0.4 * pedestrians_crossed
        reward += 2.0 * emergency_crossed
        reward += -0.05 * signal_changes

        return reward

    def get_reward_breakdown(self):
        """Return reward weight documentation."""
        return self.weights.copy()