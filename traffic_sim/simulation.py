"""Main Simulation Runner - integrates all modules"""

import pygame
import numpy as np
import random
from traffic_sim.config import *
from traffic_sim.engine import Intersection, Vehicle, Pedestrian
from rl_agent.q_learning import QLearningAgent
from rl_agent.reward_function import RewardFunction
from rl_agent.state_manager import TrafficStateManager
from rl_agent.safety_controller import SafetyController


class TrafficSimulation:
    """
    Main simulation class that integrates all modules.

    Modes:
    - 'train': Q-Learning training mode
    - 'eval': Evaluation mode (use trained model)
    - 'fixed': Fixed-time control baseline
    """

    def __init__(self, mode='train', spawn_prob=BASE_SPAWN_PROB,
                 emergency_prob=EMERGENCY_SPAWN_PROB,
                 pedestrian_prob=PEDESTRIAN_SPAWN_PROB,
                 load_model=False, speed_multiplier=1.0):
        self.mode = mode
        self.spawn_prob = spawn_prob
        self.emergency_prob = emergency_prob
        self.pedestrian_prob = pedestrian_prob
        self.speed_multiplier = speed_multiplier

        # Initialize modules
        self.intersection = Intersection()
        self.agent = QLearningAgent()
        self.reward_fn = RewardFunction()
        self.state_manager = TrafficStateManager()
        self.safety = SafetyController()

        # Load trained model if available
        if load_model:
            self.agent.load()

        # Simulation state
        self.frame = 0
        self.running = True
        self.paused = False
        self.current_action = None
        self.prev_action = None
        self.prev_state = None

        # Metrics tracking
        self.episode_metrics = {
            'vehicles_crossed': 0,
            'pedestrians_crossed': 0,
            'total_vehicle_waiting': 0,
            'total_pedestrian_waiting': 0,
            'signal_changes': 0,
            'emergency_crossed': 0,
            'total_reward': 0,
            'crossed_by_direction': {d: 0 for d in DIRECTIONS},
            'generated_by_direction': {d: 0 for d in DIRECTIONS},
        }

        # Reward tracking per step
        self.step_rewards = []
        self.step_details = []  # For decision monitor

        # Initialize first phase
        self.intersection.set_phase('NS_GREEN')

    def reset(self, keep_q_table=True):
        """Reset simulation for new episode.

        Args:
            keep_q_table: If True, preserve the trained Q-Table.
        """
        # Save Q-table if we need to keep it
        q_table_backup = None
        if keep_q_table:
            q_table_backup = dict(self.agent.q_table)
            lr_backup = self.agent.learning_rate
            gamma_backup = self.agent.discount_factor
            epsilon_backup = self.agent.epsilon
            episodes_backup = self.agent.training_episodes

        # Reinitialize modules
        self.intersection = Intersection()
        self.state_manager = TrafficStateManager()
        self.frame = 0
        self.current_action = None
        self.prev_action = None
        self.prev_state = None
        self.episode_metrics = {
            'vehicles_crossed': 0,
            'pedestrians_crossed': 0,
            'total_vehicle_waiting': 0,
            'total_pedestrian_waiting': 0,
            'signal_changes': 0,
            'emergency_crossed': 0,
            'total_reward': 0,
            'crossed_by_direction': {d: 0 for d in DIRECTIONS},
            'generated_by_direction': {d: 0 for d in DIRECTIONS},
        }
        self.step_rewards = []
        self.step_details = []
        self.intersection.set_phase('NS_GREEN')

        # Restore Q-table if needed
        if keep_q_table and q_table_backup is not None:
            self.agent.q_table = q_table_backup
            self.agent.learning_rate = lr_backup
            self.agent.discount_factor = gamma_backup
            self.agent.epsilon = epsilon_backup
            self.agent.training_episodes = episodes_backup

    def reset_q_table(self):
        """Reset the Q-Table completely (fresh training)."""
        self.agent.q_table = {}
        self.agent.training_episodes = 0
        self.agent.total_updates = 0
        self.agent.training_rewards = []
        self.agent.epsilon = EPSILON_START

    def set_speed_multiplier(self, multiplier):
        """Set simulation speed multiplier."""
        self.speed_multiplier = multiplier

    def get_state_for_agent(self):
        """Get current state for Q-Learning agent (tuple key) without updating.

        The intersection has already been updated by step(). This method just
        computes the Q-Learning state key from the current intersection state.
        """
        info = self._get_current_info()

        # Collect detailed state for monitoring
        state_dict = self.state_manager.collect_state(
            self.intersection, info['current_phase']
        )

        # Convert to Q-Learning state key (tuple)
        vehicle_counts = info.get('vehicle_counts', [0, 0, 0, 0])
        pedestrian_counts = info.get('pedestrian_counts', [0, 0, 0, 0])
        emergency_active = info.get('emergency_active', False)
        current_phase = info.get('current_phase', 'NS_GREEN')

        q_state = self.agent.get_state(
            vehicle_counts, pedestrian_counts, emergency_active, current_phase
        )

        return q_state, state_dict, info

    def _get_current_info(self):
        """Get current intersection info without calling update again."""
        return {
            'emergency_active': self.intersection.emergency_override_active,
            'emergency_dir': self.intersection.get_emergency_direction(),
            'current_phase': self.intersection.current_phase,
            'vehicle_counts': self.intersection.get_vehicle_counts(),
            'pedestrian_counts': self.intersection.get_pedestrian_counts(),
            'crossed_by_direction': self.intersection.get_crossed_by_direction(),
            'generated_by_direction': self.intersection.get_generated_counts(),
            'current_vehicle_waiting': self.intersection.get_total_waiting(),
            'current_pedestrian_waiting': self.intersection.get_total_pedestrian_waiting(),
        }

    def step(self, action=None):
        """
        Execute one simulation step.

        Args:
            action: Action index (for AI mode) or None (fixed-time)

        Returns:
            Dictionary with step results
        """
        if self.paused:
            return self._get_step_result()

        self.prev_action = self.current_action
        self.current_action = action

        # Update intersection (spawn, move vehicles/pedestrians, update lights)
        info = self.intersection.update(
            self.spawn_prob, self.emergency_prob, self.pedestrian_prob,
            speed_multiplier=self.speed_multiplier
        )

        # Collect state for monitoring
        state_dict = self.state_manager.collect_state(
            self.intersection, info['current_phase']
        )

        # Convert to Q-Learning state key (tuple)
        vehicle_counts = info.get('vehicle_counts', [0, 0, 0, 0])
        pedestrian_counts = info.get('pedestrian_counts', [0, 0, 0, 0])
        emergency_active = info.get('emergency_active', False)
        current_phase = info.get('current_phase', 'NS_GREEN')

        q_state = self.agent.get_state(
            vehicle_counts, pedestrian_counts, emergency_active, current_phase
        )

        # Emergency override check
        emergency_dir = info.get('emergency_dir', None)

        # Determine actual phase to apply
        if emergency_active and self.safety and emergency_dir:
            # Emergency priority override
            safe_phase, _ = self.safety.apply_emergency_override(
                info['current_phase'], emergency_dir
            )
            self.intersection.set_phase(safe_phase)
        elif self.mode == 'fixed':
            # Fixed-time control: cycle through phases
            if self.frame % (GREEN_TIME + YELLOW_TIME) == 0:
                idx = (self.frame // (GREEN_TIME + YELLOW_TIME)) % 2
                phase = 'NS_GREEN' if idx == 0 else 'EW_GREEN'
                self.intersection.set_phase(phase)
        elif self.mode in ['train', 'eval']:
            # AI control - Q-Learning update for previous action
            if self.prev_state is not None:
                # Compute reward for previous action
                step_reward = self.reward_fn.compute_reward(
                    self._get_prev_info(), info,
                    self.prev_action, self.prev_prev_action,
                    emergency_handled=emergency_active,
                    emergency_blocked=emergency_active and not self._emergency_green(),
                    conflict=False,
                    signal_changed=(self.prev_action != self.current_action)
                )
                self.step_rewards.append(step_reward)

                # Record step details for decision monitor
                q_values = [self.agent.get_q_value(self.prev_state, a)
                           for a in range(self.agent.num_actions)]
                self.step_details.append({
                    'frame': self.frame - 1,
                    'state': self.prev_state,
                    'action': self.prev_action,
                    'action_name': ACTIONS[self.prev_action] if self.prev_action is not None else None,
                    'reward': step_reward,
                    'q_values': q_values,
                    'vehicle_counts': self._get_prev_info().get('vehicle_counts', [0, 0, 0, 0]),
                    'pedestrian_counts': self._get_prev_info().get('pedestrian_counts', [0, 0, 0, 0]),
                    'epsilon': self.agent.epsilon,
                })
                # Keep only last 50 details
                if len(self.step_details) > 50:
                    self.step_details = self.step_details[-50:]

                # Q-Learning update
                if self.mode == 'train':
                    self.agent.update(
                        self.prev_state, self.prev_action,
                        step_reward, q_state
                    )

            # Choose next action
            if self.mode == 'train':
                action_idx = self.agent.choose_action(q_state, training=True)
            else:
                action_idx = self.agent.choose_action(q_state, training=False)

            # Apply safety checks
            allowed, reason = self.safety.can_change_phase(
                self.frame, info['current_phase'],
                ACTIONS[action_idx] if action_idx is not None else 'NS_GREEN'
            )

            if allowed:
                self.intersection._apply_action(action_idx)

        # Track metrics
        crossed_this_step = self.intersection.crossed_vehicles - self.episode_metrics['vehicles_crossed']
        ped_crossed_this_step = self.intersection.crossed_pedestrians - self.episode_metrics['pedestrians_crossed']

        self.episode_metrics['vehicles_crossed'] = self.intersection.crossed_vehicles
        self.episode_metrics['pedestrians_crossed'] = self.intersection.crossed_pedestrians
        self.episode_metrics['total_vehicle_waiting'] = self.intersection.total_vehicle_waiting
        self.episode_metrics['total_pedestrian_waiting'] = self.intersection.total_pedestrian_waiting
        self.episode_metrics['signal_changes'] = self.intersection.signal_changes
        self.episode_metrics['emergency_crossed'] = self.intersection.emergency_crossed
        self.episode_metrics['crossed_by_direction'] = dict(
            self.intersection.crossed_by_direction)
        self.episode_metrics['generated_by_direction'] = dict(
            self.intersection.generated_by_direction)

        self.frame += 1

        # Store for next iteration
        self.prev_state = q_state
        self.prev_prev_action = self.prev_action
        self._prev_info = info

        return self._get_step_result(info, crossed_this_step, ped_crossed_this_step)

    def _emergency_green(self):
        """Check if emergency vehicle direction has green."""
        emergency_dir = self.intersection.get_emergency_direction()
        if emergency_dir is None:
            return True
        light = self.intersection.lights[emergency_dir]
        return light.is_green() or light.is_yellow()

    def _get_prev_info(self):
        """Get previous step info."""
        return getattr(self, '_prev_info', {
            'current_vehicle_waiting': 0,
            'current_pedestrian_waiting': 0,
            'vehicle_counts': [0, 0, 0, 0],
            'pedestrian_counts': [0, 0, 0, 0],
        })

    def _get_step_result(self, step_info=None, crossed_vehicles=0, crossed_pedestrians=0):
        """Get standardized step result."""
        vehicle_counts = self.intersection.get_vehicle_counts()
        pedestrian_counts = self.intersection.get_pedestrian_counts()

        # Get Q-values for current state (for decision monitor)
        current_q_values = None
        current_action_name = None
        if self.prev_state is not None:
            current_q_values = [self.agent.get_q_value(self.prev_state, a)
                               for a in range(self.agent.num_actions)]
            current_action_name = ACTIONS[self.current_action] if self.current_action is not None else None

        return {
            'frame': self.frame,
            'phase': self.intersection.current_phase,
            'vehicle_counts': vehicle_counts,
            'pedestrian_counts': pedestrian_counts,
            'crossed_by_direction': self.intersection.crossed_by_direction.copy(),
            'generated_by_direction': self.intersection.generated_by_direction.copy(),
            'total_vehicles': sum(vehicle_counts),
            'total_pedestrians': sum(pedestrian_counts),
            'crossed_vehicles': self.intersection.crossed_vehicles,
            'crossed_pedestrians': self.intersection.crossed_pedestrians,
            'emergency_active': step_info.get('emergency_active', False) if step_info else False,
            'emergency_dir': step_info.get('emergency_dir', None) if step_info else None,
            'epsilon': self.agent.epsilon,
            'action': current_action_name,
            'q_values': current_q_values,
            'q_table_size': len(self.agent.q_table),
            'step_details': self.step_details[-10:] if self.step_details else [],
        }

    def draw(self, screen):
        """Draw simulation on screen."""
        self.intersection.draw(screen)

        # Draw mode indicator
        font = pygame.font.SysFont(None, 22, bold=True)
        mode_text = font.render(f"Mode: {self.mode.upper()}", True, NAVY)
        screen.blit(mode_text, (20, 20))

        # Draw speed indicator
        speed_text = font.render(f"Speed: {self.speed_multiplier}x", True, NAVY)
        screen.blit(speed_text, (20, 50))

        # Draw emergency alert
        if self.intersection.emergency_override_active:
            alert_font = pygame.font.SysFont(None, 26, bold=True)
            alert = alert_font.render("EMERGENCY VEHICLE PRIORITY ACTIVE", True, RED)
            screen.blit(alert, (WINDOW_WIDTH // 2 - 220, 20))

    def end_episode(self):
        """End current training episode and update agent metrics."""
        if self.mode == 'train':
            episode_reward = sum(self.step_rewards)
            self.agent.training_rewards.append(episode_reward)
            self.agent.training_episodes += 1
            self.agent.decay_epsilon()

        return self.episode_metrics.copy()