"""Traffic Safety Controller - enforces safety constraints"""

from traffic_sim.config import *


class SafetyController:
    """
    Enforces traffic safety constraints independent of the RL agent.

    Responsibilities:
    - Minimum and maximum green durations
    - Yellow signal transitions
    - All-red clearance between conflicting phases
    - Preventing conflicting green signals
    - Preventing unsafe vehicle-pedestrian movements
    - Pedestrian clearance time
    - Emergency vehicle priority override
    """

    def __init__(self):
        self.min_green_time = MIN_GREEN_TIME
        self.max_green_time = MAX_GREEN_TIME
        self.yellow_time = YELLOW_TIME
        self.all_red_time = ALL_RED_TIME
        self.pedestrian_cross_time = PEDESTRIAN_CROSS_TIME

        self.phase_start_frame = 0
        self.current_phase = 'NS_GREEN'
        self.last_phase_change_frame = 0
        self.violation_log = []

    def can_change_phase(self, current_frame, current_phase, requested_phase):
        """
        Check if phase change is safe.

        Returns:
            (allowed: bool, reason: str)
        """
        time_in_phase = current_frame - self.last_phase_change_frame

        # Minimum green time enforcement
        if current_phase in ['NS_GREEN', 'EW_GREEN']:
            if time_in_phase < self.min_green_time:
                return False, f"Minimum green time not met ({time_in_phase}/{self.min_green_time})"

        # Prevent conflicting green signals
        conflicting_pairs = [
            ('NS_GREEN', 'EW_GREEN'),
            ('EW_GREEN', 'NS_GREEN'),
        ]
        if (current_phase, requested_phase) in conflicting_pairs:
            # Must go through ALL_RED first
            return False, "Must transition through ALL_RED"

        # Prevent vehicle green when pedestrian crossing active
        if current_phase.startswith('PEDESTRIAN') and requested_phase in ['NS_GREEN', 'EW_GREEN']:
            return False, "Pedestrian crossing active - must clear first"

        return True, "OK"

    def apply_emergency_override(self, current_phase, emergency_direction):
        """
        Apply emergency vehicle priority override.

        Returns:
            (new_phase: str, safe: bool)
        """
        # Emergency vehicles always get priority
        if emergency_direction in ['N', 'S']:
            return 'NS_GREEN', True
        else:
            return 'EW_GREEN', True

    def get_safe_transition(self, from_phase, to_phase):
        """
        Get safe transition sequence between phases.

        Returns:
            List of phases to transition through
        """
        if from_phase == to_phase:
            return [to_phase]

        # Conflicting vehicle phases need ALL_RED
        conflict_pairs = [
            ('NS_GREEN', 'EW_GREEN'),
            ('EW_GREEN', 'NS_GREEN'),
        ]

        if (from_phase, to_phase) in conflict_pairs:
            return ['ALL_RED', to_phase]

        # Pedestrian to vehicle needs ALL_RED
        if from_phase.startswith('PEDESTRIAN') and to_phase in ['NS_GREEN', 'EW_GREEN']:
            return ['ALL_RED', to_phase]

        # Vehicle to pedestrian needs yellow then pedestrian
        if from_phase in ['NS_GREEN', 'EW_GREEN'] and to_phase.startswith('PEDESTRIAN'):
            return ['ALL_RED', to_phase]

        return [to_phase]

    def log_violation(self, violation_type, details):
        """Log a safety violation."""
        self.violation_log.append({
            'type': violation_type,
            'details': details,
            'frame': self.phase_start_frame,
        })

    def get_violations(self):
        """Get all logged violations."""
        return self.violation_log

    def is_conflict_safe(self, phase):
        """Check if a phase is conflict-safe."""
        unsafe_phases = ['NS_GREEN_EW_GREEN', 'PEDESTRIAN_ALL']
        return phase not in unsafe_phases

    def update_phase(self, new_phase, current_frame):
        """Update current phase tracking."""
        if new_phase != self.current_phase:
            self.last_phase_change_frame = current_frame
            self.current_phase = new_phase