"""Traffic State Manager - prepares RL state from simulation data"""

from traffic_sim.config import *


class TrafficStateManager:
    """
    Manages traffic state collection and preparation for Q-Learning agent.

    Responsibilities:
    - Collecting traffic conditions from intersection
    - Calculating queue lengths, waiting times, density
    - Monitoring pedestrian crowds
    - Tracking emergency vehicle positions
    - Preparing the RL state vector
    """

    def __init__(self):
        self.history = []
        self.frame_count = 0

    def collect_state(self, intersection, current_phase):
        """
        Collect all relevant traffic metrics from intersection.

        Returns:
            Dictionary with current traffic state
        """
        vehicle_counts = intersection.get_vehicle_counts()
        pedestrian_counts = intersection.get_pedestrian_counts()
        emergency_vehicles = intersection.get_emergency_vehicles()

        # Check for emergency vehicles
        emergency_present = any(len(v) > 0 for v in emergency_vehicles.values())
        emergency_direction = intersection.get_emergency_direction()

        # Calculate queue lengths (vehicles waiting at stop line)
        queue_lengths = vehicle_counts  # Simplified: all vehicles in queue

        # Calculate average waiting time
        total_waiting = sum(v.total_waiting_time for d in DIRECTIONS
                            for v in intersection.vehicles[d])
        total_vehicles = sum(vehicle_counts)
        avg_vehicle_wait = total_waiting / max(total_vehicles, 1)

        # Pedestrian waiting
        total_ped_waiting = sum(p.total_waiting_time for d in DIRECTIONS
                                for p in intersection.pedestrians[d])
        total_pedestrians = sum(pedestrian_counts)
        avg_pedestrian_wait = total_ped_waiting / max(total_pedestrians, 1)

        # Traffic density (vehicles per road length)
        density = [c / 10.0 for c in vehicle_counts]  # Normalized

        state = {
            'vehicle_counts': vehicle_counts,
            'pedestrian_counts': pedestrian_counts,
            'queue_lengths': queue_lengths,
            'emergency_present': emergency_present,
            'emergency_direction': emergency_direction,
            'avg_vehicle_waiting': avg_vehicle_wait,
            'avg_pedestrian_waiting': avg_pedestrian_wait,
            'traffic_density': density,
            'current_phase': current_phase,
            'total_vehicles': total_vehicles,
            'total_pedestrians': total_pedestrians,
            'frame': intersection.frame_count,
        }

        self.history.append(state)
        self.frame_count += 1

        return state

    def get_state_summary(self, state):
        """Get a human-readable summary of the state."""
        vc = state['vehicle_counts']
        pc = state['pedestrian_counts']
        return (
            f"Vehicles: N={vc[0]} S={vc[1]} E={vc[2]} W={vc[3]} | "
            f"Pedestrians: N={pc[0]} S={pc[1]} E={pc[2]} W={pc[3]} | "
            f"Emergency: {state['emergency_present']} | "
            f"Phase: {state['current_phase']}"
        )

    def get_recent_history(self, n=100):
        """Get last n state records."""
        return self.history[-n:]

    def compute_average_metrics(self):
        """Compute average metrics from history."""
        if not self.history:
            return {}

        avg_vehicles = np.mean([sum(s['vehicle_counts']) for s in self.history])
        avg_pedestrians = np.mean([sum(s['pedestrian_counts']) for s in self.history])
        avg_vehicle_wait = np.mean([s['avg_vehicle_waiting'] for s in self.history])
        avg_pedestrian_wait = np.mean([s['avg_pedestrian_waiting'] for s in self.history])

        return {
            'avg_vehicles': avg_vehicles,
            'avg_pedestrians': avg_pedestrians,
            'avg_vehicle_waiting': avg_vehicle_wait,
            'avg_pedestrian_waiting': avg_pedestrian_wait,
            'total_frames': self.frame_count,
        }