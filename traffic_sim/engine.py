"""Traffic Simulation Engine - Vehicles, Pedestrians, Intersection, Traffic Lights"""

import random
import numpy as np
import pygame
from traffic_sim.config import *


class Vehicle:
    """Represents a single vehicle in the simulation with smooth movement."""

    VEHICLE_ID_COUNTER = 0

    def __init__(self, direction, vehicle_type='car', is_emergency=False,
                 lane_index=0, speed_multiplier=1.0):
        Vehicle.VEHICLE_ID_COUNTER += 1
        self.id = Vehicle.VEHICLE_ID_COUNTER
        self.direction = direction  # 'N', 'S', 'E', 'W'
        self.vehicle_type = vehicle_type
        self.is_emergency = is_emergency or vehicle_type in EMERGENCY_TYPES

        props = VEHICLE_TYPES[vehicle_type]
        self.length = props['length']
        self.width = props['width']
        self.base_speed = props['base_speed']
        self.accel = props['accel']
        self.decel = props['decel']
        self.color = props['color']

        # Lane assignment within the road
        self.lane_index = lane_index
        self.lane_offset = LANE_OFFSETS[direction][lane_index % len(LANE_OFFSETS[direction])]

        # Movement state
        self.speed = 0.0  # Start from 0, accelerate up
        self.target_speed = self.base_speed * speed_multiplier
        self.max_speed = self.base_speed * speed_multiplier
        self.waiting_time = 0
        self.total_waiting_time = 0
        self.crossed = False
        self.crossing_intersection = False
        self.spawn_time = 0
        self.cross_time = None
        self.direction_crossed = False  # Track if this vehicle has crossed

        # Starting position at edge of screen
        self.pos = self._get_start_pos()

        # Vehicle state for smooth movement
        self.state = 'APPROACHING'  # APPROACHING, SLOWING, STOPPED, MOVING, CROSSING, EXITING

    def _get_start_pos(self):
        """Get starting position at the edge of the screen based on direction and lane."""
        if self.direction == 'N':
            # North-bound: starts at top, moves down
            return [INTERSECTION_X + self.lane_offset, 0]
        if self.direction == 'S':
            # South-bound: starts at bottom, moves up
            return [INTERSECTION_X + self.lane_offset, WINDOW_HEIGHT]
        if self.direction == 'E':
            # East-bound: starts at right, moves left
            return [WINDOW_WIDTH, INTERSECTION_Y + self.lane_offset]
        if self.direction == 'W':
            # West-bound: starts at left, moves right
            return [0, INTERSECTION_Y + self.lane_offset]
        return [0, 0]

    def get_stop_line(self):
        """Get the stop line position for this vehicle's direction."""
        if self.direction == 'N':
            return INTERSECTION_Y - INTERSECTION_SIZE // 2
        if self.direction == 'S':
            return INTERSECTION_Y + INTERSECTION_SIZE // 2
        if self.direction == 'E':
            return INTERSECTION_X + INTERSECTION_SIZE // 2
        if self.direction == 'W':
            return INTERSECTION_X - INTERSECTION_SIZE // 2
        return 0

    def get_exit_boundary(self):
        """Get the position where vehicle has completely exited the intersection."""
        if self.direction == 'N':
            return WINDOW_HEIGHT + self.length
        if self.direction == 'S':
            return -self.length
        if self.direction == 'E':
            return -self.width
        if self.direction == 'W':
            return WINDOW_WIDTH + self.width
        return 0

    def is_in_intersection(self):
        """Check if vehicle is currently within the intersection box."""
        if self.direction in ['N', 'S']:
            return (INTERSECTION_Y - INTERSECTION_SIZE // 2 <= self.pos[1] <=
                    INTERSECTION_Y + INTERSECTION_SIZE // 2)
        else:
            return (INTERSECTION_X - INTERSECTION_SIZE // 2 <= self.pos[0] <=
                    INTERSECTION_X + INTERSECTION_SIZE // 2)

    def get_front_distance(self, other_vehicle):
        """Get distance to the vehicle in front (same direction, same lane)."""
        if self.direction == 'N':
            return other_vehicle.pos[1] - self.pos[1]
        if self.direction == 'S':
            return self.pos[1] - other_vehicle.pos[1]
        if self.direction == 'E':
            return self.pos[0] - other_vehicle.pos[0]
        if self.direction == 'W':
            return other_vehicle.pos[0] - self.pos[0]
        return float('inf')

    def update(self, can_cross, emergency_override=False, all_red=False,
               speed_multiplier=1.0, vehicles_in_direction=None):
        """Update vehicle position with smooth acceleration/deceleration.

        Args:
            can_cross: Whether the traffic light allows crossing
            emergency_override: Whether emergency override is active
            all_red: Whether all lights are red
            speed_multiplier: Global simulation speed multiplier
            vehicles_in_direction: List of all vehicles in same direction (for following distance)

        Returns:
            True if vehicle has completely exited the simulation
        """
        if self.crossed:
            return False

        # Safety: all red means everyone stops
        if all_red:
            self._apply_deceleration(speed_multiplier)
            self.waiting_time += 1
            self.total_waiting_time += 1
            self.state = 'STOPPED'
            return False

        # Emergency vehicles get priority
        effective_can_cross = can_cross or self.is_emergency or emergency_override

        stop_line = self.get_stop_line()

        # Check following distance - don't get too close to vehicle ahead
        too_close = False
        if vehicles_in_direction:
            for other in vehicles_in_direction:
                if other is self:
                    continue
                # Check if other vehicle is in front (same lane approx)
                if abs(other.lane_offset - self.lane_offset) < 30:
                    dist = self.get_front_distance(other)
                    if 0 < dist < SAFE_FOLLOWING_DISTANCE + self.length:
                        too_close = True
                        break

        # Determine if vehicle should move
        should_move = effective_can_cross or self.is_emergency or emergency_override

        # Check if at stop line
        at_stop_line = False
        if self.direction == 'N':
            at_stop_line = self.pos[1] >= stop_line - 5
        elif self.direction == 'S':
            at_stop_line = self.pos[1] <= stop_line + 5
        elif self.direction == 'E':
            at_stop_line = self.pos[0] <= stop_line + 5
        elif self.direction == 'W':
            at_stop_line = self.pos[0] >= stop_line - 5

        # Slow down when approaching stop line and light is red
        approaching_stop = False
        if not effective_can_cross and not self.is_emergency:
            if self.direction == 'N':
                approaching_stop = (stop_line - SLOW_DOWN_DISTANCE <= self.pos[1] < stop_line)
            elif self.direction == 'S':
                approaching_stop = (stop_line < self.pos[1] <= stop_line + SLOW_DOWN_DISTANCE)
            elif self.direction == 'E':
                approaching_stop = (stop_line < self.pos[0] <= stop_line + SLOW_DOWN_DISTANCE)
            elif self.direction == 'W':
                approaching_stop = (stop_line - SLOW_DOWN_DISTANCE <= self.pos[0] < stop_line)

        # Update movement state
        if too_close or (at_stop_line and not should_move) or approaching_stop:
            self._apply_deceleration(speed_multiplier)
            self.waiting_time += 1
            self.total_waiting_time += 1
            self.state = 'STOPPED' if self.speed <= 0.1 else 'SLOWING'
        else:
            self._apply_acceleration(speed_multiplier)
            self.state = 'MOVING'

        # Apply movement
        if self.direction == 'N':
            self.pos[1] += self.speed
            if self.pos[1] > WINDOW_HEIGHT + self.length:
                self.crossed = True
                self.cross_time = self.spawn_time + self.waiting_time
                self.direction_crossed = True
                return True
        elif self.direction == 'S':
            self.pos[1] -= self.speed
            if self.pos[1] < -self.length:
                self.crossed = True
                self.cross_time = self.spawn_time + self.waiting_time
                self.direction_crossed = True
                return True
        elif self.direction == 'E':
            self.pos[0] -= self.speed
            if self.pos[0] < -self.width:
                self.crossed = True
                self.cross_time = self.spawn_time + self.waiting_time
                self.direction_crossed = True
                return True
        elif self.direction == 'W':
            self.pos[0] += self.speed
            if self.pos[0] > WINDOW_WIDTH + self.width:
                self.crossed = True
                self.cross_time = self.spawn_time + self.waiting_time
                self.direction_crossed = True
                return True

        return False

    def _apply_acceleration(self, speed_multiplier):
        """Gradually accelerate towards target speed."""
        target = self.target_speed * speed_multiplier
        if self.speed < target:
            self.speed = min(target, self.speed + self.accel * speed_multiplier)

    def _apply_deceleration(self, speed_multiplier):
        """Gradually decelerate to stop."""
        if self.speed > 0:
            self.speed = max(0, self.speed - self.decel * speed_multiplier)

    def draw(self, screen):
        """Draw vehicle on the screen with proper orientation and visibility."""
        if self.crossed:
            return

        # Calculate drawing rectangle based on direction
        if self.direction in ['N', 'S']:
            # Vertical vehicles: width is horizontal dimension, length is vertical
            if self.direction == 'N':
                x = INTERSECTION_X + self.lane_offset - self.width // 2
                y = int(self.pos[1])
            else:
                x = INTERSECTION_X + self.lane_offset - self.width // 2
                y = int(self.pos[1]) - self.length
            rect = pygame.Rect(x, y, self.width, self.length)
        else:
            # Horizontal vehicles: width is vertical dimension, length is horizontal
            if self.direction == 'E':
                x = int(self.pos[0])
                y = INTERSECTION_Y + self.lane_offset - self.width // 2
            else:
                x = int(self.pos[0]) - self.length
                y = INTERSECTION_Y + self.lane_offset - self.width // 2
            rect = pygame.Rect(x, y, self.length, self.width)

        # Ensure rectangle is within screen bounds for visibility
        rect.clamp_ip(screen.get_rect())

        # Emergency vehicle highlighting
        if self.is_emergency:
            # White body with red stripe and emergency markings
            pygame.draw.rect(screen, (255, 255, 240), rect, border_radius=4)
            pygame.draw.rect(screen, RED, rect, 3, border_radius=4)
            # Emergency stripe
            if self.direction in ['N', 'S']:
                stripe = pygame.Rect(rect.x, rect.y + 4, rect.w, 6)
            else:
                stripe = pygame.Rect(rect.x + 4, rect.y, 6, rect.h)
            pygame.draw.rect(screen, RED, stripe, border_radius=2)
            # "AMBULANCE" or "POLICE" text
            font = pygame.font.SysFont(None, 14)
            label = self.vehicle_type.upper()
            text = font.render(label, True, RED)
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
        else:
            # Normal vehicle with outline for visibility
            pygame.draw.rect(screen, self.color, rect, border_radius=4)
            pygame.draw.rect(screen, BLACK, rect, 2, border_radius=4)

            # Vehicle type label
            font = pygame.font.SysFont(None, 12)
            label = self.vehicle_type.replace('_', ' ').title()
            text = font.render(label, True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            # Background for label readability
            bg = text_rect.inflate(4, 2)
            pygame.draw.rect(screen, WHITE, bg, border_radius=2)
            screen.blit(text, text_rect)


class Pedestrian:
    """Represents a pedestrian waiting or crossing at zebra crossing."""

    PEDESTRIAN_ID_COUNTER = 0

    def __init__(self, side):
        Pedestrian.PEDESTRIAN_ID_COUNTER += 1
        self.id = Pedestrian.PEDESTRIAN_ID_COUNTER
        self.side = side  # 'N', 'S', 'E', 'W'
        self.waiting_time = 0
        self.total_waiting_time = 0
        self.crossed = False
        self.crossing = False
        self.pos = self._get_start_pos()
        self.speed = 1.5
        self.color = (200, 100, 50)

    def _get_start_pos(self):
        """Get starting position near zebra crossing."""
        offset = random.randint(-40, 40)
        if self.side == 'N':
            return [INTERSECTION_X + offset, INTERSECTION_Y - INTERSECTION_SIZE // 2 - 25]
        if self.side == 'S':
            return [INTERSECTION_X + offset, INTERSECTION_Y + INTERSECTION_SIZE // 2 + 25]
        if self.side == 'E':
            return [INTERSECTION_X + INTERSECTION_SIZE // 2 + 25, INTERSECTION_Y + offset]
        if self.side == 'W':
            return [INTERSECTION_X - INTERSECTION_SIZE // 2 - 25, INTERSECTION_Y + offset]
        return [0, 0]

    def update(self, can_cross):
        """Update pedestrian state."""
        if self.crossed:
            return False

        if not self.crossing and not can_cross:
            self.waiting_time += 1
            self.total_waiting_time += 1
            return False

        if not self.crossing and can_cross:
            self.crossing = True
            self.color = (0, 255, 0)  # Green when crossing

        if self.crossing:
            if self.side == 'N':
                self.pos[1] += self.speed
                if self.pos[1] > INTERSECTION_Y + INTERSECTION_SIZE // 2 + 25:
                    self.crossed = True
                    return True
            elif self.side == 'S':
                self.pos[1] -= self.speed
                if self.pos[1] < INTERSECTION_Y - INTERSECTION_SIZE // 2 - 25:
                    self.crossed = True
                    return True
            elif self.side == 'E':
                self.pos[0] -= self.speed
                if self.pos[0] < INTERSECTION_X - INTERSECTION_SIZE // 2 - 25:
                    self.crossed = True
                    return True
            elif self.side == 'W':
                self.pos[0] += self.speed
                if self.pos[0] > INTERSECTION_X + INTERSECTION_SIZE // 2 + 25:
                    self.crossed = True
                    return True

        return False

    def draw(self, screen):
        """Draw pedestrian on screen."""
        if self.crossed:
            return

        # Draw pedestrian as a small person figure
        x, y = int(self.pos[0]), int(self.pos[1])
        color = (0, 255, 0) if self.crossing else (200, 100, 50)

        # Body
        pygame.draw.circle(screen, color, (x, y), 7)
        pygame.draw.circle(screen, BLACK, (x, y), 7, 2)
        # Legs
        pygame.draw.line(screen, color, (x - 3, y + 5), (x - 3, y + 12), 2)
        pygame.draw.line(screen, color, (x + 3, y + 5), (x + 3, y + 12), 2)
        # Arms
        pygame.draw.line(screen, color, (x - 5, y + 2), (x + 5, y + 2), 2)


class TrafficLight:
    """Traffic light for one direction."""

    def __init__(self, direction):
        self.direction = direction
        self.state = 'RED'  # RED, YELLOW, GREEN
        self.timer = 0
        self.phase_start_time = 0

    def set(self, state, duration):
        self.state = state
        self.timer = duration

    def update(self):
        if self.timer > 0:
            self.timer -= 1
        if self.timer == 0:
            if self.state == 'GREEN':
                self.state = 'YELLOW'
                self.timer = YELLOW_TIME
            elif self.state == 'YELLOW':
                self.state = 'RED'
                self.timer = 0
            elif self.state == 'RED':
                self.state = 'RED'
                self.timer = 0

    def is_green(self):
        return self.state == 'GREEN'

    def is_yellow(self):
        return self.state == 'YELLOW'

    def is_red(self):
        return self.state == 'RED'

    def draw(self, screen):
        positions = {
            'N': (INTERSECTION_X - 90, INTERSECTION_Y - 130),
            'S': (INTERSECTION_X + 90, INTERSECTION_Y + 130),
            'E': (INTERSECTION_X + 130, INTERSECTION_Y - 90),
            'W': (INTERSECTION_X - 130, INTERSECTION_Y + 90)
        }
        x, y = positions[self.direction]
        # Draw housing
        pygame.draw.rect(screen, DARK_GRAY, (x - 20, y - 20, 40, 60), border_radius=5)
        # Draw light
        color = {'RED': RED, 'YELLOW': YELLOW, 'GREEN': GREEN}[self.state]
        pygame.draw.circle(screen, color, (x, y), LIGHT_RADIUS)
        pygame.draw.circle(screen, BLACK, (x, y), LIGHT_RADIUS, 2)
        # Draw state label
        font = pygame.font.SysFont(None, 16, bold=True)
        text = font.render(self.state[:1], True, WHITE if self.state != 'WHITE' else BLACK)
        text_rect = text.get_rect(center=(x, y + 35))
        screen.blit(text, text_rect)


class Intersection:
    """Manages the four-way intersection with vehicles, pedestrians, and lights."""

    def __init__(self):
        self.vehicles = {d: [] for d in DIRECTIONS}
        self.pedestrians = {d: [] for d in DIRECTIONS}
        self.lights = {d: TrafficLight(d) for d in DIRECTIONS}

        self.current_phase = 'NS_GREEN'  # NS_GREEN, EW_GREEN, PEDESTRIAN, ALL_RED
        self.phase_timer = 0

        # Direction-wise crossing counters
        self.crossed_by_direction = {d: 0 for d in DIRECTIONS}
        self.crossed_vehicles = 0
        self.crossed_pedestrians = 0
        self.total_vehicle_waiting = 0
        self.total_pedestrian_waiting = 0
        self.total_emergency_waiting = 0
        self.signal_changes = 0
        self.emergency_crossed = 0
        self.emergency_override_active = False
        self.frame_count = 0

        # Vehicles generated count per direction
        self.generated_by_direction = {d: 0 for d in DIRECTIONS}

        # Initialize NS green
        self.lights['N'].set('GREEN', MAX_GREEN_TIME)
        self.lights['S'].set('GREEN', MAX_GREEN_TIME)
        self.lights['E'].set('RED', 0)
        self.lights['W'].set('RED', 0)

    def spawn_vehicle(self, direction, spawn_prob, emergency_prob,
                      speed_multiplier=1.0):
        """Randomly spawn a vehicle on a road."""
        if random.random() < spawn_prob:
            # Weighted vehicle type distribution for Indian traffic
            types = ['car', 'motorcycle', 'scooter', 'auto_rickshaw', 'bus', 'truck']
            weights = [0.30, 0.25, 0.20, 0.15, 0.05, 0.05]
            vehicle_type = random.choices(types, weights=weights)[0]

            is_emergency = random.random() < emergency_prob
            if is_emergency:
                vehicle_type = random.choice(['ambulance', 'police'])

            # Assign lane (0, 1, 2)
            lane_index = random.randint(0, 2)

            v = Vehicle(direction, vehicle_type, is_emergency,
                       lane_index=lane_index, speed_multiplier=speed_multiplier)
            v.spawn_time = self.frame_count
            self.vehicles[direction].append(v)
            self.generated_by_direction[direction] += 1

    def spawn_pedestrian(self, direction, prob):
        """Randomly spawn a pedestrian on a side."""
        if random.random() < prob:
            self.pedestrians[direction].append(Pedestrian(direction))

    def set_phase(self, phase):
        """Set traffic signal phase with safety transitions."""
        if phase == self.current_phase:
            return

        # Safety: all-red clearance between conflicting phases
        if self.current_phase in ['NS_GREEN', 'EW_GREEN'] and phase in ['NS_GREEN', 'EW_GREEN']:
            # Switching between conflicting vehicle phases
            self.current_phase = 'ALL_RED'
            self.phase_timer = ALL_RED_TIME
            for d in DIRECTIONS:
                self.lights[d].set('RED', ALL_RED_TIME)
            self.signal_changes += 1
            return

        self.current_phase = phase
        self.signal_changes += 1

        if phase == 'NS_GREEN':
            self.lights['N'].set('GREEN', MAX_GREEN_TIME)
            self.lights['S'].set('GREEN', MAX_GREEN_TIME)
            self.lights['E'].set('RED', 0)
            self.lights['W'].set('RED', 0)
            self.phase_timer = MAX_GREEN_TIME

        elif phase == 'EW_GREEN':
            self.lights['E'].set('GREEN', MAX_GREEN_TIME)
            self.lights['W'].set('GREEN', MAX_GREEN_TIME)
            self.lights['N'].set('RED', 0)
            self.lights['S'].set('RED', 0)
            self.phase_timer = MAX_GREEN_TIME

        elif phase.startswith('PEDESTRIAN'):
            side = phase[-1]  # N, S, E, or W
            # Stop all vehicles first
            for d in DIRECTIONS:
                self.lights[d].set('RED', 0)
            self.phase_timer = PEDESTRIAN_CROSS_TIME

        elif phase == 'ALL_RED':
            for d in DIRECTIONS:
                self.lights[d].set('RED', ALL_RED_TIME)
            self.phase_timer = ALL_RED_TIME

    def get_vehicle_counts(self):
        return [len(self.vehicles[d]) for d in DIRECTIONS]

    def get_pedestrian_counts(self):
        return [len(self.pedestrians[d]) for d in DIRECTIONS]

    def get_generated_counts(self):
        return [self.generated_by_direction[d] for d in DIRECTIONS]

    def get_crossed_by_direction(self):
        return [self.crossed_by_direction[d] for d in DIRECTIONS]

    def get_emergency_vehicles(self):
        """Return dict of direction -> list of emergency vehicles."""
        result = {}
        for d in DIRECTIONS:
            result[d] = [v for v in self.vehicles[d] if v.is_emergency]
        return result

    def get_emergency_direction(self):
        """Get direction of nearest emergency vehicle, or None."""
        emergencies = self.get_emergency_vehicles()
        for d in DIRECTIONS:
            if emergencies[d]:
                closest = min(emergencies[d],
                             key=lambda v: v.pos[1] if v.direction in ['N', 'S'] else v.pos[0])
                return d
        return None

    def update(self, spawn_prob, emergency_prob, pedestrian_prob,
               speed_multiplier=1.0, action=None):
        """Update intersection state."""
        self.frame_count += 1

        # Spawn vehicles and pedestrians
        for d in DIRECTIONS:
            self.spawn_vehicle(d, spawn_prob, emergency_prob, speed_multiplier)
            self.spawn_pedestrian(d, pedestrian_prob)

        # Check for emergency vehicles - priority override
        emergency_dir = self.get_emergency_direction()
        emergency_active = emergency_dir is not None

        # Update traffic lights
        for d in DIRECTIONS:
            self.lights[d].update()

        # Handle phase transitions
        if self.phase_timer > 0:
            self.phase_timer -= 1

        if self.phase_timer == 0 and not emergency_active:
            if action is not None:
                self._apply_action(action)

        # Emergency override
        if emergency_active and self.current_phase not in ['NS_GREEN', 'EW_GREEN']:
            self.set_phase('NS_GREEN' if emergency_dir in ['N', 'S'] else 'EW_GREEN')

        # Update vehicles
        for direction in DIRECTIONS:
            can_cross = self.lights[direction].is_green() or self.lights[direction].is_yellow()
            all_red = self.current_phase == 'ALL_RED'

            # Emergency override for this direction
            emergency_for_dir = (emergency_active and
                                ((direction in ['N', 'S'] and emergency_dir in ['N', 'S']) or
                                 (direction in ['E', 'W'] and emergency_dir in ['E', 'W'])))

            for v in self.vehicles[direction]:
                crossed = v.update(can_cross,
                                   emergency_override=emergency_for_dir,
                                   all_red=all_red,
                                   speed_multiplier=speed_multiplier,
                                   vehicles_in_direction=self.vehicles[direction])
                if crossed:
                    self.crossed_vehicles += 1
                    self.crossed_by_direction[direction] += 1
                    self.total_vehicle_waiting += v.total_waiting_time
                    if v.is_emergency:
                        self.emergency_crossed += 1

            # Remove crossed vehicles
            self.vehicles[direction] = [v for v in self.vehicles[direction] if not v.crossed]

        # Update pedestrians
        pedestrian_can_cross = (self.current_phase.startswith('PEDESTRIAN') or
                               self.current_phase == 'ALL_RED')

        for direction in DIRECTIONS:
            for p in self.pedestrians[direction]:
                crossed = p.update(pedestrian_can_cross)
                if crossed:
                    self.crossed_pedestrians += 1
                    self.total_pedestrian_waiting += p.total_waiting_time

            self.pedestrians[direction] = [p for p in self.pedestrians[direction] if not p.crossed]

        # Count current waiting
        current_vehicle_waiting = sum(len(self.vehicles[d]) for d in DIRECTIONS)
        current_pedestrian_waiting = sum(len(self.pedestrians[d]) for d in DIRECTIONS)

        return {
            'emergency_active': emergency_active,
            'emergency_dir': emergency_dir,
            'current_phase': self.current_phase,
            'vehicle_counts': self.get_vehicle_counts(),
            'pedestrian_counts': self.get_pedestrian_counts(),
            'crossed_by_direction': self.get_crossed_by_direction(),
            'generated_by_direction': self.get_generated_counts(),
            'current_vehicle_waiting': current_vehicle_waiting,
            'current_pedestrian_waiting': current_pedestrian_waiting,
        }

    def _apply_action(self, action):
        """Apply Q-Learning action with safety checks."""
        action_name = ACTIONS[action] if isinstance(action, int) else action

        if action_name == 'NS_GREEN':
            self.set_phase('NS_GREEN')
        elif action_name == 'EW_GREEN':
            self.set_phase('EW_GREEN')
        elif action_name == 'PEDESTRIAN_N':
            self.set_phase('PEDESTRIAN_N')
        elif action_name == 'PEDESTRIAN_S':
            self.set_phase('PEDESTRIAN_S')
        elif action_name == 'PEDESTRIAN_E':
            self.set_phase('PEDESTRIAN_E')
        elif action_name == 'PEDESTRIAN_W':
            self.set_phase('PEDESTRIAN_W')
        elif action_name == 'EXTEND_GREEN':
            if self.current_phase == 'NS_GREEN':
                self.lights['N'].timer = MAX_GREEN_TIME
                self.lights['S'].timer = MAX_GREEN_TIME
                self.phase_timer = MAX_GREEN_TIME
            elif self.current_phase == 'EW_GREEN':
                self.lights['E'].timer = MAX_GREEN_TIME
                self.lights['W'].timer = MAX_GREEN_TIME
                self.phase_timer = MAX_GREEN_TIME

    def get_total_waiting(self):
        return sum(len(self.vehicles[d]) for d in DIRECTIONS)

    def get_total_pedestrian_waiting(self):
        return sum(len(self.pedestrians[d]) for d in DIRECTIONS)

    def draw_roads(self, screen):
        """Draw the intersection roads with lane markings."""
        # North road
        pygame.draw.rect(screen, ROAD_GRAY,
                         (INTERSECTION_X - ROAD_WIDTH // 2, 0,
                          ROAD_WIDTH, INTERSECTION_Y - INTERSECTION_SIZE // 2))
        # South road
        pygame.draw.rect(screen, ROAD_GRAY,
                         (INTERSECTION_X - ROAD_WIDTH // 2,
                          INTERSECTION_Y + INTERSECTION_SIZE // 2,
                          ROAD_WIDTH, WINDOW_HEIGHT - INTERSECTION_Y - INTERSECTION_SIZE // 2))
        # East road
        pygame.draw.rect(screen, ROAD_GRAY,
                         (INTERSECTION_X + INTERSECTION_SIZE // 2,
                          INTERSECTION_Y - ROAD_WIDTH // 2,
                          WINDOW_WIDTH - INTERSECTION_X - INTERSECTION_SIZE // 2,
                          ROAD_WIDTH))
        # West road
        pygame.draw.rect(screen, ROAD_GRAY,
                         (0, INTERSECTION_Y - ROAD_WIDTH // 2,
                          INTERSECTION_X - INTERSECTION_SIZE // 2,
                          ROAD_WIDTH))
        # Intersection box
        pygame.draw.rect(screen, ROAD_GRAY,
                         (INTERSECTION_X - INTERSECTION_SIZE // 2,
                          INTERSECTION_Y - INTERSECTION_SIZE // 2,
                          INTERSECTION_SIZE, INTERSECTION_SIZE))

        # Center road divider lines
        pygame.draw.line(screen, WHITE,
                        (INTERSECTION_X, 0), (INTERSECTION_X, WINDOW_HEIGHT), 3)
        pygame.draw.line(screen, WHITE,
                        (0, INTERSECTION_Y), (WINDOW_WIDTH, INTERSECTION_Y), 3)

        # Lane divider lines (dashed)
        lane_width = ROAD_WIDTH // 3
        for d in DIRECTIONS:
            if d == 'N':
                for offset in [-lane_width//2, lane_width//2]:
                    x = INTERSECTION_X + offset
                    for y in range(0, INTERSECTION_Y - INTERSECTION_SIZE//2, 30):
                        pygame.draw.line(screen, WHITE, (x, y), (x, min(y+15, WINDOW_HEIGHT)), 2)
            elif d == 'S':
                for offset in [-lane_width//2, lane_width//2]:
                    x = INTERSECTION_X + offset
                    for y in range(INTERSECTION_Y + INTERSECTION_SIZE//2, WINDOW_HEIGHT, 30):
                        pygame.draw.line(screen, WHITE, (x, y), (x, min(y+15, WINDOW_HEIGHT)), 2)
            elif d == 'E':
                for offset in [-lane_width//2, lane_width//2]:
                    y = INTERSECTION_Y + offset
                    for x in range(INTERSECTION_X + INTERSECTION_SIZE//2, WINDOW_WIDTH, 30):
                        pygame.draw.line(screen, WHITE, (x, y), (min(x+15, WINDOW_WIDTH), y), 2)
            elif d == 'W':
                for offset in [-lane_width//2, lane_width//2]:
                    y = INTERSECTION_Y + offset
                    for x in range(0, INTERSECTION_X - INTERSECTION_SIZE//2, 30):
                        pygame.draw.line(screen, WHITE, (x, y), (min(x+15, WINDOW_WIDTH), y), 2)

        # Stop lines
        stop_color = STOP_LINE_COLOR
        # North stop line
        pygame.draw.line(screen, stop_color,
                        (INTERSECTION_X - ROAD_WIDTH//2, INTERSECTION_Y - INTERSECTION_SIZE//2),
                        (INTERSECTION_X + ROAD_WIDTH//2, INTERSECTION_Y - INTERSECTION_SIZE//2), 4)
        # South stop line
        pygame.draw.line(screen, stop_color,
                        (INTERSECTION_X - ROAD_WIDTH//2, INTERSECTION_Y + INTERSECTION_SIZE//2),
                        (INTERSECTION_X + ROAD_WIDTH//2, INTERSECTION_Y + INTERSECTION_SIZE//2), 4)
        # East stop line
        pygame.draw.line(screen, stop_color,
                        (INTERSECTION_X + INTERSECTION_SIZE//2, INTERSECTION_Y - ROAD_WIDTH//2),
                        (INTERSECTION_X + INTERSECTION_SIZE//2, INTERSECTION_Y + ROAD_WIDTH//2), 4)
        # West stop line
        pygame.draw.line(screen, stop_color,
                        (INTERSECTION_X - INTERSECTION_SIZE//2, INTERSECTION_Y - ROAD_WIDTH//2),
                        (INTERSECTION_X - INTERSECTION_SIZE//2, INTERSECTION_Y + ROAD_WIDTH//2), 4)

        # Zebra crossings
        zebra_color = ZEBRA_WHITE
        # North zebra (above intersection)
        for i in range(-INTERSECTION_SIZE // 2, INTERSECTION_SIZE // 2, 12):
            pygame.draw.rect(screen, zebra_color,
                             (INTERSECTION_X + i, INTERSECTION_Y - INTERSECTION_SIZE // 2 - 20, 8, 16))
        # South zebra (below intersection)
        for i in range(-INTERSECTION_SIZE // 2, INTERSECTION_SIZE // 2, 12):
            pygame.draw.rect(screen, zebra_color,
                             (INTERSECTION_X + i, INTERSECTION_Y + INTERSECTION_SIZE // 2 + 4, 8, 16))
        # East zebra (right of intersection)
        for i in range(-INTERSECTION_SIZE // 2, INTERSECTION_SIZE // 2, 12):
            pygame.draw.rect(screen, zebra_color,
                             (INTERSECTION_X + INTERSECTION_SIZE // 2 + 4, INTERSECTION_Y + i, 16, 8))
        # West zebra (left of intersection)
        for i in range(-INTERSECTION_SIZE // 2, INTERSECTION_SIZE // 2, 12):
            pygame.draw.rect(screen, zebra_color,
                             (INTERSECTION_X - INTERSECTION_SIZE // 2 - 20, INTERSECTION_Y + i, 16, 8))

    def draw(self, screen):
        """Draw entire intersection."""
        self.draw_roads(screen)

        # Draw vehicles (sorted by distance to draw further ones first for proper overlap)
        all_vehicles = []
        for direction in DIRECTIONS:
            for v in self.vehicles[direction]:
                all_vehicles.append(v)

        # Sort by distance from camera (rough approximation)
        all_vehicles.sort(key=lambda v: (
            v.pos[1] if v.direction in ['N', 'S'] else v.pos[0]
        ), reverse=(self.current_phase == 'NS_GREEN'))

        for v in all_vehicles:
            v.draw(screen)

        # Draw pedestrians
        for direction in DIRECTIONS:
            for p in self.pedestrians[direction]:
                p.draw(screen)

        # Draw traffic lights
        for d in DIRECTIONS:
            self.lights[d].draw(screen)

        # Draw direction labels and counts
        font = pygame.font.SysFont(None, 20, bold=True)
        counts = self.get_vehicle_counts()
        crossed = self.get_crossed_by_direction()
        generated = self.get_generated_counts()
        ped_counts = self.get_pedestrian_counts()

        labels = {
            'N': (INTERSECTION_X - 110, 20),
            'S': (INTERSECTION_X - 110, WINDOW_HEIGHT - 40),
            'E': (WINDOW_WIDTH - 220, INTERSECTION_Y - 25),
            'W': (20, INTERSECTION_Y - 25)
        }
        for d, pos in labels.items():
            idx = DIRECTIONS.index(d)
            v_count = counts[idx]
            c_count = crossed[idx]
            g_count = generated[idx]
            p_count = ped_counts[idx]
            color = RED if v_count > 10 else (ORANGE if v_count > 5 else GREEN)
            # Main label
            text = font.render(f"{DIRECTION_NAMES[d]}: {v_count}", True, color)
            screen.blit(text, pos)
            # Crossed count below
            text2 = font.render(f"Crossed: {c_count}/{g_count}", True, BLACK)
            screen.blit(text2, (pos[0], pos[1] + 22))
            # Pedestrian count
            text3 = font.render(f"Peds: {p_count}", True, PURPLE)
            screen.blit(text3, (pos[0], pos[1] + 44))

        # Draw current phase prominently
        phase_font = pygame.font.SysFont(None, 28, bold=True)
        phase_text = phase_font.render(f"PHASE: {self.current_phase}", True, NAVY)
        phase_rect = phase_text.get_rect(center=(INTERSECTION_X, INTERSECTION_Y - INTERSECTION_SIZE // 2 - 50))
        # Background for phase text
        bg_rect = phase_rect.inflate(20, 10)
        pygame.draw.rect(screen, LIGHT_GREEN, bg_rect, border_radius=8)
        pygame.draw.rect(screen, NAVY, bg_rect, 2, border_radius=8)
        screen.blit(phase_text, phase_rect)

        # Draw frame counter
        info_font = pygame.font.SysFont(None, 18)
        frame_text = info_font.render(f"Frame: {self.frame_count}", True, BLACK)
        screen.blit(frame_text, (WINDOW_WIDTH - 150, 20))