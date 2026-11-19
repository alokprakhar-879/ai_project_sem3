"""Traffic Simulation Engine - Vehicles, Pedestrians, Intersection, Traffic Lights"""

import random
import numpy as np
import pygame
from traffic_sim.config import *


class Vehicle:
    """Represents a single vehicle in the simulation."""

    VEHICLE_ID_COUNTER = 0

    def __init__(self, direction, vehicle_type='car', is_emergency=False):
        Vehicle.VEHICLE_ID_COUNTER += 1
        self.id = Vehicle.VEHICLE_ID_COUNTER
        self.direction = direction  # 'N', 'S', 'E', 'W'
        self.vehicle_type = vehicle_type
        self.is_emergency = is_emergency or vehicle_type in EMERGENCY_TYPES

        props = VEHICLE_TYPES[vehicle_type]
        self.length = props['length']
        self.width = props['width']
        self.speed = props['speed']
        self.color = props['color']

        self.waiting_time = 0
        self.total_waiting_time = 0
        self.crossed = False
        self.spawn_time = 0
        self.cross_time = None

        # Starting position at edge of screen
        self.pos = self._get_start_pos()

    def _get_start_pos(self):
        """Get starting position at the edge of the screen based on direction."""
        if self.direction == 'N':
            return [INTERSECTION_X - ROAD_WIDTH // 4, 0]
        if self.direction == 'S':
            return [INTERSECTION_X + ROAD_WIDTH // 4, WINDOW_HEIGHT]
        if self.direction == 'E':
            return [WINDOW_WIDTH, INTERSECTION_Y - ROAD_WIDTH // 4]
        if self.direction == 'W':
            return [0, INTERSECTION_Y + ROAD_WIDTH // 4]
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

    def update(self, can_cross, emergency_override=False, all_red=False):
        """Update vehicle position. Returns True if vehicle crossed intersection."""
        if self.crossed:
            return False

        # Safety: all red means everyone stops
        if all_red:
            self.waiting_time += 1
            self.total_waiting_time += 1
            return False

        # Emergency vehicles get priority
        effective_can_cross = can_cross or self.is_emergency or emergency_override

        stop_line = self.get_stop_line()

        if self.direction == 'N':
            if self.pos[1] < stop_line or effective_can_cross:
                self.pos[1] += self.speed
            else:
                self.waiting_time += 1
                self.total_waiting_time += 1
            if self.pos[1] > WINDOW_HEIGHT:
                self.crossed = True
                self.cross_time = self.spawn_time + self.waiting_time
                return True

        elif self.direction == 'S':
            if self.pos[1] > stop_line or effective_can_cross:
                self.pos[1] -= self.speed
            else:
                self.waiting_time += 1
                self.total_waiting_time += 1
            if self.pos[1] < 0:
                self.crossed = True
                self.cross_time = self.spawn_time + self.waiting_time
                return True

        elif self.direction == 'E':
            if self.pos[0] > stop_line or effective_can_cross:
                self.pos[0] -= self.speed
            else:
                self.waiting_time += 1
                self.total_waiting_time += 1
            if self.pos[0] < 0:
                self.crossed = True
                self.cross_time = self.spawn_time + self.waiting_time
                return True

        elif self.direction == 'W':
            if self.pos[0] < stop_line or effective_can_cross:
                self.pos[0] += self.speed
            else:
                self.waiting_time += 1
                self.total_waiting_time += 1
            if self.pos[0] > WINDOW_WIDTH:
                self.crossed = True
                self.cross_time = self.spawn_time + self.waiting_time
                return True

        return False

    def draw(self, screen):
        """Draw vehicle on the screen."""
        if self.crossed:
            return

        font = None
        try:
            font = pygame_font = __import__('pygame').font.SysFont(None, 12)
        except Exception:
            pass

        if self.direction in ['N', 'S']:
            if self.direction == 'N':
                x = INTERSECTION_X - ROAD_WIDTH // 4
                y = int(self.pos[1])
            else:
                x = INTERSECTION_X + ROAD_WIDTH // 4 - self.length
                y = int(self.pos[1])
            rect = pygame.Rect(x, y, self.length, self.width)
        else:
            if self.direction == 'E':
                x = int(self.pos[0])
                y = INTERSECTION_Y - ROAD_WIDTH // 4
            else:
                x = int(self.pos[0])
                y = INTERSECTION_Y + ROAD_WIDTH // 4 - self.width
            rect = pygame.Rect(x, y, self.width, self.length)

        # Emergency vehicle highlighting
        if self.is_emergency:
            pygame.draw.rect(screen, YELLOW, rect, border_radius=3)
            pygame.draw.rect(screen, RED, rect, 3, border_radius=3)
            # Draw emergency stripe
            stripe = pygame.Rect(rect.x + 2, rect.y + 2, rect.w - 4, 4)
            pygame.draw.rect(screen, RED, stripe)
        else:
            pygame.draw.rect(screen, self.color, rect, border_radius=3)
            pygame.draw.rect(screen, BLACK, rect, 1, border_radius=3)

        # Vehicle type label
        if font:
            label = self.vehicle_type.replace('_', ' ').title()
            if self.is_emergency:
                label = 'EMERGENCY: ' + label.upper()
            text = font.render(label, True, BLACK if not self.is_emergency else RED)
            text_rect = text.get_rect(center=rect.center)
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
        self.speed = 1.8

    def _get_start_pos(self):
        """Get starting position near zebra crossing."""
        if self.side == 'N':
            return [INTERSECTION_X + random.randint(-30, 30), INTERSECTION_Y - INTERSECTION_SIZE // 2 - 20]
        if self.side == 'S':
            return [INTERSECTION_X + random.randint(-30, 30), INTERSECTION_Y + INTERSECTION_SIZE // 2 + 20]
        if self.side == 'E':
            return [INTERSECTION_X + INTERSECTION_SIZE // 2 + 20, INTERSECTION_Y + random.randint(-30, 30)]
        if self.side == 'W':
            return [INTERSECTION_X - INTERSECTION_SIZE // 2 - 20, INTERSECTION_Y + random.randint(-30, 30)]
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

        if self.crossing:
            if self.side == 'N':
                self.pos[1] += self.speed
                if self.pos[1] > INTERSECTION_Y + INTERSECTION_SIZE // 2 + 20:
                    self.crossed = True
                    return True
            elif self.side == 'S':
                self.pos[1] -= self.speed
                if self.pos[1] < INTERSECTION_Y - INTERSECTION_SIZE // 2 - 20:
                    self.crossed = True
                    return True
            elif self.side == 'E':
                self.pos[0] -= self.speed
                if self.pos[0] < INTERSECTION_X - INTERSECTION_SIZE // 2 - 20:
                    self.crossed = True
                    return True
            elif self.side == 'W':
                self.pos[0] += self.speed
                if self.pos[0] > INTERSECTION_X + INTERSECTION_SIZE // 2 + 20:
                    self.crossed = True
                    return True

        return False

    def draw(self, screen):
        """Draw pedestrian on screen."""
        if self.crossed:
            return

        color = (200, 100, 50) if not self.crossing else (0, 255, 0)
        pygame.draw.circle(screen, color, (int(self.pos[0]), int(self.pos[1])), 6)
        pygame.draw.circle(screen, BLACK, (int(self.pos[0]), int(self.pos[1])), 6, 1)


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
            'N': (INTERSECTION_X - 70, INTERSECTION_Y - 110),
            'S': (INTERSECTION_X + 70, INTERSECTION_Y + 110),
            'E': (INTERSECTION_X + 110, INTERSECTION_Y - 70),
            'W': (INTERSECTION_X - 110, INTERSECTION_Y + 70)
        }
        x, y = positions[self.direction]
        color = {'RED': RED, 'YELLOW': YELLOW, 'GREEN': GREEN}[self.state]
        pygame.draw.circle(screen, color, (x, y), LIGHT_RADIUS)
        pygame.draw.circle(screen, BLACK, (x, y), LIGHT_RADIUS, 2)
        # Draw state label
        font = pygame.font.SysFont(None, 14)
        text = font.render(self.state[:1], True, BLACK)
        screen.blit(text, (x - 6, y - 8))


class Intersection:
    """Manages the four-way intersection with vehicles, pedestrians, and lights."""

    def __init__(self):
        self.vehicles = {d: [] for d in DIRECTIONS}
        self.pedestrians = {d: [] for d in DIRECTIONS}
        self.lights = {d: TrafficLight(d) for d in DIRECTIONS}

        self.current_phase = 'NS_GREEN'  # NS_GREEN, EW_GREEN, PEDESTRIAN, ALL_RED
        self.phase_timer = 0
        self.all_red_timer = 0

        self.crossed_vehicles = 0
        self.crossed_pedestrians = 0
        self.total_vehicle_waiting = 0
        self.total_pedestrian_waiting = 0
        self.total_emergency_waiting = 0
        self.signal_changes = 0
        self.emergency_crossed = 0
        self.emergency_override_active = False
        self.frame_count = 0

        # Initialize NS green
        self.lights['N'].set('GREEN', MAX_GREEN_TIME)
        self.lights['S'].set('GREEN', MAX_GREEN_TIME)
        self.lights['E'].set('RED', 0)
        self.lights['W'].set('RED', 0)

    def spawn_vehicle(self, direction, spawn_prob, emergency_prob):
        """Randomly spawn a vehicle on a road."""
        if random.random() < spawn_prob:
            # Weighted vehicle type distribution for Indian traffic
            types = ['car', 'motorcycle', 'scooter', 'auto_rickshaw', 'bus', 'truck']
            weights = [0.30, 0.25, 0.20, 0.15, 0.05, 0.05]
            vehicle_type = random.choices(types, weights=weights)[0]

            is_emergency = random.random() < emergency_prob
            if is_emergency:
                vehicle_type = random.choice(['ambulance', 'police'])

            self.vehicles[direction].append(Vehicle(direction, vehicle_type, is_emergency))

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
            # Set pedestrian green for that side
            self.phase_timer = PEDESTRIAN_CROSS_TIME

        elif phase == 'ALL_RED':
            for d in DIRECTIONS:
                self.lights[d].set('RED', ALL_RED_TIME)
            self.phase_timer = ALL_RED_TIME

    def get_vehicle_counts(self):
        return [len(self.vehicles[d]) for d in DIRECTIONS]

    def get_pedestrian_counts(self):
        return [len(self.pedestrians[d]) for d in DIRECTIONS]

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
                # Return direction with closest emergency vehicle
                closest = min(emergencies[d], key=lambda v: v.pos[1] if v.direction in ['N', 'S'] else v.pos[0])
                return d
        return None

    def update(self, spawn_prob, emergency_prob, pedestrian_prob, action=None):
        """Update intersection state."""
        self.frame_count += 1

        # Spawn vehicles and pedestrians
        for d in DIRECTIONS:
            self.spawn_vehicle(d, spawn_prob, emergency_prob)
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
            # else: keep current phase

        # Emergency override
        if emergency_active and self.current_phase not in ['NS_GREEN', 'EW_GREEN']:
            # Give green to emergency direction
            self.set_phase('NS_GREEN' if emergency_dir in ['N', 'S'] else 'EW_GREEN')

        # Update vehicles
        for direction in DIRECTIONS:
            can_cross = self.lights[direction].is_green() or self.lights[direction].is_yellow()
            all_red = self.current_phase == 'ALL_RED'

            for v in self.vehicles[direction]:
                crossed = v.update(can_cross, emergency_override=emergency_active and
                                   ((direction in ['N', 'S'] and emergency_dir in ['N', 'S']) or
                                    (direction in ['E', 'W'] and emergency_dir in ['E', 'W'])),
                                   all_red=all_red)
                if crossed:
                    self.crossed_vehicles += 1
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

        # Accumulate waiting times for current waiting entities
        for direction in DIRECTIONS:
            for v in self.vehicles[direction]:
                if not v.update.__defaults__:  # skip, already updated
                    pass

        # Count current waiting
        current_vehicle_waiting = sum(len(self.vehicles[d]) for d in DIRECTIONS)
        current_pedestrian_waiting = sum(len(self.pedestrians[d]) for d in DIRECTIONS)

        return {
            'emergency_active': emergency_active,
            'emergency_dir': emergency_dir,
            'current_phase': self.current_phase,
            'vehicle_counts': self.get_vehicle_counts(),
            'pedestrian_counts': self.get_pedestrian_counts(),
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
            # Extend current phase if it's a green phase
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
        """Draw the intersection roads."""
        # North road
        pygame.draw.rect(screen, GRAY,
                         (INTERSECTION_X - ROAD_WIDTH // 2, 0,
                          ROAD_WIDTH, INTERSECTION_Y - INTERSECTION_SIZE // 2))
        # South road
        pygame.draw.rect(screen, GRAY,
                         (INTERSECTION_X - ROAD_WIDTH // 2,
                          INTERSECTION_Y + INTERSECTION_SIZE // 2,
                          ROAD_WIDTH, WINDOW_HEIGHT - INTERSECTION_Y - INTERSECTION_SIZE // 2))
        # East road
        pygame.draw.rect(screen, GRAY,
                         (INTERSECTION_X + INTERSECTION_SIZE // 2,
                          INTERSECTION_Y - ROAD_WIDTH // 2,
                          WINDOW_WIDTH - INTERSECTION_X - INTERSECTION_SIZE // 2,
                          ROAD_WIDTH))
        # West road
        pygame.draw.rect(screen, GRAY,
                         (0, INTERSECTION_Y - ROAD_WIDTH // 2,
                          INTERSECTION_X - INTERSECTION_SIZE // 2,
                          ROAD_WIDTH))
        # Intersection box
        pygame.draw.rect(screen, GRAY,
                         (INTERSECTION_X - INTERSECTION_SIZE // 2,
                          INTERSECTION_Y - INTERSECTION_SIZE // 2,
                          INTERSECTION_SIZE, INTERSECTION_SIZE))

        # Zebra crossings
        zebra_color = WHITE
        # North zebra
        for i in range(-INTERSECTION_SIZE // 2, INTERSECTION_SIZE // 2, 10):
            pygame.draw.rect(screen, zebra_color,
                             (INTERSECTION_X + i, INTERSECTION_Y - INTERSECTION_SIZE // 2 - 15, 6, 12))
        # South zebra
        for i in range(-INTERSECTION_SIZE // 2, INTERSECTION_SIZE // 2, 10):
            pygame.draw.rect(screen, zebra_color,
                             (INTERSECTION_X + i, INTERSECTION_Y + INTERSECTION_SIZE // 2 + 3, 6, 12))
        # East zebra
        for i in range(-INTERSECTION_SIZE // 2, INTERSECTION_SIZE // 2, 10):
            pygame.draw.rect(screen, zebra_color,
                             (INTERSECTION_X + INTERSECTION_SIZE // 2 + 3, INTERSECTION_Y + i, 12, 6))
        # West zebra
        for i in range(-INTERSECTION_SIZE // 2, INTERSECTION_SIZE // 2, 10):
            pygame.draw.rect(screen, zebra_color,
                             (INTERSECTION_X - INTERSECTION_SIZE // 2 - 15, INTERSECTION_Y + i, 12, 6))

        # Lane markings (dashed lines)
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.line(screen, WHITE, (INTERSECTION_X, y),
                             (INTERSECTION_X, min(y + 20, WINDOW_HEIGHT)), 2)
        for x in range(0, WINDOW_WIDTH, 40):
            pygame.draw.line(screen, WHITE, (x, INTERSECTION_Y),
                             (min(x + 20, WINDOW_WIDTH), INTERSECTION_Y), 2)

    def draw(self, screen):
        """Draw entire intersection."""
        self.draw_roads(screen)

        # Draw vehicles
        for direction in DIRECTIONS:
            for v in self.vehicles[direction]:
                v.draw(screen)

        # Draw pedestrians
        for direction in DIRECTIONS:
            for p in self.pedestrians[direction]:
                p.draw(screen)

        # Draw traffic lights
        for d in DIRECTIONS:
            self.lights[d].draw(screen)

        # Draw direction labels and counts
        font = pygame.font.SysFont(None, 18)
        counts = self.get_vehicle_counts()
        ped_counts = self.get_pedestrian_counts()
        labels = {
            'N': (INTERSECTION_X - 40, 10),
            'S': (INTERSECTION_X - 40, WINDOW_HEIGHT - 25),
            'E': (WINDOW_WIDTH - 100, INTERSECTION_Y - 15),
            'W': (10, INTERSECTION_Y - 15)
        }
        for d, pos in labels.items():
            idx = DIRECTIONS.index(d)
            v_count = counts[idx]
            p_count = ped_counts[idx]
            color = RED if v_count > 8 else (ORANGE if v_count > 4 else GREEN)
            text = font.render(f"{DIRECTION_NAMES[d]}: V{v_count} P{p_count}", True, color)
            screen.blit(text, pos)

        # Draw current phase
        phase_font = pygame.font.SysFont(None, 22, bold=True)
        phase_text = phase_font.render(f"Phase: {self.current_phase}", True, NAVY)
        screen.blit(phase_text, (INTERSECTION_X - 100, INTERSECTION_Y - INTERSECTION_SIZE // 2 - 30))