# simulation.py - Pure Python simulation engine (NO pygame)
import random

from config import (
    WINDOW_W, WINDOW_H, INTER_X, INTER_Y, INTER_SIZE,
    VEHICLE_TYPES, GREEN_MIN, GREEN_MAX, YELLOW_TIME, ALL_RED_TIME,
    SPAWN_PROB_BASE, EMERGENCY_SPAWN_PROB, PED_SPAWN_PROB,
    PED_CROSSING_TIME, PED_HIGH_DEMAND, PED_MAX_WAIT, EMERGENCY_PRIORITY_DIST,
)

DIRECTIONS = ['N', 'S', 'E', 'W']

# Global vehicle counter for unique IDs
_vehicle_counter = 0


def _next_vid():
    global _vehicle_counter
    _vehicle_counter += 1
    return _vehicle_counter


# ---------------------------------------------------------------------------
# Vehicle
# ---------------------------------------------------------------------------

class Vehicle:
    def __init__(self, direction, vtype=None):
        self.direction = direction
        if vtype is None:
            vtype = self._random_type()
        self.vtype = vtype
        info = VEHICLE_TYPES[vtype]
        self.speed = info['speed']
        self.length = info['length']
        self.width = info['width']
        self.is_emergency = info['is_emergency']
        self.waiting_time = 0
        self.crossed = False
        self.pos = float(self._start_pos())
        self.vid = _next_vid()

    @staticmethod
    def _random_type():
        types = [t for t, v in VEHICLE_TYPES.items() if not v['is_emergency']]
        weights = [VEHICLE_TYPES[t]['weight'] for t in types]
        total = sum(weights)
        r = random.uniform(0, total)
        cumulative = 0
        for t, w in zip(types, weights):
            cumulative += w
            if r <= cumulative:
                return t
        return types[-1]

    def _start_pos(self):
        d = self.direction
        if d == 'N':
            return 0.0
        elif d == 'S':
            return float(WINDOW_H)
        elif d == 'E':
            return float(WINDOW_W)
        else:  # W
            return 0.0

    def stop_line(self):
        d = self.direction
        if d == 'N':
            return INTER_Y - INTER_SIZE // 2
        elif d == 'S':
            return INTER_Y + INTER_SIZE // 2
        elif d == 'E':
            return INTER_X + INTER_SIZE // 2
        else:  # W
            return INTER_X - INTER_SIZE // 2

    def dist_to_intersection(self):
        d = self.direction
        sl = self.stop_line()
        if d == 'N':
            # Moving downward; pos increases toward stop_line
            dist = sl - self.pos
        elif d == 'S':
            # Moving upward (pos decreases toward stop_line)
            dist = self.pos - sl
        elif d == 'E':
            # Moving leftward (pos decreases toward stop_line)
            dist = self.pos - sl
        else:  # W
            # Moving rightward (pos increases toward stop_line)
            dist = sl - self.pos
        return max(0.0, dist)

    def update(self, can_cross):
        d = self.direction
        sl = self.stop_line()

        at_stop = self.dist_to_intersection() <= self.speed

        if not can_cross and at_stop and not self.crossed:
            # Blocked at stop line
            self.waiting_time += 1
            # Snap to stop line if overshot
            if d == 'N':
                self.pos = min(self.pos + self.speed, float(sl))
            elif d == 'S':
                self.pos = max(self.pos - self.speed, float(sl))
            elif d == 'E':
                self.pos = max(self.pos - self.speed, float(sl))
            else:
                self.pos = min(self.pos + self.speed, float(sl))
        else:
            # Move freely
            if d == 'N':
                self.pos += self.speed
            elif d == 'S':
                self.pos -= self.speed
            elif d == 'E':
                self.pos -= self.speed
            else:
                self.pos += self.speed

        # Check if exited screen
        if d == 'N' and self.pos >= WINDOW_H:
            self.crossed = True
        elif d == 'S' and self.pos <= 0:
            self.crossed = True
        elif d == 'E' and self.pos <= 0:
            self.crossed = True
        elif d == 'W' and self.pos >= WINDOW_W:
            self.crossed = True

    def to_dict(self):
        return {
            'pos': self.pos,
            'type': self.vtype,
            'is_emergency': self.is_emergency,
            'waiting_time': self.waiting_time,
            'direction': self.direction,
        }


# ---------------------------------------------------------------------------
# Pedestrian
# ---------------------------------------------------------------------------

class Pedestrian:
    def __init__(self, side):
        self.side = side
        self.waiting_time = 0
        self.crossing = False
        self.cross_progress = 0
        self.crossed = False

    def update(self, ped_green: bool):
        if self.crossing:
            self.cross_progress += 1
            if self.cross_progress >= PED_CROSSING_TIME:
                self.crossed = True
        elif ped_green:
            self.crossing = True
        else:
            self.waiting_time += 1


# ---------------------------------------------------------------------------
# SignalController
# ---------------------------------------------------------------------------

class SignalController:
    VALID_PHASES = ['NS', 'EW', 'PED_N', 'PED_S', 'PED_E', 'PED_W']

    def __init__(self):
        self.phase = 'NS'
        self.sig_state = 'GREEN'
        self.timer = GREEN_MIN
        self.next_phase = 'NS'
        self.phase_timer = 0
        self.changes = 0

    def request_phase(self, new_phase):
        if new_phase not in self.VALID_PHASES:
            return
        if self.sig_state == 'GREEN':
            if self.phase_timer >= GREEN_MAX:
                # Force transition
                self.sig_state = 'YELLOW'
                self.timer = YELLOW_TIME
                self.next_phase = new_phase
            elif self.phase_timer >= GREEN_MIN and new_phase != self.phase:
                self.sig_state = 'YELLOW'
                self.timer = YELLOW_TIME
                self.next_phase = new_phase

    def update(self):
        if self.sig_state == 'GREEN':
            self.phase_timer += 1
            self.timer -= 1
            # Timer ran out naturally — stay green until request_phase is called
            # but enforce GREEN_MAX
            if self.phase_timer >= GREEN_MAX and self.sig_state == 'GREEN':
                self.sig_state = 'YELLOW'
                self.timer = YELLOW_TIME
        else:
            self.timer -= 1
            if self.timer <= 0:
                if self.sig_state == 'YELLOW':
                    self.sig_state = 'ALL_RED'
                    self.timer = ALL_RED_TIME
                elif self.sig_state == 'ALL_RED':
                    self.phase = self.next_phase
                    self.sig_state = 'GREEN'
                    self.timer = GREEN_MIN
                    self.phase_timer = 0
                    self.changes += 1

    def get_lights(self):
        lights = {d: 'RED' for d in DIRECTIONS}
        ped_lights = {d: False for d in DIRECTIONS}

        phase = self.phase
        state = self.sig_state

        if phase == 'NS':
            if state == 'GREEN':
                lights['N'] = 'GREEN'
                lights['S'] = 'GREEN'
            elif state == 'YELLOW':
                lights['N'] = 'YELLOW'
                lights['S'] = 'YELLOW'
        elif phase == 'EW':
            if state == 'GREEN':
                lights['E'] = 'GREEN'
                lights['W'] = 'GREEN'
            elif state == 'YELLOW':
                lights['E'] = 'YELLOW'
                lights['W'] = 'YELLOW'
        elif phase.startswith('PED_'):
            ped_dir = phase[4:]  # 'N', 'S', 'E', or 'W'
            if state == 'GREEN':
                ped_lights[ped_dir] = True
            # all vehicle lights remain RED

        # ALL_RED: everything stays RED (default)

        return lights, ped_lights

    def can_move(self, direction):
        lights, _ = self.get_lights()
        return lights.get(direction, 'RED') in ('GREEN', 'YELLOW')

    def is_ped_phase(self):
        return self.phase.startswith('PED_')


# ---------------------------------------------------------------------------
# Intersection
# ---------------------------------------------------------------------------

class Intersection:
    def __init__(self):
        self.vehicles = {d: [] for d in DIRECTIONS}
        self.pedestrians = {d: [] for d in DIRECTIONS}
        self.signal = SignalController()
        self.frame = 0
        self.peak_hour = False
        self.emergency_active = False
        self.emergency_dir = None
        self.emergency_vtype = None
        # Metrics
        self.crossed_count = 0
        self.total_delay = 0
        self.ped_crossed = 0
        self.emergency_delay = 0
        self.signal_changes = 0

    def spawn_vehicles(self):
        base = SPAWN_PROB_BASE * (1.8 if self.peak_hour else 1.0)
        density = {'N': 1.4, 'S': 1.0, 'E': 1.3, 'W': 0.9}
        for d in DIRECTIONS:
            if random.random() < base * density[d]:
                if random.random() < EMERGENCY_SPAWN_PROB:
                    vtype = random.choice(['ambulance', 'police'])
                else:
                    # Pick from non-emergency types by weight
                    types = [t for t, v in VEHICLE_TYPES.items() if not v['is_emergency']]
                    weights = [VEHICLE_TYPES[t]['weight'] for t in types]
                    total = sum(weights)
                    r = random.uniform(0, total)
                    cumulative = 0
                    vtype = types[-1]
                    for t, w in zip(types, weights):
                        cumulative += w
                        if r <= cumulative:
                            vtype = t
                            break
                self.vehicles[d].append(Vehicle(d, vtype))

    def spawn_pedestrians(self):
        prob = PED_SPAWN_PROB * (1.5 if self.peak_hour else 1.0)
        for d in DIRECTIONS:
            if random.random() < prob:
                self.pedestrians[d].append(Pedestrian(d))

    def check_emergency(self):
        found = False
        found_dir = None
        found_vtype = None
        for d in DIRECTIONS:
            for v in self.vehicles[d]:
                if v.is_emergency and v.dist_to_intersection() <= EMERGENCY_PRIORITY_DIST and not v.crossed:
                    found = True
                    found_dir = d
                    found_vtype = v.vtype
                    break
            if found:
                break

        if found and not self.emergency_active:
            self.emergency_active = True
            self.emergency_dir = found_dir
            self.emergency_vtype = found_vtype
        elif self.emergency_active and not found:
            self.emergency_active = False
            self.emergency_dir = None
            self.emergency_vtype = None

        return self.emergency_active

    def update(self):
        self.frame += 1
        self.peak_hour = (self.frame // 400) % 2 == 0
        self.check_emergency()
        self.signal.update()
        lights, ped_lights = self.signal.get_lights()

        for d in DIRECTIONS:
            can_cross = self.signal.can_move(d)
            remaining = []
            for v in self.vehicles[d]:
                if v.is_emergency and not can_cross:
                    self.emergency_delay += 1
                v.update(can_cross)
                if v.crossed:
                    self.crossed_count += 1
                    self.total_delay += v.waiting_time
                else:
                    remaining.append(v)
            self.vehicles[d] = remaining

        for d in DIRECTIONS:
            pg = ped_lights[d]
            remaining = []
            for p in self.pedestrians[d]:
                p.update(pg)
                if p.crossed:
                    self.ped_crossed += 1
                else:
                    remaining.append(p)
            self.pedestrians[d] = remaining

    def get_state_info(self):
        queue_counts = [len(self.vehicles[d]) for d in DIRECTIONS]
        waiting_times = [sum(v.waiting_time for v in self.vehicles[d]) for d in DIRECTIONS]
        ped_counts = [len(self.pedestrians[d]) for d in DIRECTIONS]
        ped_max_wait = [
            max((p.waiting_time for p in self.pedestrians[d]), default=0)
            for d in DIRECTIONS
        ]
        return {
            'queue_counts': queue_counts,
            'waiting_times': waiting_times,
            'ped_counts': ped_counts,
            'ped_max_wait': ped_max_wait,
            'phase': self.signal.phase,
            'phase_timer': self.signal.phase_timer,
            'has_emergency': self.emergency_active,
            'emergency_dir': self.emergency_dir,
            'peak_hour': self.peak_hour,
        }

    def get_metrics(self):
        total_waiting = sum(
            v.waiting_time for d in DIRECTIONS for v in self.vehicles[d]
        )
        queue_total = sum(len(self.vehicles[d]) for d in DIRECTIONS)
        avg_wait = round(self.total_delay / max(1, self.crossed_count), 2)
        return {
            'crossed_count': self.crossed_count,
            'total_delay': self.total_delay,
            'avg_wait': avg_wait,
            'queue': queue_total,
            'ped_crossed': self.ped_crossed,
            'emergency_delay': self.emergency_delay,
            'signal_changes': self.signal.changes,
            'peak_hour': self.peak_hour,
            'frame': self.frame,
        }

    def to_dict(self):
        lights, ped_lights = self.signal.get_lights()
        vehicles_dict = {d: [v.to_dict() for v in self.vehicles[d]] for d in DIRECTIONS}
        pedestrians_dict = {}
        for d in DIRECTIONS:
            peds = self.pedestrians[d]
            waiting_peds = [p for p in peds if not p.crossing]
            crossing_peds = [p for p in peds if p.crossing]
            max_wait = max((p.waiting_time for p in peds), default=0)
            pedestrians_dict[d] = {
                'waiting': len(waiting_peds),
                'crossing': len(crossing_peds),
                'max_wait': max_wait,
                'total': len(peds),
            }
        return {
            'vehicles': vehicles_dict,
            'pedestrians': pedestrians_dict,
            'lights': {d: lights[d] for d in DIRECTIONS},
            'ped_lights': {d: ped_lights[d] for d in DIRECTIONS},
            'phase': self.signal.phase,
            'sig_state': self.signal.sig_state,
            'phase_timer': self.signal.phase_timer,
            'emergency_active': self.emergency_active,
            'emergency_dir': self.emergency_dir,
            'emergency_vtype': self.emergency_vtype,
            'peak_hour': self.peak_hour,
            'frame': self.frame,
            'metrics': self.get_metrics(),
        }
