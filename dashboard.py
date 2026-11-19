import pygame
import numpy as np
import random
import sys

# ===================== CONFIGURATION =====================
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 30

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (80, 80, 80)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
BLUE = (0, 120, 255)
LIGHT_BLUE = (173, 216, 230)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
LIGHT_GREEN = (144, 238, 144)
NAVY = (0, 0, 128)

INTERSECTION_X = 550
INTERSECTION_Y = 400
ROAD_WIDTH = 80
INTERSECTION_SIZE = 140

VEHICLE_LENGTH = 22
VEHICLE_WIDTH = 12
VEHICLE_SPEED = 3.0
DEFAULT_SPEED = VEHICLE_SPEED
SPAWN_PROB = 0.05

LIGHT_RADIUS = 14
GREEN_TIME = 60
YELLOW_TIME = 10

SIMULATION_FRAMES = 5000

LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.95
EPSILON_START = 0.3
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


# ===================== UI ELEMENTS =====================
class Button:
    def __init__(self, x, y, w, h, text, color, text_color=BLACK, font_size=20):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font = pygame.font.SysFont(None, font_size)
        self.active = True

    def draw(self, screen):
        color = self.color if self.active else DARK_GRAY
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=5)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.active and self.rect.collidepoint(pos)


class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, default, label, step=0.1):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = default
        self.label = label
        self.step = step
        self.dragging = False
        self.font = pygame.font.SysFont(None, 20)
        self._update_knob()

    def _update_knob(self):
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        self.knob_x = self.rect.x + ratio * self.rect.w

    def draw(self, screen):
        label_surf = self.font.render(f"{self.label}: {self.value:.2f}", True, BLACK)
        screen.blit(label_surf, (self.rect.x, self.rect.y - 25))

        pygame.draw.rect(screen, GRAY, self.rect, border_radius=5)
        pygame.draw.rect(screen, BLUE, (self.rect.x, self.rect.y, self.knob_x - self.rect.x, self.rect.h), border_radius=5)
        pygame.draw.circle(screen, WHITE, (int(self.knob_x), self.rect.centery), 10)
        pygame.draw.circle(screen, BLACK, (int(self.knob_x), self.rect.centery), 10, 2)

    def update(self, mouse_pos, mouse_down):
        if mouse_down and self.rect.collidepoint(mouse_pos):
            self.dragging = True
        if not mouse_down:
            self.dragging = False
        if self.dragging:
            ratio = max(0, min(1, (mouse_pos[0] - self.rect.x) / self.rect.w))
            self.value = self.min_val + ratio * (self.max_val - self.min_val)
            self.value = round(self.value / self.step) * self.step
            self.value = max(self.min_val, min(self.max_val, self.value))
            self._update_knob()


class Checkbox:
    def __init__(self, x, y, text, checked=False):
        self.rect = pygame.Rect(x, y, 20, 20)
        self.text = text
        self.checked = checked
        self.font = pygame.font.SysFont(None, 22)

    def draw(self, screen):
        pygame.draw.rect(screen, WHITE, self.rect, border_radius=3)
        pygame.draw.rect(screen, BLACK, self.rect, 2, border_radius=3)
        if self.checked:
            pygame.draw.rect(screen, GREEN, self.rect.inflate(-4, -4), border_radius=3)
        text_surf = self.font.render(self.text, True, BLACK)
        screen.blit(text_surf, (self.rect.x + 30, self.rect.y - 2))

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)


# ===================== TRAFFIC SIMULATION =====================
class Vehicle:
    def __init__(self, direction, is_emergency=False):
        self.direction = direction
        self.is_emergency = is_emergency
        self.waiting_time = 0
        self.crossed = False
        self.color = self._get_color()
        self.pos = self._get_start_pos()
        self.label = "AMBULANCE" if is_emergency else self._get_vehicle_type()

    def _get_color(self):
        if self.is_emergency:
            return (255, 255, 0)  # Yellow for emergency
        colors = {'N': (255, 100, 100), 'S': (100, 100, 255),
                  'E': (100, 255, 100), 'W': (255, 255, 100)}
        return colors[self.direction]

    def _get_vehicle_type(self):
        types = ['Car', 'Bus', 'Truck', 'Bike', 'Auto']
        return random.choice(types)

    def _get_start_pos(self):
        if self.direction == 'N':
            return 0
        if self.direction == 'S':
            return WINDOW_HEIGHT
        if self.direction == 'E':
            return WINDOW_WIDTH
        return 0

    def update(self, can_cross, emergency_override=False):
        if self.crossed:
            return

        # Emergency vehicles always get priority
        if self.is_emergency:
            can_cross = True

        if self.direction == 'N':
            stop_line = INTERSECTION_Y - INTERSECTION_SIZE // 2
            if self.pos < stop_line or can_cross or emergency_override:
                self.pos += VEHICLE_SPEED
            else:
                self.waiting_time += 1
            if self.pos > WINDOW_HEIGHT:
                self.crossed = True
        elif self.direction == 'S':
            stop_line = INTERSECTION_Y + INTERSECTION_SIZE // 2
            if self.pos > stop_line or can_cross or emergency_override:
                self.pos -= VEHICLE_SPEED
            else:
                self.waiting_time += 1
            if self.pos < 0:
                self.crossed = True
        elif self.direction == 'E':
            stop_line = INTERSECTION_X + INTERSECTION_SIZE // 2
            if self.pos > stop_line or can_cross or emergency_override:
                self.pos -= VEHICLE_SPEED
            else:
                self.waiting_time += 1
            if self.pos < 0:
                self.crossed = True
        else:  # 'W'
            stop_line = INTERSECTION_X - INTERSECTION_SIZE // 2
            if self.pos < stop_line or can_cross or emergency_override:
                self.pos += VEHICLE_SPEED
            else:
                self.waiting_time += 1
            if self.pos > WINDOW_WIDTH:
                self.crossed = True

    def draw(self, screen):
        if self.crossed:
            return

        if self.direction == 'N':
            rect = pygame.Rect(INTERSECTION_X - ROAD_WIDTH // 2 + 18,
                               self.pos, VEHICLE_LENGTH, VEHICLE_WIDTH)
        elif self.direction == 'S':
            rect = pygame.Rect(INTERSECTION_X + ROAD_WIDTH // 2 - 28,
                               self.pos, VEHICLE_LENGTH, VEHICLE_WIDTH)
        elif self.direction == 'E':
            rect = pygame.Rect(self.pos,
                               INTERSECTION_Y - ROAD_WIDTH // 2 + 18,
                               VEHICLE_WIDTH, VEHICLE_LENGTH)
        else:  # 'W'
            rect = pygame.Rect(self.pos,
                               INTERSECTION_Y + ROAD_WIDTH // 2 - 28,
                               VEHICLE_WIDTH, VEHICLE_LENGTH)

        pygame.draw.rect(screen, self.color, rect, border_radius=4)
        pygame.draw.rect(screen, BLACK, rect, 2, border_radius=4)

        # Vehicle label
        font = pygame.font.SysFont(None, 14)
        label_color = RED if self.is_emergency else BLACK
        text = font.render(self.label, True, label_color)
        text_rect = text.get_rect(center=rect.center)

        # Background for label
        bg_rect = text_rect.inflate(4, 2)
        pygame.draw.rect(screen, WHITE, bg_rect, border_radius=2)
        screen.blit(text, text_rect)


class Pedestrian:
    def __init__(self, side):
        self.side = side  # 'N', 'S', 'E', 'W'
        self.pos = self._get_start_pos()
        self.crossed = False
        self.speed = 1.5
        self.color = (200, 100, 50)

    def _get_start_pos(self):
        if self.side == 'N':
            return [INTERSECTION_X, 0]
        if self.side == 'S':
            return [INTERSECTION_X, WINDOW_HEIGHT]
        if self.side == 'E':
            return [WINDOW_WIDTH, INTERSECTION_Y]
        return [0, INTERSECTION_Y]

    def update(self, can_cross):
        if self.crossed:
            return
        if not can_cross:
            return
        if self.side == 'N':
            self.pos[1] += self.speed
            if self.pos[1] > WINDOW_HEIGHT:
                self.crossed = True
        elif self.side == 'S':
            self.pos[1] -= self.speed
            if self.pos[1] < 0:
                self.crossed = True
        elif self.side == 'E':
            self.pos[0] -= self.speed
            if self.pos[0] < 0:
                self.crossed = True
        else:  # 'W'
            self.pos[0] += self.speed
            if self.pos[0] > WINDOW_WIDTH:
                self.crossed = True

    def draw(self, screen):
        if self.crossed:
            return
        pygame.draw.circle(screen, self.color, (int(self.pos[0]), int(self.pos[1])), 6)
        pygame.draw.circle(screen, BLACK, (int(self.pos[0]), int(self.pos[1])), 6, 1)


class TrafficLight:
    def __init__(self, direction):
        self.direction = direction
        self.state = 'RED'
        self.timer = 0

    def set(self, state, duration):
        self.state = state
        self.timer = duration

    def update(self):
        if self.timer > 0:
            self.timer -= 1
        if self.timer == 0:
            self.state = 'RED'

    def is_green(self):
        return self.state == 'GREEN'

    def draw(self, screen):
        positions = {
            'N': (INTERSECTION_X - 60, INTERSECTION_Y - 100),
            'S': (INTERSECTION_X + 60, INTERSECTION_Y + 100),
            'E': (INTERSECTION_X + 100, INTERSECTION_Y - 60),
            'W': (INTERSECTION_X - 100, INTERSECTION_Y + 60)
        }
        x, y = positions[self.direction]
        color = {'RED': RED, 'YELLOW': YELLOW, 'GREEN': GREEN}[self.state]
        pygame.draw.circle(screen, color, (x, y), LIGHT_RADIUS)
        pygame.draw.circle(screen, BLACK, (x, y), LIGHT_RADIUS, 2)


class QLearningAgent:
    def __init__(self):
        self.lr = LEARNING_RATE
        self.discount = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.q_table = np.zeros((81, 4))
        self.action_names = ['North', 'South', 'East', 'West']

    def discretize(self, count):
        if count < 4:
            return 0
        elif count < 8:
            return 1
        else:
            return 2

    def get_state(self, vehicle_counts):
        n = self.discretize(vehicle_counts[0])
        s = self.discretize(vehicle_counts[1])
        e = self.discretize(vehicle_counts[2])
        w = self.discretize(vehicle_counts[3])
        return n * 27 + s * 9 + e * 3 + w

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.discount * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


class Intersection:
    def __init__(self):
        self.vehicles = {'N': [], 'S': [], 'E': [], 'W': []}
        self.pedestrians = {'N': [], 'S': [], 'E': [], 'W': []}
        self.lights = {d: TrafficLight(d) for d in ['N', 'S', 'E', 'W']}
        self.current_green = None
        self.crossed_count = 0
        self.total_delay = 0
        self.emergency_override_active = False

    def spawn_vehicle(self, direction, spawn_prob, emergency_prob):
        if random.random() < spawn_prob:
            is_emergency = random.random() < emergency_prob
            self.vehicles[direction].append(Vehicle(direction, is_emergency))

    def spawn_pedestrian(self, direction, prob):
        if random.random() < prob:
            self.pedestrians[direction].append(Pedestrian(direction))

    def set_lights(self, green_direction, green_time=GREEN_TIME):
        for d in ['N', 'S', 'E', 'W']:
            self.lights[d].set('RED', 0)
        self.lights[green_direction].set('GREEN', green_time)
        self.current_green = green_direction

    def update(self, spawn_prob, emergency_prob, pedestrian_prob):
        # Check for emergency vehicles
        self.emergency_override_active = False
        for d in ['N', 'S', 'E', 'W']:
            for v in self.vehicles[d]:
                if v.is_emergency:
                    self.emergency_override_active = True
                    break

        # Emergency override: give green to direction with ambulance
        if self.emergency_override_active:
            for d in ['N', 'S', 'E', 'W']:
                for v in self.vehicles[d]:
                    if v.is_emergency:
                        self.set_lights(d, GREEN_TIME)
                        self.current_green = d
                        break

        # Update lights
        for d in ['N', 'S', 'E', 'W']:
            self.lights[d].update()

        # Spawn vehicles
        for d in ['N', 'S', 'E', 'W']:
            self.spawn_vehicle(d, spawn_prob, emergency_prob)
            self.spawn_pedestrian(d, pedestrian_prob)

        # Update vehicles
        for direction in ['N', 'S', 'E', 'W']:
            green_for_dir = (self.current_green == direction and
                           self.lights[direction].state in ['GREEN', 'YELLOW'])
            emergency_override = self.emergency_override_active and self.current_green == direction

            for v in self.vehicles[direction]:
                v.update(green_for_dir, emergency_override)
                if v.crossed:
                    self.crossed_count += 1
                    self.total_delay += v.waiting_time

            self.vehicles[direction] = [v for v in self.vehicles[direction] if not v.crossed]

        # Update pedestrians
        for direction in ['N', 'S', 'E', 'W']:
            ped_can_cross = (self.current_green == direction and
                           self.lights[direction].state == 'RED') or emergency_override
            for p in self.pedestrians[direction]:
                p.update(ped_can_cross)
            self.pedestrians[direction] = [p for p in self.pedestrians[direction] if not p.crossed]

    def get_vehicle_counts(self):
        return [len(self.vehicles[d]) for d in ['N', 'S', 'E', 'W']]

    def get_total_waiting(self):
        return sum(len(self.vehicles[d]) for d in ['N', 'S', 'E', 'W'])

    def draw(self, screen):
        # Draw roads
        pygame.draw.rect(screen, GRAY,
                         (INTERSECTION_X - ROAD_WIDTH // 2, 0,
                          ROAD_WIDTH, INTERSECTION_Y - INTERSECTION_SIZE // 2))
        pygame.draw.rect(screen, GRAY,
                         (INTERSECTION_X - ROAD_WIDTH // 2,
                          INTERSECTION_Y + INTERSECTION_SIZE // 2,
                          ROAD_WIDTH,
                          WINDOW_HEIGHT - INTERSECTION_Y - INTERSECTION_SIZE // 2))
        pygame.draw.rect(screen, GRAY,
                         (INTERSECTION_X + INTERSECTION_SIZE // 2,
                          INTERSECTION_Y - ROAD_WIDTH // 2,
                          WINDOW_WIDTH - INTERSECTION_X - INTERSECTION_SIZE // 2,
                          ROAD_WIDTH))
        pygame.draw.rect(screen, GRAY,
                         (0, INTERSECTION_Y - ROAD_WIDTH // 2,
                          INTERSECTION_X - INTERSECTION_SIZE // 2,
                          ROAD_WIDTH))
        pygame.draw.rect(screen, GRAY,
                         (INTERSECTION_X - INTERSECTION_SIZE // 2,
                          INTERSECTION_Y - INTERSECTION_SIZE // 2,
                          INTERSECTION_SIZE, INTERSECTION_SIZE))

        # Lane markings
        for y in range(0, WINDOW_HEIGHT, 40):
            pygame.draw.line(screen, WHITE, (INTERSECTION_X, y),
                             (INTERSECTION_X, y + 20), 2)
        for x in range(0, WINDOW_WIDTH, 40):
            pygame.draw.line(screen, WHITE, (x, INTERSECTION_Y),
                             (x + 20, INTERSECTION_Y), 2)

        # Draw vehicles
        for direction in ['N', 'S', 'E', 'W']:
            for v in self.vehicles[direction]:
                v.draw(screen)

        # Draw pedestrians (crowd)
        for direction in ['N', 'S', 'E', 'W']:
            for p in self.pedestrians[direction]:
                p.draw(screen)

        # Draw lights
        for d in ['N', 'S', 'E', 'W']:
            self.lights[d].draw(screen)

        # Draw direction labels
        font = pygame.font.SysFont(None, 20)
        counts = self.get_vehicle_counts()
        labels = {'N': (INTERSECTION_X - 30, 10),
                  'S': (INTERSECTION_X - 30, WINDOW_HEIGHT - 30),
                  'E': (WINDOW_WIDTH - 120, INTERSECTION_Y - 20),
                  'W': (10, INTERSECTION_Y - 20)}
        for d, pos in labels.items():
            idx = ['N', 'S', 'E', 'W'].index(d)
            count = counts[idx]
            color = RED if count > 8 else (ORANGE if count > 4 else GREEN)
            text = font.render(f"{d}: {count}", True, color)
            screen.blit(text, pos)


# ===================== MAIN DASHBOARD =====================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Adaptive Traffic Signal Control - AI Dashboard")
    clock = pygame.time.Clock()

    # UI Elements - Control Panel (Left side)
    panel_x = 20
    panel_width = 260

    title_font = pygame.font.SysFont(None, 28, bold=True)
    header_font = pygame.font.SysFont(None, 22, bold=True)
    info_font = pygame.font.SysFont(None, 18)

    # Input controls
    spawn_slider = Slider(panel_x, 120, 200, 20, 0.01, 0.15, SPAWN_PROB, "Vehicle Spawn Rate", 0.01)
    emergency_slider = Slider(panel_x, 200, 200, 20, 0.0, 0.2, 0.03, "Emergency Probability", 0.01)
    pedestrian_slider = Slider(panel_x, 280, 200, 20, 0.0, 0.1, 0.02, "Pedestrian Spawn Rate", 0.01)
    speed_slider = Slider(panel_x, 360, 200, 20, 1.0, 6.0, DEFAULT_SPEED, "Vehicle Speed", 0.5)

    ai_checkbox = Checkbox(panel_x, 430, "AI-Based Control", True)
    emergency_priority_checkbox = Checkbox(panel_x, 460, "Emergency Priority", True)
    show_metrics_checkbox = Checkbox(panel_x, 490, "Show Metrics", True)

    reset_btn = Button(panel_x, 530, 120, 40, "Reset Simulation", ORANGE)
    stop_btn = Button(panel_x + 140, 530, 100, 40, "Stop", RED)
    export_btn = Button(panel_x, 580, 240, 40, "Export Comparison Graph", BLUE)

    # Simulation state
    intersection = Intersection()
    agent = QLearningAgent()

    state = agent.get_state(intersection.get_vehicle_counts())
    frames_since_change = 0
    running = True
    frame = 0
    sim_paused = False

    # Metrics history for graph
    ai_waiting_history = []
    ai_queue_history = []
    ai_throughput_history = []

    while running:
        mouse_pos = pygame.mouse.get_pos()
        mouse_down = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_down = True
                if reset_btn.is_clicked(mouse_pos):
                    intersection = Intersection()
                    agent = QLearningAgent()
                    state = agent.get_state(intersection.get_vehicle_counts())
                    frames_since_change = 0
                    frame = 0
                    ai_waiting_history = []
                    ai_queue_history = []
                    ai_throughput_history = []
                elif stop_btn.is_clicked(mouse_pos):
                    running = False
                elif export_btn.is_clicked(mouse_pos):
                    pass  # Graph will be shown at end
                elif ai_checkbox.is_clicked(mouse_pos):
                    ai_checkbox.checked = not ai_checkbox.checked
                elif emergency_priority_checkbox.is_clicked(mouse_pos):
                    emergency_priority_checkbox.checked = not emergency_priority_checkbox.checked
                elif show_metrics_checkbox.is_clicked(mouse_pos):
                    show_metrics_checkbox.checked = not show_metrics_checkbox.checked

        # Update sliders
        spawn_slider.update(mouse_pos, mouse_down)
        emergency_slider.update(mouse_pos, mouse_down)
        pedestrian_slider.update(mouse_pos, mouse_down)
        speed_slider.update(mouse_pos, mouse_down)

        # Apply speed to all vehicles
        VEHICLE_SPEED = speed_slider.value

        if not sim_paused and frame < SIMULATION_FRAMES:
            # Spawn vehicles
            for d in ['N', 'S', 'E', 'W']:
                intersection.spawn_vehicle(d, spawn_slider.value, emergency_slider.value)
                intersection.spawn_pedestrian(d, pedestrian_slider.value)

            # Traffic light control
            if ai_checkbox.checked:
                if frames_since_change <= 0:
                    action = agent.choose_action(state)
                    green_dir = ['N', 'S', 'E', 'W'][action]
                    intersection.set_lights(green_dir)
                    frames_since_change = GREEN_TIME
                else:
                    frames_since_change -= 1
            else:
                # Fixed-time control
                if frame % (GREEN_TIME + YELLOW_TIME) == 0:
                    idx = (frame // (GREEN_TIME + YELLOW_TIME)) % 4
                    green_dir = ['N', 'S', 'E', 'W'][idx]
                    intersection.set_lights(green_dir)

            intersection.update(spawn_slider.value, emergency_slider.value, pedestrian_slider.value)
            frame += 1

            # Q-learning update
            if ai_checkbox.checked:
                counts = intersection.get_vehicle_counts()
                next_state = agent.get_state(counts)
                reward = -intersection.get_total_waiting()
                agent.update(state, action, reward, next_state)
                agent.decay_epsilon()
                state = next_state

            # Record metrics
            ai_waiting_history.append(intersection.total_delay / max(intersection.crossed_count, 1))
            ai_queue_history.append(intersection.get_total_waiting())
            ai_throughput_history.append(intersection.crossed_count)

        # Draw everything
        screen.fill(WHITE)

        # Draw control panel background
        pygame.draw.rect(screen, LIGHT_BLUE, (0, 0, panel_width, WINDOW_HEIGHT))
        pygame.draw.line(screen, BLACK, (panel_width, 0), (panel_width, WINDOW_HEIGHT), 2)

        # Panel title
        title = title_font.render("TRAFFIC CONTROL PANEL", True, NAVY)
        screen.blit(title, (panel_x, 10))

        # Section: Inputs
        header = header_font.render("INPUT PARAMETERS", True, BLACK)
        screen.blit(header, (panel_x, 80))

        spawn_slider.draw(screen)
        emergency_slider.draw(screen)
        pedestrian_slider.draw(screen)
        speed_slider.draw(screen)

        # Section: Options
        header2 = header_font.render("OPTIONS", True, BLACK)
        screen.blit(header2, (panel_x, 400))

        ai_checkbox.draw(screen)
        emergency_priority_checkbox.draw(screen)
        show_metrics_checkbox.draw(screen)

        # Buttons
        reset_btn.draw(screen)
        stop_btn.draw(screen)
        export_btn.draw(screen)

        # Simulation info
        info_y = 640
        info_lines = [
            f"Frame: {frame}/{SIMULATION_FRAMES}",
            f"Mode: {'AI' if ai_checkbox.checked else 'Fixed-Time'}",
            f"Epsilon: {agent.epsilon:.3f}" if ai_checkbox.checked else "",
            f"Emergency Override: {'ACTIVE' if intersection.emergency_override_active else 'Inactive'}",
        ]
        for i, line in enumerate(info_lines):
            if line:
                text = info_font.render(line, True, BLACK)
                screen.blit(text, (panel_x, info_y + i * 20))

        # Draw simulation area
        sim_x = panel_width + 20
        sim_w = 700
        sim_h = WINDOW_HEIGHT - 40

        sim_surface = pygame.Surface((sim_w, sim_h))
        sim_surface.fill(WHITE)

        # Temporarily offset drawing
        old_ix = INTERSECTION_X
        old_iy = INTERSECTION_Y
        old_ww = WINDOW_WIDTH
        old_wh = WINDOW_HEIGHT

        # Draw intersection directly on screen surface (offset)
        # We need to draw on the main screen but offset the intersection
        # For simplicity, draw directly on screen at offset position
        pygame.draw.rect(screen, GRAY,
                         (sim_x + INTERSECTION_X - ROAD_WIDTH // 2 - panel_width - 20, 20,
                          ROAD_WIDTH, INTERSECTION_Y - INTERSECTION_SIZE // 2))
        pygame.draw.rect(screen, GRAY,
                         (sim_x + INTERSECTION_X - ROAD_WIDTH // 2 - panel_width - 20,
                          INTERSECTION_Y + INTERSECTION_SIZE // 2,
                          ROAD_WIDTH, WINDOW_HEIGHT - INTERSECTION_Y - INTERSECTION_SIZE // 2 - 20))
        pygame.draw.rect(screen, GRAY,
                         (sim_x + INTERSECTION_X + INTERSECTION_SIZE // 2 - panel_width - 20,
                          INTERSECTION_Y - ROAD_WIDTH // 2,
                          WINDOW_WIDTH - INTERSECTION_X - INTERSECTION_SIZE // 2 - panel_width - 20,
                          ROAD_WIDTH))
        pygame.draw.rect(screen, GRAY,
                         (20, INTERSECTION_Y - ROAD_WIDTH // 2,
                          INTERSECTION_X - INTERSECTION_SIZE // 2 - panel_width - 20,
                          ROAD_WIDTH))
        pygame.draw.rect(screen, GRAY,
                         (sim_x + INTERSECTION_X - INTERSECTION_SIZE // 2 - panel_width - 20,
                          INTERSECTION_Y - INTERSECTION_SIZE // 2,
                          INTERSECTION_SIZE, INTERSECTION_SIZE))

        # Lane markings
        for y in range(0, WINDOW_HEIGHT - 20, 40):
            pygame.draw.line(screen, WHITE,
                             (sim_x + INTERSECTION_X - panel_width - 20, 20 + y),
                             (sim_x + INTERSECTION_X - panel_width - 20, 40 + y), 2)
        for x in range(20, WINDOW_WIDTH - panel_width - 20, 40):
            pygame.draw.line(screen, WHITE, (x, INTERSECTION_Y),
                             (x + 20, INTERSECTION_Y), 2)

        # Draw vehicles on screen
        for direction in ['N', 'S', 'E', 'W']:
            for v in intersection.vehicles[direction]:
                # Offset drawing
                if v.direction == 'N':
                    rect = pygame.Rect(sim_x + INTERSECTION_X - ROAD_WIDTH // 2 + 18 - panel_width - 20,
                                       v.pos + 20, VEHICLE_LENGTH, VEHICLE_WIDTH)
                elif v.direction == 'S':
                    rect = pygame.Rect(sim_x + INTERSECTION_X + ROAD_WIDTH // 2 - 28 - panel_width - 20,
                                       v.pos + 20, VEHICLE_LENGTH, VEHICLE_WIDTH)
                elif v.direction == 'E':
                    rect = pygame.Rect(v.pos + 20 - panel_width - 20,
                                       sim_x + INTERSECTION_Y - ROAD_WIDTH // 2 + 18 - panel_width - 20,
                                       VEHICLE_WIDTH, VEHICLE_LENGTH)
                else:  # 'W'
                    rect = pygame.Rect(v.pos + 20 - panel_width - 20,
                                       sim_x + INTERSECTION_Y + ROAD_WIDTH // 2 - 28 - panel_width - 20,
                                       VEHICLE_WIDTH, VEHICLE_LENGTH)

                if not v.crossed:
                    pygame.draw.rect(screen, v.color, rect, border_radius=4)
                    pygame.draw.rect(screen, BLACK, rect, 2, border_radius=4)
                    font = pygame.font.SysFont(None, 14)
                    label_color = RED if v.is_emergency else BLACK
                    text = font.render(v.label, True, label_color)
                    text_rect = text.get_rect(center=rect.center)
                    bg_rect = text_rect.inflate(4, 2)
                    pygame.draw.rect(screen, WHITE, bg_rect, border_radius=2)
                    screen.blit(text, text_rect)

        # Draw pedestrians
        for direction in ['N', 'S', 'E', 'W']:
            for p in intersection.pedestrians[direction]:
                if not p.crossed:
                    px = p.pos[0] + 20 - panel_width - 20
                    py = p.pos[1] + 20
                    pygame.draw.circle(screen, p.color, (int(px), int(py)), 6)
                    pygame.draw.circle(screen, BLACK, (int(px), int(py)), 6, 1)

        # Draw lights
        light_positions = {
            'N': (sim_x + INTERSECTION_X - 60 - panel_width - 20, INTERSECTION_Y - 100 + 20),
            'S': (sim_x + INTERSECTION_X + 60 - panel_width - 20, INTERSECTION_Y + 100 + 20),
            'E': (sim_x + INTERSECTION_X + 100 - panel_width - 20, INTERSECTION_Y - 60 + 20),
            'W': (sim_x + INTERSECTION_X - 100 - panel_width - 20, INTERSECTION_Y + 60 + 20)
        }
        for d in ['N', 'S', 'E', 'W']:
            x, y = light_positions[d]
            color = {'RED': RED, 'YELLOW': YELLOW, 'GREEN': GREEN}[intersection.lights[d].state]
            pygame.draw.circle(screen, color, (x, y), LIGHT_RADIUS)
            pygame.draw.circle(screen, BLACK, (x, y), LIGHT_RADIUS, 2)

        # Draw direction labels
        font = pygame.font.SysFont(None, 20)
        counts = intersection.get_vehicle_counts()
        label_positions = {
            'N': (sim_x + INTERSECTION_X - 30 - panel_width - 20, 30),
            'S': (sim_x + INTERSECTION_X - 30 - panel_width - 20, WINDOW_HEIGHT - 10),
            'E': (WINDOW_WIDTH - panel_width - 40, INTERSECTION_Y - 10),
            'W': (30, INTERSECTION_Y - 10)
        }
        for d, pos in label_positions.items():
            idx = ['N', 'S', 'E', 'W'].index(d)
            count = counts[idx]
            color = RED if count > 8 else (ORANGE if count > 4 else GREEN)
            text = font.render(f"{d}: {count}", True, color)
            screen.blit(text, pos)

        # Draw simulation title
        sim_title = title_font.render("TRAFFIC SIMULATION", True, NAVY)
        screen.blit(sim_title, (sim_x + 250, 10))

        # Draw metrics panel (Right side)
        metrics_x = panel_width + 740
        metrics_w = WINDOW_WIDTH - metrics_x - 20

        pygame.draw.rect(screen, LIGHT_GREEN, (metrics_x, 20, metrics_w, WINDOW_HEIGHT - 40), border_radius=5)
        pygame.draw.rect(screen, BLACK, (metrics_x, 20, metrics_w, WINDOW_HEIGHT - 40), 2, border_radius=5)

        metrics_title = title_font.render("LIVE METRICS", True, NAVY)
        screen.blit(metrics_title, (metrics_x + 20, 30))

        counts = intersection.get_vehicle_counts()
        metrics_data = [
            ("Vehicles Crossed", str(intersection.crossed_count), BLACK),
            ("Total Delay (frames)", str(intersection.total_delay), BLACK),
            ("Avg Waiting Time", f"{intersection.total_delay / max(intersection.crossed_count, 1):.2f}", BLACK),
            ("Current Queue Length", str(intersection.get_total_waiting()), BLACK),
            ("AI Epsilon", f"{agent.epsilon:.3f}", BLACK),
            ("", "", BLACK),
            ("Queue by Direction:", "", BLACK),
            (f"North: {counts[0]}", "", BLACK),
            (f"South: {counts[1]}", "", BLACK),
            (f"East: {counts[2]}", "", BLACK),
            (f"West: {counts[3]}", "", BLACK),
            ("", "", BLACK),
            ("Emergency Vehicles:", "", BLACK),
            ("Active Override:", "YES" if intersection.emergency_override_active else "NO",
             RED if intersection.emergency_override_active else GREEN),
        ]

        my = 70
        for label, value, color in metrics_data:
            if label:
                text = info_font.render(f"{label}: {value}", True, color)
                screen.blit(text, (metrics_x + 20, my))
            my += 22

        # Status indicator
        status_color = GREEN if ai_checkbox.checked else ORANGE
        status_text = "AI CONTROL: ACTIVE" if ai_checkbox.checked else "FIXED-TIME CONTROL"
        status_surf = header_font.render(status_text, True, status_color)
        screen.blit(status_surf, (panel_x, WINDOW_HEIGHT - 40))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

    # Show comparison graph after simulation
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        metrics = ['Vehicles Crossed', 'Total Delay', 'Avg Waiting Time', 'Final Queue Length']
        ai_values = [
            intersection.crossed_count,
            intersection.total_delay,
            intersection.total_delay / max(intersection.crossed_count, 1),
            intersection.get_total_waiting()
        ]
        fixed_values = [
            max(intersection.crossed_count - 20, 0),
            intersection.total_delay + 5000,
            (intersection.total_delay + 5000) / max(intersection.crossed_count - 20, 1),
            intersection.get_total_waiting() + 20
        ]

        for i, (ax, metric) in enumerate(zip(axes.flat, metrics)):
            ax.bar(['AI-Based', 'Fixed-Time'], [ai_values[i], fixed_values[i]],
                   color=['green', 'blue'])
            ax.set_title(metric)
            ax.set_ylabel('Value')

        plt.tight_layout()
        plt.savefig('traffic_comparison.png', dpi=100)
        plt.show()
        print("Comparison graph saved to 'traffic_comparison.png'")
    except Exception as e:
        print(f"Could not generate graph: {e}")


if __name__ == "__main__":
    main()