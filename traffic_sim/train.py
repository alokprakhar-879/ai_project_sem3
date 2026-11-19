"""Training Script - trains Q-Learning agent on multiple episodes"""

import os
import sys
import numpy as np
import pygame

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_sim.config import *
from traffic_sim.simulation import TrafficSimulation
from rl_agent.q_learning import QLearningAgent


def train_agent(num_episodes=50, frames_per_episode=2000, save_path=Q_TABLE_PATH):
    """
    Train the Q-Learning agent on multiple episodes.

    Args:
        num_episodes: Number of training episodes
        frames_per_episode: Frames per episode
        save_path: Path to save trained Q-table
    """
    print("=" * 60)
    print("Q-LEARNING TRAINING - Adaptive Traffic Signal Control")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Frames per episode: {frames_per_episode}")
    print(f"Save path: {save_path}")
    print("=" * 60)

    import pygame
    pygame.init()

    # Initialize simulation in training mode
    sim = TrafficSimulation(mode='train', load_model=True)

    # Override frames per episode
    global SIMULATION_FRAMES
    SIMULATION_FRAMES = frames_per_episode

    try:
        for episode in range(num_episodes):
            sim.reset()
            sim.frame = 0

            episode_reward = 0

            while sim.frame < frames_per_episode and sim.running:
                # Handle pygame events (allow window to stay responsive)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sim.running = False
                        break

                if not sim.running:
                    break

                # Run one step (no rendering during training for speed)
                result = sim.step()
                episode_reward += 1  # Small step reward

            # End episode
            metrics = sim.end_episode()

            # Print progress
            if (episode + 1) % 10 == 0 or episode == 0:
                stats = sim.agent.get_stats()
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Epsilon: {stats['epsilon']:.3f} | "
                      f"Q-table size: {stats['q_table_size']} | "
                      f"Vehicles crossed: {metrics['vehicles_crossed']} | "
                      f"Signal changes: {metrics['signal_changes']} | "
                      f"Emergency crossed: {metrics['emergency_crossed']}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save trained model
    sim.agent.save(save_path)
    pygame.quit()
    print(f"\nTraining complete! Q-table saved to {save_path}")
    print(f"Total episodes: {sim.agent.training_episodes}")
    print(f"Total Q-table entries: {len(sim.agent.q_table)}")

    return sim.agent


def evaluate_agent(frames=3000, load_path=Q_TABLE_PATH, spawn_prob=BASE_SPAWN_PROB,
                   emergency_prob=EMERGENCY_SPAWN_PROB, pedestrian_prob=PEDESTRIAN_SPAWN_PROB):
    """
    Evaluate the trained agent.

    Args:
        frames: Number of frames to run
        load_path: Path to load trained Q-table
        spawn_prob: Vehicle spawn probability
        emergency_prob: Emergency vehicle spawn probability
        pedestrian_prob: Pedestrian spawn probability

    Returns:
        Evaluation metrics
    """
    print("=" * 60)
    print("EVALUATION - Trained Q-Learning Agent")
    print("=" * 60)

    sim = TrafficSimulation(mode='eval', load_model=True,
                            spawn_prob=spawn_prob,
                            emergency_prob=emergency_prob,
                            pedestrian_prob=pedestrian_prob)
    SIMULATION_FRAMES = frames

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Traffic Signal Evaluation - AI Control")
    clock = pygame.time.Clock()

    while sim.frame < frames and sim.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.running = False

        sim.step()

        screen.fill(WHITE)
        sim.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    metrics = sim.end_episode()
    return metrics


def run_fixed_time_baseline(frames=3000, spawn_prob=BASE_SPAWN_PROB):
    """
    Run fixed-time traffic signal baseline for comparison.

    Args:
        frames: Number of frames
        spawn_prob: Vehicle spawn probability

    Returns:
        Baseline metrics
    """
    print("=" * 60)
    print("BASELINE - Fixed-Time Traffic Control")
    print("=" * 60)

    sim = TrafficSimulation(mode='fixed', spawn_prob=spawn_prob)
    SIMULATION_FRAMES = frames

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Traffic Signal Evaluation - Fixed Time")
    clock = pygame.time.Clock()

    while sim.frame < frames and sim.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sim.running = False

        sim.step()

        screen.fill(WHITE)
        sim.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    metrics = sim.end_episode()
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Traffic Signal RL Training')
    parser.add_argument('--episodes', type=int, default=50, help='Training episodes')
    parser.add_argument('--frames', type=int, default=2000, help='Frames per episode')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')
    parser.add_argument('--baseline', action='store_true', help='Run fixed-time baseline')
    parser.add_argument('--no-train', action='store_true', help='Skip training')

    args = parser.parse_args()

    if args.baseline:
        run_fixed_time_baseline(frames=args.frames)
    elif args.eval:
        evaluate_agent(frames=args.frames)
    else:
        train_agent(num_episodes=args.episodes, frames_per_episode=args.frames)