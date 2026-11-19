"""Main Entry Point - Adaptive Traffic Signal Control System"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from traffic_sim.train import train_agent, evaluate_agent, run_fixed_time_baseline


def main():
    parser = argparse.ArgumentParser(
        description='Adaptive Traffic Signal Control Using Q-Learning'
    )
    parser.add_argument('--train', action='store_true', help='Train the Q-Learning agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate trained agent')
    parser.add_argument('--baseline', action='store_true', help='Run fixed-time baseline')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    parser.add_argument('--episodes', type=int, default=50, help='Training episodes')
    parser.add_argument('--frames', type=int, default=2000, help='Frames per episode')

    args = parser.parse_args()

    if args.dashboard:
        print("Launching Streamlit dashboard...")
        print("Run: streamlit run dashboard/app.py")
        os.system('streamlit run dashboard/app.py')
    elif args.train:
        train_agent(num_episodes=args.episodes, frames_per_episode=args.frames)
    elif args.eval:
        evaluate_agent(frames=args.frames)
    elif args.baseline:
        run_fixed_time_baseline(frames=args.frames)
    else:
        print("Adaptive Traffic Signal Control Using Q-Learning")
        print("=" * 50)
        print("Usage:")
        print("  python main.py --train          Train the agent")
        print("  python main.py --eval           Evaluate trained agent")
        print("  python main.py --baseline       Run fixed-time baseline")
        print("  python main.py --dashboard      Launch Streamlit dashboard")
        print("")
        print("Examples:")
        print("  python main.py --train --episodes 100")
        print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()