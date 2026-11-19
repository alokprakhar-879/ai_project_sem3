"""Streamlit Interactive Dashboard for Adaptive Traffic Signal Control"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_sim.config import *
from traffic_sim.simulation import TrafficSimulation
from rl_agent.q_learning import QLearningAgent
from rl_agent.state_manager import TrafficStateManager


st.set_page_config(
    page_title="Adaptive Traffic Signal Control - AI Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize Streamlit session state."""
    if 'simulation' not in st.session_state:
        st.session_state.simulation = None
    if 'agent' not in st.session_state:
        st.session_state.agent = QLearningAgent()
        st.session_state.agent.load()
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    if 'comparison_data' not in st.session_state:
        st.session_state.comparison_data = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0


def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.title("🚦 Traffic Control")
    st.sidebar.markdown("### Simulation Parameters")

    spawn_rate = st.sidebar.slider(
        "Vehicle Spawn Rate", 0.01, 0.15, BASE_SPAWN_PROB, 0.01,
        help="Probability of vehicle spawning per frame per direction"
    )

    emergency_prob = st.sidebar.slider(
        "Emergency Vehicle Probability", 0.0, 0.1, EMERGENCY_SPAWN_PROB, 0.01,
        help="Probability of emergency vehicle (ambulance/police) spawning"
    )

    pedestrian_prob = st.sidebar.slider(
        "Pedestrian Spawn Rate", 0.0, 0.1, PEDESTRIAN_SPAWN_PROB, 0.01,
        help="Probability of pedestrian spawning per frame per side"
    )

    mode = st.sidebar.selectbox(
        "Control Mode",
        ["AI-Based (Q-Learning)", "Fixed-Time (Baseline)"],
        index=0,
        help="Select traffic signal control strategy"
    )

    st.sidebar.markdown("### Q-Learning Parameters")
    lr = st.sidebar.slider("Learning Rate (Alpha)", 0.01, 0.5, LEARNING_RATE, 0.01)
    gamma = st.sidebar.slider("Discount Factor (Gamma)", 0.5, 0.99, DISCOUNT_FACTOR, 0.01)
    epsilon = st.sidebar.slider("Exploration (Epsilon)", 0.0, 1.0, EPSILON_START, 0.05)

    st.sidebar.markdown("### Actions")
    st.sidebar.info("""
    **Available Actions:**
    - 🟢 NS Green: North-South green
    - 🟢 EW Green: East-West green
    - 🚶 Pedestrian N/S/E/W: Side crossing
    - ⏱️ Extend Green: Extend current phase
    """)

    st.sidebar.markdown("### Emergency Vehicles")
    st.sidebar.warning("""
    **Priority Vehicles:**
    - 🚑 Ambulance
    - 🚓 Police

    Emergency vehicles override normal control for safe passage.
    """)

    return {
        'spawn_rate': spawn_rate,
        'emergency_prob': emergency_prob,
        'pedestrian_prob': pedestrian_prob,
        'mode': mode,
        'lr': lr,
        'gamma': gamma,
        'epsilon': epsilon,
    }


def render_metrics_panel(metrics):
    """Render live metrics panel."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Vehicles Crossed", metrics.get('crossed_vehicles', 0))
        st.metric("Total Queue", metrics.get('total_vehicles', 0))

    with col2:
        st.metric("Pedestrians Crossed", metrics.get('crossed_pedestrians', 0))
        st.metric("Total Pedestrians", metrics.get('total_pedestrians', 0))

    with col3:
        st.metric("Signal Phase", metrics.get('phase', 'NS_GREEN'))
        st.metric("Signal Changes", metrics.get('signal_changes', 0))

    with col4:
        st.metric("Frame", metrics.get('frame', 0))
        if metrics.get('emergency_active'):
            st.error(f"🚨 Emergency: {metrics.get('emergency_dir', 'N/A')}")
        else:
            st.success("✅ No Emergency")


def render_vehicle_distribution(vehicle_counts, pedestrian_counts):
    """Render vehicle distribution chart."""
    directions = ['North', 'South', 'East', 'West']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Vehicles',
        x=directions,
        y=vehicle_counts,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ))

    fig.add_trace(go.Bar(
        name='Pedestrians',
        x=directions,
        y=pedestrian_counts,
        marker_color=['#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    ))

    fig.update_layout(
        title="Traffic Distribution by Direction",
        xaxis_title="Direction",
        yaxis_title="Count",
        barmode='group',
        height=300,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_q_table_stats(agent):
    """Render Q-Learning agent statistics."""
    stats = agent.get_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Q-Table Size", stats['q_table_size'])
    with col2:
        st.metric("Training Episodes", stats['training_episodes'])
    with col3:
        st.metric("Total Updates", stats['total_updates'])

    st.metric("Current Epsilon", f"{stats['epsilon']:.3f}")
    st.metric("Avg Reward (Last 100)", f"{stats['avg_reward_last_100']:.2f}")


def render_comparison_chart(comparison_data):
    """Render performance comparison chart."""
    if comparison_data is None:
        st.info("Run comparison to see results.")
        return

    metrics = ['Vehicles Crossed', 'Avg Waiting Time', 'Queue Length', 'Signal Changes']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Fixed-Time',
        x=metrics,
        y=[comparison_data.get('fixed_vehicles', 0),
           comparison_data.get('fixed_wait', 0),
           comparison_data.get('fixed_queue', 0),
           comparison_data.get('fixed_signals', 0)],
        marker_color='#3498DB'
    ))

    fig.add_trace(go.Bar(
        name='AI-Based (Q-Learning)',
        x=metrics,
        y=[comparison_data.get('ai_vehicles', 0),
           comparison_data.get('ai_wait', 0),
           comparison_data.get('ai_queue', 0),
           comparison_data.get('ai_signals', 0)],
        marker_color='#2ECC71'
    ))

    fig.update_layout(
        title="Performance Comparison: Fixed-Time vs AI-Based",
        barmode='group',
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit dashboard."""
    init_session_state()

    st.title("🚦 Adaptive Traffic Signal Control Using Q-Learning")
    st.markdown("""
    **India-Specific Intelligent Traffic Management System**

    This system uses Reinforcement Learning (Q-Learning) to dynamically optimize
    traffic signal timings based on real-time simulated traffic conditions including
    vehicles, pedestrians, and emergency vehicles.
    """)

    # Sidebar controls
    params = render_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Simulation",
        "🧠 Q-Learning Monitor",
        "📈 Performance Comparison",
        "⚙️ Training"
    ])

    with tab1:
        st.header("Live Traffic Simulation")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Intersection View (Simulation)")

            # Initialize simulation if needed
            if st.session_state.simulation is None:
                mode = 'train' if 'AI' in params['mode'] else 'fixed'
                st.session_state.simulation = TrafficSimulation(
                    mode=mode,
                    spawn_prob=params['spawn_rate'],
                    emergency_prob=params['emergency_prob'],
                    pedestrian_prob=params['pedestrian_prob'],
                    load_model=True
                )

            sim = st.session_state.simulation

            # Run simulation step
            if st.session_state.running:
                if 'AI' in params['mode']:
                    state, info = sim.get_state_for_agent()
                    action = sim.agent.choose_action(state, training=(params['mode'] == 'AI-Based (Q-Learning)'))
                    sim.step(action)
                else:
                    sim.step(None)
                st.session_state.frame_count = sim.frame

            # Display metrics
            result = sim._get_step_result()
            render_metrics_panel(result)

            # Vehicle distribution
            render_vehicle_distribution(
                result.get('vehicle_counts', [0, 0, 0, 0]),
                result.get('pedestrian_counts', [0, 0, 0, 0])
            )

            # Emergency status
            if result.get('emergency_active'):
                st.error(f"🚨 **EMERGENCY VEHICLE DETECTED** - Direction: {result.get('emergency_dir')}")
                st.info("Priority override activated - giving green to emergency vehicle direction.")

        with col2:
            st.subheader("Controls")
            if st.button("▶️ Start Simulation", type="primary"):
                st.session_state.running = True

            if st.button("⏸️ Pause Simulation"):
                st.session_state.running = False

            if st.button("🔄 Reset Simulation"):
                st.session_state.simulation = TrafficSimulation(
                    mode='train' if 'AI' in params['mode'] else 'fixed',
                    spawn_prob=params['spawn_rate'],
                    emergency_prob=params['emergency_prob'],
                    pedestrian_prob=params['pedestrian_prob'],
                    load_model=True
                )
                st.session_state.frame_count = 0
                st.session_state.running = False

            st.markdown("---")
            st.subheader("Current Status")
            st.write(f"**Mode:** {params['mode']}")
            st.write(f"**Frame:** {st.session_state.frame_count}")
            st.write(f"**Running:** {st.session_state.running}")

    with tab2:
        st.header("Q-Learning Monitor")

        agent = st.session_state.agent

        col1, col2 = st.columns(2)

        with col1:
            render_q_table_stats(agent)

            # Q-table heatmap (sample)
            if len(agent.q_table) > 0:
                st.subheader("Q-Table Sample (First 20 entries)")
                q_items = list(agent.q_table.items())[:20]
                q_data = []
                for (state, action), q_val in q_items:
                    q_data.append({
                        'State': str(state),
                        'Action': ACTIONS[action] if isinstance(action, int) else action,
                        'Q-Value': round(q_val, 3),
                    })
                st.dataframe(pd.DataFrame(q_data), use_container_width=True)

        with col2:
            st.subheader("Reward Function Weights")
            weights = {
                'Vehicle Waiting': -0.5,
                'Pedestrian Waiting': -0.3,
                'Vehicle Crossed': +1.0,
                'Pedestrian Crossed': +0.8,
                'Emergency Crossed': +5.0,
                'Signal Change': -0.1,
                'Emergency Blocked': -10.0,
                'Conflict': -20.0,
            }
            for k, v in weights.items():
                color = "🟢" if v > 0 else "🔴"
                st.write(f"{color} **{k}:** {v}")

            st.markdown("""
            **Reward Design:**
            - Positive rewards for vehicle/pedestrian crossings
            - Negative rewards for waiting time and queue length
            - High penalty for blocking emergency vehicles
            - Penalty for unsafe conflicts
            - Small penalty for signal changes (discourage unnecessary switching)
            """)

    with tab3:
        st.header("Performance Comparison")

        st.markdown("""
        Compare adaptive Q-Learning control against traditional fixed-time signals.
        """)

        if st.button("🔄 Run Comparison"):
            with st.spinner("Running simulations..."):
                # Run fixed-time baseline
                from traffic_sim.train import run_fixed_time_baseline, evaluate_agent
                fixed_metrics = run_fixed_time_baseline(frames=3000)
                ai_metrics = evaluate_agent(frames=3000)

                st.session_state.comparison_data = {
                    'fixed_vehicles': fixed_metrics.get('vehicles_crossed', 0),
                    'fixed_wait': fixed_metrics.get('total_vehicle_waiting', 0) / max(fixed_metrics.get('vehicles_crossed', 1), 1),
                    'fixed_queue': fixed_metrics.get('total_vehicle_waiting', 0),
                    'fixed_signals': fixed_metrics.get('signal_changes', 0),
                    'ai_vehicles': ai_metrics.get('vehicles_crossed', 0),
                    'ai_wait': ai_metrics.get('total_vehicle_waiting', 0) / max(ai_metrics.get('vehicles_crossed', 1), 1),
                    'ai_queue': ai_metrics.get('total_vehicle_waiting', 0),
                    'ai_signals': ai_metrics.get('signal_changes', 0),
                }

        render_comparison_chart(st.session_state.comparison_data)

        if st.session_state.comparison_data:
            cd = st.session_state.comparison_data
            st.markdown("### Summary Table")
            summary = pd.DataFrame({
                'Metric': ['Vehicles Crossed', 'Avg Waiting Time', 'Queue Length', 'Signal Changes'],
                'Fixed-Time': [
                    cd['fixed_vehicles'],
                    f"{cd['fixed_wait']:.2f}",
                    cd['fixed_queue'],
                    cd['fixed_signals'],
                ],
                'AI-Based': [
                    cd['ai_vehicles'],
                    f"{cd['ai_wait']:.2f}",
                    cd['ai_queue'],
                    cd['ai_signals'],
                ],
                'Improvement': [
                    f"{((cd['ai_vehicles'] - cd['fixed_vehicles']) / max(cd['fixed_vehicles'], 1) * 100):.1f}%",
                    f"{((cd['fixed_wait'] - cd['ai_wait']) / max(cd['fixed_wait'], 1) * 100):.1f}%",
                    f"{((cd['fixed_queue'] - cd['ai_queue']) / max(cd['fixed_queue'], 1) * 100):.1f}%",
                    f"{((cd['fixed_signals'] - cd['ai_signals']) / max(cd['fixed_signals'], 1) * 100):.1f}%",
                ],
            })
            st.dataframe(summary, use_container_width=True)

    with tab4:
        st.header("Training")

        st.markdown("""
        Train the Q-Learning agent on multiple traffic scenarios.
        """)

        col1, col2 = st.columns(2)

        with col1:
            num_episodes = st.slider("Training Episodes", 10, 200, 50)
            frames_per_episode = st.slider("Frames per Episode", 500, 5000, 2000)

        with col2:
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training Q-Learning agent..."):
                    from traffic_sim.train import train_agent
                    agent = train_agent(
                        num_episodes=num_episodes,
                        frames_per_episode=frames_per_episode,
                        save_path=Q_TABLE_PATH
                    )
                    st.session_state.agent = agent
                    st.success("Training complete! Model saved.")

            if st.button("💾 Save Model"):
                st.session_state.agent.save()
                st.success("Model saved to models/q_table.npy")

            if st.button("📂 Load Model"):
                if st.session_state.agent.load():
                    st.success("Model loaded successfully.")
                else:
                    st.warning("No saved model found.")

        st.markdown("---")
        st.subheader("Training Scenarios")
        st.markdown("""
        The agent is trained on diverse scenarios:
        - Low, medium, and high traffic density
        - Unequal traffic across directions
        - High pedestrian crowd scenarios
        - Emergency vehicle arrivals
        - Combined heavy traffic + pedestrians + emergencies
        """)


if __name__ == "__main__":
    main()