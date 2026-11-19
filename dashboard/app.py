"""Streamlit Interactive Dashboard for Adaptive Traffic Signal Control - User Friendly"""

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


st.set_page_config(
    page_title="Adaptive Traffic Signal Control - AI Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===================== CUSTOM CSS =====================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin-bottom: 1rem;
    }
    .emergency-alert {
        background: #ffe6e6;
        border: 2px solid #ff4444;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { background-color: #ffe6e6; }
        50% { background-color: #ffcccc; }
        100% { background-color: #ffe6e6; }
    }
    .success-box {
        background: #e6ffe6;
        border: 2px solid #44aa44;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .section-divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }
    .simulation-container {
        background: #ffffff;
        border: 2px solid #d0d0d0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .control-panel {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state."""
    if 'simulation' not in st.session_state:
        st.session_state.simulation = None
    if 'agent' not in st.session_state:
        st.session_state.agent = QLearningAgent()
        st.session_state.agent.load()
    if 'comparison_data' not in st.session_state:
        st.session_state.comparison_data = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'sim_size' not in st.session_state:
        st.session_state.sim_size = 1.0
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False


def render_header():
    """Render main dashboard header."""
    st.markdown("""
    <div class="main-header">
        <h1 style='margin:0;'>🚦 Adaptive Traffic Signal Control</h1>
        <p style='margin:0.5rem 0 0 0; opacity:0.9;'>
            AI-Based Intelligent Traffic Management System using Q-Learning Reinforcement Learning
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar controls with size adjustment options."""
    st.sidebar.markdown("## 🛠️ Control Panel")

    # === SIZE & DISPLAY SETTINGS ===
    st.sidebar.markdown("### 📏 Display Settings")
    sim_size = st.sidebar.slider(
        "Simulation Display Size",
        min_value=0.7, max_value=1.5, value=st.session_state.sim_size, step=0.1,
        help="Adjust the size of the simulation visualization"
    )
    st.session_state.sim_size = sim_size

    chart_height = st.sidebar.slider(
        "Chart Height (px)",
        min_value=250, max_value=600, value=400, step=50,
        help="Adjust height of data visualization charts"
    )

    font_size = st.sidebar.selectbox(
        "Text Size",
        options=["Small", "Medium", "Large"],
        index=1,
        help="Adjust dashboard text readability"
    )

    st.sidebar.markdown("---")

    # === SIMULATION PARAMETERS ===
    st.sidebar.markdown("### 🚗 Traffic Parameters")

    spawn_rate = st.sidebar.slider(
        "Vehicle Spawn Rate",
        min_value=0.01, max_value=0.20, value=BASE_SPAWN_PROB, step=0.01,
        help="Probability of vehicle spawning per frame per direction"
    )

    emergency_prob = st.sidebar.slider(
        "Emergency Vehicle Probability",
        min_value=0.0, max_value=0.15, value=EMERGENCY_SPAWN_PROB, step=0.01,
        help="Probability of ambulance/police vehicle spawning"
    )

    pedestrian_prob = st.sidebar.slider(
        "Pedestrian Spawn Rate",
        min_value=0.0, max_value=0.15, value=PEDESTRIAN_SPAWN_PROB, step=0.01,
        help="Probability of pedestrian spawning per side"
    )

    st.sidebar.markdown("---")

    # === CONTROL MODE ===
    st.sidebar.markdown("### 🎯 Control Mode")
    mode = st.sidebar.selectbox(
        "Traffic Signal Strategy",
        options=["AI-Based (Q-Learning)", "Fixed-Time (Baseline)"],
        index=0,
        help="Choose between AI adaptive control or traditional fixed-time signals"
    )

    # === Q-LEARNING PARAMETERS ===
    with st.sidebar.expander("⚙️ Q-Learning Advanced Settings", expanded=False):
        lr = st.slider("Learning Rate (Alpha)", 0.01, 0.5, LEARNING_RATE, 0.01)
        gamma = st.slider("Discount Factor (Gamma)", 0.5, 0.99, DISCOUNT_FACTOR, 0.01)
        epsilon = st.slider("Exploration (Epsilon)", 0.0, 1.0, EPSILON_START, 0.05)

    # === EMERGENCY INFO ===
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚨 Emergency Vehicles")
    st.sidebar.warning("""
    **Priority Vehicles:**
    - 🚑 Ambulance
    - 🚓 Police

    Emergency vehicles automatically override normal traffic signals for safe passage.
    """)

    return {
        'spawn_rate': spawn_rate,
        'emergency_prob': emergency_prob,
        'pedestrian_prob': pedestrian_prob,
        'mode': mode,
        'lr': lr,
        'gamma': gamma,
        'epsilon': epsilon,
        'chart_height': chart_height,
        'font_size': font_size,
    }


def render_metrics_cards(metrics):
    """Render live metrics as clean cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 🚗 Vehicles")
        st.metric("Crossed", metrics.get('crossed_vehicles', 0), delta=None)
        st.metric("In Queue", metrics.get('total_vehicles', 0))

    with col2:
        st.markdown("### 🚶 Pedestrians")
        st.metric("Crossed", metrics.get('crossed_pedestrians', 0), delta=None)
        st.metric("Waiting", metrics.get('total_pedestrians', 0))

    with col3:
        st.markdown("### 🚦 Signals")
        st.metric("Current Phase", metrics.get('phase', 'NS_GREEN'))
        st.metric("Changes", metrics.get('signal_changes', 0))

    with col4:
        st.markdown("### ⏱️ Status")
        st.metric("Frame", metrics.get('frame', 0))
        if metrics.get('emergency_active'):
            st.markdown(
                f'<div class="emergency-alert">'
                f'<strong>🚨 EMERGENCY VEHICLE DETECTED</strong><br>'
                f'Direction: {metrics.get("emergency_dir", "N/A")}<br>'
                f'Priority override: ACTIVE'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="success-box">✅ All Systems Normal<br>No Emergency Vehicles</div>',
                unsafe_allow_html=True
            )


def render_vehicle_chart(vehicle_counts, pedestrian_counts, chart_height):
    """Render vehicle distribution bar chart."""
    directions = ['North', 'South', 'East', 'West']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='🚗 Vehicles',
        x=directions,
        y=vehicle_counts,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
        text=vehicle_counts,
        textposition='outside',
    ))

    fig.add_trace(go.Bar(
        name='🚶 Pedestrians',
        x=directions,
        y=pedestrian_counts,
        marker_color=['#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'],
        text=pedestrian_counts,
        textposition='outside',
    ))

    fig.update_layout(
        title="📊 Traffic Distribution by Direction",
        xaxis_title="Direction",
        yaxis_title="Count",
        barmode='group',
        height=chart_height,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        margin=dict(l=50, r=50, t=80, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)


def render_q_table_section(agent, chart_height):
    """Render Q-Learning monitor section."""
    st.markdown("## 🧠 Q-Learning Agent Monitor")

    stats = agent.get_stats()

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Q-Table Entries", stats['q_table_size'])
    with col2:
        st.metric("Training Episodes", stats['training_episodes'])
    with col3:
        st.metric("Total Updates", stats['total_updates'])
    with col4:
        st.metric("Epsilon", f"{stats['epsilon']:.3f}")

    st.markdown("### 📈 Average Reward (Last 100 Episodes)")
    st.metric("Reward", f"{stats['avg_reward_last_100']:.2f}")

    # Q-Table sample
    st.markdown("### 📋 Q-Table Sample (First 15 State-Action Pairs)")
    if len(agent.q_table) > 0:
        q_items = list(agent.q_table.items())[:15]
        q_data = []
        for (state, action), q_val in q_items:
            action_name = ACTIONS[action] if isinstance(action, int) else action
            q_data.append({
                'State': str(state)[:60] + '...' if len(str(state)) > 60 else str(state),
                'Action': action_name,
                'Q-Value': round(q_val, 3),
            })
        st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)
    else:
        st.info("No Q-table data yet. Train the agent first.")


def render_reward_weights():
    """Render reward function explanation."""
    st.markdown("### 🎯 Reward Function Design")

    weights = [
        ('🚗 Vehicle Crossed', '+1.0', 'Positive', 'Reward for each vehicle passing through'),
        ('🚶 Pedestrian Crossed', '+0.8', 'Positive', 'Reward for each pedestrian crossing safely'),
        ('🚑 Emergency Crossed', '+5.0', 'High Positive', 'High reward for emergency vehicle passage'),
        ('⏳ Vehicle Waiting', '-0.5', 'Negative', 'Penalty per waiting vehicle'),
        ('⏳ Pedestrian Waiting', '-0.3', 'Negative', 'Penalty per waiting pedestrian'),
        ('🔄 Signal Change', '-0.1', 'Small Negative', 'Discourage unnecessary signal switching'),
        ('🚫 Emergency Blocked', '-10.0', 'High Negative', 'Severe penalty for blocking emergency vehicles'),
        ('⚠️ Conflict', '-20.0', 'Severe Negative', 'Penalty for unsafe signal combinations'),
    ]

    for name, value, impact, desc in weights:
        color = "🟢" if impact in ['Positive', 'High Positive'] else "🔴"
        st.markdown(f"""
        <div class="metric-card">
            <strong>{color} {name}</strong> → <code>{value}</code><br>
            <em>Impact:</em> {impact} | <em>Purpose:</em> {desc}
        </div>
        """, unsafe_allow_html=True)


def render_comparison_section(chart_height):
    """Render performance comparison section."""
    st.markdown("## 📈 Performance Comparison: AI vs Fixed-Time")

    st.markdown("""
    Compare the adaptive Q-Learning traffic controller against traditional fixed-time signals
    using identical traffic scenarios for fair evaluation.
    """)

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Run Fresh Comparison", type="primary", use_container_width=True):
            st.session_state.comparison_data = None  # Reset to force recalculation
            with st.spinner("Running AI simulation..."):
                from traffic_sim.train import evaluate_agent
                ai_metrics = evaluate_agent(frames=3000)
            with st.spinner("Running fixed-time baseline..."):
                from traffic_sim.train import run_fixed_time_baseline
                fixed_metrics = run_fixed_time_baseline(frames=3000)

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
            st.success("Comparison complete!")

    if st.session_state.comparison_data:
        cd = st.session_state.comparison_data

        # Comparison chart
        metrics_names = ['Vehicles Crossed', 'Avg Waiting Time', 'Queue Length', 'Signal Changes']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='🔵 Fixed-Time',
            x=metrics_names,
            y=[cd['fixed_vehicles'], cd['fixed_wait'], cd['fixed_queue'], cd['fixed_signals']],
            marker_color='#3498DB',
            text=[cd['fixed_vehicles'], f"{cd['fixed_wait']:.1f}", cd['fixed_queue'], cd['fixed_signals']],
            textposition='outside',
        ))
        fig.add_trace(go.Bar(
            name='🟢 AI-Based (Q-Learning)',
            x=metrics_names,
            y=[cd['ai_vehicles'], cd['ai_wait'], cd['ai_queue'], cd['ai_signals']],
            marker_color='#2ECC71',
            text=[cd['ai_vehicles'], f"{cd['ai_wait']:.1f}", cd['ai_queue'], cd['ai_signals']],
            textposition='outside',
        ))
        fig.update_layout(
            title="Performance Metrics Comparison",
            barmode='group',
            height=chart_height,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            margin=dict(l=50, r=50, t=80, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.markdown("### 📊 Summary Table")
        summary = pd.DataFrame({
            'Metric': metrics_names,
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
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Click 'Run Fresh Comparison' to generate performance data.")


def render_training_section():
    """Render training section."""
    st.markdown("## ⚙️ Q-Learning Agent Training")

    st.markdown("""
    Train the reinforcement learning agent on multiple traffic scenarios including
    low/medium/high density, pedestrian crowds, and emergency vehicle arrivals.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Training Configuration")
        num_episodes = st.slider("Number of Episodes", 10, 500, 50, step=10)
        frames_per_episode = st.slider("Frames per Episode", 500, 5000, 2000, step=500)

    with col2:
        st.markdown("### Training Actions")
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            with st.spinner(f"Training Q-Learning agent ({num_episodes} episodes...)"):
                from traffic_sim.train import train_agent
                agent = train_agent(
                    num_episodes=num_episodes,
                    frames_per_episode=frames_per_episode,
                    save_path=Q_TABLE_PATH
                )
                st.session_state.agent = agent
            st.success("✅ Training Complete! Model saved to models/q_table.npy")
            st.balloons()

        if st.button("💾 Save Current Model", use_container_width=True):
            st.session_state.agent.save()
            st.success("✅ Model saved!")

        if st.button("📂 Load Saved Model", use_container_width=True):
            if st.session_state.agent.load():
                st.success("✅ Model loaded successfully!")
            else:
                st.warning("⚠️ No saved model found.")

    # Scenario info
    with st.expander("📋 Training Scenarios Covered", expanded=False):
        st.markdown("""
        The agent trains on diverse Indian traffic scenarios:
        - 🟢 Low traffic density (sparse vehicle arrival)
        - 🟡 Medium traffic density (balanced flow)
        - 🔴 Heavy traffic congestion (high vehicle density)
        - 🚶 High pedestrian crowd scenarios
        - 🚑 Emergency vehicle arrivals (ambulance/police)
        - 🔀 Combined scenarios (heavy traffic + pedestrians + emergencies)
        - 📉 Unequal traffic density across different roads
        """)


def main():
    """Main Streamlit dashboard."""
    init_session_state()

    # Header
    render_header()

    # Sidebar controls
    params = render_sidebar()

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Simulation",
        "🧠 Q-Learning Monitor",
        "📈 Performance Comparison",
        "⚙️ Agent Training",
    ])

    # ===================== TAB 1: LIVE SIMULATION =====================
    with tab1:
        st.markdown("## 📊 Live Traffic Simulation")

        # Control buttons
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        with col_btn1:
            if st.button("▶️ Start Simulation", type="primary", use_container_width=True):
                st.session_state.running = True
        with col_btn2:
            if st.button("⏸️ Pause Simulation", use_container_width=True):
                st.session_state.running = False
        with col_btn3:
            if st.button("🔄 Reset Simulation", use_container_width=True):
                mode = 'train' if 'AI' in params['mode'] else 'fixed'
                st.session_state.simulation = TrafficSimulation(
                    mode=mode,
                    spawn_prob=params['spawn_rate'],
                    emergency_prob=params['emergency_prob'],
                    pedestrian_prob=params['pedestrian_prob'],
                    load_model=True
                )
                st.session_state.frame_count = 0
                st.session_state.running = False
        with col_btn4:
            st.session_state.auto_refresh = st.checkbox("🔁 Auto-Refresh", value=st.session_state.auto_refresh)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Initialize simulation
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
        if st.session_state.running or st.session_state.auto_refresh:
            if 'AI' in params['mode']:
                state, info = sim.get_state_for_agent()
                action = sim.agent.choose_action(
                    state,
                    training=(params['mode'] == 'AI-Based (Q-Learning)')
                )
                sim.step(action)
            else:
                sim.step(None)
            st.session_state.frame_count = sim.frame

        # Metrics cards
        result = sim._get_step_result()
        render_metrics_cards(result)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Simulation visualization placeholder
        st.markdown("### 🎯 Intersection Visualization")
        st.info("""
        💡 **Simulation View:** Run `python main.py --eval` or `python main.py --train` 
        to see the full Pygame 2D visualization with moving vehicles, traffic lights, 
        zebra crossings, and emergency vehicles.

        The Streamlit dashboard shows real-time metrics, charts, and Q-Learning analytics.
        """)

        # Vehicle distribution chart
        st.markdown("### 📊 Real-Time Traffic Distribution")
        render_vehicle_chart(
            result.get('vehicle_counts', [0, 0, 0, 0]),
            result.get('pedestrian_counts', [0, 0, 0, 0]),
            params['chart_height']
        )

        # Emergency alert
        if result.get('emergency_active'):
            st.markdown(
                f'<div class="emergency-alert">'
                f'<strong>🚨 EMERGENCY VEHICLE PRIORITY ACTIVE</strong><br>'
                f'Direction: {result.get("emergency_dir", "N/A")} | '
                f'Ambulance/Police detected - signal override engaged'
                f'</div>',
                unsafe_allow_html=True
            )

        # Status info
        with st.expander("ℹ️ Current Simulation Details", expanded=False):
            st.write(f"**Control Mode:** {params['mode']}")
            st.write(f"**Current Frame:** {st.session_state.frame_count}")
            st.write(f"**Running:** {'Yes' if st.session_state.running else 'No'}")
            st.write(f"**Auto-Refresh:** {'Yes' if st.session_state.auto_refresh else 'No'}")
            st.write(f"**Q-Table Size:** {len(sim.agent.q_table)} entries")
            st.write(f"**Epsilon:** {sim.agent.epsilon:.3f}")

    # ===================== TAB 2: Q-LEARNING MONITOR =====================
    with tab2:
        render_q_table_section(st.session_state.agent, params['chart_height'])
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        render_reward_weights()

    # ===================== TAB 3: PERFORMANCE COMPARISON =====================
    with tab3:
        render_comparison_section(params['chart_height'])

    # ===================== TAB 4: AGENT TRAINING =====================
    with tab4:
        render_training_section()

    # Footer
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🚦 Adaptive Traffic Signal Control Using Q-Learning | India-Specific Intelligent Traffic Management</p>
        <p>Built with Python, NumPy, Pygame, Streamlit, Plotly</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()