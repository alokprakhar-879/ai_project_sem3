"""Streamlit Interactive Dashboard for Adaptive Traffic Signal Control - Upgraded"""

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
        margin-bottom: 0.5rem;
    }
    .emergency-alert {
        background: #ffe6e6;
        border: 2px solid #ff4444;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { background-color: #ffe6e6; box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.4); }
        70% { background-color: #ffcccc; box-shadow: 0 0 0 10px rgba(255, 68, 68, 0); }
        100% { background-color: #ffe6e6; box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
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
    .decision-box {
        background: #f0f8ff;
        border: 2px solid #4a90e2;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .q-value-card {
        background: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .q-value-best {
        background: #e8f5e9;
        border: 2px solid #4caf50;
    }
    .direction-stats {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    .direction-card {
        flex: 1;
        padding: 0.75rem;
        margin: 0 0.25rem;
        border-radius: 6px;
        text-align: center;
        background: #f8f9fa;
    }
    .direction-north { border-top: 4px solid #FF6B6B; }
    .direction-south { border-top: 4px solid #4ECDC4; }
    .direction-east { border-top: 4px solid #45B7D1; }
    .direction-west { border-top: 4px solid #FFA07A; }
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
    if 'speed_multiplier' not in st.session_state:
        st.session_state.speed_multiplier = 1.0
    if 'training_progress' not in st.session_state:
        st.session_state.training_progress = 0
    if 'training_status' not in st.session_state:
        st.session_state.training_status = "Idle"


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
    """Render sidebar controls."""
    st.sidebar.markdown("## 🛠️ Control Panel")

    # === SIMULATION CONTROLS ===
    st.sidebar.markdown("### 🎮 Simulation Controls")

    col_start, col_pause = st.sidebar.columns(2)
    with col_start:
        start_btn = st.button("▶️ Start", use_container_width=True, key="btn_start")
    with col_pause:
        pause_btn = st.button("⏸️ Pause", use_container_width=True, key="btn_pause")

    col_restart, col_reset = st.sidebar.columns(2)
    with col_restart:
        restart_btn = st.button("🔄 Restart", use_container_width=True, key="btn_restart")
    with col_reset:
        reset_stats_btn = st.button("📊 Reset Stats", use_container_width=True, key="btn_reset_stats")

    st.sidebar.markdown("---")

    # === SPEED CONTROL ===
    st.sidebar.markdown("### ⚡ Simulation Speed")
    speed_option = st.sidebar.selectbox(
        "Speed Multiplier",
        options=["0.25x", "0.5x", "1x", "2x"],
        index=2,
        key="speed_select",
        help="0.25x = Slow motion (clearly see each vehicle), 2x = Fast forward"
    )
    speed_map = {'0.25x': 0.25, '0.5x': 0.5, '1x': 1.0, '2x': 2.0}
    speed_multiplier = speed_map[speed_option]

    st.sidebar.markdown("---")

    # === TRAFFIC DENSITY ===
    st.sidebar.markdown("### 🚗 Traffic Density")

    spawn_rate = st.sidebar.slider(
        "Overall Spawn Rate",
        min_value=0.01, max_value=0.25, value=BASE_SPAWN_PROB, step=0.01,
        key="spawn_rate",
        help="Probability of vehicle spawning per frame per direction"
    )

    # Per-direction density multipliers
    st.sidebar.markdown("**Per-Direction Multipliers:**")
    col_n, col_s = st.sidebar.columns(2)
    with col_n:
        north_mult = st.slider("North", 0.0, 3.0, 1.0, 0.1, key="north_mult")
    with col_s:
        south_mult = st.slider("South", 0.0, 3.0, 1.0, 0.1, key="south_mult")
    col_e, col_w = st.sidebar.columns(2)
    with col_e:
        east_mult = st.slider("East", 0.0, 3.0, 1.0, 0.1, key="east_mult")
    with col_w:
        west_mult = st.slider("West", 0.0, 3.0, 1.0, 0.1, key="west_mult")

    pedestrian_prob = st.sidebar.slider(
        "🚶 Pedestrian Spawn Rate",
        min_value=0.0, max_value=0.15, value=PEDESTRIAN_SPAWN_PROB, step=0.01,
        key="pedestrian_prob"
    )

    emergency_prob = st.sidebar.slider(
        "🚑 Emergency Probability",
        min_value=0.0, max_value=0.10, value=EMERGENCY_SPAWN_PROB, step=0.01,
        key="emergency_prob"
    )

    st.sidebar.markdown("---")

    # === CONTROL MODE ===
    st.sidebar.markdown("### 🎯 Control Mode")
    mode = st.sidebar.selectbox(
        "Traffic Signal Strategy",
        options=["AI-Based (Q-Learning)", "Fixed-Time (Baseline)"],
        index=0,
        key="mode_select"
    )

    st.sidebar.markdown("---")

    # === Q-LEARNING PARAMETERS ===
    with st.sidebar.expander("⚙️ Q-Learning Settings", expanded=False):
        lr = st.slider("Learning Rate (Alpha)", 0.01, 0.5, LEARNING_RATE, 0.01, key="q_lr")
        gamma = st.slider("Discount Factor (Gamma)", 0.5, 0.99, DISCOUNT_FACTOR, 0.01, key="q_gamma")
        epsilon = st.slider("Exploration (Epsilon)", 0.0, 1.0, EPSILON_START, 0.05, key="q_epsilon")

    # === EMERGENCY INFO ===
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚨 Emergency Vehicles")
    st.sidebar.warning("""
    **Priority Vehicles:**
    - 🚑 Ambulance
    - 🚓 Police

    Emergency vehicles override normal traffic signals for safe passage.
    """)

    # Handle button clicks
    if start_btn:
        st.session_state.running = True
    if pause_btn:
        st.session_state.running = False
    if restart_btn:
        if st.session_state.simulation:
            st.session_state.simulation.reset(keep_q_table=True)
        st.session_state.running = False
        st.session_state.frame_count = 0
    if reset_stats_btn:
        if st.session_state.simulation:
            st.session_state.simulation.reset(keep_q_table=True)
        st.session_state.running = False
        st.session_state.frame_count = 0

    return {
        'spawn_rate': spawn_rate,
        'emergency_prob': emergency_prob,
        'pedestrian_prob': pedestrian_prob,
        'mode': mode,
        'lr': lr,
        'gamma': gamma,
        'epsilon': epsilon,
        'speed_multiplier': speed_multiplier,
        'north_mult': north_mult,
        'south_mult': south_mult,
        'east_mult': east_mult,
        'west_mult': west_mult,
        'start_btn': start_btn,
        'pause_btn': pause_btn,
        'restart_btn': restart_btn,
        'reset_stats_btn': reset_stats_btn,
    }


def render_traffic_statistics(metrics):
    """Render traffic statistics section with direction-wise breakdown."""
    st.markdown("## 📊 Traffic Statistics")

    crossed_by_dir = metrics.get('crossed_by_direction', {d: 0 for d in DIRECTIONS})
    generated_by_dir = metrics.get('generated_by_direction', {d: 0 for d in DIRECTIONS})
    vehicle_counts = metrics.get('vehicle_counts', [0, 0, 0, 0])
    pedestrian_counts = metrics.get('pedestrian_counts', [0, 0, 0, 0])

    # Direction cards
    dir_config = [
        ('N', 'North', '#FF6B6B', 'direction-north'),
        ('S', 'South', '#4ECDC4', 'direction-south'),
        ('E', 'East', '#45B7D1', 'direction-east'),
        ('W', 'West', '#FFA07A', 'direction-west'),
    ]

    cols = st.columns(4)
    for idx, (dir_key, dir_name, color, css_class) in enumerate(dir_config):
        with cols[idx]:
            v_count = vehicle_counts[idx] if idx < len(vehicle_counts) else 0
            c_count = crossed_by_dir.get(dir_key, 0)
            g_count = generated_by_dir.get(dir_key, 0)
            p_count = pedestrian_counts[idx] if idx < len(pedestrian_counts) else 0

            st.markdown(f"""
            <div class="direction-card {css_class}">
                <div style="font-size: 1.1rem; font-weight: bold; color: {color};">{dir_name}</div>
                <div style="font-size: 0.9rem; margin: 0.25rem 0;">
                    🚗 <b>{v_count}</b> waiting
                </div>
                <div style="font-size: 0.9rem; margin: 0.25rem 0;">
                    ✅ <b>{c_count}</b> crossed / <b>{g_count}</b> generated
                </div>
                <div style="font-size: 0.85rem; color: #666;">
                    🚶 {p_count} pedestrians
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Generated", sum(generated_by_dir.values()))
    with col2:
        st.metric("Total Crossed", metrics.get('crossed_vehicles', 0))
    with col3:
        st.metric("Currently Waiting", metrics.get('total_vehicles', 0))
    with col4:
        st.metric("Avg Wait Time", f"{metrics.get('total_vehicle_waiting', 0) / max(metrics.get('crossed_vehicles', 1), 1):.1f} frames")


def render_rl_decision_monitor(metrics, agent):
    """Render live Q-Learning decision monitor."""
    st.markdown("## 🧠 RL Decision Monitor")

    # Current state information
    vehicle_counts = metrics.get('vehicle_counts', [0, 0, 0, 0])
    pedestrian_counts = metrics.get('pedestrian_counts', [0, 0, 0, 0])
    current_phase = metrics.get('phase', 'NS_GREEN')
    current_action = metrics.get('action', None)
    q_values = metrics.get('q_values', None)
    epsilon = metrics.get('epsilon', 0)

    # State display
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Current Traffic State")

        # Create a visual state representation
        state_data = {
            'Direction': ['North', 'South', 'East', 'West'],
            'Vehicles': vehicle_counts,
            'Pedestrians': pedestrian_counts,
        }
        df_state = pd.DataFrame(state_data)
        st.dataframe(df_state, use_container_width=True, hide_index=True)

        st.markdown(f"""
        **Current Signal Phase:** `{current_phase}`  
        **Selected Action:** `{current_action if current_action else 'None'}`  
        **Epsilon (Exploration):** `{epsilon:.3f}`  
        **Q-Table Size:** `{metrics.get('q_table_size', 0)}` entries
        """)

    with col2:
        st.markdown("### Q-Values for Available Actions")

        if q_values is not None:
            # Get best action
            best_idx = int(np.argmax(q_values))

            for i, (action_name, q_val) in enumerate(zip(ACTIONS, q_values)):
                is_best = (i == best_idx)
                css_class = "q-value-card q-value-best" if is_best else "q-value-card"
                indicator = "🏆 SELECTED" if is_best else ""
                bar_color = "#4caf50" if is_best else "#2196f3"
                bar_width = max(10, abs(q_val) * 50)  # Visual bar width

                st.markdown(f"""
                <div class="{css_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><b>{action_name}</b> {indicator}</span>
                        <span style="font-weight: bold; color: {'#4caf50' if is_best else '#2196f3'};">
                            {q_val:+.3f}
                        </span>
                    </div>
                    <div style="background: #e0e0e0; height: 8px; border-radius: 4px; margin-top: 0.5rem;">
                        <div style="background: {bar_color}; height: 100%; width: {min(100, bar_width)}%; border-radius: 4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No Q-values available yet. Start the simulation to see decisions.")

    # Decision explanation
    st.markdown("### 💡 Decision Explanation")

    if current_action and q_values is not None:
        best_idx = int(np.argmax(q_values))
        best_action = ACTIONS[best_idx]

        # Generate explanation based on actual state
        n_vehicles = vehicle_counts[0] if len(vehicle_counts) > 0 else 0
        s_vehicles = vehicle_counts[1] if len(vehicle_counts) > 1 else 0
        e_vehicles = vehicle_counts[2] if len(vehicle_counts) > 2 else 0
        w_vehicles = vehicle_counts[3] if len(vehicle_counts) > 3 else 0
        n_peds = pedestrian_counts[0] if len(pedestrian_counts) > 0 else 0
        s_peds = pedestrian_counts[1] if len(pedestrian_counts) > 1 else 0
        e_peds = pedestrian_counts[2] if len(pedestrian_counts) > 2 else 0
        w_peds = pedestrian_counts[3] if len(pedestrian_counts) > 3 else 0

        ns_total = n_vehicles + s_vehicles
        ew_total = e_vehicles + w_vehicles
        total_peds = n_peds + s_peds + e_peds + w_peds

        explanation_parts = []

        if current_action == 'NS_GREEN':
            explanation_parts.append(
                f"North-South has {ns_total} total vehicles (N:{n_vehicles}, S:{s_vehicles}) "
                f"vs East-West {ew_total} vehicles."
            )
        elif current_action == 'EW_GREEN':
            explanation_parts.append(
                f"East-West has {ew_total} total vehicles (E:{e_vehicles}, W:{w_vehicles}) "
                f"vs North-South {ns_total} vehicles."
            )
        elif current_action and current_action.startswith('PEDESTRIAN'):
            side = current_action[-1]
            side_name = DIRECTION_NAMES.get(side, side)
            side_peds = pedestrian_counts[['N', 'S', 'E', 'W'].index(side)] if side in ['N', 'S', 'E', 'W'] else 0
            explanation_parts.append(
                f"Pedestrian crossing activated for {side_name} side ({side_peds} waiting pedestrians)."
            )
        elif current_action == 'EXTEND_GREEN':
            explanation_parts.append(
                f"Extending current green phase ({current_phase}) to clear remaining queue."
            )

        if total_peds > 15:
            explanation_parts.append(
                f"High pedestrian demand detected ({total_peds} total waiting). "
                f"Pedestrian crossing will be prioritized soon."
            )

        if metrics.get('emergency_active'):
            explanation_parts.append(
                f"🚨 Emergency vehicle detected in {metrics.get('emergency_dir', 'unknown')} direction - "
                f"emergency override active."
            )

        explanation = " ".join(explanation_parts) if explanation_parts else \
            "Agent is evaluating traffic conditions and selecting the optimal signal phase."

        st.markdown(f"""
        <div class="decision-box">
            <b>Action:</b> {current_action}<br>
            <b>Q-Value:</b> {q_values[ACTIONS.index(current_action)] if current_action in ACTIONS else 0:+.3f}<br>
            <b>Explanation:</b> {explanation}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Start the simulation to see real-time decision explanations.")


def render_metrics_cards(metrics):
    """Render live metrics as clean cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 🚗 Vehicles")
        st.metric("Crossed", metrics.get('crossed_vehicles', 0))
        st.metric("In Queue", metrics.get('total_vehicles', 0))

    with col2:
        st.markdown("### 🚶 Pedestrians")
        st.metric("Crossed", metrics.get('crossed_pedestrians', 0))
        st.metric("Waiting", metrics.get('total_pedestrians', 0))

    with col3:
        st.markdown("### 🚦 Signals")
        st.metric("Current Phase", metrics.get('phase', 'NS_GREEN'))
        st.metric("Changes", metrics.get('signal_changes', 0))

    with col4:
        st.markdown("### ⏱️ Status")
        st.metric("Frame", metrics.get('frame', 0))
        st.metric("Speed", f"{metrics.get('_speed_multiplier', 1.0)}x")
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
        name='🚗 Vehicles Waiting',
        x=directions,
        y=vehicle_counts,
        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
        text=vehicle_counts,
        textposition='outside',
    ))

    fig.add_trace(go.Bar(
        name='🚶 Pedestrians Waiting',
        x=directions,
        y=pedestrian_counts,
        marker_color=['#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'],
        text=pedestrian_counts,
        textposition='outside',
    ))

    fig.update_layout(
        title="📊 Real-Time Traffic Distribution by Direction",
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


def render_crossed_chart(crossed_by_dir, chart_height):
    """Render vehicles crossed by direction chart."""
    directions = ['North', 'South', 'East', 'West']
    crossed_values = [crossed_by_dir.get(d, 0) for d in DIRECTIONS]

    fig = go.Figure(data=[
        go.Bar(
            x=directions,
            y=crossed_values,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
            text=crossed_values,
            textposition='outside',
        )
    ])

    fig.update_layout(
        title="✅ Vehicles Crossed by Direction",
        xaxis_title="Direction",
        yaxis_title="Vehicles Crossed",
        height=chart_height,
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
                'State': str(state)[:70] + '...' if len(str(state)) > 70 else str(state),
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
        if st.button("🔄 Run Fresh Comparison", type="primary", use_container_width=True, key="btn_compare"):
            st.session_state.comparison_data = None
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
        num_episodes = st.slider("Number of Episodes", 10, 500, 50, step=10, key="train_episodes")
        frames_per_episode = st.slider("Frames per Episode", 500, 5000, 2000, step=500, key="train_frames")

    with col2:
        st.markdown("### Training Actions")
        if st.button("🚀 Start Training", type="primary", use_container_width=True, key="btn_train"):
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

        if st.button("💾 Save Current Model", use_container_width=True, key="btn_save"):
            st.session_state.agent.save()
            st.success("✅ Model saved!")

        if st.button("📂 Load Saved Model", use_container_width=True, key="btn_load"):
            if st.session_state.agent.load():
                st.success("✅ Model loaded successfully!")
            else:
                st.warning("⚠️ No saved model found.")

        if st.button("🗑️ Reset Q-Table", use_container_width=True, key="btn_reset_q"):
            st.session_state.agent.reset_q_table()
            st.success("✅ Q-Table reset!")

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

    # Apply speed multiplier to simulation if running
    if st.session_state.simulation:
        st.session_state.simulation.set_speed_multiplier(params['speed_multiplier'])

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Simulation",
        "🚦 Traffic Statistics",
        "🧠 RL Decision Monitor",
        "📈 Performance Comparison",
        "⚙️ Agent Training",
    ])

    # ===================== TAB 1: LIVE SIMULATION =====================
    with tab1:
        st.markdown("## 📊 Live Traffic Simulation")

        # Control buttons
        col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
        with col_btn1:
            if st.button("▶️ Start Simulation", type="primary", use_container_width=True, key="tab1_start"):
                st.session_state.running = True
        with col_btn2:
            if st.button("⏸️ Pause Simulation", use_container_width=True, key="tab1_pause"):
                st.session_state.running = False
        with col_btn3:
            if st.button("🔄 Restart Simulation", use_container_width=True, key="tab1_restart"):
                if st.session_state.simulation:
                    st.session_state.simulation.reset(keep_q_table=True)
                st.session_state.running = False
                st.session_state.frame_count = 0
        with col_btn4:
            if st.button("📊 Reset Statistics", use_container_width=True, key="tab1_reset"):
                if st.session_state.simulation:
                    st.session_state.simulation.reset(keep_q_table=True)
                st.session_state.running = False
                st.session_state.frame_count = 0
        with col_btn5:
            st.session_state.auto_refresh = st.checkbox("🔁 Auto-Refresh", value=st.session_state.auto_refresh, key="tab1_autorefresh")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Initialize simulation
        if st.session_state.simulation is None:
            mode = 'train' if 'AI' in params['mode'] else 'fixed'
            st.session_state.simulation = TrafficSimulation(
                mode=mode,
                spawn_prob=params['spawn_rate'],
                emergency_prob=params['emergency_prob'],
                pedestrian_prob=params['pedestrian_prob'],
                load_model=True,
                speed_multiplier=params['speed_multiplier']
            )

        sim = st.session_state.simulation

        # Run simulation step
        if st.session_state.running or st.session_state.auto_refresh:
            if 'AI' in params['mode']:
                # Use per-direction multipliers for spawn rates
                # For simplicity, use base spawn rate with per-direction modifiers applied in engine
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
        result['_speed_multiplier'] = params['speed_multiplier']
        render_metrics_cards(result)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Traffic statistics (direction-wise)
        render_traffic_statistics(result)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Vehicle distribution chart
        st.markdown("### 📊 Real-Time Traffic Distribution")
        render_vehicle_chart(
            result.get('vehicle_counts', [0, 0, 0, 0]),
            result.get('pedestrian_counts', [0, 0, 0, 0]),
            params.get('chart_height', 400) if 'chart_height' in params else 400
        )

        # Crossed by direction chart
        st.markdown("### ✅ Vehicles Crossed by Direction")
        render_crossed_chart(
            result.get('crossed_by_direction', {d: 0 for d in DIRECTIONS}),
            params.get('chart_height', 400) if 'chart_height' in params else 400
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
            st.write(f"**Speed Multiplier:** {params['speed_multiplier']}x")
            st.write(f"**Q-Table Size:** {len(sim.agent.q_table)} entries")
            st.write(f"**Epsilon:** {sim.agent.epsilon:.3f}")
            st.write(f"**Vehicles Generated:** {sum(result.get('generated_by_direction', {}).values())}")
            st.write(f"**Vehicles Crossed:** {result.get('crossed_vehicles', 0)}")

    # ===================== TAB 2: TRAFFIC STATISTICS =====================
    with tab2:
        st.markdown("## 🚦 Detailed Traffic Statistics")

        result = sim._get_step_result() if st.session_state.simulation else {
            'vehicle_counts': [0, 0, 0, 0],
            'pedestrian_counts': [0, 0, 0, 0],
            'crossed_by_direction': {d: 0 for d in DIRECTIONS},
            'generated_by_direction': {d: 0 for d in DIRECTIONS},
            'crossed_vehicles': 0,
            'crossed_pedestrians': 0,
            'total_vehicles': 0,
            'total_pedestrians': 0,
            'total_vehicle_waiting': 0,
        }
        render_traffic_statistics(result)

    # ===================== TAB 3: RL DECISION MONITOR =====================
    with tab3:
        result = sim._get_step_result() if st.session_state.simulation else {
            'vehicle_counts': [0, 0, 0, 0],
            'pedestrian_counts': [0, 0, 0, 0],
            'phase': 'NS_GREEN',
            'action': None,
            'q_values': None,
            'epsilon': 0.4,
            'q_table_size': 0,
            'emergency_active': False,
        }
        render_rl_decision_monitor(result, st.session_state.agent)

    # ===================== TAB 4: PERFORMANCE COMPARISON =====================
    with tab4:
        chart_height = params.get('chart_height', 400) if 'chart_height' in params else 400
        render_comparison_section(chart_height)

    # ===================== TAB 5: AGENT TRAINING =====================
    with tab5:
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