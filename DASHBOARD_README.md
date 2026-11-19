# Adaptive Traffic Signal Control Dashboard

## Quick Start

### Option 1: Double-click (Windows)
Double-click **`run_dashboard.bat`** in the project folder.

### Option 2: Command Line
Open PowerShell or Command Prompt in `D:\Desktop\demo_omni` and run:

```powershell
python -m streamlit run dashboard/app.py
```

### Option 3: Install dependencies first (if needed)
```powershell
pip install -r requirements.txt
python -m streamlit run dashboard/app.py
```

## Access the Dashboard

Once running, open your browser to:
**http://localhost:8501**

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `streamlit` not recognized | Use `python -m streamlit` instead of `streamlit` |
| Email/prompt on first run | Config already set to headless mode |
| Port 8501 busy | Change port in `.streamlit/config.toml` |
| Dashboard not loading | Make sure you're in `D:\Desktop\demo_omni` directory |

## Dashboard Tabs

1. **📊 Live Simulation** - Real-time traffic metrics, vehicle distribution, emergency alerts
2. **🧠 Q-Learning Monitor** - Agent statistics, Q-table, reward function
3. **📈 Performance Comparison** - AI vs Fixed-time traffic control comparison
4. **⚙️ Agent Training** - Train the Q-Learning agent, save/load models

## Project Files

- `dashboard/app.py` - Main Streamlit dashboard
- `traffic_sim/` - Traffic simulation engine
- `rl_agent/` - Q-Learning reinforcement learning agent
- `models/q_table.npy` - Saved trained model
- `main.py` - Command-line entry point
- `requirements.txt` - Python dependencies