"""
Spectra-Sovereignty Real-Time Dashboard

A Streamlit-based visualization dashboard for monitoring the signal
processing pipeline and agent decisions in real-time.

Features:
- Live signal charts (frequency, amplitude, noise)
- Threat detection indicators
- Agent decision panel with mode status
- Split view: Raw vs Processed signal
"""

import os
import sys
import json
import time
import requests
from collections import deque
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_simulator import SignalSimulator, Scenario

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Spectra-Sovereignty | Signal Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #0f0f23 100%);
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }

    /* Status cards */
    .status-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }

    .threat-indicator-safe {
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        color: #000;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
    }

    .threat-indicator-danger {
        background: linear-gradient(135deg, #ff3366, #ff0044);
        color: #fff;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        animation: pulse 1s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* Mode badges */
    .mode-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .mode-alpha { background: #00d4ff; color: #000; }
    .mode-beta { background: #7b2ff7; color: #fff; }
    .mode-gamma { background: #ff6b35; color: #fff; }
    .mode-delta { background: #ffd700; color: #000; }
    .mode-normal { background: #00ff88; color: #000; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state."""
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = deque(maxlen=100)
    if 'threat_history' not in st.session_state:
        st.session_state.threat_history = deque(maxlen=50)
    if 'agent_decisions' not in st.session_state:
        st.session_state.agent_decisions = deque(maxlen=20)
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "MODE_NORMAL"
    if 'threat_count' not in st.session_state:
        st.session_state.threat_count = 0
    if 'total_readings' not in st.session_state:
        st.session_state.total_readings = 0


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

def create_signal_chart(history: list) -> go.Figure:
    """Create real-time signal visualization chart."""
    if not history:
        # Empty chart
        fig = go.Figure()
        fig.add_annotation(
            text="Waiting for signal data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#666")
        )
    else:
        df = pd.DataFrame(list(history))

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Frequency (Hz)", "Amplitude & Noise"),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5]
        )

        # Frequency trace
        colors = ['#ff3366' if t > 0.5 else '#00d4ff' for t in df['threat_level']]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df['frequency'],
                mode='lines+markers',
                name='Frequency',
                line=dict(color='#00d4ff', width=2),
                marker=dict(size=6, color=colors)
            ),
            row=1, col=1
        )

        # Amplitude trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df['amplitude'],
                mode='lines',
                name='Amplitude',
                line=dict(color='#7b2ff7', width=2)
            ),
            row=2, col=1
        )

        # Noise trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=df['noise'],
                mode='lines',
                name='Noise',
                line=dict(color='#ffd700', width=1, dash='dot'),
                fill='tozeroy',
                fillcolor='rgba(255,215,0,0.1)'
            ),
            row=2, col=1
        )

    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(20,20,40,0.8)',
        font=dict(color='#ffffff'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=60, b=50)
    )

    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')

    return fig


def create_threat_gauge(anomaly_score: float) -> go.Figure:
    """Create a gauge chart for threat level."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=anomaly_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Threat Level", 'font': {'size': 16, 'color': '#fff'}},
        number={'suffix': '%', 'font': {'size': 32, 'color': '#fff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#fff'},
            'bar': {'color': '#00d4ff'},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 2,
            'bordercolor': 'rgba(255,255,255,0.3)',
            'steps': [
                {'range': [0, 40], 'color': 'rgba(0,255,136,0.3)'},
                {'range': [40, 70], 'color': 'rgba(255,215,0,0.3)'},
                {'range': [70, 100], 'color': 'rgba(255,51,102,0.3)'}
            ],
            'threshold': {
                'line': {'color': '#ff3366', 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))

    fig.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#fff'),
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig


def get_mode_badge_html(mode: str) -> str:
    """Generate HTML badge for agent mode."""
    mode_class = mode.lower().replace("mode_", "mode-")
    mode_display = mode.replace("_", " ")
    return f'<span class="mode-badge {mode_class}">{mode_display}</span>'


# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def main():
    """Main dashboard entry point."""
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">🛡️ SPECTRA-SHIELD</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Real-Time Electromagnetic Signal Defense System</p>', unsafe_allow_html=True)

    # Spectra-Shield Status Banner
    st.markdown('''
    <div style="
        background: linear-gradient(90deg, rgba(0,212,255,0.2), rgba(123,47,247,0.2));
        border: 2px solid #00d4ff;
        border-radius: 12px;
        padding: 1rem 2rem;
        margin: 1rem 0 2rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div>
            <span style="color: #00d4ff; font-size: 1.2rem; font-weight: 700;">⚡ SPECTRA-SHIELD STATUS</span>
        </div>
        <div style="
            background: linear-gradient(135deg, #00ff88, #00cc6a);
            color: #000;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            font-weight: 800;
            font-size: 1.1rem;
            box-shadow: 0 0 20px rgba(0,255,136,0.5);
        ">✓ ACTIVE</div>
    </div>
    ''', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Control Panel")

        scenario = st.selectbox(
            "Scenario",
            ["defense", "health", "agriculture"],
            format_func=lambda x: {
                "defense": "🛡️ Defense (Ghost Unit)",
                "health": "❤️ Health (Contactless ICU)",
                "agriculture": "🌾 Agriculture (Wireless Pesticide)"
            }.get(x, x)
        )

        threat_prob = st.slider(
            "Threat Probability",
            min_value=0.0,
            max_value=0.5,
            value=0.1,
            step=0.05,
            help="How often threats are injected into the signal stream"
        )

        sample_rate = st.slider(
            "Sample Rate (ms)",
            min_value=100,
            max_value=1000,
            value=200,
            step=100
        )

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("▶️ Start", use_container_width=True, type="primary")
        with col2:
            stop_btn = st.button("⏹️ Stop", use_container_width=True)

        if start_btn and not st.session_state.is_running:
            st.session_state.simulator = SignalSimulator(
                scenario=Scenario(scenario),
                threat_probability=threat_prob,
                sample_interval_ms=sample_rate
            )
            st.session_state.is_running = True
            st.session_state.signal_history.clear()
            st.rerun()

        if stop_btn:
            st.session_state.is_running = False
            st.rerun()

        st.divider()

        # Stats
        st.subheader("📊 Statistics")
        st.metric("Total Readings", st.session_state.total_readings)
        st.metric("Threats Detected", st.session_state.threat_count)

        if st.session_state.total_readings > 0:
            threat_rate = (st.session_state.threat_count / st.session_state.total_readings) * 100
            st.metric("Threat Rate", f"{threat_rate:.1f}%")

    # Main content area
    col_main, col_status = st.columns([3, 1])

    with col_main:
        # Signal chart
        st.subheader("📡 Signal Monitor")
        chart_placeholder = st.empty()

        # Agent Defense Logs
        st.subheader("🛡️ Agent Defense Logs")
        decisions_placeholder = st.empty()

    with col_status:
        # Current status
        st.subheader("🎯 Status")
        status_placeholder = st.empty()

        # Threat gauge
        gauge_placeholder = st.empty()

        # Current mode
        mode_placeholder = st.empty()

    # Main simulation loop
    if st.session_state.is_running and st.session_state.simulator:
        for reading in st.session_state.simulator.stream():
            if not st.session_state.is_running:
                break

            # Store reading
            st.session_state.signal_history.append(reading.to_dict())
            st.session_state.total_readings += 1

            # Check for threat
            is_threat = reading.threat_level > 0.5
            if is_threat:
                st.session_state.threat_count += 1
                st.session_state.current_mode = "MODE_GAMMA" if reading.threat_level > 0.8 else "MODE_BETA"

                # Log decision
                decision = {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "mode": st.session_state.current_mode,
                    "threat_type": reading.threat_type,
                    "confidence": reading.threat_level
                }
                st.session_state.agent_decisions.append(decision)
            else:
                st.session_state.current_mode = "MODE_NORMAL"

            # Update visualizations
            chart_placeholder.plotly_chart(
                create_signal_chart(st.session_state.signal_history),
                use_container_width=True,
                key=f"chart_{st.session_state.total_readings}"
            )

            gauge_placeholder.plotly_chart(
                create_threat_gauge(reading.threat_level),
                use_container_width=True,
                key=f"gauge_{st.session_state.total_readings}"
            )

            # Update status
            status_class = "threat-indicator-danger" if is_threat else "threat-indicator-safe"
            status_text = "⚠️ THREAT DETECTED" if is_threat else "✅ ALL CLEAR"
            status_placeholder.markdown(
                f'<div class="{status_class}">{status_text}</div>',
                unsafe_allow_html=True
            )

            # Update mode
            mode_placeholder.markdown(
                f'<p style="text-align: center; margin-top: 1rem;">Current Mode:<br/>'
                f'{get_mode_badge_html(st.session_state.current_mode)}</p>',
                unsafe_allow_html=True
            )

            # Update decisions log
            if st.session_state.agent_decisions:
                decisions_df = pd.DataFrame(list(st.session_state.agent_decisions))
                decisions_placeholder.dataframe(
                    decisions_df.iloc[::-1],  # Reverse order (newest first)
                    use_container_width=True,
                    hide_index=True
                )

            time.sleep(0.1)  # Small delay for UI responsiveness
    else:
        # Static display when not running
        chart_placeholder.plotly_chart(
            create_signal_chart(list(st.session_state.signal_history)),
            use_container_width=True
        )
        gauge_placeholder.plotly_chart(
            create_threat_gauge(0.0),
            use_container_width=True
        )
        status_placeholder.markdown(
            '<div class="threat-indicator-safe">⏸️ MONITORING PAUSED</div>',
            unsafe_allow_html=True
        )
        mode_placeholder.markdown(
            f'<p style="text-align: center; margin-top: 1rem;">Current Mode:<br/>'
            f'{get_mode_badge_html("MODE_NORMAL")}</p>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
