"""
Spectra-Shield Dashboard with REAL AI Agent (Gemini)

This version integrates the actual Gemini LLM to make intelligent
decisions about threats, not just simple if-else rules.
"""

import os
import sys
import time
import json
from collections import deque
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import dotenv

# Load environment
dotenv.load_dotenv()

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signal_simulator import SignalSimulator, Scenario

# Try to import litellm for Gemini
try:
    from litellm import completion
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

AGENT_PROMPT = """You are Spectra-Sentinel, an AI agent monitoring electromagnetic signals.

RESPONSE MODES:
- MODE_ALPHA: Noise-Cancellation (blend into background)
- MODE_BETA: Frequency-Shift (evade detection)
- MODE_GAMMA: Active-Counter (emit jamming signal)
- MODE_DELTA: Alert-Only (monitor, no action)
- MODE_NORMAL: No threat detected

Respond with JSON only:
{"mode": "MODE_X", "confidence": 0.0-1.0, "rationale": "Why this mode", "action": "What to do"}"""


def get_ai_decision(freq, amp, noise, anomaly, scenario):
    """Get decision from Gemini AI agent."""
    api_key = os.environ.get("GEMINI_API_KEY")

    # If no threat, skip LLM call
    if anomaly < 0.35:
        return {
            "mode": "MODE_NORMAL",
            "confidence": round(1.0 - anomaly, 2),
            "rationale": "Signal within normal parameters - no anomaly detected",
            "action": "Continue standard monitoring"
        }, False

    # Try Gemini LLM
    if GEMINI_AVAILABLE and api_key and len(api_key) > 10:
        try:
            user_msg = f"""Signal: Freq={freq:.1f}Hz, Amp={amp:.1f}, Noise={noise:.1f}, Anomaly={anomaly:.2f}
Scenario: {scenario.upper()}
What defensive mode should activate? JSON only."""

            response = completion(
                model="gemini/gemini-2.5-flash-lite",
                messages=[
                    {"role": "system", "content": AGENT_PROMPT},
                    {"role": "user", "content": user_msg}
                ],
                api_key=api_key,
                max_tokens=200,
                temperature=0.1
            )

            content = response.choices[0].message.content
            # Parse JSON from response
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content.strip())
            return result, True  # True = used LLM

        except Exception as e:
            st.session_state.llm_error = str(e)[:100]

    # Fallback: Rule-based
    if anomaly > 0.8:
        mode, action = "MODE_GAMMA", "Deploying active counter-measures"
    elif anomaly > 0.6:
        mode, action = "MODE_BETA", "Initiating frequency shift"
    else:
        mode, action = "MODE_ALPHA", "Blending into background noise"

    return {
        "mode": mode,
        "confidence": round(anomaly, 2),
        "rationale": f"[FALLBACK] High anomaly score: {anomaly:.2f}",
        "action": action
    }, False


def calc_anomaly(freq, amp, scenario):
    """Calculate anomaly score based on scenario baseline."""
    baselines = {
        "defense": (300.0, 45.0),
        "health": (1.0, 70.0),
        "agriculture": (200.0, 30.0)
    }
    base_f, base_a = baselines.get(scenario, (300.0, 45.0))
    f_dev = abs(freq - base_f) / max(base_f, 1)
    a_dev = abs(amp - base_a) / max(base_a, 1)
    return min(1.0, f_dev * 0.7 + a_dev * 0.3)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Spectra-Shield | AI Agent",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%); }
    .main-header {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .agent-box {
        background: rgba(123,47,247,0.15);
        border: 3px solid #7b2ff7;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1.1rem;
    }
    .rationale-text {
        background: rgba(0,212,255,0.15);
        border-left: 5px solid #00d4ff;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-style: italic;
        font-size: 1.1rem;
        border-radius: 0 12px 12px 0;
    }
    .status-active {
        background: linear-gradient(135deg, #00ff88, #00cc6a);
        color: #000;
        padding: 0.6rem 1.5rem;
        border-radius: 25px;
        font-weight: 800;
        font-size: 1.2rem;
    }
    .status-inactive { background: #444; color: #fff; padding: 0.6rem 1.5rem; border-radius: 25px; font-size: 1.2rem; }
    .llm-active { color: #00ff88; font-weight: bold; font-size: 1.1rem; }
    .llm-fallback { color: #ffd700; font-weight: bold; font-size: 1.1rem; }
    .threat-badge {
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
    }
    .threat-safe { background: linear-gradient(135deg, #00ff88, #00cc6a); color: #000; }
    .threat-danger { background: linear-gradient(135deg, #ff3366, #ff0044); color: #fff; animation: pulse 1s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
    .mode-display {
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        font-weight: 800;
        font-size: 1.4rem;
    }
    .stMetric { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# STATE
# =============================================================================

def init_state():
    defaults = {
        'history': deque(maxlen=50),
        'logs': deque(maxlen=20),
        'running': False,
        'mode': "MODE_NORMAL",
        'threats': 0,
        'readings': 0,
        'llm_calls': 0,
        'llm_error': None,
        'last_rationale': "Waiting for signal...",
        'scenario': 'defense',
        'threat_prob': 15,
        'sample_rate': 400
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset():
    st.session_state.history.clear()
    st.session_state.logs.clear()
    st.session_state.threats = 0
    st.session_state.readings = 0
    st.session_state.llm_calls = 0
    st.session_state.mode = "MODE_NORMAL"
    st.session_state.last_rationale = "Reset - waiting for signal..."


# =============================================================================
# CHARTS
# =============================================================================

def make_chart(history):
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Frequency", "Amplitude"), vertical_spacing=0.15)
    if history:
        df = pd.DataFrame(list(history))
        x = list(range(len(df)))
        colors = ['#ff3366' if t > 0.5 else '#00d4ff' for t in df['threat_level']]
        fig.add_trace(go.Scatter(x=x, y=df['frequency'], mode='lines+markers', line=dict(color='#00d4ff', width=3), marker=dict(color=colors, size=8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df['amplitude'], mode='lines', line=dict(color='#7b2ff7', width=3)), row=2, col=1)
    else:
        fig.add_annotation(text="Click START to begin...", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=20, color="#888"))
    fig.update_layout(height=420, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(20,20,40,0.8)', font=dict(color='#fff', size=14), showlegend=False, margin=dict(l=50, r=30, t=50, b=40))
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    return fig


def make_gauge(val):
    """Create threat level gauge."""
    color = '#00ff88' if val < 0.4 else '#ffd700' if val < 0.7 else '#ff3366'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val*100,
        domain={'x': [0,1], 'y': [0,1]},
        title={'text': "Threat Level", 'font': {'size': 16, 'color': '#fff'}},
        number={'suffix': '%', 'font': {'size': 36, 'color': '#fff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 12}},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': 'rgba(0,0,0,0)',
            'steps': [
                {'range': [0,40], 'color': 'rgba(0,255,136,0.2)'},
                {'range': [40,70], 'color': 'rgba(255,215,0,0.2)'},
                {'range': [70,100], 'color': 'rgba(255,51,102,0.2)'}
            ]
        }
    ))
    fig.update_layout(height=220, paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=20, r=20, t=50, b=20))
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    init_state()

    # Header
    st.markdown('<h1 class="main-header">🛡️ SPECTRA-SHIELD</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#888;">AI-Powered Electromagnetic Signal Defense</p>', unsafe_allow_html=True)

    # Check Gemini status
    api_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_ready = GEMINI_AVAILABLE and len(api_key) > 10

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")

        # LLM Status
        if gemini_ready:
            st.markdown('<span class="llm-active">🤖 GEMINI AI: CONNECTED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="llm-fallback">⚠️ GEMINI: NOT CONFIGURED</span>', unsafe_allow_html=True)
            st.caption("Add GEMINI_API_KEY to .env for AI responses")

        st.divider()

        scenario = st.selectbox("Scenario", ["defense", "health", "agriculture"],
            format_func=lambda x: {"defense": "🛡️ Defense", "health": "❤️ Health", "agriculture": "🌾 Agriculture"}[x])

        threat_prob = st.slider("Threat %", 5, 50, 15, 5, format="%d%%")
        sample_rate = st.slider("Speed (ms)", 300, 1000, 400, 100)

        st.divider()

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶️", use_container_width=True, help="Start"):
                reset()
                st.session_state.running = True
                st.session_state.scenario = scenario
                st.session_state.threat_prob = threat_prob
                st.session_state.sample_rate = sample_rate
                st.rerun()
        with c2:
            if st.button("⏹️", use_container_width=True, help="Stop"):
                st.session_state.running = False
                st.rerun()
        with c3:
            if st.button("🔄", use_container_width=True, help="Reset"):
                reset()
                st.session_state.running = False
                st.rerun()

        st.divider()
        st.metric("Readings", st.session_state.readings)
        st.metric("Threats", st.session_state.threats)
        st.metric("AI Calls", st.session_state.llm_calls)

    # Status banner
    status = "ACTIVE" if st.session_state.running else "STANDBY"
    status_cls = "status-active" if st.session_state.running else "status-inactive"
    st.markdown(f'''<div style="display:flex;justify-content:space-between;align-items:center;padding:0.8rem 1.5rem;background:rgba(0,212,255,0.1);border:2px solid {'#00d4ff' if st.session_state.running else '#444'};border-radius:12px;margin:1rem 0;">
        <span style="color:#00d4ff;font-weight:700;">⚡ SPECTRA-SHIELD</span>
        <span class="{status_cls}">{status}</span>
    </div>''', unsafe_allow_html=True)

    # Main layout
    col1, col2 = st.columns([2, 1])

    with col1:
        chart_ph = st.empty()

        # AI Agent Decision Box
        st.markdown("### 🤖 AI Agent Decision")
        agent_ph = st.empty()

        st.markdown("### 🛡️ Defense Logs")
        logs_ph = st.empty()

    with col2:
        st.markdown("### 🎯 Threat Level")
        threat_ph = st.empty()
        gauge_ph = st.empty()

        st.markdown("### 📊 Current Mode")
        mode_ph = st.empty()

    # Simulation loop
    if st.session_state.running:
        simulator = SignalSimulator(
            scenario=Scenario(st.session_state.scenario),
            threat_probability=st.session_state.threat_prob / 100.0,
            sample_interval_ms=st.session_state.sample_rate
        )

        for reading in simulator.stream():
            if not st.session_state.running:
                break

            st.session_state.history.append(reading.to_dict())
            st.session_state.readings += 1

            # Calculate anomaly
            anomaly = calc_anomaly(reading.frequency, reading.amplitude, st.session_state.scenario)

            # Get AI decision
            decision, used_llm = get_ai_decision(
                reading.frequency, reading.amplitude, reading.noise,
                anomaly, st.session_state.scenario
            )

            if used_llm:
                st.session_state.llm_calls += 1

            st.session_state.mode = decision["mode"]
            st.session_state.last_rationale = decision.get("rationale", "No rationale provided")

            is_threat = anomaly > 0.35
            if is_threat:
                st.session_state.threats += 1
                st.session_state.logs.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "Mode": decision["mode"],
                    "AI": "✓" if used_llm else "—",
                    "Action": decision.get("action", "N/A")[:40]
                })

            # Update displays
            chart_ph.plotly_chart(make_chart(st.session_state.history), use_container_width=True, key=f"c{st.session_state.readings}")

            # Agent box with rationale
            llm_badge = '<span class="llm-active">🤖 GEMINI</span>' if used_llm else '<span class="llm-fallback">📋 FALLBACK</span>'
            agent_ph.markdown(f'''<div class="agent-box">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:1.3rem;font-weight:700;">{decision["mode"]}</span>
                    {llm_badge}
                </div>
                <div style="margin-top:0.5rem;color:#aaa;">Confidence: {decision.get("confidence", 0):.0%}</div>
                <div class="rationale-text">💭 {decision.get("rationale", "...")}</div>
                <div style="color:#00d4ff;font-weight:600;">▶ {decision.get("action", "...")}</div>
            </div>''', unsafe_allow_html=True)

            threat_cls = "threat-danger" if is_threat else "threat-safe"
            threat_txt = "⚠️ THREAT" if is_threat else "✅ CLEAR"
            threat_ph.markdown(f'<div class="threat-badge {threat_cls}">{threat_txt}</div>', unsafe_allow_html=True)

            gauge_ph.plotly_chart(make_gauge(anomaly), use_container_width=True, key=f"g{st.session_state.readings}")

            mode_colors = {"MODE_NORMAL": "#00ff88", "MODE_ALPHA": "#00d4ff", "MODE_BETA": "#7b2ff7", "MODE_GAMMA": "#ff6b35", "MODE_DELTA": "#ffd700"}
            mode_ph.markdown(f'<div style="text-align:center;padding:1rem;background:{mode_colors.get(st.session_state.mode, "#666")};border-radius:12px;font-weight:700;font-size:1.1rem;">{st.session_state.mode}</div>', unsafe_allow_html=True)

            if st.session_state.logs:
                logs_ph.dataframe(pd.DataFrame(list(st.session_state.logs))[::-1], use_container_width=True, hide_index=True)

            time.sleep(0.05)
    else:
        # Static view
        chart_ph.plotly_chart(make_chart(st.session_state.history), use_container_width=True)
        agent_ph.markdown(f'''<div class="agent-box">
            <div style="font-size:1.2rem;font-weight:700;">{st.session_state.mode}</div>
            <div class="rationale-text">💭 {st.session_state.last_rationale}</div>
        </div>''', unsafe_allow_html=True)
        threat_ph.markdown('<div class="threat-badge threat-safe">⏸️ PAUSED</div>', unsafe_allow_html=True)
        gauge_ph.plotly_chart(make_gauge(0), use_container_width=True)
        mode_ph.markdown(f'<div style="text-align:center;padding:1rem;background:#00ff88;border-radius:12px;font-weight:700;">MODE_NORMAL</div>', unsafe_allow_html=True)
        if st.session_state.logs:
            logs_ph.dataframe(pd.DataFrame(list(st.session_state.logs))[::-1], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
