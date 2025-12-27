# 🛡️ Spectra-Sovereignty

> **Real-Time Electromagnetic Signal AI Agent powered by Pathway + Google Gemini**

*"Transform the invisible electromagnetic spectrum into a real-time intelligent shield for our soldiers, our crops, and our lives."*

---

## 🌟 What Is This?

Spectra-Sovereignty is an **Agentic AI system** that monitors electromagnetic signals in real-time and makes intelligent decisions to counteract threats. Unlike traditional signal processing which just cleans noise, our agent **thinks and acts** on the signal stream.

### The Innovation

| Traditional AI | Spectra-Sovereignty |
|---------------|---------------------|
| Processes signals in batches | **Real-time streaming** via Pathway |
| Fixed pattern matching | **LLM-powered threat classification** |
| Manual response configuration | **Autonomous agent decisions** |
| Seconds of latency | **Millisecond response times** |

---

## 🎯 Use Cases

| Domain | Scenario | What It Does |
|--------|----------|--------------|
| 🛡️ **Defense** | Ghost Unit | Detects enemy radar, emits counter-frequency to appear as static noise |
| ❤️ **Health** | Contactless ICU | Monitors bio-electrical patterns, detects cardiac anomalies before they happen |
| 🌾 **Agriculture** | Wireless Pesticide | Detects pest swarm EM signatures, emits frequency to confuse their navigation |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────┐     ┌─────────────────┐
│                 │     │         PATHWAY ENGINE               │     │                 │
│  Signal         │────▶│  ┌─────────────┐  ┌──────────────┐  │────▶│   Dashboard     │
│  Simulator      │     │  │ Anomaly     │  │ Gemini Agent │  │     │   (Streamlit)   │
│  (EM Stream)    │     │  │ Detection   │──│ via LiteLLM  │  │     │                 │
└─────────────────┘     │  └─────────────┘  └──────────────┘  │     └─────────────────┘
                        └──────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd spectra_sovereignty
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Gemini API key
# GEMINI_API_KEY=your_key_here
```

### 3. Run the Dashboard (Standalone Demo)

```bash
streamlit run ui/dashboard.py
```

This runs the signal simulator + dashboard without needing the Pathway engine.

### 4. Run Full Pipeline (with Pathway Engine)

**Terminal 1 - Start the Signal Engine:**
```bash
python signal_engine.py --scenario defense
```

**Terminal 2 - Start the Simulator:**
```bash
python app.py --scenario defense --threat-prob 0.15
```

**Terminal 3 - Open Dashboard:**
```bash
streamlit run ui/dashboard.py
```

---

## 📁 Project Structure

```
spectra_sovereignty/
├── __init__.py              # Package definition
├── app.py                   # Main orchestrator
├── signal_simulator.py      # EM signal stream generator
├── signal_engine.py         # Pathway pipeline + Gemini agent
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
├── ui/
│   ├── __init__.py
│   └── dashboard.py         # Streamlit visualization
└── README.md                # This file
```

---

## 🤖 Agent Response Modes

| Mode | Description | When Activated |
|------|-------------|----------------|
| `MODE_ALPHA` | Noise-Cancellation | Blend into background EM |
| `MODE_BETA` | Frequency-Shift | Evade by shifting bands |
| `MODE_GAMMA` | Active-Counter | Emit jamming frequency |
| `MODE_DELTA` | Alert-Only | Monitor and log only |
| `MODE_NORMAL` | Standard | No threat detected |

---

## 🔧 Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | Required |
| `LLM_MODEL` | Model identifier | `gemini/gemini-2.5-flash-lite` |
| `HOST` | Engine bind address | `0.0.0.0` |
| `PORT` | Engine port | `8080` |
| `ACTIVE_SCENARIO` | Default scenario | `defense` |
| `THREAT_PROBABILITY` | Threat injection rate | `0.05` |

---

## 🧪 Testing

```bash
# Run signal simulator standalone (10 seconds)
python signal_simulator.py --scenario defense --duration 10

# Run with high threat probability for testing
python app.py --simulator-only --scenario health --threat-prob 0.3 --duration 20
```

---

## 🏆 Why This Wins

1. **Post-Transformer Innovation**: "Standard Transformers can't handle high-frequency signal streams—we use **Pathway** to create a zero-latency Agent that lives *inside* the signal."

2. **Out-of-the-Box Thinking**: Most teams build chatbots. We're building an **Agent that controls the invisible physical world.**

3. **Multi-Domain Impact**: One architecture, three critical sectors (Defense, Health, Agriculture).

---

## 📜 License

MIT License - Built for IIT Madras Hackathon

---

## 🙏 Credits

- **Pathway** - Real-time streaming engine
- **Google Gemini** - LLM intelligence via LiteLLM
- **Streamlit** - Dashboard visualization
