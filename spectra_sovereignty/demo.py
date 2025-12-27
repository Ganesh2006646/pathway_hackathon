"""
Spectra-Shield Demo Mode

A standalone demo that works WITHOUT Pathway (for Windows/demo purposes).
Uses direct LiteLLM calls to simulate the agent behavior.

For production on Linux, use signal_engine.py with full Pathway pipeline.
"""

import os
import json
import time
from typing import Optional
from datetime import datetime

import dotenv

# Load environment
dotenv.load_dotenv()

# Try to import litellm
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    print("⚠️ LiteLLM not available. Running in MOCK mode.")


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are Spectra-Sentinel, an AI agent monitoring electromagnetic signals for threats.

You analyze signal patterns and decide on the appropriate defensive response.

AVAILABLE RESPONSE MODES:
- MODE_ALPHA (Noise-Cancellation): Blend into background EM noise
- MODE_BETA (Frequency-Shift): Evade detection by shifting frequency bands
- MODE_GAMMA (Active-Counter): Emit jamming/counter-frequency
- MODE_DELTA (Alert-Only): Monitor and log, no active response
- MODE_NORMAL: No threat, continue standard monitoring

Always respond with a JSON object containing:
{
    "mode": "MODE_X",
    "confidence": 0.0-1.0,
    "rationale": "Brief explanation",
    "action": "Specific action to take"
}"""


def calculate_anomaly_score(
    frequency: float,
    amplitude: float,
    baseline_freq: float = 300.0,
    baseline_amp: float = 45.0
) -> float:
    """Calculate anomaly score based on deviation from baseline."""
    freq_deviation = abs(frequency - baseline_freq) / baseline_freq
    amp_deviation = abs(amplitude - baseline_amp) / baseline_amp
    raw_score = (freq_deviation * 0.7) + (amp_deviation * 0.3)
    return min(1.0, raw_score)


def get_agent_decision(
    frequency: float,
    amplitude: float,
    noise: float,
    anomaly_score: float,
    scenario: str = "defense",
    use_llm: bool = True
) -> dict:
    """
    Get agent decision for a signal reading.

    Uses Gemini LLM if available and use_llm=True, otherwise returns mock response.
    """
    # If no threat detected, return normal mode
    if anomaly_score < 0.4:
        return {
            "mode": "MODE_NORMAL",
            "confidence": 1.0 - anomaly_score,
            "rationale": "Signal within normal parameters",
            "action": "Continue standard monitoring"
        }

    # Build prompt for LLM
    user_prompt = f"""CURRENT SIGNAL READING:
- Frequency: {frequency:.2f} Hz
- Amplitude: {amplitude:.2f}
- Noise Floor: {noise:.2f}
- Anomaly Score: {anomaly_score:.3f}
- Scenario: {scenario.upper()}

Based on this signal pattern, what response mode should be activated?
Respond with JSON only."""

    # Try LLM call
    if use_llm and LITELLM_AVAILABLE:
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key:
                response = completion(
                    model="gemini/gemini-2.5-flash-lite",
                    messages=[
                        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    api_key=api_key,
                    max_tokens=300,
                    temperature=0.1
                )

                # Parse response
                content = response.choices[0].message.content
                # Try to extract JSON from response
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]

                return json.loads(content.strip())
        except Exception as e:
            print(f"⚠️ LLM call failed: {e}")

    # Mock response based on anomaly score
    if anomaly_score > 0.8:
        return {
            "mode": "MODE_GAMMA",
            "confidence": anomaly_score,
            "rationale": f"Critical threat detected - anomaly score {anomaly_score:.2f}",
            "action": "Activating active counter-measures"
        }
    elif anomaly_score > 0.6:
        return {
            "mode": "MODE_BETA",
            "confidence": anomaly_score,
            "rationale": f"High threat level - anomaly score {anomaly_score:.2f}",
            "action": "Shifting to alternate frequency band"
        }
    else:
        return {
            "mode": "MODE_ALPHA",
            "confidence": anomaly_score,
            "rationale": f"Moderate threat - anomaly score {anomaly_score:.2f}",
            "action": "Blending into background noise"
        }


def run_demo(
    scenario: str = "defense",
    duration: int = 30,
    use_llm: bool = True
):
    """
    Run the Spectra-Shield demo.

    Generates simulated signals and shows agent decisions.
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from signal_simulator import SignalSimulator, Scenario

    print("\n" + "=" * 60)
    print("🛡️  SPECTRA-SHIELD DEMO")
    print("   Real-Time Electromagnetic Signal Defense System")
    print("=" * 60)

    llm_status = "🟢 GEMINI ACTIVE" if (use_llm and LITELLM_AVAILABLE and os.environ.get("GEMINI_API_KEY")) else "🟡 MOCK MODE"
    print(f"\n⚡ SPECTRA-SHIELD STATUS: ✓ ACTIVE")
    print(f"🤖 LLM Agent: {llm_status}")
    print(f"📡 Scenario: {scenario.upper()}")
    print(f"⏱️  Duration: {duration}s")
    print("-" * 60)

    # Baselines per scenario
    baselines = {
        "defense": {"freq": 300.0, "amp": 45.0},
        "health": {"freq": 1.0, "amp": 70.0},
        "agriculture": {"freq": 200.0, "amp": 30.0}
    }
    baseline = baselines.get(scenario, baselines["defense"])

    # Create simulator
    simulator = SignalSimulator(
        scenario=Scenario(scenario),
        threat_probability=0.15,
        sample_interval_ms=500  # Slower for visibility
    )

    # Defense log
    defense_log = []

    # Run simulation
    reading_count = 0
    for reading in simulator.stream(duration_sec=duration):
        reading_count += 1

        # Calculate anomaly
        anomaly = calculate_anomaly_score(
            reading.frequency,
            reading.amplitude,
            baseline["freq"],
            baseline["amp"]
        )

        # Get agent decision
        decision = get_agent_decision(
            reading.frequency,
            reading.amplitude,
            reading.noise,
            anomaly,
            scenario,
            use_llm
        )

        # Display
        is_threat = anomaly > 0.4
        status = "🔴 THREAT" if is_threat else "🟢 NORMAL"

        print(f"\n[{reading_count:03d}] {status}")
        print(f"     Freq: {reading.frequency:7.2f} Hz | Amp: {reading.amplitude:5.1f} | Anomaly: {anomaly:.2f}")
        print(f"     Agent: {decision['mode']} (confidence: {decision['confidence']:.2f})")
        print(f"     Action: {decision['action']}")

        # Log threat detections
        if is_threat:
            log_entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "reading": reading_count,
                "mode": decision["mode"],
                "confidence": decision["confidence"],
                "action": decision["action"]
            }
            defense_log.append(log_entry)

    # Summary
    print("\n" + "=" * 60)
    print("📊 DEMO COMPLETE")
    print(f"   Total Readings: {reading_count}")
    print(f"   Threats Handled: {len(defense_log)}")
    print("=" * 60)

    if defense_log:
        print("\n🛡️ AGENT DEFENSE LOG:")
        for entry in defense_log[-10:]:  # Last 10
            print(f"   [{entry['time']}] {entry['mode']} - {entry['action'][:50]}")

    return defense_log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spectra-Shield Demo")
    parser.add_argument("--scenario", choices=["defense", "health", "agriculture"],
                        default="defense", help="Simulation scenario")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM calls (mock mode)")

    args = parser.parse_args()

    run_demo(
        scenario=args.scenario,
        duration=args.duration,
        use_llm=not args.no_llm
    )
