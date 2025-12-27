"""
Spectra-Sovereignty: Main Application Entry Point

This is the main orchestrator that runs both:
1. The Signal Engine (Pathway pipeline with Gemini LLM)
2. The Signal Simulator (generating test data)

For production, the simulator would be replaced with real SDR input.
"""

import os
import sys
import json
import time
import argparse
import threading
import requests
from typing import Optional

import dotenv

# Load environment variables
dotenv.load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_simulator import SignalSimulator, Scenario


def send_signal_to_engine(
    signal_data: dict,
    host: str = "localhost",
    port: int = 8080
) -> Optional[dict]:
    """Send a signal reading to the Pathway engine via HTTP."""
    try:
        url = f"http://{host}:{port}/"
        response = requests.post(
            url,
            json=signal_data,
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"⚠️ Engine returned status {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to engine at {host}:{port}")
        return None
    except Exception as e:
        print(f"❌ Error sending signal: {e}")
        return None


def run_simulator_thread(
    scenario: str,
    threat_prob: float,
    sample_rate_ms: int,
    engine_host: str,
    engine_port: int,
    duration: Optional[int] = None
):
    """Run the signal simulator in a separate thread."""
    simulator = SignalSimulator(
        scenario=Scenario(scenario),
        threat_probability=threat_prob,
        sample_interval_ms=sample_rate_ms
    )

    print(f"\n📡 Starting Signal Simulator")
    print(f"   Scenario: {scenario.upper()}")
    print(f"   Threat Probability: {threat_prob * 100:.1f}%")
    print(f"   Sample Rate: {sample_rate_ms}ms")
    print("-" * 50)

    start_time = time.time()
    reading_count = 0
    threat_count = 0

    for reading in simulator.stream(duration_sec=duration):
        reading_count += 1

        # Send to engine
        signal_data = reading.to_dict()
        response = send_signal_to_engine(
            signal_data,
            host=engine_host,
            port=engine_port
        )

        # Display status
        is_threat = reading.threat_level > 0.5
        if is_threat:
            threat_count += 1

        status_icon = "🔴" if is_threat else "🟢"
        threat_type = reading.threat_type or "NORMAL"

        print(f"{status_icon} [{reading_count:04d}] "
              f"Freq: {reading.frequency:7.2f} Hz | "
              f"Amp: {reading.amplitude:5.1f} | "
              f"Type: {threat_type:15} | ", end="")

        if response:
            try:
                result = json.loads(response.get("result", "{}"))
                agent_mode = result.get("agent_decision", {}).get("mode", "N/A")
                print(f"Agent: {agent_mode}")
            except:
                print("Agent: Processing...")
        else:
            print("Agent: Offline")

    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"📊 Simulation Complete")
    print(f"   Total Readings: {reading_count}")
    print(f"   Threats Detected: {threat_count}")
    print(f"   Duration: {elapsed:.1f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Spectra-Sovereignty: Real-Time Signal AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulator only (for testing without engine)
  python app.py --simulator-only --scenario defense --duration 30

  # Run with custom engine address
  python app.py --engine-host localhost --engine-port 8080

  # Health monitoring scenario with high threat probability
  python app.py --scenario health --threat-prob 0.2
        """
    )

    parser.add_argument(
        "--scenario",
        choices=["defense", "health", "agriculture"],
        default=os.environ.get("ACTIVE_SCENARIO", "defense"),
        help="Simulation scenario (default: defense)"
    )
    parser.add_argument(
        "--threat-prob",
        type=float,
        default=float(os.environ.get("THREAT_PROBABILITY", "0.1")),
        help="Threat probability 0.0-1.0 (default: 0.1)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=int(os.environ.get("SIGNAL_INTERVAL_MS", "200")),
        help="Sample rate in milliseconds (default: 200)"
    )
    parser.add_argument(
        "--engine-host",
        default=os.environ.get("HOST", "localhost"),
        help="Signal engine host (default: localhost)"
    )
    parser.add_argument(
        "--engine-port",
        type=int,
        default=int(os.environ.get("PORT", "8080")),
        help="Signal engine port (default: 8080)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in seconds (default: infinite)"
    )
    parser.add_argument(
        "--simulator-only",
        action="store_true",
        help="Run simulator without connecting to engine"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("🛡️  SPECTRA-SOVEREIGNTY")
    print("   Real-Time Electromagnetic Signal AI Agent")
    print("=" * 60)

    if args.simulator_only:
        # Just run the simulator for testing
        simulator = SignalSimulator(
            scenario=Scenario(args.scenario),
            threat_probability=args.threat_prob,
            sample_interval_ms=args.sample_rate
        )

        print(f"\n📡 Running in SIMULATOR-ONLY mode")
        print(f"   Scenario: {args.scenario.upper()}")
        print("-" * 50)

        for reading in simulator.stream(duration_sec=args.duration):
            status = "🔴 THREAT" if reading.threat_level > 0.5 else "🟢 NORMAL"
            print(f"{status} | Freq: {reading.frequency:7.2f} Hz | "
                  f"Amp: {reading.amplitude:5.1f} | "
                  f"Threat: {reading.threat_level:.2f}")
    else:
        # Run simulator and send to engine
        run_simulator_thread(
            scenario=args.scenario,
            threat_prob=args.threat_prob,
            sample_rate_ms=args.sample_rate,
            engine_host=args.engine_host,
            engine_port=args.engine_port,
            duration=args.duration
        )


if __name__ == "__main__":
    main()
