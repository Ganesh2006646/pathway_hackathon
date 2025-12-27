"""
Signal Simulator for Spectra-Sovereignty

Generates realistic simulated electromagnetic signal data streams with
configurable threat injection for testing the detection pipeline.

Threats are simulated as:
- Defense: Radar sweep patterns (high-frequency spikes)
- Health: Cardiac anomalies (irregular rhythm patterns)
- Agriculture: Pest swarm EM signatures (specific frequency bands)
"""

import json
import math
import random
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Generator


class Scenario(Enum):
    """Available simulation scenarios."""
    DEFENSE = "defense"
    HEALTH = "health"
    AGRICULTURE = "agriculture"


@dataclass
class SignalReading:
    """A single signal reading from the EM spectrum."""
    timestamp: float
    frequency: float          # Hz (0-1000)
    amplitude: float          # Signal strength (0-100)
    noise: float              # Noise floor (0-50)
    phase: float              # Phase angle (0-360)
    threat_level: float       # Ground truth: 0.0 (none) to 1.0 (critical)
    threat_type: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string for streaming."""
        return json.dumps(asdict(self))

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


class SignalSimulator:
    """
    Generates continuous signal stream with configurable threat injection.

    Each scenario has different baseline characteristics and threat patterns:

    - Defense: Baseline at 200-400 Hz, threats are radar sweeps at 800+ Hz
    - Health: Baseline at 0.8-1.2 Hz (heartbeat), threats are arrhythmias
    - Agriculture: Baseline at 100-300 Hz, threats are pest EM at 50-80 Hz
    """

    def __init__(
        self,
        scenario: Scenario = Scenario.DEFENSE,
        threat_probability: float = 0.05,
        sample_interval_ms: int = 100
    ):
        self.scenario = scenario
        self.threat_probability = threat_probability
        self.sample_interval_sec = sample_interval_ms / 1000.0

        # Scenario-specific baselines
        self._configure_scenario()

        # State for continuous patterns
        self._tick = 0
        self._threat_duration = 0
        self._in_threat = False

    def _configure_scenario(self):
        """Configure baseline parameters per scenario."""
        if self.scenario == Scenario.DEFENSE:
            self.base_freq = 300.0       # Normal comm frequency
            self.freq_variance = 50.0
            self.base_amplitude = 45.0
            self.amp_variance = 10.0
            self.base_noise = 15.0
            self.threat_freq = 850.0     # Radar sweep frequency
            self.threat_name = "RADAR_SWEEP"

        elif self.scenario == Scenario.HEALTH:
            self.base_freq = 1.0         # Normal heart rate ~60 bpm
            self.freq_variance = 0.15
            self.base_amplitude = 70.0   # Strong heartbeat signal
            self.amp_variance = 5.0
            self.base_noise = 8.0
            self.threat_freq = 2.5       # Tachycardia/arrhythmia
            self.threat_name = "CARDIAC_ANOMALY"

        elif self.scenario == Scenario.AGRICULTURE:
            self.base_freq = 200.0       # Normal environmental EM
            self.freq_variance = 40.0
            self.base_amplitude = 30.0
            self.amp_variance = 8.0
            self.base_noise = 20.0
            self.threat_freq = 65.0      # Insect navigation frequency
            self.threat_name = "PEST_SWARM"

    def _generate_baseline_signal(self) -> tuple[float, float, float, float]:
        """Generate normal baseline signal values."""
        # Sinusoidal pattern with noise for realism
        time_factor = self._tick * 0.1

        frequency = (
            self.base_freq +
            self.freq_variance * math.sin(time_factor * 0.5) +
            random.gauss(0, self.freq_variance * 0.1)
        )

        amplitude = (
            self.base_amplitude +
            self.amp_variance * math.sin(time_factor * 0.3) +
            random.gauss(0, self.amp_variance * 0.2)
        )

        noise = self.base_noise + random.gauss(0, 3.0)
        phase = (time_factor * 30) % 360

        return max(0, frequency), max(0, amplitude), max(0, noise), phase

    def _inject_threat(self) -> tuple[float, float, float, float, float, str]:
        """Inject a threat pattern based on scenario."""
        # Threat patterns are more aggressive
        time_factor = self._tick * 0.1

        if self.scenario == Scenario.DEFENSE:
            # Radar sweep: very high frequency spike
            frequency = self.threat_freq + random.gauss(0, 30)
            amplitude = 90 + random.gauss(0, 5)  # Strong signal
            noise = self.base_noise * 2  # Increased EM noise
            threat_level = 0.85 + random.uniform(0, 0.15)

        elif self.scenario == Scenario.HEALTH:
            # Arrhythmia: irregular frequency and amplitude
            frequency = self.threat_freq + random.uniform(-0.5, 1.5)
            amplitude = 40 + random.gauss(0, 20)  # Erratic amplitude
            noise = self.base_noise + 5
            threat_level = 0.7 + random.uniform(0, 0.3)

        else:  # Agriculture
            # Pest swarm: characteristic low frequency signature
            frequency = self.threat_freq + random.gauss(0, 5)
            amplitude = 55 + random.gauss(0, 10)
            noise = self.base_noise + 10  # Bioelectric noise
            threat_level = 0.6 + random.uniform(0, 0.3)

        phase = (time_factor * 60) % 360  # Faster phase change during threat

        return frequency, amplitude, noise, phase, threat_level, self.threat_name

    def generate_reading(self) -> SignalReading:
        """Generate a single signal reading."""
        self._tick += 1
        timestamp = time.time()

        # Threat state machine
        if self._in_threat:
            self._threat_duration -= 1
            if self._threat_duration <= 0:
                self._in_threat = False
        else:
            # Random threat injection
            if random.random() < self.threat_probability:
                self._in_threat = True
                self._threat_duration = random.randint(3, 10)  # 3-10 readings

        if self._in_threat:
            freq, amp, noise, phase, threat_level, threat_type = self._inject_threat()
        else:
            freq, amp, noise, phase = self._generate_baseline_signal()
            threat_level = 0.0
            threat_type = None

        return SignalReading(
            timestamp=timestamp,
            frequency=round(freq, 2),
            amplitude=round(amp, 2),
            noise=round(noise, 2),
            phase=round(phase, 2),
            threat_level=round(threat_level, 3),
            threat_type=threat_type
        )

    def stream(self, duration_sec: Optional[float] = None) -> Generator[SignalReading, None, None]:
        """
        Generate continuous stream of signal readings.

        Args:
            duration_sec: Optional duration limit. None for infinite stream.

        Yields:
            SignalReading objects at configured interval
        """
        start_time = time.time()

        while True:
            if duration_sec and (time.time() - start_time) >= duration_sec:
                break

            yield self.generate_reading()
            time.sleep(self.sample_interval_sec)


def create_test_stream(scenario: str = "defense", duration: int = 10) -> list[dict]:
    """Create a test stream for development purposes."""
    scenario_enum = Scenario(scenario.lower())
    simulator = SignalSimulator(
        scenario=scenario_enum,
        threat_probability=0.15,  # Higher for testing
        sample_interval_ms=100
    )

    readings = []
    for reading in simulator.stream(duration_sec=duration):
        readings.append(reading.to_dict())
        print(f"[{reading.threat_type or 'NORMAL':15}] "
              f"Freq: {reading.frequency:7.2f} Hz | "
              f"Amp: {reading.amplitude:5.2f} | "
              f"Threat: {reading.threat_level:.2f}")

    return readings


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Signal Simulator for Spectra-Sovereignty")
    parser.add_argument("--scenario", choices=["defense", "health", "agriculture"],
                        default="defense", help="Simulation scenario")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration in seconds")
    parser.add_argument("--threat-prob", type=float, default=0.1,
                        help="Threat probability (0.0-1.0)")

    args = parser.parse_args()

    print(f"\n🛡️ Spectra-Sovereignty Signal Simulator")
    print(f"📡 Scenario: {args.scenario.upper()}")
    print(f"⏱️  Duration: {args.duration}s")
    print(f"⚠️  Threat Probability: {args.threat_prob * 100:.1f}%")
    print("-" * 60)

    simulator = SignalSimulator(
        scenario=Scenario(args.scenario),
        threat_probability=args.threat_prob,
        sample_interval_ms=200
    )

    for reading in simulator.stream(duration_sec=args.duration):
        status = "🔴 THREAT" if reading.threat_level > 0.5 else "🟢 NORMAL"
        print(f"{status} | Freq: {reading.frequency:7.2f} Hz | "
              f"Amp: {reading.amplitude:5.1f} | "
              f"Noise: {reading.noise:4.1f} | "
              f"Threat: {reading.threat_level:.2f}")
