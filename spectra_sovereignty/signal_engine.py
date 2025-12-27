"""
Signal Engine for Spectra-Sovereignty

Core Pathway-powered streaming pipeline that:
1. Ingests real-time signal data
2. Detects anomalies using statistical analysis
3. Uses Google Gemini (via LiteLLMChat) for intelligent threat classification
4. Outputs appropriate counter-response actions

This is the "brain" of the Spectra-Sovereignty system.
"""

import os
import json
from typing import Optional

import dotenv
import pathway as pw
from pathway.xpacks.llm.llms import LiteLLMChat, prompt_chat_single_qa

# Load environment variables
dotenv.load_dotenv()

# Pathway license for demo (optional, can be removed for community edition)
pw.set_license_key("demo-license-key-with-telemetry")


# =============================================================================
# SCHEMAS
# =============================================================================

class SignalInputSchema(pw.Schema):
    """Schema for incoming signal data stream."""
    timestamp: float
    frequency: float
    amplitude: float
    noise: float
    phase: float
    threat_level: float  # Ground truth for testing (in real system, this is unknown)


class QueryInputSchema(pw.Schema):
    """Schema for REST API queries."""
    query: str
    user: str


# =============================================================================
# AGENT PROMPTS
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are Spectra-Sentinel, an AI agent monitoring electromagnetic signals for threats.

You analyze signal patterns and decide on the appropriate defensive response.

AVAILABLE RESPONSE MODES:
- MODE_ALPHA (Noise-Cancellation): Blend into background EM noise
- MODE_BETA (Frequency-Shift): Evade detection by shifting frequency bands
- MODE_GAMMA (Active-Counter): Emit jamming/counter-frequency
- MODE_DELTA (Alert-Only): Monitor and log, no active response
- MODE_NORMAL: No threat, continue standard monitoring

SCENARIO CONTEXT:
- Defense: Watching for enemy radar/surveillance signals
- Health: Monitoring for cardiac/bio-signal anomalies
- Agriculture: Detecting pest swarm EM signatures

Always respond with a JSON object containing:
{
    "mode": "MODE_X",
    "confidence": 0.0-1.0,
    "rationale": "Brief explanation",
    "action": "Specific action to take"
}
"""


def build_threat_analysis_prompt(
    frequency: float,
    amplitude: float,
    noise: float,
    anomaly_score: float,
    scenario: str
) -> str:
    """Build prompt for LLM threat analysis."""
    return f"""{AGENT_SYSTEM_PROMPT}

CURRENT SIGNAL READING:
- Frequency: {frequency:.2f} Hz
- Amplitude: {amplitude:.2f}
- Noise Floor: {noise:.2f}
- Anomaly Score: {anomaly_score:.3f}
- Scenario: {scenario.upper()}

Based on this signal pattern, what response mode should be activated?
Respond with JSON only."""


# =============================================================================
# PATHWAY UDFs (User Defined Functions)
# =============================================================================

@pw.udf
def calculate_anomaly_score(
    frequency: float,
    amplitude: float,
    noise: float,
    baseline_freq: float = 300.0,
    baseline_amp: float = 45.0
) -> float:
    """
    Calculate an anomaly score based on deviation from baseline.
    Uses a simple z-score-like metric.

    Returns: 0.0 (normal) to 1.0 (highly anomalous)
    """
    freq_deviation = abs(frequency - baseline_freq) / baseline_freq
    amp_deviation = abs(amplitude - baseline_amp) / baseline_amp

    # Combined score with frequency weighted higher
    raw_score = (freq_deviation * 0.7) + (amp_deviation * 0.3)

    # Normalize to 0-1 range
    return min(1.0, raw_score)


@pw.udf
def is_threat_detected(anomaly_score: float, threshold: float = 0.4) -> bool:
    """Determine if anomaly score exceeds threat threshold."""
    return anomaly_score > threshold


@pw.udf
def build_prompt(frequency: float, amplitude: float, noise: float,
                 anomaly_score: float, scenario: str) -> str:
    """Build the LLM prompt for threat analysis."""
    return build_threat_analysis_prompt(frequency, amplitude, noise, anomaly_score, scenario)


@pw.udf
def parse_agent_response(response: str) -> dict:
    """Parse the JSON response from the agent."""
    try:
        # Try to extract JSON from response
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code blocks
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])

        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # Fallback response
        return {
            "mode": "MODE_DELTA",
            "confidence": 0.0,
            "rationale": "Failed to parse agent response",
            "action": "Alert only - manual review required"
        }


@pw.udf
def format_output(
    timestamp: float,
    frequency: float,
    amplitude: float,
    anomaly_score: float,
    is_threat: bool,
    agent_response: dict
) -> str:
    """Format the final output for the dashboard."""
    output = {
        "timestamp": timestamp,
        "signal": {
            "frequency": frequency,
            "amplitude": amplitude
        },
        "analysis": {
            "anomaly_score": anomaly_score,
            "is_threat": is_threat
        },
        "agent_decision": agent_response
    }
    return json.dumps(output)


# =============================================================================
# SIGNAL ENGINE CLASS
# =============================================================================

class SignalEngine:
    """
    Main Pathway-powered signal processing engine.

    Uses LiteLLMChat with Google Gemini for intelligent threat analysis.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        gemini_api_key: Optional[str] = None,
        model: str = "gemini/gemini-2.5-flash-lite",
        scenario: str = "defense"
    ):
        self.host = host
        self.port = port
        self.scenario = scenario

        # Configure Gemini via LiteLLM
        api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided. Set environment variable or pass to constructor.")

        # LiteLLM uses GEMINI_API_KEY env var automatically for gemini/ models
        os.environ["GEMINI_API_KEY"] = api_key

        # Initialize LiteLLM Chat with Gemini
        self.llm = LiteLLMChat(
            model=model,
            retry_strategy=pw.udfs.ExponentialBackoffRetryStrategy(max_retries=3),
            cache_strategy=pw.udfs.DefaultCache(),
            temperature=0.1,  # Low temperature for consistent responses
            max_tokens=500
        )

        # Scenario-specific baselines
        self.baselines = {
            "defense": {"freq": 300.0, "amp": 45.0},
            "health": {"freq": 1.0, "amp": 70.0},
            "agriculture": {"freq": 200.0, "amp": 30.0}
        }

    def build_pipeline(self):
        """
        Build the Pathway streaming pipeline.

        Pipeline stages:
        1. Ingest signals from HTTP stream
        2. Calculate anomaly scores
        3. Detect threats
        4. Query LLM agent for threat analysis (only for detected threats)
        5. Output decisions
        """
        baseline = self.baselines.get(self.scenario, self.baselines["defense"])

        # Stage 1: Input connector - receive signals via HTTP
        signals, response_writer = pw.io.http.rest_connector(
            host=self.host,
            port=self.port,
            schema=SignalInputSchema,
            autocommit_duration_ms=50,
            delete_completed_queries=True
        )

        # Stage 2: Calculate anomaly scores
        signals = signals.select(
            pw.this.timestamp,
            pw.this.frequency,
            pw.this.amplitude,
            pw.this.noise,
            pw.this.phase,
            pw.this.threat_level,  # Ground truth for testing
            anomaly_score=calculate_anomaly_score(
                pw.this.frequency,
                pw.this.amplitude,
                pw.this.noise,
                baseline["freq"],
                baseline["amp"]
            )
        )

        # Stage 3: Threat detection
        signals = signals.select(
            pw.this.timestamp,
            pw.this.frequency,
            pw.this.amplitude,
            pw.this.noise,
            pw.this.anomaly_score,
            pw.this.threat_level,
            is_threat=is_threat_detected(pw.this.anomaly_score)
        )

        # Stage 4: LLM Agent analysis (for threats)
        # Build prompt for threat analysis
        signals = signals.select(
            pw.this.timestamp,
            pw.this.frequency,
            pw.this.amplitude,
            pw.this.noise,
            pw.this.anomaly_score,
            pw.this.is_threat,
            pw.this.threat_level,
            prompt=build_prompt(
                pw.this.frequency,
                pw.this.amplitude,
                pw.this.noise,
                pw.this.anomaly_score,
                self.scenario
            )
        )

        # Query LLM only when threat is detected
        # For non-threats, use a default response
        signals = signals.select(
            pw.this.timestamp,
            pw.this.frequency,
            pw.this.amplitude,
            pw.this.anomaly_score,
            pw.this.is_threat,
            pw.this.threat_level,
            agent_raw=pw.if_else(
                pw.this.is_threat,
                self.llm(prompt_chat_single_qa(pw.this.prompt)),
                pw.cast(str, '{"mode": "MODE_NORMAL", "confidence": 1.0, "rationale": "No threat detected", "action": "Continue standard monitoring"}')
            )
        )

        # Stage 5: Parse agent response and format output
        signals = signals.select(
            pw.this.timestamp,
            pw.this.frequency,
            pw.this.amplitude,
            pw.this.anomaly_score,
            pw.this.is_threat,
            pw.this.threat_level,
            agent_response=parse_agent_response(pw.this.agent_raw)
        )

        # Final output formatting
        output = signals.select(
            result=format_output(
                pw.this.timestamp,
                pw.this.frequency,
                pw.this.amplitude,
                pw.this.anomaly_score,
                pw.this.is_threat,
                pw.this.agent_response
            )
        )

        # Write responses back
        response_writer(output)

        return signals

    def run(self):
        """Start the Pathway engine."""
        print(f"\n🛡️  Spectra-Sovereignty Signal Engine")
        print(f"📡 Scenario: {self.scenario.upper()}")
        print(f"🤖 LLM: Google Gemini (gemini-2.5-flash-lite)")
        print(f"🌐 Listening on {self.host}:{self.port}")
        print("-" * 50)

        self.build_pipeline()
        pw.run(monitoring_level=pw.MonitoringLevel.NONE)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run(
    host: str = os.environ.get("HOST", "0.0.0.0"),
    port: int = int(os.environ.get("PORT", "8080")),
    scenario: str = os.environ.get("ACTIVE_SCENARIO", "defense"),
    **kwargs
):
    """Run the Signal Engine."""
    engine = SignalEngine(
        host=host,
        port=port,
        scenario=scenario
    )
    engine.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spectra-Sovereignty Signal Engine")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--scenario", choices=["defense", "health", "agriculture"],
                        default="defense", help="Active scenario")

    args = parser.parse_args()

    run(host=args.host, port=args.port, scenario=args.scenario)
