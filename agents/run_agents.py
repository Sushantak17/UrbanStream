#!/usr/bin/env python3
"""
UrbanStream — Multi-Agent System Entry Point

Usage:
    REDIS_HOST=localhost python3 agents/run_agents.py

Starts the Coordinator which launches all 4 agents:
  1. Monitor Agent  (anomaly detection,  15s interval)
  2. Forecaster Agent (LSTM AQI forecast, 30s interval)
  3. Router Agent   (LinUCB routing,      30s interval)
  4. Alert Agent    (safety briefings,    30s interval)

Press Ctrl+C for graceful shutdown.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.coordinator import Coordinator


def main():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    coordinator = Coordinator(
        redis_host=redis_host,
        redis_port=redis_port,
    )
    coordinator.run_forever()


if __name__ == "__main__":
    main()
