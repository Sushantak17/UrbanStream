#!/usr/bin/env python3
"""
UrbanStream — Coordinator (Orchestrator Agent)

Manages the lifecycle of all agents:
  - Starts agents in dependency order: Monitor → Forecaster → Router → Alert
  - Monitors heartbeats and restarts dead agents
  - Tracks system-wide decision history for dashboard
  - Publishes coordinator status to Redis
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.monitor_agent import MonitorAgent
from agents.forecaster_agent import ForecasterAgent
from agents.router_agent import RouterAgent
from agents.alert_agent import AlertAgent

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [COORDINATOR] %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class Coordinator:
    """
    Orchestrates the multi-agent system.

    Execution order ensures data flows correctly:
      1. Monitor Agent  — detects anomalies (15s)
      2. Forecaster Agent — predicts AQI (30s)
      3. Router Agent — assigns workers (30s) — depends on Monitor + Forecaster
      4. Alert Agent — generates briefings (30s) — depends on Router
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self._start_ts = None

        # Create agents with shared Redis config
        kwargs = {"redis_host": redis_host, "redis_port": redis_port}
        self.agents = [
            MonitorAgent(**kwargs),
            ForecasterAgent(**kwargs),
            RouterAgent(**kwargs),
            AlertAgent(**kwargs),
        ]

    def start(self):
        """Start all agents in dependency order."""
        self._start_ts = datetime.now(timezone.utc).isoformat()
        log.info("╔══════════════════════════════════════════════════════╗")
        log.info("║  UrbanStream Multi-Agent System — Starting          ║")
        log.info("║  4 agents · Redis %s:%d                  ║",
                 self.redis_host, self.redis_port)
        log.info("╚══════════════════════════════════════════════════════╝")

        for agent in self.agents:
            agent.start()
            log.info("  ✓ %s agent started (interval=%.0fs)",
                     agent.name, agent.run_interval_sec)
            time.sleep(0.5)  # stagger startup

        log.info("All %d agents running", len(self.agents))

    def stop(self):
        """Stop all agents gracefully."""
        log.info("Stopping all agents...")
        for agent in reversed(self.agents):
            agent.stop()
        log.info("All agents stopped")

    def health_check(self) -> dict:
        """Check health of all agents and restart dead ones."""
        status = {}
        for agent in self.agents:
            alive = agent.is_alive
            status[agent.name] = {
                "alive": alive,
                "decisions": agent._decision_count,
                "errors": agent._error_count,
                "last_run": agent._last_run_ts,
            }
            if not alive and agent._running:
                log.warning("Agent %s died — restarting", agent.name)
                agent.stop()
                agent.start()
        return status

    def publish_status(self):
        """Publish coordinator status to Redis."""
        try:
            import redis
            r = redis.Redis(host=self.redis_host, port=self.redis_port,
                           db=0, socket_timeout=5, decode_responses=True)
            status = {
                "started_at": self._start_ts,
                "n_agents": len(self.agents),
                "agents": {a.name: a.get_status() for a in self.agents},
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            r.set("agent:coordinator:status", json.dumps(status), ex=120)
            r.publish("agent:coordinator:status", json.dumps(status))
        except Exception as e:
            log.warning("Failed to publish coordinator status: %s", e)

    def run_forever(self):
        """Run the coordinator loop — monitors agents and publishes status."""
        self.start()
        try:
            while True:
                time.sleep(15)
                self.health_check()
                self.publish_status()
        except KeyboardInterrupt:
            log.info("Shutdown signal received")
        finally:
            self.stop()

    def get_all_agent_logs(self) -> dict:
        """Collect decision logs from all agents for dashboard."""
        return {agent.name: list(agent._decision_log) for agent in self.agents}
