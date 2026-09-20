#!/usr/bin/env python3
"""
UrbanStream — BaseAgent Abstract Class

All agents inherit from this. Provides:
  - Lifecycle management (start / stop / heartbeat)
  - perceive() → decide() → act() loop
  - Structured reasoning logs
  - Redis-based health tracking
"""

import json
import logging
import threading
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from collections import deque
from typing import Any, Optional

import redis

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base agent that all UrbanStream agents extend.

    Subclasses must implement:
      - perceive()  → read state from Redis
      - decide()    → run ML model, produce action
      - act()       → publish decision to Redis
    """

    def __init__(self, name: str, role: str, run_interval_sec: float = 30.0,
                 redis_host: str = "localhost", redis_port: int = 6379):
        self.name = name
        self.role = role
        self.run_interval_sec = run_interval_sec
        self.redis_host = redis_host
        self.redis_port = redis_port

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._decision_count = 0
        self._error_count = 0
        self._last_run_ts: Optional[str] = None
        self._last_error: Optional[str] = None
        self._decision_log: deque = deque(maxlen=20)  # last 20 decisions
        self._start_ts: Optional[str] = None

    # ── Redis ─────────────────────────────────────────────────────────────────
    def _get_redis(self) -> redis.Redis:
        return redis.Redis(host=self.redis_host, port=self.redis_port, db=0,
                           socket_timeout=5, decode_responses=True)

    # ── Abstract methods ──────────────────────────────────────────────────────
    @abstractmethod
    def perceive(self, r: redis.Redis) -> dict:
        """Read state from Redis. Return a dict of observations."""
        ...

    @abstractmethod
    def decide(self, observations: dict) -> dict:
        """Run ML model or logic on observations. Return a dict of decisions."""
        ...

    @abstractmethod
    def act(self, r: redis.Redis, decisions: dict) -> None:
        """Publish decisions to Redis keys/channels."""
        ...

    # ── Logging ───────────────────────────────────────────────────────────────
    def log_reasoning(self, summary: str, details: dict = None):
        """Log a structured reasoning entry for the dashboard."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "agent": self.name,
            "summary": summary,
            "details": details or {},
        }
        self._decision_log.append(entry)
        log.info("[%s] %s", self.name, summary)

    # ── Heartbeat ─────────────────────────────────────────────────────────────
    def _publish_heartbeat(self, r: redis.Redis):
        """Write agent status to Redis for dashboard visibility."""
        status = {
            "name":           self.name,
            "role":           self.role,
            "alive":          self._running,
            "decisions":      self._decision_count,
            "errors":         self._error_count,
            "last_run":       self._last_run_ts,
            "last_error":     self._last_error,
            "started_at":     self._start_ts,
            "interval_sec":   self.run_interval_sec,
        }
        r.set(f"agent_status:{self.name}", json.dumps(status), ex=120)
        # Store last 20 decisions for the dashboard log feed
        r.set(f"agent_log:{self.name}",
              json.dumps(list(self._decision_log)), ex=300)

    # ── Main loop ─────────────────────────────────────────────────────────────
    def _loop(self):
        """The agent's main perceive→decide→act loop."""
        while self._running:
            try:
                r = self._get_redis()
                observations = self.perceive(r)
                decisions = self.decide(observations)
                self.act(r, decisions)
                self._decision_count += 1
                self._last_run_ts = datetime.now(timezone.utc).isoformat()
                self._publish_heartbeat(r)
            except redis.ConnectionError as e:
                self._error_count += 1
                self._last_error = f"Redis: {e}"
                log.warning("[%s] Redis connection error: %s", self.name, e)
            except Exception as e:
                self._error_count += 1
                self._last_error = str(e)
                log.error("[%s] Error: %s\n%s", self.name, e,
                          traceback.format_exc())

            time.sleep(self.run_interval_sec)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start(self):
        """Start the agent in a background thread."""
        if self._running:
            log.warning("[%s] Already running", self.name)
            return
        self._running = True
        self._start_ts = datetime.now(timezone.utc).isoformat()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"agent-{self.name}")
        self._thread.start()
        log.info("[%s] Started (interval=%.0fs)", self.name, self.run_interval_sec)

    def stop(self):
        """Signal the agent to stop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.run_interval_sec + 2)
        log.info("[%s] Stopped (decisions=%d, errors=%d)",
                 self.name, self._decision_count, self._error_count)

    @property
    def is_alive(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    def get_status(self) -> dict:
        return {
            "name":       self.name,
            "role":       self.role,
            "alive":      self.is_alive,
            "decisions":  self._decision_count,
            "errors":     self._error_count,
            "last_run":   self._last_run_ts,
        }
