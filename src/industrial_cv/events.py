from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .types import Observation


class EventState(str, Enum):
    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


@dataclass(frozen=True)
class EventUpdate:
    state: EventState
    started: bool = False
    ended: bool = False


class TemporalEventMachine:
    """Debounces frame-level evidence into stable process events.

    It requires sustained positive evidence before activation and sustained
    negative evidence before termination. Time is used instead of raw frame
    counts so behavior remains stable when inference FPS changes.
    """

    def __init__(self, start_after_s: float = 0.6, end_after_s: float = 0.8, cooldown_s: float = 1.0, min_evidence: float = 0.5) -> None:
        if min(start_after_s, end_after_s, cooldown_s) < 0:
            raise ValueError("durations must be non-negative")
        self.start_after_s = start_after_s
        self.end_after_s = end_after_s
        self.cooldown_s = cooldown_s
        self.min_evidence = min_evidence
        self.state = EventState.IDLE
        self._since: float | None = None

    def update(self, obs: Observation) -> EventUpdate:
        positive = obs.present and obs.evidence >= self.min_evidence
        t = obs.timestamp_s
        started = ended = False

        if self.state == EventState.IDLE:
            if positive:
                self.state = EventState.CANDIDATE
                self._since = t

        elif self.state == EventState.CANDIDATE:
            if not positive:
                self.state, self._since = EventState.IDLE, None
            elif self._since is not None and t - self._since >= self.start_after_s:
                self.state, self._since, started = EventState.ACTIVE, None, True

        elif self.state == EventState.ACTIVE:
            if not positive:
                if self._since is None:
                    self._since = t
                elif t - self._since >= self.end_after_s:
                    self.state, self._since, ended = EventState.COOLDOWN, t, True
            else:
                self._since = None

        elif self.state == EventState.COOLDOWN:
            if self._since is not None and t - self._since >= self.cooldown_s:
                self.state, self._since = EventState.IDLE, None

        return EventUpdate(self.state, started=started, ended=ended)
