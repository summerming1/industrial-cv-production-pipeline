from __future__ import annotations

import math
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
    reset_due_to_gap: bool = False


class TemporalEventMachine:
    """Debounce frame-level evidence into stable process events.

    Time is used instead of frame counts, but continuity is still explicit:
    observations separated by more than ``max_observation_gap_s`` reset the
    state instead of being treated as uninterrupted evidence.
    """

    def __init__(
        self,
        start_after_s: float = 0.6,
        end_after_s: float = 0.8,
        cooldown_s: float = 1.0,
        min_evidence: float = 0.5,
        max_observation_gap_s: float | None = 2.0,
    ) -> None:
        values = (start_after_s, end_after_s, cooldown_s, min_evidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("durations and min_evidence must be finite")
        if min(start_after_s, end_after_s, cooldown_s) < 0:
            raise ValueError("durations must be non-negative")
        if not 0 <= min_evidence <= 1:
            raise ValueError("min_evidence must be in [0, 1]")
        if max_observation_gap_s is not None:
            if not math.isfinite(max_observation_gap_s) or max_observation_gap_s <= 0:
                raise ValueError("max_observation_gap_s must be positive or None")

        self.start_after_s = start_after_s
        self.end_after_s = end_after_s
        self.cooldown_s = cooldown_s
        self.min_evidence = min_evidence
        self.max_observation_gap_s = max_observation_gap_s
        self.state = EventState.IDLE
        self._since: float | None = None
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        """Reset temporal state, for example after an explicit stream reconnect."""
        self.state = EventState.IDLE
        self._since = None
        self._last_timestamp = None

    def update(self, obs: Observation) -> EventUpdate:
        t = float(obs.timestamp_s)
        evidence = float(obs.evidence)
        if not math.isfinite(t) or not math.isfinite(evidence):
            raise ValueError("timestamp and evidence must be finite")
        if not 0 <= evidence <= 1:
            raise ValueError("evidence must be in [0, 1]")

        reset_due_to_gap = False
        if self._last_timestamp is not None:
            if t < self._last_timestamp:
                raise ValueError("timestamps must be monotonic")
            if (
                self.max_observation_gap_s is not None
                and t - self._last_timestamp > self.max_observation_gap_s
            ):
                self.state = EventState.IDLE
                self._since = None
                reset_due_to_gap = True
        self._last_timestamp = t

        positive = obs.present and evidence >= self.min_evidence
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

        return EventUpdate(
            self.state,
            started=started,
            ended=ended,
            reset_due_to_gap=reset_due_to_gap,
        )
