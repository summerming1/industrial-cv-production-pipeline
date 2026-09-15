import math

import pytest

from industrial_cv.events import EventState, TemporalEventMachine
from industrial_cv.types import Observation


def obs(t, positive=True, evidence=None):
    value = 0.9 if evidence is None and positive else (0.0 if evidence is None else evidence)
    return Observation(t, value, positive)


def test_temporal_debounce_and_end():
    m = TemporalEventMachine(start_after_s=0.5, end_after_s=0.5, cooldown_s=0.5)
    assert m.update(obs(0.0)).state == EventState.CANDIDATE
    u = m.update(obs(0.6))
    assert u.started and u.state == EventState.ACTIVE
    assert m.update(obs(0.8, False)).state == EventState.ACTIVE
    u = m.update(obs(1.4, False))
    assert u.ended and u.state == EventState.COOLDOWN
    assert m.update(obs(2.0, False)).state == EventState.IDLE


def test_long_observation_gap_resets_continuity():
    m = TemporalEventMachine(start_after_s=0.5, max_observation_gap_s=1.0)
    assert m.update(obs(0.0)).state == EventState.CANDIDATE
    u = m.update(obs(10.0))
    assert u.reset_due_to_gap
    assert not u.started
    assert u.state == EventState.CANDIDATE


def test_active_state_does_not_survive_stream_gap():
    m = TemporalEventMachine(start_after_s=0.1, max_observation_gap_s=1.0)
    m.update(obs(0.0))
    assert m.update(obs(0.2)).state == EventState.ACTIVE
    u = m.update(obs(5.0, False))
    assert u.reset_due_to_gap
    assert not u.ended
    assert u.state == EventState.IDLE


def test_timestamp_must_be_monotonic_and_finite():
    m = TemporalEventMachine()
    m.update(obs(1.0))
    with pytest.raises(ValueError, match="monotonic"):
        m.update(obs(0.9))
    with pytest.raises(ValueError, match="finite"):
        TemporalEventMachine().update(obs(math.nan))


def test_explicit_reset_for_reconnect():
    m = TemporalEventMachine(start_after_s=0.1)
    m.update(obs(0.0))
    assert m.update(obs(0.2)).state == EventState.ACTIVE
    m.reset()
    assert m.state == EventState.IDLE
    assert m.update(obs(100.0)).state == EventState.CANDIDATE
