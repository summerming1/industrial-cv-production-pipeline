from industrial_cv.events import EventState, TemporalEventMachine
from industrial_cv.types import Observation


def obs(t, positive=True):
    return Observation(t, 0.9 if positive else 0.0, positive)


def test_temporal_debounce_and_end():
    m = TemporalEventMachine(start_after_s=0.5, end_after_s=0.5, cooldown_s=0.5)
    assert m.update(obs(0.0)).state == EventState.CANDIDATE
    u = m.update(obs(0.6))
    assert u.started and u.state == EventState.ACTIVE
    assert m.update(obs(0.8, False)).state == EventState.ACTIVE
    u = m.update(obs(1.4, False))
    assert u.ended and u.state == EventState.COOLDOWN
    assert m.update(obs(2.0, False)).state == EventState.IDLE
