"""Synthetic temporal replay. No customer video, model weights, or measured benchmark data."""

from industrial_cv.events import TemporalEventMachine
from industrial_cv.types import Observation


def main() -> None:
    machine = TemporalEventMachine(
        start_after_s=0.5,
        end_after_s=0.5,
        cooldown_s=0.5,
        max_observation_gap_s=1.0,
    )
    timeline = [
        (0.0, False),
        (0.2, True),
        (0.8, True),
        (1.0, False),
        (1.2, True),
        (1.5, False),
        (2.1, False),
        (2.7, False),
        (5.0, True),
    ]

    print("synthetic_replay=true")
    print("t\tpresent\tstate\tstarted\tended\tgap_reset")
    for timestamp, present in timeline:
        update = machine.update(
            Observation(timestamp, 0.9 if present else 0.0, present)
        )
        print(
            f"{timestamp:.1f}\t{int(present)}\t{update.state.value}\t"
            f"{int(update.started)}\t{int(update.ended)}\t"
            f"{int(update.reset_due_to_gap)}"
        )


if __name__ == "__main__":
    main()
