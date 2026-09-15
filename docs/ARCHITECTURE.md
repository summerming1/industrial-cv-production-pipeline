# Architecture and engineering boundaries

This repository is a clean-room public reference implementation inspired by patterns used in real industrial CV work. It intentionally excludes customer video, production thresholds, equipment identifiers, stream URLs, broker routes, and proprietary process rules.

```text
Camera / RTSP / file
        |
        v
segmentation detector
        |
        v
typed detections + geometry
        |
        v
spatial policy
        |
        v
continuity checks
        |
        v
time-based event state machine
        |
        v
structured event / FrameResult
        |
        +--> UI / metrics / audit log
        +--> deployment adapter
```

## Spatial layer

The public runtime supports three explicit policies:

- `center`: detection bbox center must be inside the ROI.
- `bbox_approx`: overlap is computed against the ROI bounding rectangle. This is dependency-free and not exact polygon intersection.
- `polygon_vertex`: the fraction of segmentation-polygon vertices inside the ROI must pass a threshold. This uses segmentation geometry but is not exact area IoU.

Real deployments can substitute OpenCV/Shapely mask/polygon intersection without changing the temporal layer.

## Why time-based state is not enough by itself

Using wall-clock durations avoids hard-coding business semantics to inference FPS, but elapsed time alone can incorrectly bridge a dropped stream. `TemporalEventMachine` therefore also enforces monotonic timestamps and an optional maximum observation gap. A gap beyond that bound resets continuity instead of treating the next detection as sustained evidence.

An explicit `reset()` is available for known reconnect/restart events. Gap reset does not synthesize an `ended` event because the system lacks evidence about what happened while observations were missing; downstream deployment policy may choose to emit a separate stream-health event.

## Event semantics

The state machine separates:

- `IDLE`
- `CANDIDATE`: positive evidence has begun but has not yet met the start duration
- `ACTIVE`: the process event is considered active
- `COOLDOWN`: the event has ended and re-triggering is temporarily suppressed

Brief negative detector noise can be tolerated by `end_after_s`. Long observation gaps are handled separately from negative evidence.

## Deployment boundary

The public project stops at structured event output. A real deployment still needs stream decoding/reconnect policy, buffering/backpressure, multi-camera scheduling, batching, monitoring, model-version rollout, MES/MQ publishing, secrets, and site-specific thresholds.

## Export boundary

The Ultralytics export helper exposes ONNX/TensorRT export integration only. No performance or accuracy parity is claimed without a reproducible benchmark and redistributable test media.

## Verification

Core tests use synthetic detections and timestamps so spatial and temporal rules are deterministic and reviewable without a GPU. `examples/synthetic_replay.py` provides a small end-to-end temporal trace and is run in CI.
