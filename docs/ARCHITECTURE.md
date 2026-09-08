# Architecture

This repository is a clean-room public reference implementation inspired by patterns used in real industrial CV work. It intentionally does **not** contain customer video, production thresholds, equipment identifiers, stream URLs, message-broker routes, or proprietary state logic.

```text
Camera / RTSP / file
        |
        v
Segmentation detector (optional Ultralytics adapter)
        |
        v
Typed detections -> ROI/spatial filtering
        |
        v
Time-based event debouncing / state machine
        |
        v
Structured FrameResult / process event
        |
        +--> UI / metrics / audit log
        +--> integration adapter (implemented per deployment)
```

## Why the state machine is time-based

Industrial pipelines often change processing FPS according to load or activity. Using wall-clock durations rather than fixed frame counts prevents business semantics from changing when the inference sampling rate changes.

## Deployment boundary

The public project stops at structured event output. In a real deployment, RTSP reconnect policy, backpressure, batching, observability, MES/MQ publishing, secrets and device-specific thresholds belong in environment-specific adapters.

## What is intentionally simplified

- ROI overlap uses a dependency-free bounding-box approximation; exact polygon-mask overlap can be added with OpenCV/Shapely.
- The Ultralytics adapter is optional so tests run without model weights or GPU libraries.
- No performance numbers are claimed without a reproducible benchmark and redistributable test media.
