# Industrial Computer Vision Production Pipeline

A **clean-room, runnable public showcase** of the engineering patterns behind production computer-vision systems: segmentation inference, ROI logic, temporal event detection, model export, and testable orchestration.

> This is not a copy of a private/customer repository. It contains no customer code, factory media, equipment IDs, production thresholds, RTSP URLs, MES/MQ routes, or proprietary process rules.

## What this demonstrates

- YOLO-style **segmentation inference** through an optional Ultralytics adapter
- Typed detection outputs: class, confidence, bbox and mask polygon
- Polygon **ROI / spatial gating**
- **Time-based temporal state machine** for stable process-event detection
- A small orchestration layer designed for unit testing and replacement of inference backends
- ONNX / TensorRT export entry point through Ultralytics
- Dependency-light tests that run without model weights or a GPU

## Architecture

```text
Video / RTSP / Camera
        |
        v
Segmentation model
        |
        v
Detections + masks
        |
        v
ROI / spatial evidence
        |
        v
Temporal event state machine
        |
        v
Structured process events
        |
        +--> monitoring / UI
        +--> deployment-specific integration adapter
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design boundaries.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python examples/demo.py
```

To use a real Ultralytics segmentation model:

```bash
pip install -e '.[vision]'
```

```python
from industrial_cv.inference import UltralyticsSegmenter

detector = UltralyticsSegmenter(
    "weights/example-seg.pt",
    conf=0.35,
    imgsz=640,
    device="0",
)
```

## Model export

```python
from industrial_cv.export import export_ultralytics_model

export_ultralytics_model("weights/model.pt", format="onnx")
# TensorRT host:
export_ultralytics_model("weights/model.pt", format="engine", half=True)
```

## Why this is more than a YOLO demo

A production CV system usually fails outside the model if temporal semantics, stream handling and integration boundaries are not designed explicitly. This showcase isolates those concerns so they can be tested independently:

1. **Model layer** — replaceable detector backend.
2. **Spatial layer** — ROI logic converts detections into process evidence.
3. **Temporal layer** — debounces noisy frame predictions using elapsed time rather than hard-coded FPS assumptions.
4. **Integration layer** — returns structured events; deployment-specific MQ/MES adapters remain outside the public repository.

## Private-work boundary

The architecture is based on experience building industrial video analytics for process-state recognition, but the implementation here was written from scratch for public demonstration. No private source code was copied.

## Portfolio relevance

This repository supports work involving **Computer Vision, YOLO, segmentation, real-time video analytics, ROI/event logic, ONNX/TensorRT deployment, and production ML engineering**.
