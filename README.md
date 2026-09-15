# Industrial Computer Vision Production Pipeline

[![core-tests](https://github.com/summerming1/industrial-cv-production-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/summerming1/industrial-cv-production-pipeline/actions/workflows/ci.yml)

A clean-room, runnable public showcase of production-style computer-vision engineering: segmentation inference, spatial/ROI logic, temporal event semantics, model export adapters, and testable orchestration.

> This is not a copy of a private/customer repository. It contains no customer code, factory media, equipment IDs, production thresholds, RTSP URLs, MES/MQ routes, or proprietary process rules.

## 15-second reviewer map

| Capability | Verify here |
|---|---|
| YOLO segmentation adapter | [`src/industrial_cv/inference.py`](src/industrial_cv/inference.py) |
| Typed detections / frame contracts | [`src/industrial_cv/types.py`](src/industrial_cv/types.py) |
| Polygon ROI + spatial policies | [`src/industrial_cv/roi.py`](src/industrial_cv/roi.py), [`src/industrial_cv/runtime.py`](src/industrial_cv/runtime.py) |
| Time-based event state machine | [`src/industrial_cv/events.py`](src/industrial_cv/events.py) |
| Stream-gap / timestamp safety | [`tests/test_state_machine.py`](tests/test_state_machine.py) |
| End-to-end orchestration | [`src/industrial_cv/runtime.py`](src/industrial_cv/runtime.py) |
| ONNX / TensorRT export adapter | [`src/industrial_cv/export.py`](src/industrial_cv/export.py) |
| Synthetic process replay | [`examples/synthetic_replay.py`](examples/synthetic_replay.py) |
| Runnable behavior | [`tests/`](tests/) and [`examples/demo.py`](examples/demo.py) |

## What this demonstrates

- YOLO-style segmentation through an optional Ultralytics adapter
- typed detection outputs: class, confidence, bbox, mask polygon
- explicit spatial policies: **center-in-ROI**, dependency-free **bbox approximation**, or **segmentation-polygon vertex ratio**
- time-based state transitions that do not depend on a fixed FPS
- explicit protection against timestamp reversal and long observation gaps
- an explicit `reset()` path for stream reconnect handling
- structured event output suitable for UI, metrics, or deployment-specific integration
- ONNX/TensorRT export entry point through Ultralytics
- lightweight tests that run without model weights or a GPU

The public polygon and bbox policies are intentionally simple. `bbox_approx` uses the ROI bounding rectangle, and `polygon_vertex` measures the fraction of segmentation vertices inside the ROI. Neither is presented as exact mask-area intersection.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python examples/demo.py
python examples/synthetic_replay.py
```

The synthetic replay demonstrates event continuity, brief detector dropout, event termination, cooldown, and a long stream gap that starts a fresh candidate instead of silently extending prior evidence.

## Architecture

```text
Video / RTSP / camera
        |
        v
segmentation detector
        |
        v
typed detections + optional mask polygon
        |
        v
spatial policy
(center / bbox approximation / polygon vertices)
        |
        v
observation continuity checks
(timestamp monotonicity / max gap)
        |
        v
temporal event state machine
        |
        v
structured FrameResult / process event
        |
        +--> UI / metrics / audit
        +--> deployment-specific integration
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for boundaries and production extensions.

## Spatial-policy example

```python
from industrial_cv.events import TemporalEventMachine
from industrial_cv.roi import PolygonROI
from industrial_cv.runtime import VisionPipeline

roi = PolygonROI("slot-a", ((100, 100), (500, 100), (500, 400), (100, 400)))

pipeline = VisionPipeline(
    detector,
    roi,
    TemporalEventMachine(max_observation_gap_s=1.0),
    target_classes={"material"},
    spatial_policy="polygon_vertex",
    min_spatial_ratio=0.5,
)
```

This makes the business rule visible and testable rather than hiding it inside detector code.

## Real model adapter

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

## Export boundary

```python
from industrial_cv.export import export_ultralytics_model

export_ultralytics_model("weights/model.pt", format="onnx")
export_ultralytics_model("weights/model.pt", format="engine", half=True)
```

The export helper proves integration with the backend API. It is **not** presented as a TensorRT accuracy/latency benchmark. A production claim would require fixed media, hardware, preprocessing, warmup, output-equivalence checks, and measured latency/throughput.

## Why this is more than a YOLO demo

Production video systems often fail outside the neural network. A useful implementation needs explicit semantics for geometry, noisy detections, sampling gaps, reconnects, and event timing. This project isolates those concerns so they can be reviewed and tested independently of model weights.

## Private-work boundary

The architecture is informed by industrial process-state, ROI and event-detection work, but this implementation was written from scratch for public demonstration. No private source code or factory media is included.

## Portfolio relevance

This repository supports remote work involving **computer vision, YOLO segmentation, real-time video analytics, ROI/spatial logic, temporal event detection, inference-pipeline design, ONNX/TensorRT integration, and edge/production ML engineering**.

Related public work:

- [27B LLM Fine-Tuning, Evaluation & vLLM Deployment](https://github.com/summerming1/llm-posttraining-case-study)
- [Production RAG Agent](https://github.com/summerming1/production-rag-agent)
