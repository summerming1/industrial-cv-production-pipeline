from __future__ import annotations

from pathlib import Path


def export_ultralytics_model(model_path: str, *, format: str = "onnx", imgsz: int = 640, half: bool = False, dynamic: bool = False) -> Path:
    """Export a YOLO model through Ultralytics' supported backends.

    TensorRT (`format="engine"`) requires a compatible CUDA/TensorRT host.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install with: pip install -e '.[vision]'") from exc
    model = YOLO(model_path)
    exported = model.export(format=format, imgsz=imgsz, half=half, dynamic=dynamic)
    return Path(str(exported))
