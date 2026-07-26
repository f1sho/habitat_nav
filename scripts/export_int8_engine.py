from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path("scripts/yolo26n-seg.pt")
DATA_PATH = Path("scripts/int8_calibration/replica_int8.yaml")
DEFAULT_ENGINE_PATH = Path("scripts/yolo26n-seg.engine")
TARGET_ENGINE_PATH = Path("scripts/yolo26n-seg_int8.engine")
CALIBRATION_CACHE_PATH = Path("scripts/yolo26n-seg.cache")


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Calibration YAML not found: {DATA_PATH.resolve()}"
        )

    # if CALIBRATION_CACHE_PATH.exists():
    #     CALIBRATION_CACHE_PATH.unlink()
    #     print("Removed old calibration cache:", CALIBRATION_CACHE_PATH.resolve())

    if DEFAULT_ENGINE_PATH.exists():
        DEFAULT_ENGINE_PATH.unlink()

    model = YOLO(str(MODEL_PATH), task="segment")
    exported_path = Path(
        model.export(
            format="engine",
            imgsz=640,
            batch=1,
            dynamic=False,
            int8=True,
            data=str(DATA_PATH),
            fraction=1.0,
            workspace=4,
            device=0,
            simplify=False,
            nms=False,
        )
    )

    if not exported_path.is_file():
        raise FileNotFoundError(
            f"Export did not create an engine: {exported_path.resolve()}"
        )

    if TARGET_ENGINE_PATH.exists():
        TARGET_ENGINE_PATH.unlink()

    shutil.move(str(exported_path), str(TARGET_ENGINE_PATH))
    print("\nINT8 TensorRT engine ready.")
    print("Engine:", TARGET_ENGINE_PATH.resolve())
    print(
        "Size:",
        f"{TARGET_ENGINE_PATH.stat().st_size / (1024 ** 2):.2f} MB",
    )


if __name__ == "__main__":
    main()
