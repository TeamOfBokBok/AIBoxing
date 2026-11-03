"""Project paths and runtime settings."""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = SRC_ROOT / "data"
REAL_VIDEO_DIR = DATA_DIR / "videos"
REAL_IMAGE_DIR = DATA_DIR / "real_images"
SYNTHETIC_IMAGE_DIR = DATA_DIR / "synthetic_images"
ANNOTATIONS_PATH = DATA_DIR / "annotations.json"
MODEL_PATH = PROJECT_ROOT / "pose_landmarker_heavy.task"
WEIGHTS_DIR = SRC_ROOT / "weights"
PYTORCH_WEIGHTS_PATH = WEIGHTS_DIR / "best_model.pth"
ONNX_WEIGHTS_PATH = WEIGHTS_DIR / "best_model.onnx"
OUTPUT_DIR = SRC_ROOT / "output"
IMAGE_SIZE = 256
NUM_KEYPOINTS = 9
JOINT_NAMES = ("nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
                "l_shoulder", "l_elbow", "l_wrist", "waist")
TRAIN_BATCH_SIZE = int(os.environ.get("AI_BOXING_BATCH_SIZE", "4"))
EPOCHS = int(os.environ.get("AI_BOXING_EPOCHS", "20"))
if TRAIN_BATCH_SIZE < 1 or EPOCHS < 1:
    raise ValueError("AI_BOXING_BATCH_SIZE와 AI_BOXING_EPOCHS는 1 이상이어야 합니다.")


@dataclass(frozen=True)
class RuntimeConfig:
    device: Any
    backend: str
    loader_kwargs: dict[str, Any]


def _cpu_loader_kwargs() -> dict[str, Any]:
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    return {"num_workers": 0, "pin_memory": False,
            "persistent_workers": False, "non_blocking": False}


def resolve_runtime() -> RuntimeConfig:
    import torch
    if torch.cuda.is_available():
        return RuntimeConfig(torch.device("cuda"), "cuda",
                              {"num_workers": 0 if os.name == "nt" else 2,
                               "pin_memory": True, "persistent_workers": False,
                               "non_blocking": True})
    if importlib.util.find_spec("torch_directml") is not None:
        try:
            directml = __import__("torch_directml")
            if directml.is_available():
                return RuntimeConfig(directml.device(), "directml",
                                      {"num_workers": 0, "pin_memory": False,
                                       "persistent_workers": False,
                                       "non_blocking": False})
        except (ImportError, RuntimeError):
            pass
    return RuntimeConfig(torch.device("cpu"), "cpu", _cpu_loader_kwargs())


def available_onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    available = set(ort.get_available_providers())
    return [name for name in ("CUDAExecutionProvider", "DmlExecutionProvider",
                              "CPUExecutionProvider") if name in available]
