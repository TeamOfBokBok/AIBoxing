"""공통 경로, 장치 선택, 실행 환경 설정을 제공합니다."""

import importlib.util
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATA_DIR = SRC_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
WEIGHTS_DIR = SRC_ROOT / "weights"
OUTPUT_DIR = SRC_ROOT / "output"
MODEL_PATH = PROJECT_ROOT / "pose_landmarker_heavy.task"
PYTORCH_WEIGHTS_PATH = WEIGHTS_DIR / "best_model.pth"
ONNX_WEIGHTS_PATH = WEIGHTS_DIR / "best_model.onnx"


@dataclass(frozen=True)
class RuntimeConfig:
    """현재 실행 환경에서 사용할 PyTorch 장치와 DataLoader 옵션입니다."""

    device: Any
    backend: str
    loader_kwargs: dict[str, Any]


def resolve_runtime() -> RuntimeConfig:
    """CUDA/ROCm, DirectML, CPU 순서로 사용 가능한 학습 장치를 선택합니다."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "cuda/rocm"
        loader_kwargs = {
            "num_workers": min(4, os.cpu_count() or 2),
            "pin_memory": True,
            "persistent_workers": True,
            "non_blocking": True,
        }
    elif importlib.util.find_spec("torch_directml") is not None:
        try:
            torch_dml = __import__("torch_directml")
            if not torch_dml.is_available():
                raise RuntimeError("DirectML is unavailable")
            device = torch_dml.device()
            backend = "directml"
            loader_kwargs = {
                "num_workers": min(2, os.cpu_count() or 2),
                "pin_memory": False,
                "persistent_workers": True,
                "non_blocking": False,
            }
        except Exception:
            device = torch.device("cpu")
            backend = "cpu"
            loader_kwargs = _cpu_loader_kwargs()
    else:
        device = torch.device("cpu")
        backend = "cpu"
        loader_kwargs = _cpu_loader_kwargs()

    if mp.current_process().name == "MainProcess":
        device_name = str(device)
        if backend == "cuda/rocm":
            device_name = f"{device_name} ({torch.cuda.get_device_name(0)})"
        print(f"[Runtime] backend={backend}, device={device_name}")
    return RuntimeConfig(device, backend, loader_kwargs)


def _cpu_loader_kwargs() -> dict[str, Any]:
    """CPU 환경에서 프로세스와 메모리 오버헤드를 최소화한 옵션을 반환합니다."""
    import torch

    torch.set_num_threads(os.cpu_count() or 4)
    return {
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "non_blocking": False,
    }


def available_onnx_providers() -> list[str]:
    """설치된 ONNX Runtime provider 중 안정적인 우선순위를 반환합니다."""
    try:
        import onnxruntime as ort
    except ImportError:
        return []

    available = set(ort.get_available_providers())
    priority = ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider")
    return [provider for provider in priority if provider in available]
