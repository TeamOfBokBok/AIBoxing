"""Dataset loader for real pseudo-label, real GT, and synthetic samples."""

import json
import os
from pathlib import Path
import cv2
import torch
from torch.utils.data import Dataset
from geometry import preprocess_image
from schema import parse_annotation


class PoseDataset(Dataset):
    def __init__(self, annotation_path: str | list[str],
                 image_roots: list[str] | None = None,
                 sources: set[str] | None = None):
        self.image_roots = [Path(p) for p in (image_roots or [])]
        paths = [annotation_path] if isinstance(annotation_path, str) else annotation_path
        raw = []
        for path in paths:
            with open(path, encoding="utf-8") as file:
                raw.extend(json.load(file))
        self.annotations = []
        for item in raw:
            annotation = parse_annotation(item)
            if sources is not None and annotation.source not in sources:
                continue
            if self._find_image(annotation.image_path) is not None:
                self.annotations.append(annotation)
        if not self.annotations:
            raise ValueError("사용 가능한 annotation/image 샘플이 없습니다.")

    def _find_image(self, image_path: str) -> Path | None:
        candidate = Path(image_path)
        for root in self.image_roots:
            path = candidate if candidate.is_absolute() else root / candidate
            if path.exists():
                return path
            prefixed_root = root.parent / candidate
            if not candidate.is_absolute() and prefixed_root.exists():
                return prefixed_root
        return None

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        annotation = self.annotations[index]
        path = self._find_image(annotation.image_path)
        if path is None:
            raise FileNotFoundError(annotation.image_path)
        image = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"이미지를 읽을 수 없습니다: {path}")
        return (torch.from_numpy(preprocess_image(image)),
                torch.from_numpy(annotation.keypoints_3d),
                torch.from_numpy(annotation.visibility), annotation.source)
