"""Annotation schema shared by real pseudo-labels and synthetic GT."""

from dataclasses import dataclass
from typing import Any
import numpy as np

SOURCES = {"real_pseudo", "synthetic_gt", "real_gt"}


@dataclass(frozen=True)
class Annotation:
    image_path: str
    keypoints_3d: np.ndarray
    visibility: np.ndarray
    source: str
    sequence_id: str
    frame_index: int

    def to_dict(self) -> dict[str, Any]:
        if self.source not in SOURCES:
            raise ValueError(f"지원하지 않는 annotation source: {self.source}")
        return {"image_path": self.image_path,
                "keypoints_3d": self.keypoints_3d.astype(float).tolist(),
                "visibility": self.visibility.astype(float).tolist(),
                "source": self.source, "sequence_id": self.sequence_id,
                "frame_index": self.frame_index}


def parse_annotation(item: dict[str, Any]) -> Annotation:
    points = np.asarray(item["keypoints_3d"], dtype=np.float32)
    visibility = np.asarray(item.get("visibility", [1.0] * 9), dtype=np.float32)
    source = str(item.get("source", "real_pseudo"))
    if points.shape != (9, 3) or visibility.shape != (9,):
        raise ValueError("annotation은 keypoints_3d=(9,3), visibility=(9)이어야 합니다.")
    if source not in SOURCES:
        raise ValueError(f"지원하지 않는 annotation source: {source}")
    return Annotation(str(item["image_path"]), points, np.clip(visibility, 0, 1),
                      source, str(item.get("sequence_id", "unknown")),
                      int(item.get("frame_index", -1)))
