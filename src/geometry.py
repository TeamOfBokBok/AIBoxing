"""Shared coordinate and image preprocessing rules."""

import cv2
import numpy as np

TARGET_ASPECT_RATIO = 16 / 9


def letterbox(image: np.ndarray, color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("유효한 컬러 이미지가 필요합니다.")
    height, width = image.shape[:2]
    target_width = max(width, int(round(height * TARGET_ASPECT_RATIO)))
    target_height = max(height, int(round(width / TARGET_ASPECT_RATIO)))
    pad_left = (target_width - width) // 2
    pad_right = target_width - width - pad_left
    pad_top = (target_height - height) // 2
    pad_bottom = target_height - height - pad_top
    return cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                              cv2.BORDER_CONSTANT, value=color)


def normalize_keypoints(keypoints: np.ndarray, waist_index: int = 8) -> np.ndarray:
    points = np.asarray(keypoints, dtype=np.float32)
    if points.shape != (9, 3):
        raise ValueError(f"keypoints shape must be (9, 3), got {points.shape}")
    return np.clip(points - points[waist_index], -1.0, 1.0)


def preprocess_image(image_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    image = cv2.resize(letterbox(image_bgr), (size, size), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32)
    return np.ascontiguousarray(image.transpose(2, 0, 1))
