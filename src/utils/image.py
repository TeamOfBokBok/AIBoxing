"""이미지 입력에 공통으로 적용하는 전처리 유틸리티입니다."""

import cv2
import numpy as np

TARGET_ASPECT_RATIO = 16 / 9


def letterbox(
    image: np.ndarray,
    target_aspect_ratio: float = TARGET_ASPECT_RATIO,
    color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """종횡비를 유지한 채 검정 여백을 추가해 목표 종횡비로 맞춥니다."""
    if image.ndim != 3 or image.shape[1] <= 0 or image.shape[0] <= 0:
        raise ValueError("유효한 컬러 이미지가 필요합니다.")
    if target_aspect_ratio <= 0:
        raise ValueError("target_aspect_ratio는 0보다 커야 합니다.")

    height, width = image.shape[:2]
    current_aspect_ratio = width / height

    if current_aspect_ratio < target_aspect_ratio:
        target_width = max(width, int(round(height * target_aspect_ratio)))
        target_height = height
        pad_left = (target_width - width) // 2
        pad_right = target_width - width - pad_left
        return cv2.copyMakeBorder(
            image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color
        )

    target_width = width
    target_height = max(height, int(round(width / target_aspect_ratio)))
    pad_top = (target_height - height) // 2
    pad_bottom = target_height - height - pad_top
    return cv2.copyMakeBorder(
        image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=color
    )
