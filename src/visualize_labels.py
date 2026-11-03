"""Visualize MediaPipe labels as original, X-Y, and Z-Y views."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from config import ANNOTATIONS_PATH, IMAGE_SIZE, OUTPUT_DIR, REAL_IMAGE_DIR
SKELETON = (
    (0, 1),  # nose-neck
    (1, 2), (2, 3), (3, 4),  # right shoulder-elbow-wrist
    (1, 5), (5, 6), (6, 7),  # left shoulder-elbow-wrist
    (1, 8),  # neck-waist
)
JOINT_NAMES = (
    "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist", "waist",
)


def _draw_pose(canvas: np.ndarray, points: np.ndarray, visibility: np.ndarray,
               horizontal_axis: int, title: str) -> np.ndarray:
    height, width = canvas.shape[:2]
    center_x, center_y = width // 2, height // 2
    projected = []
    for point in points:
        horizontal = float(point[horizontal_axis])
        vertical = float(point[1])
        projected.append((int(center_x + horizontal * width * 0.48),
                          int(center_y + vertical * height * 0.48)))

    for start, end in SKELETON:
        if visibility[start] > 0.05 and visibility[end] > 0.05:
            cv2.line(canvas, projected[start], projected[end], (0, 190, 255), 2)
    for index, (x, y) in enumerate(projected):
        if visibility[index] <= 0.05:
            continue
        color = (0, 0, 255) if index in (4, 7) else (0, 255, 0)
        cv2.circle(canvas, (x, y), 7, color, -1)
        cv2.putText(canvas, JOINT_NAMES[index], (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(canvas, title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    return canvas


def render(annotation: dict, image_dir: Path) -> np.ndarray:
    image_path = image_dir / annotation["image_path"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    display_size = IMAGE_SIZE * 2
    image = cv2.resize(image, (display_size, display_size),
                       interpolation=cv2.INTER_NEAREST)
    points = np.asarray(annotation["keypoints_3d"], dtype=np.float32)
    visibility = np.asarray(annotation.get("visibility", [1.0] * 9), dtype=np.float32)
    original = image.copy()
    points_2d = annotation.get("keypoints_2d")
    if points_2d is not None:
        for index, (x_norm, y_norm) in enumerate(points_2d):
            x = int(float(x_norm) * image.shape[1])
            y = int(float(y_norm) * image.shape[0])
            color = (0, 0, 255) if index in (4, 7) else (0, 255, 0)
            cv2.circle(original, (x, y), 7, color, -1)
            cv2.putText(original, JOINT_NAMES[index], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(original, "Original + label", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    if annotation.get("side_consistent") is False:
        cv2.putText(original, "CHECK: side assignment", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
    xy = _draw_pose(np.zeros_like(image), points, visibility, 0, "X-Y (front)")
    zy = _draw_pose(np.zeros_like(image), points, visibility, 2, "Z-Y (depth)")
    return np.hstack((original, xy, zy))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0,
                        help="처음 표시할 annotation index")
    parser.add_argument("--save-video", action="store_true",
                        help="전체 라벨 시각화를 src/output/label_visualization.mp4로 저장")
    parser.add_argument("--no-display", action="store_true",
                        help="창을 열지 않고 저장만 수행")
    args = parser.parse_args()

    with open(ANNOTATIONS_PATH, encoding="utf-8") as file:
        annotations = json.load(file)
    if not annotations:
        raise ValueError(f"annotation이 없습니다: {ANNOTATIONS_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.save_video:
        first = render(annotations[0], REAL_IMAGE_DIR)
        output_path = OUTPUT_DIR / "label_visualization.mp4"
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 10.0, (first.shape[1], first.shape[0]))
        try:
            for annotation in annotations:
                writer.write(render(annotation, REAL_IMAGE_DIR))
        finally:
            writer.release()
        print(f"시각화 영상 저장: {output_path}")

    if args.no_display:
        return

    index = max(0, min(args.index, len(annotations) - 1))
    while True:
        frame = render(annotations[index], REAL_IMAGE_DIR)
        cv2.imshow("AIBoxing Label Visualization", frame)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), 32):
            index = (index + 1) % len(annotations)
        elif key == ord("p"):
            index = (index - 1) % len(annotations)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
