"""Generate real-image pseudo labels with MediaPipe."""

import json
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import ANNOTATIONS_PATH, IMAGE_SIZE, MODEL_PATH, REAL_IMAGE_DIR, REAL_VIDEO_DIR
from geometry import letterbox, normalize_keypoints


class MediaPipeLabeler:
    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe 모델이 없습니다: {model_path}")
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    def _smooth_sequence(items: list[dict]) -> list[dict]:
        """Suppress isolated landmark jumps without removing genuine motion."""
        if len(items) < 3:
            return items
        raw = [np.asarray(item["keypoints_3d"], dtype="float32")
               for item in items]
        filtered = [points.copy() for points in raw]
        for frame_index in range(1, len(items) - 1):
            local = np.median(np.stack(raw[frame_index - 1:frame_index + 2]), axis=0)
            jump = np.linalg.norm(raw[frame_index] - local, axis=1)
            neighbor_span = np.linalg.norm(
                raw[frame_index - 1] - raw[frame_index + 1], axis=1)
            isolated_jump = (jump > 0.18) & (neighbor_span < 0.18)
            filtered[frame_index][isolated_jump] = local[isolated_jump]
        for item, points in zip(items, filtered):
            item["keypoints_3d"] = points.tolist()
        return items

    @staticmethod
    def _repair_side_swap(raw: list[list[float]], points_2d: list[list[float]],
                          visibility: list[float]) -> tuple[list[str], bool]:
        """Repair an unambiguous left/right swap using separated shoulders."""
        neck_x = points_2d[1][0]
        right_shoulder_x = points_2d[2][0]
        left_shoulder_x = points_2d[5][0]
        if abs(right_shoulder_x - left_shoulder_x) < 0.08:
            return [], True

        right_is_left = right_shoulder_x < left_shoulder_x
        repaired = []
        side_consistent = True
        for joint_name, right_index, left_index in (
            ("elbow", 3, 6), ("wrist", 4, 7)
        ):
            right_point, left_point = points_2d[right_index], points_2d[left_index]
            right_on_expected_side = ((right_point[0] < neck_x) if right_is_left
                                      else (right_point[0] > neck_x))
            left_on_expected_side = ((left_point[0] > neck_x) if right_is_left
                                     else (left_point[0] < neck_x))
            swapped_is_valid = (((left_point[0] < neck_x) and
                                 (right_point[0] > neck_x)) if right_is_left
                                else ((left_point[0] > neck_x) and
                                      (right_point[0] < neck_x)))
            if not (right_on_expected_side and left_on_expected_side) and swapped_is_valid:
                raw[right_index], raw[left_index] = raw[left_index], raw[right_index]
                points_2d[right_index], points_2d[left_index] = (
                    points_2d[left_index], points_2d[right_index])
                visibility[right_index], visibility[left_index] = (
                    visibility[left_index], visibility[right_index])
                repaired.append(joint_name)
            elif not (right_on_expected_side and left_on_expected_side):
                side_consistent = False
        return repaired, side_consistent

    def label_videos(self, sample_rate: int = 5) -> int:
        REAL_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        REAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        annotations = []
        if ANNOTATIONS_PATH.exists() and ANNOTATIONS_PATH.stat().st_size > 0:
            with open(ANNOTATIONS_PATH, encoding="utf-8") as file:
                try:
                    annotations = json.load(file)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"annotation JSON이 손상되었습니다: {ANNOTATIONS_PATH}"
                    ) from error
        existing = {item.get("image_path") for item in annotations}
        landmark_ids = {
            "nose": 0, "r_shoulder": 12, "r_elbow": 14, "r_wrist": 16,
            "l_shoulder": 11, "l_elbow": 13, "l_wrist": 15,
            "r_hip": 24, "l_hip": 23,
        }
        for video_path in sorted(REAL_VIDEO_DIR.iterdir()):
            if video_path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
                continue
            capture = cv2.VideoCapture(str(video_path))
            frame_index = 0
            video_annotations = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_rate:
                    frame_index += 1
                    continue
                boxed = letterbox(frame)
                timestamp_ms = int(frame_index * 1000 / max(
                    capture.get(cv2.CAP_PROP_FPS), 1.0))
                result = self.detector.detect_for_video(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)), timestamp_ms)
                if result.pose_world_landmarks:
                    landmarks = result.pose_world_landmarks[0]
                    image_landmarks = result.pose_landmarks[0]
                    r_shoulder = landmarks[landmark_ids["r_shoulder"]]
                    l_shoulder = landmarks[landmark_ids["l_shoulder"]]
                    r_hip = landmarks[landmark_ids["r_hip"]]
                    l_hip = landmarks[landmark_ids["l_hip"]]
                    raw = [
                        [landmarks[landmark_ids["nose"]].x,
                         landmarks[landmark_ids["nose"]].y,
                         landmarks[landmark_ids["nose"]].z],
                        [(r_shoulder.x + l_shoulder.x) / 2,
                         (r_shoulder.y + l_shoulder.y) / 2,
                         (r_shoulder.z + l_shoulder.z) / 2],
                        [r_shoulder.x, r_shoulder.y, r_shoulder.z],
                        [landmarks[landmark_ids["r_elbow"]].x,
                         landmarks[landmark_ids["r_elbow"]].y,
                         landmarks[landmark_ids["r_elbow"]].z],
                        [landmarks[landmark_ids["r_wrist"]].x,
                         landmarks[landmark_ids["r_wrist"]].y,
                         landmarks[landmark_ids["r_wrist"]].z],
                        [l_shoulder.x, l_shoulder.y, l_shoulder.z],
                        [landmarks[landmark_ids["l_elbow"]].x,
                         landmarks[landmark_ids["l_elbow"]].y,
                         landmarks[landmark_ids["l_elbow"]].z],
                        [landmarks[landmark_ids["l_wrist"]].x,
                         landmarks[landmark_ids["l_wrist"]].y,
                         landmarks[landmark_ids["l_wrist"]].z],
                        [(r_hip.x + l_hip.x) / 2,
                         (r_hip.y + l_hip.y) / 2,
                         (r_hip.z + l_hip.z) / 2],
                    ]
                    r_shoulder_2d = image_landmarks[landmark_ids["r_shoulder"]]
                    l_shoulder_2d = image_landmarks[landmark_ids["l_shoulder"]]
                    r_hip_2d = image_landmarks[landmark_ids["r_hip"]]
                    l_hip_2d = image_landmarks[landmark_ids["l_hip"]]
                    points_2d = [
                        [image_landmarks[landmark_ids["nose"]].x,
                         image_landmarks[landmark_ids["nose"]].y],
                        [(r_shoulder_2d.x + l_shoulder_2d.x) / 2,
                         (r_shoulder_2d.y + l_shoulder_2d.y) / 2],
                        [r_shoulder_2d.x, r_shoulder_2d.y],
                        [image_landmarks[landmark_ids["r_elbow"]].x,
                         image_landmarks[landmark_ids["r_elbow"]].y],
                        [image_landmarks[landmark_ids["r_wrist"]].x,
                         image_landmarks[landmark_ids["r_wrist"]].y],
                        [l_shoulder_2d.x, l_shoulder_2d.y],
                        [image_landmarks[landmark_ids["l_elbow"]].x,
                         image_landmarks[landmark_ids["l_elbow"]].y],
                        [image_landmarks[landmark_ids["l_wrist"]].x,
                         image_landmarks[landmark_ids["l_wrist"]].y],
                        [(r_hip_2d.x + l_hip_2d.x) / 2,
                         (r_hip_2d.y + l_hip_2d.y) / 2],
                    ]
                    visibility = [float(getattr(landmarks[i], "visibility", 1.0) or 0.0)
                                  for i in (
                                      landmark_ids["nose"], landmark_ids["r_shoulder"],
                                      landmark_ids["r_shoulder"], landmark_ids["r_elbow"],
                                      landmark_ids["r_wrist"], landmark_ids["l_shoulder"],
                                      landmark_ids["l_elbow"], landmark_ids["l_wrist"],
                                      landmark_ids["r_hip"],
                                  )]
                    visibility[1] = min(visibility[2], visibility[5])
                    visibility[8] = min(
                        float(getattr(landmarks[landmark_ids["r_hip"]], "visibility", 1.0) or 0.0),
                        float(getattr(landmarks[landmark_ids["l_hip"]], "visibility", 1.0) or 0.0),
                    )
                    points = normalize_keypoints(raw)
                    name = f"{video_path.stem}_{frame_index:06d}.jpg"
                    if name not in existing:
                        cv2.imwrite(str(REAL_IMAGE_DIR / name),
                                     cv2.resize(boxed, (IMAGE_SIZE, IMAGE_SIZE)))
                        video_annotations.append({
                            "image_path": name, "keypoints_3d": points.tolist(),
                            "keypoints_2d": points_2d,
                            "visibility": visibility, "source": "real_pseudo",
                            "sequence_id": video_path.stem, "frame_index": frame_index,
                            "side_consistent": True})
                        existing.add(name)
                frame_index += 1
            capture.release()
            annotations.extend(self._smooth_sequence(video_annotations))
        with open(ANNOTATIONS_PATH, "w", encoding="utf-8") as file:
            json.dump(annotations, file, ensure_ascii=False, indent=2)
        return len(annotations)


if __name__ == "__main__":
    print(f"생성된 누적 pseudo-label: {MediaPipeLabeler().label_videos()}개")
