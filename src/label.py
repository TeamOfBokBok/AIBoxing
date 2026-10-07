import os
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import DATA_DIR, MODEL_PATH

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PoseLabelGenerator:
    """MediaPipe로 영상 프레임의 상체 3D 관절 라벨을 생성합니다."""

    def __init__(self, data_dir: str = str(DATA_DIR), video_dir: str = str(DATA_DIR / "videos"), model_path: str = str(MODEL_PATH)):
        self.output_dir = os.path.join(PROJECT_ROOT, data_dir) if not os.path.isabs(data_dir) else data_dir
        self.video_dir = os.path.join(PROJECT_ROOT, video_dir) if not os.path.isabs(video_dir) else video_dir
        self.img_out_dir = os.path.join(self.output_dir, "images")
        
        os.makedirs(self.img_out_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        model_full_path = os.path.join(PROJECT_ROOT, model_path) if not os.path.isabs(model_path) else model_path
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"❌ MediaPipe 모델 파일('{model_full_path}')을 찾을 수 없습니다.")

        base_options = python.BaseOptions(model_asset_path=model_full_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.mp_map = {
            "nose": 0, "r_shoulder": 12, "r_elbow": 14, "r_wrist": 16,
            "l_shoulder": 11, "l_elbow": 13, "l_wrist": 15, "r_hip": 24, "l_hip": 23
        }

    def process_all_new_videos(self, sample_rate: int = 5):
        """아직 처리하지 않은 영상만 샘플링해 이미지와 JSON 라벨을 누적합니다."""
        json_out_path = os.path.join(self.output_dir, "annotations.json")
        processed_log_path = os.path.join(self.output_dir, "processed_videos.json")

        annotations = []
        if os.path.exists(json_out_path):
            with open(json_out_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)

        processed_videos = set()
        if os.path.exists(processed_log_path):
            with open(processed_log_path, 'r', encoding='utf-8') as f:
                processed_videos = set(json.load(f))

        video_files = [f for f in os.listdir(self.video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        new_videos = [f for f in video_files if f not in processed_videos]

        if not new_videos:
            print("✨ 새롭게 추가된 영상이 없습니다. 기존 라벨을 유지합니다.")
            return

        print(f"🎬 신규 처리 대상 영상: {len(new_videos)}개 {new_videos}")

        for video_name in new_videos:
            video_stem = os.path.splitext(video_name)[0]
            video_full_path = os.path.join(self.video_dir, video_name)
            
            cap = cv2.VideoCapture(video_full_path)
            frame_idx = 0
            saved_in_video = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    detection_result = self.detector.detect(mp_image)

                    if detection_result.pose_world_landmarks and len(detection_result.pose_world_landmarks) > 0:
                        landmarks = detection_result.pose_world_landmarks[0]
                        r_sh = landmarks[self.mp_map["r_shoulder"]]
                        l_sh = landmarks[self.mp_map["l_shoulder"]]

                        neck_x = (r_sh.x + l_sh.x) / 2.0
                        neck_y = (r_sh.y + l_sh.y) / 2.0
                        neck_z = (r_sh.z + l_sh.z) / 2.0

                        r_hip = landmarks[self.mp_map["r_hip"]]
                        l_hip = landmarks[self.mp_map["l_hip"]]

                        waist_x = (r_hip.x + l_hip.x) / 2.0
                        waist_y = (r_hip.y + l_hip.y) / 2.0
                        waist_z = (r_hip.z + l_hip.z) / 2.0

                        keypoints_raw = [
                            [landmarks[self.mp_map["nose"]].x, landmarks[self.mp_map["nose"]].y, landmarks[self.mp_map["nose"]].z],
                            [neck_x, neck_y, neck_z],
                            [r_sh.x, r_sh.y, r_sh.z],
                            [landmarks[self.mp_map["r_elbow"]].x, landmarks[self.mp_map["r_elbow"]].y, landmarks[self.mp_map["r_elbow"]].z],
                            [landmarks[self.mp_map["r_wrist"]].x, landmarks[self.mp_map["r_wrist"]].y, landmarks[self.mp_map["r_wrist"]].z],
                            [l_sh.x, l_sh.y, l_sh.z],
                            [landmarks[self.mp_map["l_elbow"]].x, landmarks[self.mp_map["l_elbow"]].y, landmarks[self.mp_map["l_elbow"]].z],
                            [landmarks[self.mp_map["l_wrist"]].x, landmarks[self.mp_map["l_wrist"]].y, landmarks[self.mp_map["l_wrist"]].z],
                            [waist_x, waist_y, waist_z]
                        ]

                        keypoints_norm = [
                            [
                                max(-1.0, min(1.0, round(kp[0] - waist_x, 4))),
                                max(-1.0, min(1.0, round(kp[1] - waist_y, 4))),
                                max(-1.0, min(1.0, round(kp[2] - waist_z, 4)))
                            ]
                            for kp in keypoints_raw
                        ]

                        img_filename = f"{video_stem}_frame_{saved_in_video:05d}.jpg"
                        cv2.imwrite(os.path.join(self.img_out_dir, img_filename), frame)

                        annotations.append({"image_path": img_filename, "keypoints_3d": keypoints_norm})
                        saved_in_video += 1

                frame_idx += 1

            cap.release()
            processed_videos.add(video_name)
            print(f"  └─ {video_name}: {saved_in_video}개 프레임 추출 완료")

        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)

        with open(processed_log_path, 'w', encoding='utf-8') as f:
            json.dump(list(processed_videos), f, indent=2)

        print(f"\n🎉 전체 라벨링 업데이트 완료! 누적 샘플 수: {len(annotations)}개 ({json_out_path})")

if __name__ == "__main__":
    PoseLabelGenerator().process_all_new_videos(sample_rate=5)