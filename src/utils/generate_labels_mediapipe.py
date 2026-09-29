import os
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MediaPipeLabelGenerator:
    def __init__(self, output_dir="src/data", model_path="pose_landmarker_heavy.task"):
        self.output_dir = output_dir
        self.img_out_dir = os.path.join(output_dir, "images")
        os.makedirs(self.img_out_dir, exist_ok=True)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ MediaPipe 모델 파일('{model_path}')을 찾을 수 없습니다.\n"
                "다음 URL에서 파일 다운로드 후 프로젝트 루트에 놓아주세요:\n"
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            )

        # 최신 MediaPipe Tasks API PoseLandmarker 초기화
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # MediaPipe 인덱스 -> 상체 8개 관절 매핑
        self.mp_map = {
            "nose": 0,
            "r_shoulder": 12,
            "r_elbow": 14,
            "r_wrist": 16,
            "l_shoulder": 11,
            "l_elbow": 13,
            "l_wrist": 15
        }

    def process_video(self, video_path: str, sample_rate: int = 5):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"❌ 영상 파일을 찾을 수 없습니다: {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        saved_count = 0
        annotations = []

        print(f"⏳ 영상 분석 중: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # 최신 추론 API
                detection_result = self.detector.detect(mp_image)

                if detection_result.pose_world_landmarks and len(detection_result.pose_world_landmarks) > 0:
                    landmarks = detection_result.pose_world_landmarks[0]

                    # 1. 8개 관절 3D World Landmarks 추출
                    r_sh = landmarks[self.mp_map["r_shoulder"]]
                    l_sh = landmarks[self.mp_map["l_shoulder"]]

                    # 목(Neck) 위치: 양 어깨의 중점으로 계산
                    neck_x = (r_sh.x + l_sh.x) / 2.0
                    neck_y = (r_sh.y + l_sh.y) / 2.0
                    neck_z = (r_sh.z + l_sh.z) / 2.0

                    keypoints_raw = [
                        [landmarks[self.mp_map["nose"]].x, landmarks[self.mp_map["nose"]].y, landmarks[self.mp_map["nose"]].z],
                        [neck_x, neck_y, neck_z],
                        [r_sh.x, r_sh.y, r_sh.z],
                        [landmarks[self.mp_map["r_elbow"]].x, landmarks[self.mp_map["r_elbow"]].y, landmarks[self.mp_map["r_elbow"]].z],
                        [landmarks[self.mp_map["r_wrist"]].x, landmarks[self.mp_map["r_wrist"]].y, landmarks[self.mp_map["r_wrist"]].z],
                        [l_sh.x, l_sh.y, l_sh.z],
                        [landmarks[self.mp_map["l_elbow"]].x, landmarks[self.mp_map["l_elbow"]].y, landmarks[self.mp_map["l_elbow"]].z],
                        [landmarks[self.mp_map["l_wrist"]].x, landmarks[self.mp_map["l_wrist"]].y, landmarks[self.mp_map["l_wrist"]].z]
                    ]

                    # 2. Neck(목) 중심 상대 좌표계로 정규화 및 [-1, 1] 범위 클리핑
                    keypoints_norm = []
                    for kp in keypoints_raw:
                        rel_x = round(kp[0] - neck_x, 4)
                        rel_y = round(kp[1] - neck_y, 4)
                        rel_z = round(kp[2] - neck_z, 4)
                        keypoints_norm.append([
                            max(-1.0, min(1.0, rel_x)),
                            max(-1.0, min(1.0, rel_y)),
                            max(-1.0, min(1.0, rel_z))
                        ])

                    # 3. 프레임 이미지 및 라벨 등록
                    img_filename = f"frame_{saved_count:05d}.jpg"
                    img_save_path = os.path.join(self.img_out_dir, img_filename)
                    cv2.imwrite(img_save_path, frame)

                    annotations.append({
                        "image_path": img_filename,
                        "keypoints_3d": keypoints_norm
                    })
                    saved_count += 1

            frame_idx += 1

        cap.release()

        # 4. annotations_3d.json 저장
        json_out_path = os.path.join(self.output_dir, "annotations_3d.json")
        with open(json_out_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2)

        print(f"\n🎉 [MediaPipe 라벨링 완료] 총 {saved_count}개 프레임 구축 완료!")
        print(f"- 이미지 저장 경로: {self.img_out_dir}")
        print(f"- JSON 저장 경로: {json_out_path}")


if __name__ == "__main__":
    generator = MediaPipeLabelGenerator(output_dir="src/data")
    print("✅ 최신 MediaPipe Tasks API 기반 라벨 생성기가 준비되었습니다.")