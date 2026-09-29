import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pose_net_3d import BoxingPose3DNet

class RealtimePoseInference:
    def __init__(self, weight_path="src/weights/best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 추론 디바이스: {self.device}")

        # 1. 모델 선언 및 학습 가중치 로드
        self.model = BoxingPose3DNet(num_keypoints=8, depth_res=64).to(self.device)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"❌ 가중치 파일을 찾을 수 없습니다: {weight_path}")

        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        print("✅ 학습된 3D Pose 가중치 로드 완료!")

        # 2. 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 관절 연결 구조 (Bone Index Pair)
        self.skeleton = [
            (0, 1), # Head -> Neck
            (1, 2), # Neck -> R_Shoulder
            (2, 3), # R_Shoulder -> R_Elbow
            (3, 4), # R_Elbow -> R_Wrist
            (1, 5), # Neck -> L_Shoulder
            (5, 6), # L_Shoulder -> L_Elbow
            (6, 7)  # L_Elbow -> L_Wrist
        ]

    def predict_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        """단일 BGR 이미지 Frame에서 3D 관절 좌표 추론 [8, 3]"""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        img_tensor: torch.Tensor = self.transform(pil_img) # type: ignore
        input_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            coords_3d, _ = self.model(input_tensor)

        coords_numpy: np.ndarray = coords_3d.squeeze(0).cpu().numpy()
        return coords_numpy

    def draw_skeleton_canvas(self, frame_shape: tuple, coords_3d: np.ndarray) -> np.ndarray:
        """우측에 단독 디스플레이할 검은색 캔버스 위 3D 스켈레톤 시각화"""
        h, w, _ = frame_shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        neck_x_px, neck_y_px = w // 2, h // 2

        points_2d = []
        for i in range(8):
            px = int(neck_x_px + coords_3d[i][0] * (w * 0.4))
            py = int(neck_y_px + coords_3d[i][1] * (h * 0.4))
            points_2d.append((px, py))

        # 1. 뼈대 선 그리기 (초록색)
        for p1_idx, p2_idx in self.skeleton:
            pt1 = points_2d[p1_idx]
            pt2 = points_2d[p2_idx]
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 3)

        # 2. 관절 원 그리기 (손목: 빨간색, 나머지: 노란색)
        for idx, pt in enumerate(points_2d):
            color = (0, 0, 255) if idx in [4, 7] else (0, 255, 255)
            cv2.circle(canvas, pt, 6, color, -1)

        cv2.putText(canvas, "3D Skeleton Prediction", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return canvas

    def run_stream(self, video_path: str = "boxing_sample.mp4", display_scale: float = 0.25):
        """
        비디오 파일 또는 웹캠 입력으로부터 나란히(Side-by-Side) 시각화 수행
        display_scale: 모니터 디스플레이 표시 배율 (0.25 -> 25% 크기로 축소)
        """
        use_webcam = False

        if os.path.exists(video_path):
            print(f"🎬 지정된 영상 파일로 추론을 시작합니다: {video_path}")
            cap = cv2.VideoCapture(video_path)
        else:
            print("⚠️ 'boxing_sample.mp4' 파일이 없어 웹캠 실시간 모드로 전환합니다.")
            cap = cv2.VideoCapture(0)
            use_webcam = True

        if not cap.isOpened():
            raise RuntimeError("❌ 비디오 또는 웹캠 장치를 열 수 없습니다.")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 1. 저장용 동영상 파일 설정 (고해상도 원본 크기 유지)
        os.makedirs("src/output", exist_ok=True)
        out_path = "src/output/side_by_side_result.mp4"
        fourcc = getattr(cv2, 'VideoWriter_fourcc')(*'mp4v') # type: ignore
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 2, h))

        # 2. 화면 표시용 크기 계산 (25% 축소)
        display_w = int(w * 2 * display_scale)
        display_h = int(h * display_scale)

        print(f"💡 실시간 화면 창이 열립니다 ({display_w}x{display_h}). 종료하려면 'q' 키를 누르세요.")

        frame_cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if use_webcam:
                frame = cv2.flip(frame, 1)

            # 3D 좌표 추론 및 시각화
            coords_3d = self.predict_image(frame)

            left_frame = frame.copy()
            cv2.putText(left_frame, "Original Input", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            right_frame = self.draw_skeleton_canvas(frame.shape, coords_3d)

            # 가로 병합 (Side-by-Side)
            combined_frame = np.hstack((left_frame, right_frame))

            # 파일에는 고해상도 그대로 저장
            out_writer.write(combined_frame)

            # 모니터 디스플레이용으로만 25% Resize 적용
            display_frame = cv2.resize(combined_frame, (display_w, display_h))
            cv2.imshow("Boxing AI - Left: Input | Right: 3D Skeleton", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_cnt += 1

        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()

        print(f"\n🎉 추론 종료! 총 {frame_cnt}개 프레임 추론 완료.")
        print(f"  --> 💾 저장 파일: {out_path}")


if __name__ == "__main__":
    inferencer = RealtimePoseInference(weight_path="src/weights/best_model.pth")
    # display_scale=0.25 (필요시 0.2, 0.15 등으로 더 줄일 수도 있습니다)
    inferencer.run_stream(video_path="boxing_sample.mp4", display_scale=0.25)