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

    def draw_skeleton_front(self, frame_shape: tuple, coords_3d: np.ndarray) -> np.ndarray:
        """[중앙] 정면 3D 스켈레톤 시각화 (X, Y 좌표 기반)"""
        h, w, _ = frame_shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        neck_x_px, neck_y_px = w // 2, h // 2

        points_2d = []
        for i in range(8):
            px = int(neck_x_px + coords_3d[i][0] * (w * 0.4))
            py = int(neck_y_px + coords_3d[i][1] * (h * 0.4))
            points_2d.append((px, py))

        # 뼈대 선 및 관절 그리기
        for p1_idx, p2_idx in self.skeleton:
            cv2.line(canvas, points_2d[p1_idx], points_2d[p2_idx], (0, 255, 0), 3)

        for idx, pt in enumerate(points_2d):
            color = (0, 0, 255) if idx in [4, 7] else (0, 255, 255)
            cv2.circle(canvas, pt, 6, color, -1)

        cv2.putText(canvas, "3D Front View (X-Y)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return canvas

    def draw_skeleton_side(self, frame_shape: tuple, coords_3d: np.ndarray) -> np.ndarray:
        """[우측] 측면 3D 스켈레톤 시각화 (Z-깊이, Y 좌표 기반)"""
        h, w, _ = frame_shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        neck_x_px, neck_y_px = w // 2, h // 2

        points_2d = []
        for i in range(8):
            # Z축(깊이)을 화면 가로축으로 투영 (우측이 전방)
            px = int(neck_x_px + coords_3d[i][2] * (w * 0.4))
            py = int(neck_y_px + coords_3d[i][1] * (h * 0.4))
            points_2d.append((px, py))

        # 뼈대 선 및 관절 그리기 (측면은 주황색 뼈대로 구별)
        for p1_idx, p2_idx in self.skeleton:
            cv2.line(canvas, points_2d[p1_idx], points_2d[p2_idx], (0, 165, 255), 3)

        for idx, pt in enumerate(points_2d):
            color = (0, 0, 255) if idx in [4, 7] else (255, 255, 0)
            cv2.circle(canvas, pt, 6, color, -1)

        cv2.putText(canvas, "3D Side View (Z-Y)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return canvas

    def run_stream(self, video_path: str = "boxing_sample.mp4", display_scale: float = 0.25):
        """3분할(Original | Front 3D | Side 3D) 시각화 추론 실행"""
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

        # 1. 저장용 동영상 파일 설정 (가로 3배 3분할 고해상도)
        os.makedirs("src/output", exist_ok=True)
        out_path = "src/output/3panel_result.mp4"
        fourcc = getattr(cv2, 'VideoWriter_fourcc')(*'mp4v') # type: ignore
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (w * 3, h))

        # 2. 화면 디스플레이용 크기 계산 (가로 3배 반영 후 display_scale 축소)
        display_w = int(w * 3 * display_scale)
        display_h = int(h * display_scale)

        print(f"💡 실시간 3분할 화면 창이 열립니다 ({display_w}x{display_h}). 종료하려면 'q' 키를 누르세요.")

        frame_cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if use_webcam:
                frame = cv2.flip(frame, 1)

            # 3D 좌표 추론
            coords_3d = self.predict_image(frame)

            # 1칸: 원본 입력
            panel1 = frame.copy()
            cv2.putText(panel1, "Original Input", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 2칸: 정면 스켈레톤 (X-Y)
            panel2 = self.draw_skeleton_front(frame.shape, coords_3d)

            # 3칸: 측면 스켈레톤 (Z-Y)
            panel3 = self.draw_skeleton_side(frame.shape, coords_3d)

            # 가로 방향 3분할 병합 (3-Panel Horizontal Stack)
            combined_frame = np.hstack((panel1, panel2, panel3))

            # 고해상도 비디오 저장
            out_writer.write(combined_frame)

            # 모니터 디스플레이용 축소
            display_frame = cv2.resize(combined_frame, (display_w, display_h))
            cv2.imshow("Boxing AI - [Left: Input | Center: Front 3D | Right: Side 3D]", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_cnt += 1

        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()

        print(f"\n🎉 3분할 추론 완료! 총 {frame_cnt}개 프레임 추론 완료.")
        print(f"  --> 💾 저장 파일: {out_path}")


if __name__ == "__main__":
    inferencer = RealtimePoseInference(weight_path="src/weights/best_model.pth")
    # 3분할 가로 폭을 고려해 25% 비율(0.25)로 축소 디스플레이
    inferencer.run_stream(video_path="boxing_sample.mp4", display_scale=0.25)