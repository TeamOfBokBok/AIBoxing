import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "src", "weights", "best_model.onnx")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "src", "output", "side_by_side_directml.mp4")

# OpenCV 내부 멀티스레딩 활성화
cv2.setNumThreads(4)

class DirectMLPoseInference:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX 모델을 찾을 수 없습니다: {model_path}\n"
                "먼저 `python src/export_onnx.py`를 실행하세요."
            )

        # ONNX 그래프 최적화 옵션 설정
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            session_options,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        print(f"추론 execution providers: {self.session.get_providers()}")
        self.skeleton = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]

    @staticmethod
    def preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        """PIL을 배제하고 OpenCV/NumPy만으로 처리하는 고속 전처리"""
        # 1. 256x256 고속 Resize & RGB 변환
        resized = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 2. Normalization & [H, W, C] -> [C, H, W]
        img_data = rgb.astype(np.float32) / 255.0
        img_data = (img_data - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        img_data = img_data.transpose(2, 0, 1)

        # 3. 메모리 연속성(C-Contiguous) 보장하여 GPU 버스 대기시간 제거
        return np.ascontiguousarray(img_data[np.newaxis, ...], dtype=np.float32)

    def predict_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        # ONNX 추론 수행
        outputs = self.session.run(None, {self.input_name: self.preprocess(frame_bgr)})
        
        # 1. outputs[0]은 3D 관절 좌표 [1, 8, 3] NumPy 배열입니다.
        coords_batch: np.ndarray = outputs[0] # type: ignore
        
        # 2. 첫 번째 배치 요소 [8, 3] 반환
        return coords_batch[0].astype(np.float32)

    def draw_skeleton_front(self, frame_shape: tuple, coords_3d: np.ndarray) -> np.ndarray:
        height, width, _ = frame_shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2
        points = [
            (int(center_x + point[0] * width * 0.4), int(center_y + point[1] * height * 0.4))
            for point in coords_3d
        ]
        for start, end in self.skeleton:
            cv2.line(canvas, points[start], points[end], (0, 255, 0), 3)
        for index, point in enumerate(points):
            color = (0, 0, 255) if index in (4, 7) else (0, 255, 255)
            cv2.circle(canvas, point, 6, color, -1)
        cv2.putText(canvas, "3D Front View (X-Y)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return canvas

    def draw_skeleton_side(self, frame_shape: tuple, coords_3d: np.ndarray) -> np.ndarray:
        height, width, _ = frame_shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        center_x, center_y = width // 2, height // 2
        points = [
            (int(center_x + point[2] * width * 0.4), int(center_y + point[1] * height * 0.4))
            for point in coords_3d
        ]
        for start, end in self.skeleton:
            cv2.line(canvas, points[start], points[end], (0, 165, 255), 3)
        for index, point in enumerate(points):
            color = (0, 0, 255) if index in (4, 7) else (255, 255, 0)
            cv2.circle(canvas, point, 6, color, -1)
        cv2.putText(canvas, "3D Side View (Z-Y)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return canvas

    def run_stream(self, video_path: str = "boxing_sample.mp4", display_scale: float = 0.25, save_video: bool = True) -> None:
        video_path = os.path.join(PROJECT_ROOT, video_path) if not os.path.isabs(video_path) else video_path
        use_webcam = not os.path.exists(video_path)
        cap = cv2.VideoCapture(0 if use_webcam else video_path)
        print("boxing_sample.mp4가 없어 웹캠 모드로 전환합니다." if use_webcam else f"입력 영상: {video_path}")
        if not cap.isOpened():
            raise RuntimeError("비디오 또는 웹캠 장치를 열 수 없습니다.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("입력 영상의 해상도를 확인할 수 없습니다.")

        # 1. 저장용 해상도 설정 (전체 가로 1920px 고정)
        TARGET_TOTAL_WIDTH = 1920
        PANEL_WIDTH = TARGET_TOTAL_WIDTH // 3  # 패널당 가로 640px
        aspect_ratio = height / width
        PANEL_HEIGHT = int(PANEL_WIDTH * aspect_ratio)
        if PANEL_HEIGHT % 2 != 0:
            PANEL_HEIGHT += 1

        writer = None
        if save_video:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            fourcc = getattr(cv2, 'VideoWriter_fourcc')(*'mp4v') # type: ignore
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (TARGET_TOTAL_WIDTH, PANEL_HEIGHT))

        frame_count = 0
        prev_time = time.time()
        curr_fps = 0.0

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                if use_webcam:
                    frame = cv2.flip(frame, 1)

                # 추론 수행
                coords_3d = self.predict_image(frame)

                panel1 = frame.copy()
                cv2.putText(panel1, "Original Input", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                panel2 = self.draw_skeleton_front(frame.shape, coords_3d)
                panel3 = self.draw_skeleton_side(frame.shape, coords_3d)

                p1_resized = cv2.resize(panel1, (PANEL_WIDTH, PANEL_HEIGHT))
                p2_resized = cv2.resize(panel2, (PANEL_WIDTH, PANEL_HEIGHT))
                p3_resized = cv2.resize(panel3, (PANEL_WIDTH, PANEL_HEIGHT))

                combined = np.hstack((p1_resized, p2_resized, p3_resized))

                if writer is not None:
                    writer.write(combined)

                # FPS 실시간 측정
                now = time.time()
                time_diff = now - prev_time
                prev_time = now
                curr_fps = 0.9 * curr_fps + 0.1 * (1.0 / (time_diff + 1e-6)) if curr_fps > 0 else 1.0 / (time_diff + 1e-6)

                display_frame = cv2.resize(combined, (int(1920 * display_scale), int(PANEL_HEIGHT * display_scale)))
                cv2.putText(display_frame, f"FPS: {curr_fps:.1f}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Boxing AI - [Left: Input | Center: Front 3D | Right: Side 3D]", display_frame)
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

        print(f"3분할 추론 완료: {frame_count}개 프레임")
        if save_video:
            print(f"저장 파일: {OUTPUT_PATH}")


if __name__ == "__main__":
    # display_scale=1.0으로 설정하면 가로 1920px 크기 그대로 모니터에 디스플레이됩니다.
    DirectMLPoseInference().run_stream(display_scale=1.0, save_video=True)