import os
import sys
import time
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(PROJECT_ROOT, "src", "weights", "best_model.onnx")
PTH_PATH = os.path.join(PROJECT_ROOT, "src", "weights", "best_model.pth")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "src", "output", "3panel_result.mp4")

cv2.setNumThreads(4)

class FastBoxingInference:
    def __init__(self, use_onnx: bool = True):
        self.use_onnx = use_onnx
        self.skeleton = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7)]

        if self.use_onnx and os.path.exists(ONNX_PATH):
            import onnxruntime as ort
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            
            self.session = ort.InferenceSession(
                ONNX_PATH,
                session_options,
                providers=["DmlExecutionProvider", "CPUExecutionProvider"]
            )
            self.input_name = self.session.get_inputs()[0].name
            print(f"🚀 [ONNX DirectML Engine] 할당 디바이스: {self.session.get_providers()}")
        else:
            import torch
            from models.pose_net import BoxingPose3DNet
            self.use_onnx = False
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = BoxingPose3DNet(num_keypoints=8, depth_res=64).to(self.device)
            self.model.load_state_dict(torch.load(PTH_PATH, map_location=self.device))
            self.model.eval()
            print(f"🚀 [PyTorch Engine] 할당 디바이스: {self.device}")

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_data = rgb.astype(np.float32) / 255.0
        img_data = (img_data - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        img_data = img_data.transpose(2, 0, 1)
        return np.ascontiguousarray(img_data[np.newaxis, ...], dtype=np.float32)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        input_array = self.preprocess(frame_bgr)

        if self.use_onnx:
            outputs = self.session.run(["coords_3d"], {self.input_name: input_array})
            coords_batch: np.ndarray = outputs[0]  # type: ignore
            return coords_batch[0].astype(np.float32)
        else:
            import torch
            input_tensor = torch.from_numpy(input_array).to(self.device)
            with torch.no_grad():
                coords_3d, _ = self.model(input_tensor)
            return coords_3d.squeeze(0).cpu().numpy()

    def draw_front_view_fast(self, canvas_shape: tuple, coords_3d: np.ndarray) -> np.ndarray:
        h, w = canvas_shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        points = [(int(cx + p[0] * w * 0.4), int(cy + p[1] * h * 0.4)) for p in coords_3d]

        for s, e in self.skeleton:
            cv2.line(canvas, points[s], points[e], (0, 255, 0), 2)
        for idx, pt in enumerate(points):
            color = (0, 0, 255) if idx in (4, 7) else (0, 255, 255)
            cv2.circle(canvas, pt, 5, color, -1)

        cv2.putText(canvas, "3D Front View (X-Y)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas

    def draw_side_view_fast(self, canvas_shape: tuple, coords_3d: np.ndarray) -> np.ndarray:
        h, w = canvas_shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        points = [(int(cx + p[2] * w * 0.4), int(cy + p[1] * h * 0.4)) for p in coords_3d]

        for s, e in self.skeleton:
            cv2.line(canvas, points[s], points[e], (0, 165, 255), 2)
        for idx, pt in enumerate(points):
            color = (0, 0, 255) if idx in (4, 7) else (255, 255, 0)
            cv2.circle(canvas, pt, 5, color, -1)

        cv2.putText(canvas, "3D Side View (Z-Y)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return canvas

    def run(self, video_path: str = "boxing_sample.mp4", display_scale: float = 1.0, save_video: bool = False) -> None:
        video_path = os.path.join(PROJECT_ROOT, video_path) if not os.path.isabs(video_path) else video_path
        use_webcam = not os.path.exists(video_path)
        cap = cv2.VideoCapture(0 if use_webcam else video_path)

        if not cap.isOpened():
            raise RuntimeError("❌ 비디오 또는 웹캠 장치를 열 수 없습니다.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        TARGET_WIDTH = 1920
        PANEL_W = TARGET_WIDTH // 3
        PANEL_H = int(PANEL_W * (h / w))
        if PANEL_H % 2 != 0:
            PANEL_H += 1

        writer = None
        if save_video:
            base_dir = os.path.dirname(OUTPUT_PATH)
            os.makedirs(base_dir, exist_ok=True)

            filename, ext = os.path.splitext(os.path.basename(OUTPUT_PATH))
            final_output_path = OUTPUT_PATH
            counter = 1

            while os.path.exists(final_output_path):
                final_output_path = os.path.join(base_dir, f"{filename}_{counter}{ext}")
                counter += 1

            fourcc = getattr(cv2, 'VideoWriter_fourcc')(*'mp4v') # type: ignore
            writer = cv2.VideoWriter(final_output_path, fourcc, fps, (TARGET_WIDTH, PANEL_H))
            print(f"💾 저장 지정: {final_output_path}")

        prev_time = time.time()
        curr_fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if use_webcam:
                    frame = cv2.flip(frame, 1)

                coords_3d = self.predict(frame)

                panel1 = cv2.resize(frame, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)
                cv2.putText(panel1, "Original Input", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                panel2 = self.draw_front_view_fast((PANEL_H, PANEL_W), coords_3d)
                panel3 = self.draw_side_view_fast((PANEL_H, PANEL_W), coords_3d)

                combined = np.hstack((panel1, panel2, panel3))

                if writer is not None:
                    writer.write(combined)

                now = time.time()
                time_diff = now - prev_time
                prev_time = now
                curr_fps = 0.9 * curr_fps + 0.1 * (1.0 / (time_diff + 1e-6)) if curr_fps > 0 else 1.0 / (time_diff + 1e-6)

                disp_frame = cv2.resize(combined, (int(TARGET_WIDTH * display_scale), int(PANEL_H * display_scale)), interpolation=cv2.INTER_NEAREST) if display_scale != 1.0 else combined
                cv2.putText(disp_frame, f"FPS: {curr_fps:.1f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Boxing AI 3-Panel Stream", disp_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    FastBoxingInference(use_onnx=True).run(display_scale=1.0, save_video=True)