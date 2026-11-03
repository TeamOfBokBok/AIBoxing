"""Run real-time relative 3D pose inference."""

import os
import sys
import cv2
import numpy as np
from config import PYTORCH_WEIGHTS_PATH, resolve_runtime
from geometry import preprocess_image
from models.pose_net import BoxingPose3DNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_labels import _draw_pose

DISPLAY_SIZE = 512


class PoseInference:
    def __init__(self):
        import torch
        self.torch = torch
        runtime = resolve_runtime()
        self.device = runtime.device
        self.model = BoxingPose3DNet(pretrained=False).to(self.device)
        if not PYTORCH_WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"가중치가 없습니다: {PYTORCH_WEIGHTS_PATH}")
        self.model.load_state_dict(torch.load(PYTORCH_WEIGHTS_PATH,
                                              map_location=self.device,
                                              weights_only=True))
        self.model.eval()

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        image = self.torch.from_numpy(preprocess_image(frame_bgr)).unsqueeze(0)
        with self.torch.no_grad():
            coords, _ = self.model(image.to(self.device))
        return coords[0].cpu().numpy()


def run(source: str = "0") -> None:
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise RuntimeError(f"입력을 열 수 없습니다: {source}")
    predictor = PoseInference()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        coords = predictor.predict(frame)
        original = cv2.resize(frame, (DISPLAY_SIZE, DISPLAY_SIZE),
                              interpolation=cv2.INTER_NEAREST)
        cv2.putText(original, "Original Input", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        visibility = np.ones(coords.shape[0], dtype=np.float32)
        xy = _draw_pose(np.zeros_like(original), coords, visibility, 0,
                        "X-Y (front)")
        zy = _draw_pose(np.zeros_like(original), coords, visibility, 2,
                        "Z-Y (depth)")
        combined = np.hstack((original, xy, zy))
        cv2.imshow("AIBoxing Inference Visualization", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(os.environ.get("AI_BOXING_SOURCE", "0"))
