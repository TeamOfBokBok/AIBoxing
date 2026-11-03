"""Export the trained model to ONNX."""

import torch
from config import ONNX_WEIGHTS_PATH, PYTORCH_WEIGHTS_PATH, IMAGE_SIZE
from models.pose_net import BoxingPose3DNet


def export_onnx() -> None:
    if not PYTORCH_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"가중치가 없습니다: {PYTORCH_WEIGHTS_PATH}")
    model = BoxingPose3DNet(pretrained=False)
    model.load_state_dict(torch.load(PYTORCH_WEIGHTS_PATH, map_location="cpu",
                                      weights_only=True))
    model.eval()
    ONNX_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
                      ONNX_WEIGHTS_PATH, input_names=["image"],
                      output_names=["coords_3d", "heatmaps"], opset_version=17,
                      dynamo=False,
                      dynamic_axes={"image": {0: "batch"},
                                    "coords_3d": {0: "batch"},
                                    "heatmaps": {0: "batch"}})
    print(f"ONNX 저장: {ONNX_WEIGHTS_PATH}")


if __name__ == "__main__":
    export_onnx()
