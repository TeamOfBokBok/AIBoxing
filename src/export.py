import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.pose_net import BoxingPose3DNet
from config import ONNX_WEIGHTS_PATH, PYTORCH_WEIGHTS_PATH

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHT_PATH = str(PYTORCH_WEIGHTS_PATH)
OUTPUT_PATH = str(ONNX_WEIGHTS_PATH)

def export_onnx(weight_path: str = WEIGHT_PATH, output_path: str = OUTPUT_PATH) -> None:
    """학습된 PyTorch 가중치를 CPU에서 호환 가능한 ONNX 모델로 변환합니다."""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"❌ 가중치 파일을 찾을 수 없습니다: {weight_path}")

    model = BoxingPose3DNet(num_keypoints=8, depth_res=64, pretrained=False)
    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        dynamo=False,
        opset_version=11,
        input_names=["image"],
        output_names=["coords_3d", "volumetric_heatmaps"],
        dynamic_axes={
            "image": {0: "batch"},
            "coords_3d": {0: "batch"},
            "volumetric_heatmaps": {0: "batch"},
        },
    )
    print(f"✅ ONNX 내보내기 완료: {output_path}")

if __name__ == "__main__":
    export_onnx()