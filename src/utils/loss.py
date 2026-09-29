import torch
import torch.nn as nn

class MPJPELoss(nn.Module):
    """
    Mean Per Joint Position Error (MPJPE)
    각 관절별 3차원 유클리드 거리 오차의 평균을 계산
    """
    def __init__(self):
        super(MPJPELoss, self).__init__()

    def forward(self, pred_coords, target_coords):
        """
        pred_coords:   [B, K, 3] (모델이 예측한 3D 좌표)
        target_coords: [B, K, 3] (정답 3D 좌표)
        """
        # 관절별 유클리드 거리 계산: sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
        error = torch.norm(pred_coords - target_coords, dim=-1) # [B, K]
        loss = torch.mean(error)                               # 전체 배치 및 관절의 평균 오차
        return loss


if __name__ == "__main__":
    loss_fn = MPJPELoss()
    pred = torch.randn(2, 8, 3)
    target = torch.randn(2, 8, 3)
    loss_val = loss_fn(pred, target)
    print(f"✅ MPJPE Loss 계산 테스트 성공: {loss_val.item():.4f}")