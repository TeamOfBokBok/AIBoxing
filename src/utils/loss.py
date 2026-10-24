import torch
import torch.nn as nn
import torch.nn.functional as F

class MPJPELoss(nn.Module):
    """
    Mean Per Joint Position Error (MPJPE)
    각 관절별 3차원 유클리드 거리 오차의 평균을 계산
    """
    def __init__(
        self,
        max_neck_head_distance: float | None = None,
        neck_index: int = 1,
        head_index: int = 0,
        constraint_weight: float = 1.0,
    ):
        super(MPJPELoss, self).__init__()
        if max_neck_head_distance is not None and max_neck_head_distance <= 0:
            raise ValueError("max_neck_head_distance는 0보다 커야 합니다.")
        if constraint_weight < 0:
            raise ValueError("constraint_weight는 0 이상이어야 합니다.")

        self.max_neck_head_distance = max_neck_head_distance
        self.neck_index = neck_index
        self.head_index = head_index
        self.constraint_weight = constraint_weight

    def forward(self, pred_coords, target_coords):
        """관절별 유클리드 거리의 평균인 MPJPE를 계산합니다."""
        """
        pred_coords:   [B, K, 3] (모델이 예측한 3D 좌표)
        target_coords: [B, K, 3] (정답 3D 좌표)
        """
        # 관절별 유클리드 거리 계산: sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)
        error = torch.norm(pred_coords - target_coords, dim=-1) # [B, K]
        mpjpe = torch.mean(error)                               # 전체 배치 및 관절의 평균 오차

        if self.max_neck_head_distance is None or self.constraint_weight == 0:
            return mpjpe

        neck = pred_coords[:, self.neck_index, :]
        head = pred_coords[:, self.head_index, :]
        neck_head_distance = torch.norm(head - neck, dim=-1)
        excess_distance = F.relu(neck_head_distance - self.max_neck_head_distance)
        constraint_loss = torch.mean(excess_distance)
        return mpjpe + self.constraint_weight * constraint_loss


if __name__ == "__main__":
    loss_fn = MPJPELoss()
    pred = torch.randn(2, 8, 3)
    target = torch.randn(2, 8, 3)
    loss_val = loss_fn(pred, target)
    print(f"✅ MPJPE Loss 계산 테스트 성공: {loss_val.item():.4f}")