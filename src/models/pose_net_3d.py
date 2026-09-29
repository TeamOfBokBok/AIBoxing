import torch
import torch.nn as nn
import torchvision.models as models

# 1. 백본 모듈 (ResNet-18)
class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.backbone(x)

# 2. 3D 디코더 모듈 (Volumetric Heatmap Head)
class Decoder3D(nn.Module):
    def __init__(self, num_keypoints=8, depth_res=64):
        super(Decoder3D, self).__init__()
        self.num_keypoints = num_keypoints
        self.depth_res = depth_res

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(64, num_keypoints * depth_res, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        out = self.deconv_layers(x)
        out = self.final_conv(out)
        volumetric_heatmap = out.view(B, self.num_keypoints, self.depth_res, 64, 64)
        return volumetric_heatmap

# 3. Soft-Argmax 3D 레이어
class SoftArgmax3D(nn.Module):
    def __init__(self, depth=64, height=64, width=64):
        super(SoftArgmax3D, self).__init__()
        self.depth = depth
        self.height = height
        self.width = width

        pos_x = torch.linspace(-1.0, 1.0, width)
        pos_y = torch.linspace(-1.0, 1.0, height)
        pos_z = torch.linspace(-1.0, 1.0, depth)

        z_grid, y_grid, x_grid = torch.meshgrid(pos_z, pos_y, pos_x, indexing='ij')

        # 타입 체커(Pylance) 오경고 방지를 위한 버퍼 등록
        self.register_buffer('x_grid', x_grid.reshape(1, 1, -1))
        self.register_buffer('y_grid', y_grid.reshape(1, 1, -1))
        self.register_buffer('z_grid', z_grid.reshape(1, 1, -1))

    def forward(self, heatmaps):
        B, K, D, H, W = heatmaps.shape
        flat_heatmaps = heatmaps.view(B, K, -1)
        probs = nn.functional.softmax(flat_heatmaps, dim=-1)

        # torch.Tensor로 명시적 타입 캐스팅(또는 getattr/Tensor 표기)하여 Pylance 에러 제거
        x_grid: torch.Tensor = self.x_grid  # type: ignore
        y_grid: torch.Tensor = self.y_grid  # type: ignore
        z_grid: torch.Tensor = self.z_grid  # type: ignore

        x = torch.sum(probs * x_grid, dim=-1, keepdim=True)
        y = torch.sum(probs * y_grid, dim=-1, keepdim=True)
        z = torch.sum(probs * z_grid, dim=-1, keepdim=True)

        coords_3d = torch.cat([x, y, z], dim=-1)
        return coords_3d

# 4. 메인 3D 포즈 추정 모델
class BoxingPose3DNet(nn.Module):
    def __init__(self, num_keypoints=8, depth_res=64):
        super(BoxingPose3DNet, self).__init__()
        self.backbone = ResNetBackbone(pretrained=True)
        self.decoder = Decoder3D(num_keypoints=num_keypoints, depth_res=depth_res)
        self.soft_argmax = SoftArgmax3D(depth=depth_res, height=64, width=64)

    def forward(self, x):
        features = self.backbone(x)
        volumetric_heatmaps = self.decoder(features)
        coords_3d = self.soft_argmax(volumetric_heatmaps)
        return coords_3d, volumetric_heatmaps


# ----------------------------------------------------
# 🚀 5. 단독 실행 테스트 (드라이런 코드)
# ----------------------------------------------------
if __name__ == "__main__":
    print("⏳ 모델 생성 및 가중치 로드 중...")
    model = BoxingPose3DNet(num_keypoints=8, depth_res=64)
    model.eval()

    # 가짜 256x256 RGB 이미지 1장 생성
    dummy_image = torch.randn(1, 3, 256, 256)

    print("⏳ 3D 추론 계산 수행 중...")
    with torch.no_grad():
        coords_3d, heatmaps = model(dummy_image)

    print("\n==========================================")
    print("🎉 [드라이런 성공] 모델 연산이 정상 작동합니다!")
    print(f"- 입력 이미지 Shape: {dummy_image.shape}")
    print(f"- 출력 3D 좌표 Shape: {coords_3d.shape}  --> [배치, 8개관절, (X,Y,Z)]")
    print(f"- 출력 3D Heatmap Shape: {heatmaps.shape} --> [배치, 8개관절, Depth, Height, Width]")
    print("==========================================\n")
    print("추출된 가짜 3D 좌표 샘플 (첫 번째 관절):")
    print(f"X: {coords_3d[0][0][0]:.4f}, Y: {coords_3d[0][0][1]:.4f}, Z: {coords_3d[0][0][2]:.4f}")