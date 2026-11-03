"""RGB to relative 3D upper-body pose model."""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class BoxingPose3DNet(nn.Module):
    def __init__(self, num_keypoints: int = 9, depth_res: int = 64,
                 pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, num_keypoints * depth_res, 1))
        self.num_keypoints = num_keypoints
        self.depth_res = depth_res
        z, y, x = torch.meshgrid(torch.linspace(-1, 1, depth_res),
                                  torch.linspace(-1, 1, 64),
                                  torch.linspace(-1, 1, 64), indexing="ij")
        self.register_buffer("x_grid", x.reshape(1, 1, -1))
        self.register_buffer("y_grid", y.reshape(1, 1, -1))
        self.register_buffer("z_grid", z.reshape(1, 1, -1))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        heatmaps = self.decoder(self.encoder(images)).view(
            images.shape[0], self.num_keypoints, self.depth_res, 64, 64)
        probabilities = torch.softmax(heatmaps.flatten(2), dim=-1)
        coords = torch.cat([
            (probabilities * self.x_grid).sum(-1, keepdim=True),
            (probabilities * self.y_grid).sum(-1, keepdim=True),
            (probabilities * self.z_grid).sum(-1, keepdim=True)], dim=-1)
        return coords, heatmaps
