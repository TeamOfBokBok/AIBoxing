import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pose_net_3d import BoxingPose3DNet
from utils.dataset_3d import BoxingDataset3D
from utils.loss import MPJPELoss

def train_real_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 💡 에포크 수 및 배치 사이즈 설정
    num_epochs = 15
    batch_size = 8
    learning_rate = 1e-4

    print(f"🚀 학습 디바이스: {device}")

    # 실제 수집된 데이터셋 로드
    json_path = "src/data/annotations_3d.json"
    img_dir = "src/data/images"

    dataset = BoxingDataset3D(json_path=json_path, img_dir=img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"📦 총 로드된 학습 샘플 수: {len(dataset)}개 (배치 크기: {batch_size})")

    model = BoxingPose3DNet(num_keypoints=8, depth_res=64).to(device)
    criterion = MPJPELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    os.makedirs("src/weights", exist_ok=True)
    best_loss = float('inf')

    print("⏳ 실제 복싱 데이터셋 기반 모델 학습을 시작합니다...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)

            pred_coords, _ = model(images)
            loss = criterion(pred_coords, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] - Loss (MPJPE): {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "src/weights/best_model.pth")
            print(f"  --> 💾 최적 가중치 갱신 및 저장 완료 (Loss: {best_loss:.4f})")

    print("\n🎉 실제 복싱 데이터셋 모델 학습이 완료되었습니다!")


if __name__ == "__main__":
    train_real_dataset()