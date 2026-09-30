import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 CUDA 사용 가능: GPU({torch.cuda.get_device_name(0)}) 할당")
else:
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"🚀 CUDA 미지원: CPU 할당 (스레드 수: {torch.get_num_threads()}개)")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pose_net import BoxingPose3DNet
from utils.dataset import BoxingDataset
from utils.loss import MPJPELoss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train():
    num_epochs = 20
    batch_size = 16
    learning_rate = 1e-4
    val_ratio = 0.2

    # 💡 annotations.json 경로 변경 (하위 호환성 체크 적용)
    json_path = os.path.join(PROJECT_ROOT, "src", "data", "annotations.json")

    img_dir = os.path.join(PROJECT_ROOT, "src", "data", "images")

    full_dataset = BoxingDataset(json_path=json_path, img_dir=img_dir, is_train=True)
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    num_workers = 2 if device.type == "cuda" else 0
    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    model = BoxingPose3DNet(num_keypoints=8, depth_res=64, pretrained=True).to(device)

    weights_dir = os.path.join(PROJECT_ROOT, "src", "weights")
    weight_path = os.path.join(weights_dir, "best_model.pth")
    if os.path.exists(weight_path):
        print(f"🔄 기존 가중치 로드 완료: {weight_path} (이어서 학습을 시작합니다)")
        model.load_state_dict(torch.load(weight_path, map_location=device))
    else:
        print("✨ 기존 가중치 없음: 처음부터 신규 학습을 시작합니다.")

    criterion = MPJPELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    os.makedirs(weights_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            pred_coords, _ = model(images)
            loss = criterion(pred_coords, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                pred_coords, _ = model(images)
                val_loss += criterion(pred_coords, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), weight_path)
            print(f"  --> 💾 최적 가중치 갱신 (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()