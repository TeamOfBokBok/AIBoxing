import os
import sys
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pose_net import BoxingPose3DNet
from utils.dataset import BoxingDataset
from utils.loss import MPJPELoss
from config import DATA_DIR, IMAGE_DIR, PYTORCH_WEIGHTS_PATH, SRC_ROOT, resolve_runtime
from export import export_onnx

def setup_device_and_config():
    """학습 장치와 DataLoader 옵션을 공통 런타임 설정에서 가져옵니다."""
    runtime = resolve_runtime()
    return runtime.device, runtime.loader_kwargs

def train():
    """데이터셋을 분할해 3D 포즈 모델을 학습하고 최적 가중치를 저장합니다."""
    device, config = setup_device_and_config()

    num_epochs = 20
    batch_size = 32
    learning_rate = 1e-4
    val_ratio = 0.2

    json_path = DATA_DIR / "annotations.json"
    legacy_json_path = DATA_DIR / "annotations_3d.json"

    if not os.path.exists(json_path) and os.path.exists(legacy_json_path):
        os.rename(legacy_json_path, json_path)
        print("🔄 기존 annotations_3d.json을 annotations.json으로 이름을 변경했습니다.")

    img_dir = IMAGE_DIR

    full_dataset = BoxingDataset(
        json_path=str(json_path),
        img_dir=str(img_dir),
        is_train=True,
    )
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        persistent_workers=config["persistent_workers"]
    )

    model = BoxingPose3DNet(num_keypoints=9, depth_res=64, pretrained=True).to(device)

    weights_dir = SRC_ROOT / "weights"
    weight_path = PYTORCH_WEIGHTS_PATH

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

    non_blocking = config["non_blocking"]

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

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
                images = images.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)

                pred_coords, _ = model(images)
                val_loss += criterion(pred_coords, targets).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), weight_path)
            print(f"  --> 💾 최적 가중치 갱신 (Val Loss: {best_val_loss:.4f})")

    print("🔄 학습 완료: 최적 가중치를 ONNX로 변환합니다.")
    export_onnx()

if __name__ == "__main__":
    train()