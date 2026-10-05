import os
import sys
import torch
import multiprocessing as mp
import importlib
import importlib.util  # 💡 서브모듈 직접 명시 (Import error 방지)
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pose_net import BoxingPose3DNet
from utils.dataset import BoxingDataset
from utils.loss import MPJPELoss

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_device_and_config():
    """
    NVIDIA CUDA, Linux ROCm, Windows DirectML, CPU를 자동 감지하고
    하드웨어 특성에 맞는 DataLoader 설정 파라미터를 반환합니다.
    """
    device_type = "cpu"
    msg = ""

    # 1. NVIDIA CUDA 또는 Linux AMD ROCm 감지
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        msg = f"🚀 [CUDA/ROCm] 가속 장치 감지: GPU ({gpu_name})"

    # 2. Windows AMD GPU (DirectML) 동적 감지 시도
    else:
        dml_spec = importlib.util.find_spec("torch_directml")
        if dml_spec is not None:
            try:
                torch_dml = importlib.import_module("torch_directml")
                if torch_dml.is_available():
                    device = torch_dml.device()
                    device_type = "directml"
                    msg = f"🚀 [DirectML] AMD/Intel 가속 장치 감지: {torch_dml.device_name(0)}"
                else:
                    raise ImportError
            except Exception:
                device = torch.device("cpu")
                torch.set_num_threads(os.cpu_count() or 4)
                msg = f"🚀 [CPU Mode] 가속 장치 미감지: CPU 사용 (스레드 {torch.get_num_threads()}개)"
        else:
            device = torch.device("cpu")
            torch.set_num_threads(os.cpu_count() or 4)
            msg = f"🚀 [CPU Mode] 가속 장치 미감지: CPU 사용 (스레드 {torch.get_num_threads()}개)"

    # 메인 프로세스에서만 디바이스 정보 출력
    if mp.current_process().name == 'MainProcess':
        print("=" * 65)
        print(msg)
        print("=" * 65)

    # 3. 디바이스별 DataLoader 하드웨어 최적화 파라미터 구성
    if device_type == "cuda":
        loader_kwargs = {
            "num_workers": min(4, os.cpu_count() or 2),
            "pin_memory": True,
            "persistent_workers": True,
            "non_blocking": True
        }
    elif device_type == "directml":
        loader_kwargs = {
            "num_workers": min(2, os.cpu_count() or 2),
            "pin_memory": False,
            "persistent_workers": True,
            "non_blocking": False
        }
    else:  # CPU
        loader_kwargs = {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "non_blocking": False
        }

    return device, loader_kwargs

def train():
    device, config = setup_device_and_config()

    num_epochs = 20
    batch_size = 32
    learning_rate = 1e-4
    val_ratio = 0.2

    json_path = os.path.join(PROJECT_ROOT, "src", "data", "annotations.json")
    legacy_json_path = os.path.join(PROJECT_ROOT, "src", "data", "annotations_3d.json")

    if not os.path.exists(json_path) and os.path.exists(legacy_json_path):
        os.rename(legacy_json_path, json_path)
        print("🔄 기존 annotations_3d.json을 annotations.json으로 이름을 변경했습니다.")

    img_dir = os.path.join(PROJECT_ROOT, "src", "data", "images")

    full_dataset = BoxingDataset(json_path=json_path, img_dir=img_dir, is_train=True)
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

if __name__ == "__main__":
    train()