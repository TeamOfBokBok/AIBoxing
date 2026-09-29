import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple

class BoxingDataset3D(Dataset):
    """
    3D 복싱 포즈 추정을 위한 Custom PyTorch Dataset
    """
    def __init__(self, json_path: str, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. JSON 라벨 파일 검증 및 로드
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ 라벨 파일을 찾을 수 없습니다: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        # 2. 기본 이미지 전처리 설정 (256x256 Resize & Normalization)
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.annotations[idx]
        
        # 이미지 경로 병합 및 로드
        img_path = os.path.join(self.img_dir, item["image_path"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ 이미지 파일을 찾을 수 없습니다: {img_path}")
            
        pil_image = Image.open(img_path).convert('RGB')
        
        # 전처리 적용 및 텐서 변환 명시
        if self.transform is not None:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = transforms.ToTensor()(pil_image)
            
        # Pylance 타입 힌트 보장
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.tensor(image_tensor)
            
        # 3D 관절 좌표 텐서 변환 [8, 3]
        keypoints_3d = torch.tensor(item["keypoints_3d"], dtype=torch.float32)
        
        return image_tensor, keypoints_3d


# ----------------------------------------------------
# 🧪 데이터셋 단독 검증 (Dummy File 생성 및 검증)
# ----------------------------------------------------
if __name__ == "__main__":
    print("⏳ 데이터셋 모듈 검증 시작...")
    
    test_data_dir = "src/data"
    os.makedirs(os.path.join(test_data_dir, "images"), exist_ok=True)
    
    dummy_json_path = os.path.join(test_data_dir, "test_annotations.json")
    dummy_img_name = "test_frame.jpg"
    dummy_img_path = os.path.join(test_data_dir, "images", dummy_img_name)
    
    # 1. 테스트용 더미 이미지 저장
    Image.new('RGB', (640, 480), color=(128, 128, 128)).save(dummy_img_path)
    
    # 2. 테스트용 더미 JSON 라벨 저장
    dummy_anno = [{
        "image_path": dummy_img_name,
        "keypoints_3d": [[0.0] * 3 for _ in range(8)]
    }]
    with open(dummy_json_path, 'w', encoding='utf-8') as f:
        json.dump(dummy_anno, f)

    try:
        # 3. Dataset 인스턴스화 및 검증
        dataset = BoxingDataset3D(json_path=dummy_json_path, img_dir=os.path.join(test_data_dir, "images"))
        
        # 반환 튜플 받기
        img_tensor, kp_tensor = dataset[0]
        
        print("\n==========================================")
        print("🎉 [Dataset 검증 성공] 데이터 로딩이 정상 작동합니다!")
        print(f"- 로드된 이미지 Tensor Shape: {img_tensor.shape} (Resize 256x256 적용됨)")
        print(f"- 로드된 3D 관절 좌표 Shape: {kp_tensor.shape} (8개 관절 x 3D 좌표)")
        print("==========================================\n")
        
    finally:
        # 테스트 후 생성했던 임시 파일 정리
        if os.path.exists(dummy_json_path): 
            os.remove(dummy_json_path)
        if os.path.exists(dummy_img_path): 
            os.remove(dummy_img_path)