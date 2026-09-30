import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class BoxingDataset(Dataset):
    def __init__(self, json_path: str, img_dir: str, is_train: bool = True):
        self.img_dir = img_dir
        self.is_train = is_train

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ 라벨 파일('{json_path}')을 찾을 수 없습니다.")

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_annotations = json.load(f)

        # 이미지 파일 미존재로 인한 예외 방지 필터링
        self.annotations = [
            item for item in raw_annotations 
            if os.path.exists(os.path.join(self.img_dir, item["image_path"]))
        ]

        if len(self.annotations) == 0:
            raise ValueError(f"❌ 유효한 이미지 샘플이 0개입니다. 이미지 경로를 확인해 주세요: {img_dir}")

        print(f"📦 데이터셋 로드 완료: 전체 {len(raw_annotations)}개 라벨 중 유효 이미지 {len(self.annotations)}개 사용")

        self.transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.img_dir, item["image_path"])

        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)

        keypoints = torch.tensor(item["keypoints_3d"], dtype=torch.float32)

        return image_tensor, keypoints