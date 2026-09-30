import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple

class BoxingDataset(Dataset):
    def __init__(self, json_path: str, img_dir: str, is_train: bool = True):
        self.img_dir = img_dir
        self.is_train = is_train
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ 라벨 파일을 찾을 수 없습니다: {json_path}")
            
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.annotations[idx]
        img_path = os.path.join(self.img_dir, item["image_path"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ 이미지 파일 미존재: {img_path}")
            
        pil_image = Image.open(img_path).convert('RGB')
        image_tensor: torch.Tensor = self.transform(pil_image) # type: ignore
        keypoints_3d = torch.tensor(item["keypoints_3d"], dtype=torch.float32)
        
        return image_tensor, keypoints_3d