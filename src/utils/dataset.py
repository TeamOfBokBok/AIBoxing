import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from utils.image import letterbox

class BoxingDataset(Dataset):
    """이미지와 정규화된 3D 관절 라벨을 PyTorch 텐서로 제공합니다."""

    def __init__(self, json_path: str, img_dir: str, is_train: bool = True):
        self.img_dir = img_dir
        self.is_train = is_train

        # 라벨 파일명 하위 호환성 자동 체크
        if not os.path.exists(json_path):
            legacy_json_path = json_path.replace("annotations.json", "annotations_3d.json")
            if os.path.exists(legacy_json_path):
                os.rename(legacy_json_path, json_path)
            else:
                raise FileNotFoundError(f"❌ 라벨 파일('{json_path}')을 찾을 수 없습니다.")

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_annotations = json.load(f)

        # 실존하는 이미지 경로만 필터링 (FileNotFoundError 방지)
        self.annotations = [
            item for item in raw_annotations 
            if os.path.exists(os.path.join(self.img_dir, item["image_path"]))
        ]

        if len(self.annotations) == 0:
            raise ValueError(f"❌ 유효한 이미지 샘플이 0개입니다. 이미지 경로를 확인해 주세요: {img_dir}")

        print(f"📦 데이터셋 로드 완료: 전체 {len(raw_annotations)}개 중 유효 이미지 {len(self.annotations)}개 사용")

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.img_dir, item["image_path"])

        # 💡 유니코드/한글 경로 및 OS 종속성 문제 해결을 위한 numpy binary read
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            img_bgr = None

        if img_bgr is None:
            raise FileNotFoundError(f"❌ 이미지 파일을 읽을 수 없습니다: {img_path}")

        image_tensor = self.transform(cv2.cvtColor(letterbox(img_bgr), cv2.COLOR_BGR2RGB))
        keypoints = torch.tensor(item["keypoints_3d"], dtype=torch.float32)

        return image_tensor, keypoints