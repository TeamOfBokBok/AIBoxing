# AIBoxing

단일 RGB 영상에서 허리 기준 상체 9개 관절의 상대 3D 좌표를 추정하는 프로젝트입니다.
현재 출력은 카메라와의 절대 거리가 아니라 `waist = (0, 0, 0)` 기준의 정규화 좌표입니다.

## 보존해야 하는 환경 파일

리빌드 시 다음은 삭제하거나 수정하지 않습니다.

- `.git/`, `.gitignore`
- `.venv/`
- `requirements/`
- 루트의 `pose_landmarker_heavy.task`

## 데이터 준비

실제 영상을 `src/data/videos/`에 넣고 라벨을 생성합니다.

```powershell
python src\label.py
```

가상환경 데이터는 `src/data/synthetic_images/`에 이미지로 넣고,
`src/data/synthetic_annotations.json`에 아래 형식으로 저장합니다.

```json
{
  "image_path": "synthetic_000001.jpg",
  "keypoints_3d": [[0.0, 0.1, -0.2]],
  "visibility": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  "source": "synthetic_gt",
  "sequence_id": "synthetic_sequence_001",
  "frame_index": 0
}
```

`keypoints_3d` 순서는 `nose, neck, r_shoulder, r_elbow, r_wrist,
l_shoulder, l_elbow, l_wrist, waist`이며 반드시 허리 기준 상대 좌표로 변환합니다.
`source`는 `synthetic_gt`, `real_pseudo`, `real_gt` 중 하나입니다.

## 학습과 추론

```powershell
$env:AI_BOXING_BATCH_SIZE="4"
$env:AI_BOXING_EPOCHS="20"
python src\train.py
python src\export.py
python src\inference.py
```

학습은 `annotations.json`의 실제 pseudo-label과
`synthetic_annotations.json`의 synthetic GT를 자동으로 합쳐 sequence 단위로
train/validation을 나누며, MediaPipe pseudo-label은
visibility와 낮은 신뢰도 가중치로 반영합니다. 실제 GT 평가셋을 별도로 유지해야
최종 depth 성능을 올바르게 측정할 수 있습니다.

## 사용자가 준비할 것

1. `src/data/videos/`에 사람·배경·조명·거리·복싱 동작이 다양한 실제 영상
2. 실제 영상과 동일한 좌표계의 synthetic 이미지 및 정확한 3D 관절 GT
3. 최종 검증용 실제 GT 데이터(가능하면 RGB-D, 다중 카메라, 또는 수동 보정)
4. synthetic과 real 데이터가 동일한 9개 관절 순서와 좌표계인지 확인
