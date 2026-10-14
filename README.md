# 🥊 AIBoxing 3D 기술 로드맵

## 현재 구현 기준 빠른 시작

이 프로젝트는 단일 RGB 영상에서 상체 9개 관절의 상대 3D 좌표를 추정합니다.
Windows의 AMD GPU에서는 ONNX Runtime DirectML을 우선 사용하고, ONNX 파일이 없을 때는 PyTorch 추론으로 자동 전환합니다.

### 1. 가상환경 및 의존성 설치

Python 3.11.9 설치

python -m venv .venv
.venv\Scripts\activate.bat


PowerShell 기준입니다.

```powershell
python -m pip install -r requirements/base.txt
python -m pip install -r requirements/cpu.txt  # CPU 또는 Intel/AMD GPU가 없는 환경
```

NVIDIA 환경은 `cuda11.txt` 또는 `cuda12.txt`, Windows AMD/Intel GPU 환경은
`amd.txt`를 CPU 프로필 대신 설치합니다. 한 환경에서 하드웨어 프로필은 하나만
선택하세요. Docker는 필요하지 않으며, 각 스크립트가 CUDA/ROCm, DirectML, CPU
순서로 실행 장치를 자동 선택합니다.

수동 설치가 필요하면 다음 패키지를 설치합니다.

```powershell
pip install onnx onnxscript onnxruntime-directml opencv-python pillow mediapipe
```

### 2. 데이터 라벨 생성

학습할 영상을 `src/data/videos/`에 넣습니다. 폴더가 없으면 라벨 스크립트가 생성합니다.
MediaPipe 모델 파일 `pose_landmarker_heavy.task`는 프로젝트 루트에 있어야 합니다.

```powershell
python src/label.py
```

처리된 영상은 `src/data/processed_videos.json`에 기록되며, 프레임 이미지는 `src/data/images/`, 라벨은 `src/data/annotations.json`에 누적됩니다.

### 3. 모델 학습

```powershell
# GTX 1650 4GB 기준 기본값은 4입니다. 부족하면 2 또는 1로 낮추세요.
$env:AI_BOXING_BATCH_SIZE="4"
python src/train.py
```

배치 크기는 `src/config.py`의 `TRAIN_BATCH_SIZE`에서 관리하며,
`AI_BOXING_BATCH_SIZE` 환경변수로 실행 시 덮어쓸 수 있습니다.
Windows의 CUDA/DirectML 학습은 PyTorch DLL 초기화 충돌을 피하기 위해
DataLoader worker를 사용하지 않습니다. 학습은 `src/data/annotations.json`과
`src/data/images/`를 사용하며, 최적 가중치를 `src/weights/best_model.pth`에 저장합니다.
학습이 정상적으로 끝나면 `src/weights/best_model.onnx`로 자동 변환됩니다.

### 4. 실시간 3분할 추론

프로젝트 루트의 `boxing_sample.mp4`를 입력 영상으로 사용합니다. 파일이 없으면 웹캠으로 자동 전환됩니다.

```powershell
python src/inference.py
```

기본 실행은 ONNX Runtime의 `DmlExecutionProvider`를 우선 사용하고, 화면에 원본·정면(X-Y)·측면(Z-Y) 패널을 표시합니다. 종료하려면 `q`를 누릅니다. `save_video=True`로 실행되므로 결과는 `src/output/result.mp4`에 저장되며, 같은 이름이 있으면 `_1`, `_2`가 붙습니다.

### ONNX 변환 확인

학습이 끝나면 `src/weights/best_model.onnx`가 자동으로 생성됩니다. 입력은 `[1, 3, 256, 256]`, 주요 출력 `coords_3d`는 `[1, 8, 3]`이며 ONNX opset 11을 사용합니다.
이미 학습된 가중치만 다시 변환해야 하는 경우에는 다음 명령을 사용합니다.

**모델이 최신화가 안 된 것 같으면 실행**

```powershell
python src/export.py
```


### 현재 파일 구조

```text
AIBoxing/
├── boxing_sample.mp4
├── pose_landmarker_heavy.task
├── requirements/
│   ├── base.txt
│   ├── cpu.txt
│   ├── amd.txt
│   ├── cuda11.txt
│   └── cuda12.txt
├── README.md
└── src/
   ├── data/
   │   ├── annotations_3d.json
   │   ├── images/
   │   └── videos/                 # 라벨링할 원본 영상
   ├── models/
   │   └── pose_net.py              # ResNet-18 기반 3D 포즈 모델
   ├── utils/
   │   ├── dataset.py
   │   └── loss.py
   ├── weights/
   │   ├── best_model.pth
   │   ├── best_model.onnx
   │   └── best_model.onnx.data
   ├── output/                      # 추론 결과 영상
   ├── label.py                     # MediaPipe 기반 프레임 라벨 생성
   ├── train.py                     # 학습 및 검증
   ├── export.py                    # PyTorch 가중치 -> ONNX 변환
   └── inference.py                 # DirectML 우선 실시간 추론
```

---

## 1. 프로젝트 개요

> **목표:** 사전 학습된 백본 네트워크(ResNet)를 기반으로 상체 관절의 입체적 위치를 예측하는 **3D Volumetric Heatmap 디코더**를 직접 설계합니다. 단일 RGB 이미지에서 상체 관절의 3차원 상대 좌표 $(X, Y, Z)$를 추출하고, **언리얼 엔진 5(Unreal Engine 5)**와 실시간 연동하는 AI 파이프라인을 구축합니다.

---

## 2. 모델 아키텍처 설계

전체 신경망은 **특징 추출기(Encoder)**, **3D 확률 지도 생성기(3D Decoder)**, **좌표 추출기(Soft-Argmax 3D)**의 3단계 구조로 결합합니다.

* **[입력]** RGB 이미지  
   │  
   ▼
* **[1. Encoder: Pre-trained Backbone]**
   ImageNet으로 학습된 ResNet 계열 활용 (특징 추출)  
   │  
   ▼
* **[2. Decoder: Custom 3D Volumetric Head]**
  입체 공간 격자(Depth x Height x Width) 생성을 위해 직접 설계  
   │  
   ▼
* **[3. Soft-Argmax 3D Layer]**
  3D 확률 지도로부터 미분 가능한 3차원 좌표 $(X, Y, Z)$ 계산  
   │  
   ▼
* **[출력]** 3D 관절 좌표 ($K \times 3$)
  $K$개 상체 관절 각각의 $(X, Y, Z)$ 3차원 좌표

---

### 2.1. Encoder (백본 네트워크)
* **네트워크:** `ResNet-18` 활용 (ImageNet Pre-trained)
* **역할:** 입력 이미지의 고차원 특징(Feature Map)을 추출하는 기저 네트워크

### 2.2. Decoder (커스텀 3D 딥러닝 헤드)
* **네트워크:** `2D Transpose Convolution` 및 `3D Reshape Layer`로 직접 구성
* **역할:** 특징 Map의 해상도를 복원하고, 관절 $K$개와 깊이 해상도 $D$를 결합하여 **$[K, D, H, W]$ 차원의 3D Volumetric Heatmap** 출력

### 2.3. Soft-Argmax 3D Layer
* **네트워크:** PyTorch 기반 커스텀 Layer 직접 구현
* **역할:** 3D 확률 지도에 Softmax 및 공간 기대값(Expectation) 연산을 적용하여, argmax의 미분 불가능 문제를 해결하고 학습 가능한 형태로 **3차원 관절 좌표 $(X, Y, Z)$** 직접 추출

---

## 3. 데이터 파이프라인 구축

1. **데이터셋 수집 및 필터링**
   * **Human3.6M / MPI-INF-3DHP** 등 표준 3D Pose 데이터셋 중 **상체 관절 3D 좌표 $(X, Y, Z)$만 필터링**하여 활용
   * 필요 시 RGB-D 카메라 또는 사전 추정 기법을 활용해 복싱 동작(잽, 스트레이트, 가드 등) 3D 커스텀 데이터셋 구축
2. **Ground Truth 3D Heatmap 생성**
   * 각 관절의 3D 정답 좌표 $(x, y, z)$를 중심으로 3차원 **가우시안 분포(3D Gaussian Distribution)**를 적용한 Target Volumetric Heatmap을 수학적으로 생성

---

## 4. 손실 함수 및 학습 파이프라인

### 4.1. 손실 함수 (Loss Function)
* **3D MPJPE (Mean Per Joint Position Error)** 사용
* 예측된 3D 좌표 ($\hat{J}$)와 실제 3D 정답 좌표 ($J$) 사이의 3차원 유클리드 거리 오차 최소화

$$\mathcal{L}_{3D} = \frac{1}{K} \sum_{k=1}^{K} \Vert{} \hat{J}_k - J_k \Vert{}_2$$

### 4.2. 학습 루프 (Training Loop)
1. **Forward Pass:** 이미지를 모델에 입력하여 3D Volumetric Heatmap 및 Soft-Argmax 3D를 통해 $(X, Y, Z)$ 좌표 예측
2. **Loss Computation:** 예측 3D 좌표와 실제 정답 3D 좌표 간 MPJPE Loss 계산
3. **Backward Pass & Optimization:** 역전파를 수행하고 AdamW 알고리즘을 사용해 AI의 내부 가중치를 3D 위치 오차가 줄어드는 방향으로 최적화

---

## 5. 실시간 통신 및 언리얼 엔진 5 연동

1. **UDP Socket 데이터 전송 (Python)**
   * 추론된 $K$개 관절의 $3D (X, Y, Z)$ 좌표를 언리얼 스케일(cm 단위) 및 좌표계(Z-up)에 맞춰 변환
   * JSON 패킷 구조로 가공하여 Localhost UDP 통신으로 실시간 전송
2. **언리얼 엔진 5 바인딩 (Unreal Engine 5)**
   * UDP Receiver 블루프린트/C++ 모듈을 통해 패킷 수신 및 좌표 파싱
   * **Control Rig 및 Two-Bone IK** 노드를 활용해 3D 메타휴먼/캐릭터 스켈레톤의 위치 및 관절 회전 제어

---

## 6. 프로젝트 평가/발표 핵심 포인트

* **3D Volumetric AI 모델 직접 설계:** 단순 2D 포즈 추정을 넘어, 단일 RGB 이미지에서 3차원 깊이($Z$축)까지 예측하는 **3D Volumetric Heatmap 디코더 및 Soft-Argmax 3D 구조를 PyTorch로 직접 설계 및 구현**
* **도메인 특화 3D 파이프라인:** 복싱 동작의 입체적 가드 자세 및 펀치 깊이감을 정밀하게 추적하는 **상체 중심 커스텀 3D 데이터 파이프라인** 구축
* **엔드투엔드(End-to-End) 게임 엔진 연동:** PyTorch AI 모델 추론부터 UDP 실시간 통신, 언리얼 엔진 5 Control Rig 3D 캐릭터 스켈레톤 제어까지 연결되는 **완전한 3D 모션 캡처 파이프라인** 완성


---

## 7. 현재 상태 및 주의사항

* 학습 스크립트는 기존 `best_model.pth`가 있으면 해당 가중치를 불러와 이어서 학습합니다.
* `inference.py`는 입력 프레임을 256x256으로 전처리하고, 3분할 화면의 전체 기준 폭을 1920픽셀로 구성합니다.
* `DmlExecutionProvider`가 설치되어 있지 않거나 ONNX 모델이 없으면 `inference.py`가 PyTorch 경로로 fallback합니다.
* Unreal Engine UDP 전송 모듈은 현재 파일 구조에 포함되어 있지 않습니다.

---

## 8. 다중 PC 환경 분산 학습 (가중치 릴레이 방식)

대용량 이미지 데이터(수 GB~)를 PC 간에 직접 전송하지 않고, **경량화된 가중치 파일(`best_model.pth`)**만 공유하여 파인튜닝을 이어가는 방식입니다. 느린 네트워크 환경에서도 병목 없이 여러 디바이스에서 효율적으로 분산 학습을 진행할 수 있습니다.

---

### 🔄 분산 학습 워크플로우

```text
각 PC에서 학습 진행
▼
가중치 파일을 메인 PC로 모음
▼
메인 PC에서 src/merge/ 폴더 생성
▼
merge/ 폴더에 가중치 파일 넣기
▼
merge_weights.py 실행
▼
merge/ 폴더에 merged_model.pth 파일 생성
▼
다시 각 PC에 가중치 파일 업데이트

※ 주의사항
 : 가중치 파일에 인덱싱 할 거 아니면 전체 PC가 학습이 끝난 뒤 가중치 파일 업데이트하기
```

### 학습 후 원본 이미지 파일의 처리방법

```text
images/
annotations.json
processed_videos.json
```
압축 후 클라우드에 보관