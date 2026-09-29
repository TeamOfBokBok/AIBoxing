# 🥊 AIBoxing 기술 로드맵

---

## 1. 프로젝트 개요

> **목표:** 범용 사전 학습(Pre-trained) 백본 네트워크(뼈대)를 기반으로 상체 관절(Keypoints) 예측을 위한 Heatmap 디코더를 직접 설계하고, 학습 파이프라인 및 손실 함수를 구현하여 복싱 동작 특화 포즈 추정 AI 모델을 구축한다.

---

## 2. 모델 아키텍처 설계 (Architecture Design)

전체 신경망은 **특징 추출기(Encoder)**와 **좌표 예측기(Decoder)**의 2단계 구조로 직접 설계 및 결합합니다.

```
[입력 이미지 (RGB)]
        │
        ▼
[1. Encoder: Pre-trained Backbone]  ─── ImageNet으로 학습된 ResNet 계열 활용 (특징 추출)
        │
        ▼
[2. Decoder: Custom Heatmap Head]   ─── 상체 관절 확률 지도 출력을 위해 직접 설계한 신경망
        │
        ▼
[출력: Heatmaps (K x H x W)]        ─── 상체 K개 관절 각각의 위치 확률 지도
```

### 2.1. Encoder (백본 네트워크)
* **네트워크:** `ResNet-18` 또는 `ResNet-34` 활용 (ImageNet Pre-trained)
* **역할:** 이미지의 고차원 특징(Feature Map)을 추출하는 기저 네트워크

### 2.2. Decoder (커스텀 딥러닝 헤드)
* **네트워크:** `Transpose Convolution` 및 `Conv2d` 레이어로 직접 구성
* **역할:** 특징 Map의 해상도를 복원하고, 상체 주요 관절 $K$개(머리, 목, 양 어깨, 양 팔꿈치, 양 손목 등)에 대응하는 **$K$개의 Heatmap Channel** 출력

---

## 3. 데이터 파이프라인 구축 (Data Pipeline)

1. **데이터셋 수집 및 필터링**
   * COCO Keypoint 데이터셋 중 **상체 관절 좌표만 필터링**하여 활용
   * 필요 시 복싱 동작(잽, 스트레이트, 가드 등) 자체 수집 영상 데이터 추가 라벨링
2. **Ground Truth Heatmap 생성**
   * 각 관절의 실제 $(x, y)$ 좌표를 중심으로 2차원 **가우시안 분포(2D Gaussian Distribution)**를 적용한 Target Heatmap을 수학적으로 생성

---

## 4. 손실 함수 및 학습 파이프라인 (Training Pipeline)

### 4.1. 손실 함수 (Loss Function)
* **Mean Squared Error (MSE) Loss** 사용
* 예측된 Heatmap ($\hat{M}$)과 정답 Target Heatmap ($M$) 사이의 픽셀 단위 오차 최소화

$$\mathcal{L}_{MSE} = \frac{1}{K} \sum_{k=1}^{K} \| \hat{M}_k - M_k \|^2$$

### 4.2. 학습 루프 (Training Loop)
1. **Forward Pass:** 이미지를 모델에 입력하여 관절별 Heatmap 예측
2. **Loss Computation:** 예측 값과 실제 값 간 MSE Loss 계산
3. **Backward Pass & Optimization:** 신경망을 거꾸로 짚어가며(역전파), AdamW 알고리즘을 사용해 AI의 내부 가중치를 오차가 줄어드는 방향으로 최적화

---

## 5. 후처리 및 좌표 복원 (Post-Processing)

1. **Soft-Argmax 또는 Max Heatmap 추출**
   * AI가 출력한 $K$개의 Heatmap에서 가장 확률 값이 높은 픽셀 좌표 $(x', y')$ 탐색
2. **원본 좌표 변환**
   * 입력 이미지 크기로 스케일링하여 최종 상체 관절 좌표 $(X, Y)$ 복원 및 시각화

---

## 6. 프로젝트 평가/발표 핵심 포인트 (Key Selling Points)

* **AI 모델 직접 설계:** 기존 프레임워크(MediaPipe 등)를 단순 호출하는 것에 그치지 않고, PyTorch 기반으로 **Heatmap 예측 디코더 신경망 구조를 직접 설계 및 구현**
* **도메인 특화 학습:** 일반 전신 포즈 추정과 달리 **복싱 동작에 필요한 상체 관절 중심의 커스텀 파이프라인** 구축
* **학습 루프 및 손실 함수 자체 작성:** PyTorch의 `nn.Module`, `DataLoader`, `MSELoss`, Custom Training Loop를 100% 직접 코딩하여 학습 전 과정을 통제
