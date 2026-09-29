# 🥊 AIBoxing 3D 기술 로드맵

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
* **네트워크:** `ResNet-18` 또는 `ResNet-34` 활용 (ImageNet Pre-trained)
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