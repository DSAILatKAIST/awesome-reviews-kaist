---
description: >-
  WU, Size, et al. / Graph-Based 3D Multi-Person Pose Estimation Using
  Multi-View Images / ICCV-2021
---

# graph based 3d multi person pose estimation using multi view images

저는 현재 공사 현장에서 카메라 기반으로 사람들의 행동을 분석하고, 추락 사고를 예방하는 프로젝트를 진행하고 있습니다. 최근 카메라 기반으로 사람의 행동을 분석하는 데 pose estimation 기술이 많이 사용되고 있습니다. 저는 한 대의 카메라가 아닌 여러 대의 카메라를 사용했을 때, 하나의 공간에 다른 카메라에서 포착한 같은 사람을 어떻게 일치시키고, 입체적으로 여러 사람을 어떻게 표현할 수 있는지 궁금하였습니다.

이번에 소개드릴 논문은 2021년 ICCV에 발표된 논문으로, 다양한 각도의 이미지를 활용하여 그래프 기반으로 Multi-person 3D pose estimation 모델을 제안하고 있습니다.

논문 링크 https://arxiv.org/abs/2109.05885

## **1. Problem Definition**

이 논문은 **다수의 카메라로부터 다수의 사람들의 human pose를 추정하는 문제를 3가지 task-specific한 그래프 뉴럴 네트워크를 기반으로 접근하고 있습니다.** 우선, 3D pose estimation 과제는 person localization과 pose estimation로 분해됩니다. 즉, 개별 사람의 인스턴스를 식별하고 지역화하면서 시작합니다. 그리고 나서, 한 사람의 포즈를 추정합니다.

본 연구에서는 3차원 공간에서 person localization을 위해 Multi-view Matching Graph Module과 Center Refinement Graph Module을 적용하였습니다. 이 후, 3D pose estimation을 위해서 Pose Regression Graph Module을 적용하여 multi-view geometry와 관절 사이 구조적 관계를 학습시켰습니다.

## **2. Motivation**

여러 대의 카메라로부터 여러 사람의 3D pose estimation 작업은 꽤 오랜 문제였습니다. **Multi-view images를 이용한 3D multi-person pose estimation**에 관한 최근 연구는 일반적으로 2가지 흐름을 따라갑니다.

1. **2D-to-3D lifting 기반의 접근법(a)**
2. **Direct 3D estimation 접근법(b)**

![Figure1](https://blog.kakaocdn.net/dn/cJa9Kf/btrz5wE7mMd/pfkkwdINwSbnSyKKprpWXK/img.png)

* (a)와 같이, **2D-to-3D lifting 접근법**은 먼저 **각각의 시점에서 2D joints를 추정하고, 추정된 2D poses들을 같은 사람끼리 매치시킵니다**. 이후, triangulation 이나 Pictorial Structure Models(PSM)을 통해 매치된 2D single-view poses를 3D로 변환합니다. 이 접근법은 일반적으로 실시간일때 효율적입니다. 그러나, 3D reconstruction 정확도는 폐색에 견고하지 않은 2D pose estimation에 의해 제한을 받습니다.
* (b)와 같이, **Direct 3D 접근법**은 **multi-view features를 모아서 discretized 3D volumetric representations을 만들고, 3D 공간에 직접적으로 작동합니다**. 이 접근법은 2D camera views에서의 부정확한 결정을 내리는 걸 피할 수 있습니다. 그러나, 공간의 크기가 커지면 계산량도 증가합니다. 또한, space discretization에 의해 발생되는 quantization errors를 겪습니다.
* (c)와 같이, **본 연구는 두 가지 접근법을 결합합니다.** 첫번째 단계에서 효율적인 3D human center 탐지를 위해 2D-to-3D lifting을 사용하고, 두번째 단계에서 정확한 single-person 3D pose estimation을 위해 direct 3D estimation 접근법을 사용합니다. 정확성과 효율성 모두를 위해, 두 단계 모두 task-specific한 그래프 뉴럴 네트워크와 함께 coarse-to-fine 방식으로 처리됩니다. &#x20;

첫번째 단계에서, **multi-view matching을 통해 3D human center을 예측합니다**. 이전의 방법들은, multi-view geometric constraints와 appearance similarity를 통해 매칭합니다. 그러나, 매칭 기준은 수동으로 하는 것이었습니다. 이 문제를 해결하기 위해, **본 연구에서는 Multi-view Matching Graph Module(MMG)를 제안합니다**. 이 모듈은 visual과 geometric cues를 모두 고려하여 views들간 사람들을 매치하기 위해 데이터로부터 학습합니다.

또한, 본 연구는 더 **상세한 3D human center 탐지를 위해 Center Refinement Graph Module(CRG)를 제안합니다**. 이전의 연구들은 공간을 voxel로 나누었고, 일반 그리드 위에서 작동합니다. 그러나 CRG는 implicit field representations을 적용하고, 각각의 점이 human center인지 아닌지 예측하기 위해 연속적인 3D 공간 위에서 바로 작동합니다.

초기 3D poses를 만들기 위해 off-the-shelf pose estimator를 사용합니다. 본 연구는 **상세한 수준의 single person pose estimation을 위해, Pose Regression Graph Module(PRG)를 제안합니다**. 이 모듈은 몸 관절 사이 공간적 관계와 multiple views들 간 기하학적 관계를 이용하여 초기 3D poses를 정제합니다.

&#x20;

> **본 논문의 핵심 기여점**은 다음과 같습니다.

1. **Multi-view 3D pose estimation을 위해 task-specific한 그래프 뉴럴 네트워크를 사용한 것은 최초입니다.** 정확도와 효율성 면에서 이전 연구들을 능가하는 새로운 coarse-to-fine 프레임워크를 제안합니다.
2. Learnable matching을 통해 multi-view human association의 성능을 향상시키는 Multi-view Matching Graph Module(MMG)를 제안합니다.
3. Multi-view features들을 종합하는 point-based human center refinement를 위한 Center Refinement Graph Module(CRG)를 제안합니다.
4. 3D human pose refinement를 위한 그래프 기반 모델인 Pose Regression Graph(PRG)를 제안합니다. 더 정확한 3D human poses를 만들기 위해 human body 구조 정보와 multi-view geometry를 이용합니다.

## **3. Method**

### 3.1 Overview

본 연구는 Tu et al.의 사전 훈련된 **2D bottom-up pose estimator를 통해 각각의 카메라 시점에서 2D human centers를 지역화하고 feature maps를 제공합니다.**

이렇게 얻은 2D 위치로부터 3D human centers를 예측하기 위해, 본 연구는 MMG를 제안합니다. **MMG를 통해 다른 카메라 시점으로부터 같은 사람의 중심을 매치시킵니다.** 이후, triangulation을 통해 대략적인 3D human center 위치를 얻습니다. **여러개의 3D human center 후보들 중 CRG에 의해 선택됩니다.**

3D human centers가 예측된 후, 본 연구는 초기 3D poses를 만들기 위해 3D pose estimator를 적용합니다. pose estimation 정확도를 향상시키기 위해, **예측된 초기 3D poses들은 제안한 방법인 PRG에 의해 정제됩니다.**

&#x20;

### 3.2 Multi-view Matching Graph Module (MMG)

![Figure2-a](https://blog.kakaocdn.net/dn/XBxkD/btrAbLmBG1o/n65Nm9GYWKRmaAyps5FOqK/img.png)

2D pose estimator을 통해 만들어진 2D human center가 주어졌을 때, **MMG는 다른 카메라들 간 같은 사람의 center를 매칭시킵니다.**

* Vertices : 2D human centers(후보들 포함)
* Edges : 다른 카메라 시점들 간 한 쌍의 2D human centers들의 연결
* Target edge connectivity : 0 또는 1(같은 사람이면)

그러므로, multi-view matching은 edge connectivity가 중요합니다. MMG가 이 문제를 해결하기 위해 그래프 기반 모델을 적용시킨 것입니다.

##

#### - EdgeConv-E와 edge 속성 통합

EdgeConv는 graph convolution prediction입니다. 수학적으로 표현하면,

![formula1](https://blog.kakaocdn.net/dn/c7lyIk/btrAa4UMRlC/uPYYdSd7ERfnxiWNJxDKuk/img.png)

xv = 노드 피쳐 at v , xv' = 노드 피쳐 at v' , N(v) = v의 이웃 vertices, h = neural network (multi-layer perceptron)

일반적인 EdgeConv에서, feature 집합 과정은 노드 피쳐와 이웃한 노드들 간 상대적인 관계만을 고려하고, edge attributes들은 고려하지 않습니다.

**따라서, EdgeConv-E 모델에서는 edge attributes를 집합 과정에 추가했습니다.**

![Formula2](https://blog.kakaocdn.net/dn/kW42r/btrz7zPgMjt/T0XzTWhkVVraqDsORM9aok/img.png)

또한, Overfitting을 피하기 위해 ground-truth 2D human center 좌표에 uniform noises(0-25pixels 범위)를 추가하여 보강하였습니다.

Target edge connectivity와 예측한 edge connectivity간 Binary cross-entropy loss는 training에 사용됩니다. 모델을 학습하기 위해 Adam optimizer을 적용했습니다.

##

### 3.3 Center Refinement Graph Module (CRG)

![Figure2-b](https://blog.kakaocdn.net/dn/wJbNj/btrz9gn9oPe/NXtGafKDDUJZfecNNae2nk/img.png)

**CRG는 MMG에서 나온 3D human center detection 결과들을 refine합니다.** CRG는 3D search space에서 query points(아까 찾은 3D human centers)를 샘플링하고, query point가 human center이 될 확률을 예측합니다.

![implicit field representation](https://blog.kakaocdn.net/dn/cpX1nq/btrAaN0urkk/OHYndFh1lLsHm1UjcBcbrk/img.png)

이 모델은 기존에 사용된 volumetric representations 대신에 **implicit field representation을 사용합니다.** 위의 왼쪽 그림이 기존에 사용된 volumetric representations이고, 오른쪽인 implicit field representation입니다. Implicit field representation은 연속적이기 때문에 3D space에서 더 정확한 localization을 위한 real-value point를 찾는것을 가능케 합니다.

**Search space는 전체 3D 공간이 아니라, MMG로부터 매칭 결과에 기반한 공간으로 제한합니다.** 한쌍의 매칭된 2d human centers에 대하여, triangulation을 통해 coarse한 3d human center를 만드는데, 본 연구에서는 반지름이 300mm 안의 각각의 3D human center proposal를 둘러싸는 3D ball 을 만듭니다. 즉, search space는 3D balls의 집합입니다.

**각각의 query 3D point는 먼저 모든 2D camera views에 project되어, 2D 위치를 얻습니다.** 그 다음, 2D feature map을 통해 2D 위치의 point-wise feature representations을 얻습니다. 실제 2D 위치에 대한 피쳐들은 regular grid위에 위치한 4개의 가장 가까운 것들 사용해서 bilinear interpolation을 통해 얻어집니다.

Baseline model은 이렇게 다른 시점들로부터 얻은 point-wise features들을 연결하고, MLP로 처리합니다. 각각의 후보 점들에 대해, MLP는 human center가 되는 것에 대한 confidence score를 출력합니다. 이 접근법을 **MLP-baseline**이라 합니다.

&#x20;

> 그러나, **이 접근법은 2가지 문제점이 있습니다.**

1. 모든 시점에 동일한 가중치를 할당하고, 몇몇 시점에서 폐색을 처리할 수 없습니다
2. 다른 카메라 세팅(다른 수의 카메라)에 일반화 할 수 없습니다.

&#x20;

**이런 문제를 해결하기 위해, CRG에서는 각각의 3D query point에 대해 단순히 연결하는게 아니라 multi-view graph를 만듭니다.** Vertices는 각각의 카메라 시점의 2D projection을 나타냅니다. **Vertex 피쳐들은 3가지를 포함합니다.**

1. Image plane에서 추출된 visual features
2. Query point의 정규화된 3차원 좌표
3. 2D backbone으로부터 2D center confidence score

Edge는 이 2D projections들을 서로 연결시켜서, view들간 feature들을 종합합니다. CRG는 edge 피쳐가 없기 때문에 기본 Edge-Conv를 사용합니다.

##

> #### Point Selection

**MMG로부터 search region이 주어졌을 때, coarse-to-fine 방식으로 human center를 찾습니다.**

T=0일때 search space(전체 집합)에서부터 시작합니다.

T=t일때, search space에서 query points들을 샘플링합니다. **그래프 모델은 추출된 queries들을 처리하고, 점들의 human center 가능성을 예측합니다**. 가**장 높은 confidence score를 가진 점이 refined human center Xt로 선정됩니다.**

그 다음, search space를 업데이트합니다(Xt를 둘러싸는 3d ball subspace). Step size가 desired precision에 닿을때까지 반복됩니다.

그 결과, 기존에는 search space가 O(LWH)이었지만, MMG와 CRG적용 결과, search space 크기는 O(N)으로 감소했습니다.

&#x20;

> #### Training

**이 모델은 각각의 query point에 대해 confidence score을 예측합니다.** 본 연구는 CRG를 훈련하기 위해 training samples를 선정하는데 효율적인 방법을 고안했습니다. 샘플에는 2가지 종류가 있습니다: Positive samples(ground-truth human centers주위에 위치한) & Negative samples(far away from human locations). Positive와 Negative 비율은 4:1입니다.

X에 위치한 샘플에 대해, **target confidence score은 다음과 같이 계산됩니다**

![formula3](https://blog.kakaocdn.net/dn/dPh1gt/btrz7i09ci1/1AbZmsCTUxdZ5sNwAgKJW0/img.png)

CRG에 대한 training loss는 target confidence score과 예측된 confidence score사이의 loss입니다.

##

### 3.4 Pose Regression Graph Module (PRG)

사람은 쉽게 가려진 자세들을 인식할 수 있습니다, 왜냐하면 몸 구조에 대한 사전 지식과 multi-view geometry를 알기 때문입니다. 이것을 토대로, **본 연구는 multi-view geometry와 human joints들 간 structural relations을 고려하여 joint 위치들을 refine하는 PRG를 고안했습니다.**

**3D pose estimation 과정은 다음과 같습니다.**

![Figure4](https://blog.kakaocdn.net/dn/btR81g/btrAb8vDm4c/fcIivYL2O1i27H7bFXXKc1/img.png)

각각의 사람에 대해 PRG를 적용시킵니다. 먼저, 입력값으로 initial 3D pose를 받습니다. 본 연구에서, 초기 3D pose를 만들기 위해 간단한 pose regressor를 사용하였습니다. 초기 3D pose는 multiple 2D poses를 얻기 위해 모든 camera views에 project됩니다. Projected 2D poses에 기반해서 multi-view pose graph를 만듭니다. **이 그래프가 3D space에서 각각의 keypoint에 대한 offset를 예측합니다.**

**Multi-view pose 그래프에 대해, vertices는 특정 카메라 시점에서 2D keypoint를 나타냅니다**. 노드들은 다음을 포함합니다.

1. Projected 2D location에서 2D backbone networks의 feature map으로부터 얻은 visual features
2. One hot representation of joint type
3. 정규화된 초기 3D 좌표

**Multi-view pose 그래프는 2종류의 edges를 포함합니다.**

1. Single-view edges (특정 카메라 시점에서 다른 종류의 joints간 연결)
2. Cross-view edges(다른 시점에서 같은 종류의 two keypoints 연결)

&#x20;

PRG의 그래프 모델은 먼저 neighboring body joints와 multiple camera views 사이 message passing을 위해 2개의 연속적인 EdgeConv-E layers를 사용합니다. 그 다음, max-pooling layer가 cross-view features를 종합하고, 그래프를 만듭니다. Max-pooled features들은 body joints들간 효율적인 information flow을 통해 3개의 EdgeConv-E를 따르면서 업데이트됩니다. **마지막으로, 추출된 피쳐들이 2개의 fully connected layers를 가진 MLP를 통과하면서 각각의 관절에 대한 refinement vector를 만든다.**

Target offset은 초기 3D pose와 ground-truth 3D pose사이의 차이점입니다. 모델은 predicted offset과 target offset사이 loss를 사용합니다.

&#x20;

## **4. Experiment**

### **Dataset**

* **CMU Panoptic**
  * multi-person 3D pose estimation에 대한 제일 큰 데이터셋
  * 다수의 사람들이 실내에서 사회적 활동을 함
  * 평가지표: Mean Average Precision(mAP), Mean Average Recall(mAR), Mean Per Joint Position Error(MPJPE)
* **Shelf**
  * 5개의 카메라에 포착된 4명의 사람들이 선반을 분해하는 동영상
  * 복잡한 환경과 가려짐 현상이 반영됨
  * 평가지표: Percentage of correctly estimated parts(PCP3D)

### Result(Comparisons to the state-of-the-arts)

![table1](https://blog.kakaocdn.net/dn/bFvcN0/btrAa57O3dS/qMIVJ3snWb02cpKDkBxd3k/img.png)

* 기존 방법에 비해, 본 연구 모델이 **더 높은 정확도와 더 높은 재현율을 나타냄**
* Quantization error가 감소함 (Lower MPJPE)

### Ablation study

![table3](https://blog.kakaocdn.net/dn/2u5Ns/btrAbLHwK9k/bZD6HUy0wXBSMiqins4790/img.png)

* **Effect of MMG, CRG**
  * 기존 방법인 epipolar constraints과 비교 결과, **matching performace에서 향상되었음** (higher mAP)
  * MLP-Baseline과 비교 결과, **human detection 정확도**와 **3D human pose estimation 정확도 면에서 향상되었음**

![table3](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2\&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdaXPed%2FbtrAbwDTqGh%2FXZitYM2gJBglwQgksRvMN1%2Fimg.png)

* **Effect of PRG**
  * **3D pose estimation 정확도 면에서, PRG를 적용하니 향상되었음**

### Qualitative study

![qs](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2\&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbgBtcP%2FbtrAbG0vVAp%2F4tqjJjwKSkUmgiKxF5sYw0%2Fimg.png)

* 첫번째 행이 본 연구 모델 적용 결과, 두번째 행이 Tu et al. 모델 적용 결과
* Tu et al 에서는, 모든 시점에 대해 동일한 가중치를 적용하였습니다. 그 결과, 한개의 카메라에만 찍힌 여성은 탐지가 되지 않았습니다.
* 본 연구 접근법은 GCN을 통해 multi-view feature fusion을 적용하였습니다. 그 결과, **더 적은 False Negative을 얻었고, 높은 정밀도와 함께 human pose를 예측했습니다.**

### Memory and Runtime Analysis

![table6](https://blog.kakaocdn.net/dn/7mxop/btrAb8icvJP/Vc2aD8rjJAG9eOGfJVx9Gk/img.png)

* 기존에 제시된 방법과 비교했을 때, MMG,CRG,PRG 모두 **작아진 searching space덕분에 메모리 비용과 계산 시간이 감소했습니다.**

&#x20;

## **5. Conclusion**

본 논문에서는 **multi-view multi-person pose estimation을 위한 새로운 모델을 제안하였습니다**. **Multi-view features를 이용하기 위해 3개의 task-specific 그래프 뉴럴 네트워크를 디자인했습니다.** Human centers를 찾기 위해 MMG와 CRG를 제안하였고, 더 정확한 pose estimation 결과를 위해 PRG를 제안하였습니다. 그 결과, 새로운 접근법이 **기존의 접근법에 비해 정확도도 증가하였고, 메모리와 시간 측면에서도 효율성이 증가하였습니다.**

#### 개인적인 의견

다양한 각도의 이미지를 활용하여 다양한 사람의 3차원 pose estimation을 할 수 있는 접근법이 놀라웠습니다. Graph neural network를 적용하여 human center을 탐지하고, pose estimation을 refine하는데 사용된다는 것이 인상깊었습니다. 다만, 사전 훈련된 2D bottom-up pose estimator를 통해 각각의 카메라 시점에서 2D human centers를 지역화하는 것이나, 추출된 feature를 통해 initial 3D pose를 만드는 과정에 대해 좀 더 자세히 서술되었으면 좋겠다는 생각이 들었습니다. 추후에 이 과정에 대해서 자세히 연구해볼 예정입니다.

또한, 서로 다른 카메라 시점에서 얻은 geometric information을 어떻게 적용했는지도 자세히 공부할 예정입니다.

공사현장에서 추락 탐지나 장비 검사를 하는 과정에서 complex background나 multiple people 때문에 occlusion이 자주 발생되어서 pose estimation에 문제가 자주 생깁니다. 본 논문에서 나온 모델들이 이런 문제를 해결하기 위한 좋은 접근법이라 생각합니다. Multiple camera images를 활용한 3D pose estimation을 통해 공사현장에서 실제 사람들이 어떤 행동을 하는지 예측하는 연구에 적용시켜보고 싶습니다.

***

## **Author Information**

* Doil Kim
  * Affiliation : Master Course in KAIST KSE program
  * Research Topic : Data science, Pose estimation, Human factors

## **6. Reference & Additional materials**

* Reference : Graph based 3D multi person pose estimation using multi view images
