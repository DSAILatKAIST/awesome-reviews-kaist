## Description

* Tete Xiao et al. / WHAT SHOULD NOT BE CONTRAST IN CONTRASTIVE LEARNING / ICLR 2021

---



##  1. Contrastive Learning

우선 Contrastive Learning 이 무엇인지 간략히 소개하겠습니다

Contrastive Learning 은 레이블 정보 없이 이미지의 임베딩을 학습하는 Self-supervised Learning의 하나로,

하나의 이미지를 여러 Augmentation 으로 만들어 이들을 유사하게 (positive pair),  다른 이미지들가는 다르게, (negative pair)

하도록 학습하는 것 입니다.


![1](/.gitbook/2022-spring-assets/KanghoonYoon_1/figure1.png) 


그림1은 Contrastive Learning의 대표적인 모델인 SIMCLR의 구조를 나타낸 그림입니다.

Dog의 Augmentation은 Attract 하고, Chair 의 Augmentation 끼리도 Attract 하며 Dog-Chair 간에는 모두 Repel 하여 임베딩 벡터를 배우는 것 입니다.

SIMCLR를 포함해 Contrastive Learning 을 수행하기 위한 손실함수는, 아래의 InfoNCE 입니다.


![2](/.gitbook/2022-spring-assets/KanghoonYoon_1/figure2.png) 

query Image를 기준으로 positive key 를 끌어당기고, negative key를 멀게하는 형태의 최적화를 수행합니다. 





## 2. Motivation

이제 본 논문의 제목 "What Should Not Be CONTRAST IN CONTRASTIVE LEARNING" 에 대해서 생각해 보겠습니다.

기존에는 어떠한 Positive Pair 든 당기고, 어떠한 Negative Pair를 멀도록 학습하였습니다.

저자는, Augmentation + Contrastive Learning 이 Inductive Bias를 사람이 인위적으로 정한다고 합니다.


*Inductive Bias*

Inductive Bias라 함은, 학습한 모델이 가지고 있는 특성 (Bias) 입니다.

예를들어, CNN 의 경우, 이미지를 Translation 해도 같은 결과가 나오도록 하는 "Translation Invariance" 한 Inductive Bias를

지역적인 정보를 취합하는 "Locality" 와 같은 Bias를 가진 모델입니다.



Contrastive Learning + Augmentation 에서의 Inductive Bias 는 아래 그림을 보면서 설명해보겠습니다.

![3](/.gitbook/2022-spring-assets/KanghoonYoon_1/figure3.png) 


그림에서는 Color / Rotation / Texture 변화 3가지 Augmentation을 예로 들고 있습니다.

Contrastive Learning은 같은 이미지로 파생된 Augmentation은 모두 Positive Pair로 가깝게 학습할 것입니다

그렇다는 것은, 사용자가 Augmentation을 선택하는 것에 따라, 모델이 배우는 Inductive Bias가 달라지는 것입니다.

e.g., Color Augmentation을 수행한다면, Color가 다른 모든 새들도 모양이 같다면, 같은 새로 학습할 것입니다 (Color Invariant)


### Challenge

하지만, Color (또는 다른 Augmentation)이 Fine-grained Classification을 위해서는, 필요한 정보일 수 있습니다.

예를들면, 거의 유사한 새의 형태여도, 색깔에 따라 다른 새인 경우가 많아 Color Invariance는 모델의 Generalization을 저해할 수 있습니다

또한, 많은 종류의 Augmentation을 사람이 모두 Tuning 해야하는 것은 상당히 힘든 작업 입니다.





### Idea

따라서 본 논문은 다음과 같은 아이디어로 위 문제를 해결합니다

**Augmentation에 따라 다른 Embedding Space를 만들자**

기존의 SIMCLR와 같은 모델에서 사용하던 임베딩 공간을 General Embedding Space로 학습하고,

그 공간으로 부터 다른 Augmentation별로 분리된 임베딩공간을 만들어, Augmentation을 수행해도

손실되던 정보 (Color Invariant) 를 줄이고자 합니다.


## 3. Method

아래 그림은, LooC (LEAVE-ONE-OUT-CONTRASTIVE LEARNING) 의 구조입니다.

하나의 View (ONE )와 나머지 View들을 일일히 Contrast 해주는 구조여서 Leave-one-out 이라고 칭하는 것으로 보입니다

![4](/.gitbook/2022-spring-assets/KanghoonYoon_1/figure4.png) 

본 그림에서는 편의를 2가지 Augmentation (Rotation / Color )만을 사용했을 때를 나타내었습니다

각각 Component 별로 자세히 살펴보도록 하겠습니다

### 1. View Generation (Random Rotation / Color Jittering)

#### Query view q 생성.

우선 Reference Image I (그림의 새) 를 Augment하여 Query view q 를 만듭니다.

이 때, Augmentation 별로 강도를 Sampling 합니다. e.g. {Rotation - 270도 / 색깔 - Red}

(여기서는 Rotatation/색깔만 고려하여 2 dimension이고,
 texture / cropping 같은 Augmentation을 한다면 2가지 옵션이 더 sample 될것입니다)

해당 Sample된 Augmentation Reference에 적용하여 Query View 를 얻습니다.

#### Key View k0, ..., kn 생성 (현재는 Rotation, Color만 사용해서 n=2)

First Key view k0 는 query view와 마찬가지의 과정을 반복하여 (다른 augment parameter로) First Key view를 생성합니다

이 q, k0 를 contrast 하는 파트가 기존 General Contrastive Learning과 동일한 과정입니다

k1 은 Rotation 을 공유하는 View 입니다. query와 같은 Rotation을 가지고 나머지 Aug parameter는 새롭게 Sample 합니다
k2 는 Color를 공유하는 View 입니다. query와 Color를 동일하게 가지고, 나머지 Aug parameter를 새롭게 Sample 합니다.

### Contrastive Embeddign Space

해당 q, k0, k1, k2 View를 Encoder f 에 통과시켜 임베딩공간 V 에서는 Contrast 한다면, 기존의 방법들과 동일 합니다

본 논문은 그렇게 하지않고, Projection Head h를 두어서, Augmentation 별 다른 Representation을 학습하게 합니다.

#### Projection Head

그림에서는 1) all-invariant, 2) Rotation-invariant, 3) Color-invariant 와 같이 3개의 Projection Head가 있습니다

3가지 Projected embedding space의 차이는 "positive pair를 묶어주는 기준" 입니다.

1) 에서는 q, k0, k1, k2 를 모두 positive pair로 간주하고 (기존 Contrastive Learning과 동일)
2) 에서는 Rotation은 동일하지만, 색깔이 다른 것을 Positive.
3) 에서는 Color는 동일하지만, Rotation이 다른 것을 Positive.

로 두어 학습하게됩니다

본 논문에서는,
General Embeddig space의 임베딩 벡터나 또는 모든 Project space 의 임베딩 벡터를 Concat 한 것이 학습된 Representation
으로 사용하면 된다고 합니다

제안된 방법을 요약해보자면, Augmentation이 1,..., i, ..., n 가 있을 때

i번째 Project Head는 Augmentation i만 고정한채 나머지를 바꿨을 때의 표현을 학습하는 것입니다.
즉 i번째 Head가 담고 있는 정보는, i에 관한 augment parameter가 바뀌면 표현이 바뀌도록 합니다

위의 예에서 들면, Rotation-variant Project Space에서는, 다른 Augmentation은 파라미터가 바뀌어도 동일한 임베딩이 나오지만
회전이 바뀐다면, 다른 임베딩이 나오게 될 것 입니다



## 4. Experiment & Result

실험에서는 Augmentation의 Inductive Bias에 대한 확인과, Fine-grained Representation을 인식하는지에 대해서 검증합니다


### Inductive Bias of Augmentation

이 실험에서는 Augmentation에 대해 민감한 Task를 수행하여, 제안된 임베딩이 Bias를 없앴는지 확인합니다

**Task A: Rotation degree를 맞추는 태스크. (Rotation에 민감)**
**Task B: 100-category Classification.**

하나의 임베딩에 Linear Classifier를 각각 만들어 두가지 태스크를 수행하여 진행하며, 결과는 아래와 같습니다

![5](/.gitbook/2022-spring-assets/KanghoonYoon_1/table1.png) 


Rotation augmentation을 수행하면, 성능이 떨어진 MoCo와는 달리,
제안된 모델은 Rotation에 민감한 Task에서 더 잘 수행하며, 동시에 Classification 성능도 기존방법과 유사한 수준이기 때문에
모든 Augmentation을 수행하고도 정보를 잘 담고 있는 표현을 배웠다고 할 수 있습니다.

### Fine-grained Representation

Table 2는 iNat-1k, CUB-200, Flowers-102 데이터를 통해 Transferability를 확인하였습니다.


같은 Color Augmentation만 수행해도, MoCO보다 LooC가 더 많은 정보를 보존하며,
LooC는 다른 Augmentation을 함에 따라 계속해서 성능이 오르는 것 을 확인할 수 있습니다



## 5. Conclusion

본 LooC 논문에서는 Contrastive Learning 에서 Augmentation이 수반하는 Inductive Bias 문제를
제시하고 다루었습니다. 모델이 Augmentation을 학습해도, 해당 Property를 잃지 않고 인지할 수 있도록 프레임워크를 제안하였습니다

### Take home message 
이미지 뿐만 아니라 다른 도메인에서도, Augmentation은 특별하게 중요시 여겨지는 경우가 많습니다
그 상황에 맞게 Inductive Bias가 무엇인지 생각해보고, 개선해본다면 충분히 적용가능하거나 확장가능한 논문으로 보입니다


### Author

**윤강훈 \(Kanghoon Yoon\)** 

* Affiliation \(KAIST Industrial Engineering Department\)
* \(optional\) ph.D students in DSAIL

## Reference & Additional materials

1. A Simple Framework for Contrastive Learning of Visual Representations. ICML 2020
2. Github- A Simple Framework for Contrastive Learning of Visual Representations.