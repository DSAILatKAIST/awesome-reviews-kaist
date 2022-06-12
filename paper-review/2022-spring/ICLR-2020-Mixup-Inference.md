---
description: >-
  Tianyu Pang, Kun Xu, Jun Zhu / Mixup Inference Better Exploiting Mixup to
  Defend Adversarial Attacks / 2020 The International Conference on Learning
  Representations (ICLR)
---

# Mixup-Inference

본 논문은 믹스업을 통해 adversarial Robustness를 향상시키는 방법을 제시한다. 기존 믹스업과 달리 Training 단계에서 input에 noise를 추가하는 방법이 아닌 inference된 결과를 mixup하는 방법이다.

논문링크 https://arxiv.org/abs/1909.11515

## **1. Problem Definition**

## 1) Adversarial Attacks

data label pair (x, y)에 추가로 adversaial binary 변수 z(1일 경우 adversarial)를 추가해서 사용한다.\
본 논문에서는 $l\_p$-norm 어택을 사용하며 $(||\delta||\_p \leq\epsilon)$, clean sample $x\_0$에 대해 노이즈가 추가된 $x$값은 다음과 같다.\


![adversarial input](../../.gitbook/2022-spring-assets/junghurnkim\_2/pre\_adv\_notation1.png)\


## 2) Mixup in Training

mixup 방법은 [Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)에서 처음 제시된 방법으로 두 샘플 $(x\_i, y\_i), (x\_j, y\_j)$의 선형결합을 통한 data augmentation기법이라고 볼 수 있다. 새롭게 만들어진 가상의 데이터 $(\tilde{x}, \tilde{y})$는 다음과 같다.\
$\tilde x = \lambda x\_i + (1-\lambda) x\_j$, $\tilde y = \lambda y\_i + (1-\lambda) y\_j$, where $\lambda \sim Beta(\alpha, \alpha)$\
두 샘플간의 빈공간에 새로운 가상 데이터를 채워넣음으로서 다양한 샘플을 학습하기 때문에 model의 전반적인 성능을 향상시키는 동시에 adversarial robustness도 향상시킨다고 알려져 있다. 또한, 일반적인 Adversarial Training과 비교했을때 연산량이 현전히 낮고, clean data에 대한 성능도 보장하기 때문에 이 부분에서 장점을 가지고 있다.

## **2. Motivation**

본 논문에서는 기존의 mixup training처럼 input sample을 섞어서 training data로만 활용하는 경우, input의 locality에서 크게 벗어나지 못하기 때문에 mixup 본연의 'globally linear behavior'를 극대화하지 못 한다고 주장한다. (여기서 'globally linear behavior'란 위에서 얘기한 두 샘플의 빈공간에 존재하는 새로운 가상 데이터를 사용하는 것을 의미한다.) 즉, 저자는 mixup의 효과를 더 극대화하기 위해 inference phase에서 mixup 할 것을 제안한다.

## **3. Method**

## **1) Notations**

$y$ : ground truth\
$\hat y$ : predicted label\
$y\_s \sim p\_x(y)$ : sampled label\
$x\_s \sim p\_s(x|y\_s)$: sampled data\
$\tilde x = \lambda x + (1-\lambda) x\_s$\
$z$ : adversarial flag

$F$ : mixup-trained model\
$H$ : linear function\
$G$ : extra non-linear part of F\


## **2) Mixup Inference**

저자는 잘 training 된 mixup 모델은 아래와 같이 각 clean input들의 선형함수의 결합으로 나타낼 수 있다고 설명한다.\
![method\_H](../../.gitbook/2022-spring-assets/junghurnkim\_2/method\_H.png)\
본 논문에서는 이러한 선형결합으로 전개되는 내용에 대한 자세한 내용은 언급되어 있지 않고, [MixUp as Locally Linear Out-Of-Manifold Regularization](https://arxiv.org/abs/1809.02499) 논문을 참조하여 바로 사용하는 것으로 설명되어 있다.\


다만, Adversarial Training 의 경우 noise에 대한 non-linear part G가 추가되고,\


![method\_H\_andG](../../.gitbook/2022-spring-assets/junghurnkim\_2/method\_H\_andG.png)\


최종적으로 mixup 값 $\tilde x$에 대한 결과는 다음과 같이 전개된다.\


![method\_H\_xtilde](../../.gitbook/2022-spring-assets/junghurnkim\_2/method\_H\_xtilde.png)\


Mixup Inference는 이 $F(\tilde x)$값의 N번 평균을 사용해서 model을 업데이트 하는 방법이다.

![method\_MI](../../.gitbook/2022-spring-assets/junghurnkim\_2/method\_MI.png)

(5)식을 더 정리하면,\
clean data $x\_0$와 sampled data $x\_s$에 대해 $H\_y(x\_0) = 1$, $H\_{y\_s}(x\_S) = 1$ 이고, 아래 수식은 $F$ 결과를 각각 $y$, $\hat y$에 대해 아래와 같이 나타낼 수 있다.\


![method\_MI\_y\_yhat](../../.gitbook/2022-spring-assets/junghurnkim\_2/method\_MI\_y\_yhat.png)\


$y$, $\hat y$ 두 경우 모두 $y\_s$(sampled label)의 영향을 받기 때문에 논문에서는 MI-PL($y=\hat y$), MI-OL($y\neq\hat y$) 두 가지 버전을 나눠서 함께 살펴볼 필요가 있다고 설명한다\


![method\_PL\_OL](../../.gitbook/2022-spring-assets/junghurnkim\_2/method\_PL\_OL.png)\


각각의 경우 $F$값을 요약하면 다음과 같다. ($z=1$인 경우 adversarial sample, $z=0$인 경우 clean sample을 의미한다.)\


![method\_PL\_OL](../../.gitbook/2022-spring-assets/junghurnkim\_2/method\_PL\_OL\_tab.png)\


추가로 Mixup Inference 전/후 robustness 향상정도, 실제 attack된 샘플 탐지정도에 대한 평가지표로 각각 Robustness Improving Condition(RIC)와 Detection Gap(DG)를 정의했다.\
![method\_RIC\_DG](../../.gitbook/2022-spring-assets/junghurnkim\_2/method\_RIC\_DG.png)\
RIC(10번 식)는 adversarial sample에 대해 학습 이후의 예측된 F값 즉, confidence가 낮아질수록, DG(11번 식)는 adversarial atack이 된 sample과 아닌 sample간의 confidence 차이가 클 수록 학습이 잘 된 결과임을 의미한다.

## **3) Theoretical Analysis**

위에서 제시한 RIC 식은 MI-PL, MI-OL 각각 아래와 같이 정리할 수 있다.

* MI-PL (Predicted Label)\
  ![MI\_analysis\_PL](../../.gitbook/2022-spring-assets/junghurnkim\_2/MI\_analysis\_PL.png)\

* MI-OL (Other Label)\
  ![MI\_analysis\_OL](../../.gitbook/2022-spring-assets/junghurnkim\_2/MI\_analysis\_OL.png)\

* Analysis results\
  ![MI\_analysis](../../.gitbook/2022-spring-assets/junghurnkim\_2/MI\_analysis.png)\


가장 왼쪽 plot과 가운데 plot에서 adversarial inputs(주황색 실선)를 보면 실제로 MI를 적용하지 않았을 때($\lambda = 1$)보다 MI를 적용했을 때($\lambda \neq 1$), $F\_y$는 증가하고 $F\_{\hat y}$은 감소하기 때문에 이는 RIC(10번식)을 만족하는 결과임을 알 수 있다. 가장 오른쪽의 plot은 \[$G\_k(\delta;x\_0)-G\_k(\lambda\delta;\tilde x\_0)$]을 그린 그래프이다. 그래프 값을 보면 위의 12, 15(편의상 원래 수식의 minus값을 그래프에 표시함) 식에서 제시한 조건을 모두 만족하고 있기 때문에(즉, RIC 성질을 만족한다는 의미) MI 방법이 adversarial training에 효과적인 방법임을 보여준다.

## **4. Experiment**

### **Experiment setup**

* Dataset : CIFAR-10, CIFAR-100
* Model : ResNet-50
* Adversarial Attack : Gaussian noise, Random rotation, Random cropping and resizing, Random cropping and padding
* Baseline : Mixup(기본적인 mixup training 후 attack에 대한 accuracy 측정), Interpolated AT([Interpolated Adversarial Training](https://arxiv.org/abs/1906.06784)에서 소개된 Mixup 방법을 이용한 AT method)

### **Result**

논문에서는 MI-PL과 MI-OL을 결합한 Mi-Combined 버전도 실험결과에 포함시켰다. MI\_PL을 적용하다가 adversarial input detection 값이 특정 임계값을 넘어가면 MI\_OL을 적용하는 방법이다.\


아래 실험결과를 통해, Mixup, Interpolated AT 모두 Mixup Inference method를 함께 사용했을때 더 좋은 성능을 보여주는 것을 알 수 있다.

### **CIFAR-10**

![result1](../../.gitbook/2022-spring-assets/junghurnkim\_2/result1.png)\


### **CIFAR-100**

![result2](../../.gitbook/2022-spring-assets/junghurnkim\_2/result2.png)\


## **5. Conclusion**

모델의 예측값을 Mixup 하는 발상이 새로워서 관심있게 본 논문이었다. Input mixup, Manifold Mixup에 이어서 새로운 방법의 mixup 방법으로 생각 할 수 있을 것 같다. 하지만 motivation에서 제시했듯이 Inference단계에서 Mixup하는 것이 Mixup 본연의 'globally linear behavior' 성질을 확대시킬 수 있을 거라는 주장에 대한 근거는 명확하게 증명되지 않고 실험적으로 성능비교만 제시된 것이 아쉬운 점이었다.

***

## **Author Information**

* 김정헌(JUNGHURN KIM): Master student, KSE, KAIST

## **6. Reference & Additional materials**

* github https://github.com/P2333/Mixup-Inference
