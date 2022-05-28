---
description : Dan Hendrycks / AugMix - A Simple Data Processing Method to Improve Robustness and Uncertainty / ICLR-2020
---

# **Title** 

AugMix

## **1. Problem Definition**  

대부분의 머신 러닝 모델들은 training data의 represeting 능력에 의존하고 있다. 즉, training data의 분포에만 의존해서 test data에 대해 추론한다.
그러나 train data와 test data 사이의  간의 분포 불일치는 매우 빈번히 발생한다.
이러한 경우에, 모델은 데이터 분포의 변동 (data distribution shift)에 대해 강건하게(robustly) 일반화하지 못한다.
모델이 test 데이터의 달라진 분포를 인지하거나 불확실성을 정확히 추정할 수 있다면 분포 변동에 대한 취약성을 개선할 수 있을 것이다.
그러나, 대부분의 모델들은 이미 training data가 test data와 독립적이고 동등한 분포를 따를 때 overconfident 한 예측값을 생성한다.
이러한 overconfidence와 miscalibration 문제는 실제로 trainig data와 test data의 분포가 다른 경우에는 더 악화된다.
따라서, 본 논문에서는 모델이 이러한 분포 변동의 영향에 강건할 수 있도록 하는 데이터 augmenration 방법을 제안하고자 한다.

## **2. Motivation**  

데이터 분포에 작은 변형을 주는 것만으로도 기존의 classifier들은 크게 영향을 받지만, 변형에 대한 강건성(corruption robustness)을 향샹시키기 위한 기법이 
많이 제시되어 오지 않았다. 실제로 저자는 2019년 논문에서 기존 ImageNet test 데이터에 다양한 변형을 준 ImageNet-C에 대해서 test할 경우 modern model들의 분류 에러가 22%에서
64%로까지 증가함을 보였다.[1] 또한 Bayesian Neural Networks와 같이 불확실성(uncertainty)을 추정하는 확률적 방법론조차 데이터 변동이 일어난 경우 불확실성을 잘 추정하지 못하였다.[2]
이러한 데이터 변동 세팅에서의 성능을 향상시키기 위한 방법이 몇몇 제안되어 왔는데, 가장 기본적으로 여러 변동을 포함한 training 데이터로 학습하는 방법이 있다.
그러나 이 방법은 네트워크가 training 과정에서 특정 변동들을 외우도록 하여 새로운 변동이 있는 데이터에는 일반화가 잘 되지 않는다. 
또한, translation augmentation (ex. 이미지의 위치를 이동) 을 적용하는 방법은 image의 single pixel의 변동에도 매우 민감하게 반응하는 문제점이 존재한다.
이 외에도 많은 데이터 augmentation 방법들이 제안되었지만, 대부분 강건성과 불확실성 추정 간에 trade-off 관계를 가져 두 영역에 있어 모두 성능을 향상시키지 못해왔다.

따라서, 본 논문에서는 데이터의 분포 변동 하에서 강건성과 불확실성 추정 모두 향상시키는 데이터 augmenation 방법 AugMix를 제안한다.
AugMix는 표준 벤치마크 데이터셋에 대해 분류 정확도를 유지하면서 강건성과 불확실성 추정을 모두 향상시킨다.
AugMix는 확률성(stochasticity)과 다양한 augementation 기법들과 함께 Jensen-Shannon Divergence consistency loss를 사용하여 여러 개의 augmented image들을 mix하는 방법이다.


## **3. Method**  

AugMix는 간단한 augmentation 방법들을 consistency loss와 함께 사용한 점이 특징이다. 여러 augmentation 방법들이 확률적으로 샘플된 후 층층히 적용됨으로써 매우 다양한 augmented image를 생성한다. 이 후, 같은 input image에 대한 다양한 augmented image들이 classifier에 의해 consistent embedding (일관성 있는 embedding)을 갖도록 Jensen-Shannon divergence consistency loss를 이용하여 학습시킨다.

augmentation들을 섞는 것은 다양한 변형을 생성하는데, 이는 모델의 강건성을 향상시키는 데에 매우 중요한 요소이다. 대부분의 Deep Network 모델들이 변동에 대해 강건하지 못한 이유는 모델이 고정된 augmentation 방법들을 외우기 때문이다. 이를 해결하기 위해 이전 연구들은 augmentation 방법들을 chain으로 구성하여 바로 적용하는 시도를 해왔지만, 이는 이미지가 data manifold 상에서 너무 동떨어진 이미지를 생성해낸다. 다음 그림에서 확인할 수 있듯이, 이러한 방법들은 image degradation을 초래한다.
![figure1](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure1.PNG?raw=true)

AugMix는 여러 개의 augmentation chain들로부터의 결과 이미지를 convex combination을 통해 믹스함으로써 image degradation 문제를 해결하면서 augmentation 다양성은 유지할 수 있다. 구체적인 AugMix 알고리즘은 아래 pseudo-code에서 확인할 수 있다.
![figure1](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure1.PNG?raw=true)

### **Augmentations**  
앞서 언급하였듯이, AugMix는 여러 개의 augmentation 기법들로 이루어진 augmentation chain으로부터의 결과를 mix하는 방식이다. 이 때 augmentation 기법은 AutoAugment 방법을 이용한다. ImageNet-C 에 대해서 test하기 때문에 ImageNet-C에 적용된 변동들과 중복되는 augmentation operation(contrast, color, brightness, sharpness, cutout, noising, blurring)은 제외하였다. 따라서 ImageNet-C에 적용된 변동들은 모델이 test시에 처음 마주치도록 하였다. 
Rotation과 같은 augmentation operation적용 시에는 2도 에서 -15도 등 severity(강도)를 각 적용 시마다 랜덤하게 샘플링하여 적용하였다.   
이 후 k개의 augmentation chain을 샘플링하는데, k=3을 기본값으로 설정하였다. 각 augmentation chain은 랜덤으로 선택된 1~3개의 augmentation operation들로 이루어져 있다.
여기서 augmentation chain과 augmentation operation이 헷갈릴 수 있는데, 여러 개의 augmentation operation으로 구성된 하나의 chain이 augmentation chain이고, 이러한 augmentation chain을 다시 여러 개 사용하는 것이다.   

### **Mixing**  
각 augmentation chain으로부터 생성된 이미지들은 mixing을 통해 결합된다. 즉 k=3인 경우 각 augmentation chain들로부터 생성된 3개의 이미지들이 결합된다. AugMix는 간단하게 elementwise convex combination을 이용하여 이미지들을 결합하였는데, 이 때 사용되는 k개의 convex coefficients(계수)들은 Dirichelet 분포로부터 랜덤하게 샘플링된다. k개의 이미지들이 mix되고 나면, skip-connection을 이용하여 mix된 이미지와 원본 이미지를 결합한다. 이 때에도 convex combinatioin을 이용하여 결합하며, convex 계수는 Beta 분포로부터 샘플링된다. 이렇게 mix된 이미지와 원본 이미지가 결합된 이미지가 최종 augmented image이다. 
따라서, 최종 augmented image는  
(1) augmentation operation 선택에 대한 randomness  
(2) 각 operation의 severity (강도) 선택에 대한 randomness  
(3) 각 augmentation chain의 길이 (몇 개의 operation으로 구성할지)에 대한 randomness  
(4) mixing weights (어떤 비율로 mix할지)에 대한 randomness  
를 통합하고 있다.  

### **Jensen-Shannon Divergence Consistency Loss**
Augmix로 augemented된 image들이 주어질 때 모델은 Jensen-Shannon Divergence Loss를 이용하여 학습한다. AugMix를 통해 원본 이미지의 의미 정보(semantic content)가 거의 유지되었다는 가정 하에, 모델은 $$x_{orig}(원본 이미지), x_{augmix1}, x_{augmix2}$$를 유사하게 임베딩하도록 훈련된다. 이를 위해 
Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  

## **4. Experiment**  

In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

### **Experiment setup**  
* Dataset  
* baseline  
* Evaluation Metric  

### **Result**  
Then, show the experiment results which demonstrate the proposed method.  
You can attach the tables or figures, but you don't have to cover all the results.  
  



## **5. Conclusion**  

Please summarize the paper.  
It is free to write all you want. e.g, your opinion, take home message(오늘의 교훈), key idea, and etc.

---  
## **Author Information**  

* TaeMi, Kim
    * KAIST, Industrial and Systems Engineering

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation
https://github.com/google-research/augmix  
* Reference  
