---
description : Dan Hendrycks / AugMix - A Simple Data Processing Method to Improve Robustness and Uncertainty / ICLR-2020
---

# **Title** 

AugMix

## **1. Problem Definition**  

대부분의 머신 러닝 모델은 training data의 representing 능력에만 의존하고 있다. 즉, training data의 분포에 따라서만 test data에 대해 추론하기 때문에 test data의 분포가 training data와 일치하지 않을 경우 좋은 성능을 내지 못한다. 그러나 train data와 test data간의 분포 불일치는 매우 빈번히 발생한다. 대부분의 모델이 이러한 데이터 분포의 변동 (data distribution shift)에 대해 강건하게 (robustly) 일반화하지 못하고 있다. 따라서, 모델이 test 데이터의 달라진 분포를 인지하거나 불확실성을 정확히 추정할 수 있다면 분포 변동에 대한 취약성을 개선할 수 있을 것이다.

그러나, 대부분의 머신 러닝 모델은 이미 training data와 test data가 독립적이고 동등한 분포를 따를 때 test data에 대해 overconfident 한 예측값을 생성한다. 이러한 overconfidence와 miscalibration 문제는 실제로 trainig data와 test data의 분포가 다른 경우 더 악화된다. 따라서, 본 논문에서는 모델이 분포 변동의 영향에 강건할 수 있도록 하는 데이터 augmentation 방법을 제안하고자 한다.

## **2. Motivation**  

데이터 분포에 작은 변형을 주는 것만으로도 기존의 classifier들은 크게 영향을 받지만, 변형에 대한 강건성(corruption robustness)을 향샹시키기 위한 기법이 많이 연구되어 오지 않았다. 실제로 저자는 2019년 논문에서 modern deep neural model들을 기존 ImageNet test 데이터에 다양한 변형(corruption)을 준 ImageNet-C 데이터에 대해서 test할 경우 분류 에러가 22%에서 64%로까지 증가함을 보였다.\[1\] 또한 Bayesian Neural Networks와 같이 불확실성(uncertainty)을 추정하는 확률적 방법론조차 데이터 변동이 일어나면 불확실성을 잘 추정하지 못하였다.\[2\]

이러한 data corruption 세팅에서 모델의 성능을 향상시키기 위한 방법이 몇몇 제안되어 왔는데, 가장 기본적으로 여러 corruption을 포함한 training 데이터로 학습하는 방법이 있다.
그러나 이 방법은 네트워크가 training 과정에서 특정 corruption들을 외우도록 하여 새로운 corruption이 있는 데이터에는 일반화 성능이 떨어진다. 또한, translation augmentation (ex. 이미지의 위치를 이동)을 적용하는 경우에는 image의 single pixel의 변동에도 매우 민감하게 반응하는 문제점이 존재하였다. 이 외에도 많은 데이터 augmentation 방법들이 제안되었지만, 대부분 강건성(robustness)과 불확실성 추정(uncertainty estimate) 간에 trade-off 관계를 가져 두 영역에 있어 모두 성능을 향상시키지 못해왔다.

따라서, 본 논문에서는 모델이 데이터의 분포 변동에 대해 강건하고 불확실성을 좀 더 정확하게 추정할 수 있도록 하는 새로운 데이터 augmentation 방법 AugMix를 제안한다.
AugMix는 표준 벤치마크 데이터셋에 대해 분류 정확도를 유지하면서 강건성과 불확실성 추정을 모두 향상시킨다. AugMix는 확률성(stochasticity)과 다양한 augementation 기법들을 적용하여 여러 개의 augmented image들을 mix한 후 Jensen-Shannon Divergence consistency loss를 통해 네트워크를 학습하는 방법론이다.


## **3. Method**  

AugMix는 간단한 augmentation 방법들을 consistency loss와 함께 사용한 점이 특징이다. 여러 augmentation 방법들이 확률적으로 샘플된 후 층층이 적용됨으로써 매우 다양한 augmented image를 생성한다. 이 후, 같은 input image에 대한 여러 개의 augmented image들이 classifier에 의해 consistent embedding (일관성 있는 embedding)을 갖도록 Jensen-Shannon divergence consistency loss를 이용하여 학습시킨다.

augmentation을 섞는 것은 다양한 변형을 생성하는데, 이는 모델의 강건성을 향상시키는 데에 매우 중요한 요소이다. 대부분의 Deep Network 모델들이 변동에 대해 강건하지 못한 이유는 모델이 고정된 augmentation 방법들을 외우기 때문이다. 이를 해결하기 위해 이전 연구들은 augmentation 방법들을 chain으로 구성하여 바로 적용하는 시도를 해왔지만, 이는 이미지가 data manifold 상에서 너무 동떨어진 이미지를 생성해낸다. 즉, 다음 그림에서와 같이 image degradation을 초래한다.
![figure1](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim_2/fig1.PNG?raw=true)

AugMix는 여러 개의 augmentation chain들로부터의 결과 이미지를 convex combination을 통해 믹스함으로써 image degradation 문제를 해결하면서 augmentation 다양성은 유지할 수 있다. 구체적인 AugMix 알고리즘은 아래 pseudo-code에서 확인할 수 있다.
![figure2](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim_2/fig2.PNG?raw=true)

### **Augmentations**  
앞서 언급하였듯이, AugMix는 여러 개의 augmentation 기법들로 이루어진 augmentation chain으로부터의 결과를 mix하는 방식이다. 이 때 augmentation 기법은 AutoAugment 방법을 이용한다. ImageNet-C 에 대해서 test하기 때문에 ImageNet-C에 적용된 변동들과 중복되는 augmentation operation(contrast, color, brightness, sharpness, cutout, noising, blurring)은 제외하였다. 따라서 ImageNet-C에 적용된 변동들은 모델이 test시에 처음 마주치도록 하였다. 
Rotation과 같은 augmentation operation적용 시에는 2도 에서 -15도 등 severity(강도)를 각 적용 시마다 랜덤하게 샘플링하여 적용하였다.   
이 후 k개의 augmentation chain을 샘플링하는데, k=3을 기본값으로 설정하였다. 각 augmentation chain은 랜덤으로 선택된 1~3개의 augmentation operation들로 이루어져 있다.
여기서 augmentation chain과 augmentation operation이 헷갈릴 수 있는데, 여러 개의 augmentation operation으로 구성된 하나의 chain이 augmentation chain이고, 이러한 augmentation chain을 다시 여러 개 사용하는 것이다.   

### **Mixing**  
각 augmentation chain으로부터 생성된 이미지들은 mixing을 통해 결합된다. 즉 k=3인 경우 각 augmentation chain들로부터 생성된 3개의 이미지들이 결합된다. AugMix는 간단하게 elementwise convex combination을 이용하여 이미지들을 결합하였는데, 이 때 사용되는 k개의 convex coefficients(계수)들은 Dirichelet 분포로부터 랜덤하게 샘플링된다. k개의 이미지들이 mix되고 나면, skip-connection을 이용하여 mix된 이미지와 원본 이미지를 결합한다. 이 때에도 convex combinatioin을 이용하여 결합하며, convex 계수는 Beta 분포로부터 샘플링된다. 이렇게 mix된 이미지와 원본 이미지가 결합된 이미지가 최종 augmented image이다. 
따라서, 최종 augmented image는  

`(1) augmentation operation 선택에 대한 randomness`  
`(2) 각 operation의 severity (강도) 선택에 대한 randomness`  
`(3) 각 augmentation chain의 길이 (몇 개의 operation으로 구성할지)에 대한 randomness`  
`(4) mixing weights (어떤 비율로 mix할지)에 대한 randomness`  

를 통합하고 있다.  

### **Jensen-Shannon Divergence Consistency Loss**
Augmix로 augemented된 image들이 주어질 때 모델은 `Jensen-Shannon Divergence Loss`를 이용하여 학습한다. AugMix를 통해 원본 이미지의 의미 정보(semantic content)가 거의 유지되었다는 가정 하에, 모델은 
$$x_{orig}$$
$$x_{augmix1}$$
$$x_{augmix2}$$
원본 이미지와 augmented image들을 유사하게 임베딩하도록 훈련된다. 
이는 원본 데이터와 augmented data의 사후 분포 (posterior distribution) 간에 Jensen-Shannon Divergence를 최소화하도록 함으로써 구현된다. 여기서 각 posterior 분포는 다음과 같다.  
$$p_{orig}=\hat{p}(y|x_{orig})$$
$$p_{augmix1}=\hat{p}(y|x_{augmix1})$$
$$p_{augmix2}=\hat{p}(y|x_{augmix2})$$

따라서, 원래의 loss _L_ 은 다음과 같은 loss로 대체된다.  
$$L(p_{orig}, y) + \lambda JS(p_{orig};p_{augmix1};p_{augmix2})$$
$$JS(p_{orig};p_{augmix1};p_{augmix2}) = \frac{1}{3}\[ KL(p_{orig}||M) + KL(p_{augmix1}||M) + KL(p_{augmix2}||M) \]$$
$$M = (p_{orig} + p_{augmix1} + p_{augmix2}) / 3$$

결국, Jensen-Shannon Consistency Loss는 모델이 다양한 분포의 input에 대해서 안정적이고 일관성있는 output을 생성하도록 한다.
 

## **4. Experiment**  

In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

### **Experiment setup**  
* **Dataset**
  * Training Dataset   
    * `CIFAR-10` : 32x32 사이즈의 컬러 natural images로 10개의 카테고리로 구성됨. (50000 training images / 10000 testing images)
    * `CIFAR-100` : 32x32 사이즈의 컬러 natural images로 100개의 카테고리로 구성됨. (50000 training images / 10000 testing images)
    * `ImageNet` : 1000개의 카테고리로 구성됨
  * Teset Dataset
    * `CIFAR-10-C` : original CIFAR-10 데이터에 변형(corruption)을 준 데이터셋
    * `CIFAR-100-C` : original CIFAR-100 데이터에 변형을 준 데이터셋
    * `ImageNet-C` : original ImageNet 데이터에 변형을 준 데이터셋
    각 데이터셋은 noise, blur, weather, digital corruption을 각각 5가지의 강도로 주어 총 15가지의 corruption으로 이루어진 데이터셋이다. 데이터 변동에 대한 모델의 영향을 확인하기 위한 실험이므로 training 과정에서는 이 15가지의 corruption은 포함하지 않았다.

* **Baseline**  
  * `CIFAR-10 & CIFAR-100` : AllConvNet, DenseNet, WideResNet, ResNeXt 아키텍쳐에 대해서 Standard, Cutout, Mixup, CutMix, AutoAugment, Adversarial Training 등의 다양한 augmentation 방법을 적용한 결과와 AugMix를 적용한 결과를 비교하였다.
  * `ImageNet` : ResNet50에 Standard, Patch Uniform, AutoAugment, Random AA, MaxBlur Pooling, SIN을 적용한 결과와 AugMix를 비교하였다.

* **Evaluation Metric**  
  * `Clean Error` : corruption이 추가되지 않은 clean data에 대한 classification error
  * `Corruption Error` : corruption이 추가된 data에 대한 classification error  
  * `RMS Calibration Error` : 모델의 불확실성 추정에 대한 평가 지표   
   
#### **`(1) Corruption Error (CE)`**  
$$E_{c,s}$$
- corruption c가 severity s로 주어졌을 때의 error rate

`(i) CIFAR-10-C & CIFAR-100-C`
$$uCE_{c} = \sum_{s=1}^5 E_{c,s}$$ 
- corruption c에 대한 unnormalized Corruption Error. corruption c에서 각 severity마다의 Error 값들의 평균을 의미한다.   
- 15개의 corruption들의 uCE 값의 평균을 최종 error로 사용하였다.  

`(ii) ImageNet-C`  
$$CE_{c} =  {\sum_{s=1}^5 E_{c,s}}/{\sum_{s=1}^5 E_{c,s}^{AlexNet}}$$
- corruption error를 AlexNet의 corruption error로 normalizing 해준다.  
- 15개의 normalized CE 값의 평균을 최종 error로 사용하였다.  

#### **`(2) RMS Calibaration Error`**   
모델의 불확실성 추정에 대한 평가로서 miscalibration을 측정하였다. Calibration 이란 모형의 출력값이 실제 confidence를 반영하도록 만드는 것이다. 예를 들어, 어떤 input의 특정 class에 대한 모델의 output이 0.8이라면, 80 % 확률로 그 class이다 라는 의미를 갖도록 만드는 것이다. 따라서 miscalibation error는 주어진 confidence level에서의 classification accuracy와 실제 confidence level에서의 classification accuracy 간의 RMS를 통해 측정하였다. 이렇게 정의된 RMS Calibarion Error의 수식은 다음과 같다.

$$\sqrt {E_{C}\[(P(Y=\hat{Y}|C=c)-c)^{2}\]}$$


### **Result**  

* **`CIFAR-10-C & CIFAR-100-C`**
![figure5](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim_2/fig5.PNG?raw=true)

위의 그림은 ResNeXt backbone에 다양한 방법의 augmentation을 적용하여 훈련시킨 후 CIFAR-10-C test dataset에 대한 standard clean error rate을 나타낸 것이다. AugMix가 기존의 augmentation 방법들인 Standard, Cutout, Mixup, CutMix, AutoAugment, Adversarial Training보다 절반 이하 수준의 error rate을 보여주고 있다. 다음은 ResNeXt이외의 backbone에 augmentation 방법들을 적용했을 때의 average classification error를 비교한 표이다. AugMix는 CIFAR-10-C, CIFAR-100-C 두 test dataset 모두 backbone 네트워크에 상관없이 가장 낮은 error rate을 보여주었다. 
![figure6](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim_2/fig6.PNG?raw=true)

다음 그림은 ResNeXt backbone에 Standard, Cutmix, AugMix를 적용하여 훈련시킨 모델의 CIFAR-10-C에 대한 RMS Calibration Error를 나타낸다. AugMix는 corruption이 없는 CIFAR-10 데이터와 corruption이 존재하는 CIFAR-10-C 모두에 대해서 calibration error를 감소시킴을 알 수 있다. 특히, corrption이 있는 데이터셋에 대해서 다른 augmentation 방법론에 비해 매우 큰 차이로 error를 줄였다.  
![figure10](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim_2/fig10.PNG?raw=true)

* **`ImageNet`**
다음은 ImageNet-C testdataset에 대한 여러 augmentation 방법의 효과를 Clean Error, Corruption Error(CE), mCE를 통해 평가한 표이다. mCE는 앞서 metric에서 설명하였듯이 15가지 corruption의 CE를 평균낸 것이다. 모든 augmentation은 ResNet-50 backbone으로 훈련되었다.

AugMix는 다른 augmentation 방법론들에 비해 Clean Error뿐만 아니라 Corruption Error를 감소시켰다. 특히, AugMix를 SIN과 결합하여 적용하였을 때 가장 corruption에 강건함을 보여주었다. 여기서 SIN은 Stylized ImageNet으로 원본 ImageNet 데이터뿐만 아니라 style transfer가 적용된 데이터에도 모델을 훈련시킴으로써 corruption에 대한 강건성을 높이는 augmentation 방법론이다.

또한 AugMix는 데이터 corruption의 강도(severity)가 점점 높아질 때, RMS claibration error에 대해 매우 안정적이고 강건함을 보여주었다. severity가 높아질수록 classification error가 증가함에도 불구하고 calibartion error는 거의 유지됨을 알 수 있다.
![figure11](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim_2/fig11.PNG?raw=true)


## **5. Conclusion**  
* Summary
AugMix는 랜덤하게 생성된 augmented image를 mix하고 Jensen-Shannon loss를 사용하여 데이터의 consistency를 유지하는 데이터 처리 기법이다. CIFAR-10-C, CIFAR-100-C, ImageNet-C 데이터셋 모두에 대해서 기존의 존재하던 augmentation 방법들보다 좋은 성능을 보여주었다. 특히, AugMix는 데이터 변동이 일어나도 calibration을 유지하며 안정성과 강건성을 보여주었다. 따라서, AugMix는 모델을 더 신뢰할 수 있도록 하므로, safety-critical 환경에서 효과적으로 적용될 수 있을 것으로 기대된다.

* Opinion

---  
## **Author Information**  

* TaeMi, Kim
    * KAIST, Industrial and Systems Engineering

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation
https://github.com/google-research/augmix  
* Reference  
  * \[1\] Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. ICLR, 2019.
  * \[2\] Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D Sculley, Sebastian Nowozin, Joshua V Dillon, Balaji Lakshminarayanan, and Jasper Snoek. Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift. NeurIPS, 2019.
  * https://3months.tistory.com/490

