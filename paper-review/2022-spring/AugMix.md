---
description : Dan Hendrycks / AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty / ICLR-2020(description)  
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
데이퍼 분포에 작은 변형을 주는 것만으로도 기존의 classifier들은 크게 영향을 받지만, 변형에 대한 강건성(corruption robustness)을 향샹시키기 위한 기법이 
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

* Author name  
    * Affiliation  
    * Research Topic

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation  
* Reference  
