---
description : Guansong Pang / Deep Anomaly Detection with Deviation Network / 25th 2019 ACM SIGKDD international conference on knowledge discovery & data mining  
---

# **Deep Anomaly Detection with Deviation Network** 

이번에 포스팅할 논문은 Pang, G. et. al의 Deep Anomaly Detection with Deviation Network입니다. 

> Pang, G., Shen, C., & van den Hengel, A. (2019, July). Deep anomaly detection with deviation networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 353-362).

다음 논문 리뷰와 관련해서는 제 개인 블로그[**personal blog post(1)**](https://yscho.tistory.com/97) [**personal blog post(2)**](https://yscho.tistory.com/98)와 유튜브 영상[**personal Youtube Review**](https://www.youtube.com/watch?v=1lEtPCn-lcY)으로도 올려놓았으니 참고 바랍니다. 

## **0. What is Anomaly Detection**  

우선 본 논문의 구체적인 내용에 들어가기 앞서 Anomaly Detection의 개요 부분을 말씀드리고자 합니다.

그럼 차근차근 살펴보도록 하겠습니다. 

자 그렇다면 Anomaly Detection (AD)는 무엇을 의미하는 걸까요? Survey 논문에 나와있는 문구를 인용하면 다음과 같습니다. 

>_Anomaly Detection (AD) is the task of detecting samples and events which rarely appear or even do not exist in the available training data_

즉, 말 그대로 일반적으로 발생하는 event들과는 다른 특이한, 일반적인 특징을 띄고 있지 않은 샘플을 탐지하는 것을 의미합니다. 그렇다면 이런 궁금증이 들겁니다.

<br>

일반적인 Classification과 다른 것이 뭐지?

<br>

위 영어 문장에 잘 보면 'rarely appear or even do not exist in training data'라는 문구가 포인트입니다.

즉, 거의 학습 데이터에 존재하지 않거나 심지어 아예 존재하지 않는 경우의 이상치 데이터를 탐지하는 것을 의미합니다. 일반적으로 Supervised 기반 분류 모형은 분류할 클래스에 대한 충분한 데이터가 있는 것을 바탕으로 합니다. 하지만 Anomaly Detection은 anomaly로 분류되는 데이터가 거의 없기 때문에, 또 anomaly 데이터 간의 distribution이 유사하는 것을 보장할 수 없기 때문에 일반적인 classification과 차이점이 발생하게 됩니다. 

아래 그림을 보겠습니다. 

<br>

<center><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOAIob%2Fbtry4T85ixy%2F0gNRdGJJ4JLFaKR83upaK1%2Fimg.png" width="500"/></center>

#####  reference : Mohammadi, B., Fathy, M., & Sabokrou, M. (2021). Image/Video Deep anomaly detection: A survey. arXiv preprint arXiv:2103.01739

<br>

본 그림을 보면, 일반적으로 파란색 원 안에 들어가 있는 데이터들이 normal data라고 할 수 있습니다. 우리는 이 데이터들을 F라고 하는 feature representation을 통해 F1, F2를 사용하여 다음과 같은 plot을 그릴 수 있습니다. 녹색 점들의 경우 일반적으로 오토바이의 이미지들을 표현한 데이터로 볼 수 있습니다. 하지만 빨간색 자동차의 경우 우리가 일반적으로 관찰한 오토바이와는 조금 다른 특징을 가지게 됩니다. 이 경우 F를 통해 representation을 하게 되면 일반적으로 다른 위치에 데이터가 존재하게 되고 그 '격차'가 바로 이 데이터를 anomaly라고 측정하게 되는 기준이 됩니다. 이러한 방법을 통해 anomaly detection은 일반적으로 수행되게 됩니다. 

<br>

자, 그렇다면 Anomaly Detection을 수행하기 위해서는 다양한 데이터 셋에서 수행이 될 수 있는데 이에 따라 여러가지 케이스로 분류할 수 있습니다. 크게는 '**Supervised**', '**Unsupervised**' 그리고 '**Semi-supervised**' 케이스로 나뉘게 됩니다. 

<br>

그리고 용이한 표현을 위해 각각의 기호를 다음과 같이 정의해보겠습니다. 

U : Unlabeled data

N : Normal labeled data

A : Abnormal labeled data


U는 label이 되지 않은 데이터를 의미하고, N과 A는 각각 label이 된 정상 데이터와 비정상 데이터를 의미합니다.
이러한 상황에서 위 3가지 케이스는 다음과 같이 정리할 수 있습니다.

[1] Supervised Lerning ( N + A )

Supervised Learning의 경우 데이터가 충분히 많을 경우 세가지 케이스 중 가장 강력한 정확도를 가지게 됩니다. 데이터가 있는 상태에서의 예측은 데이터가 없는 경우보다 예측이 정확한 건 make sense하죠. 하지만, 문제는 데이터가 거의 없다는 것이 문제입니다. 실제 Real world에서는 labeled 된 데이터가 많지도 않을 뿐더러 labeled 데이터를 가지고 있다고 해도, abnormal 데이터의 경우는 극히 드문 경우가 많기 때문입니다. 이러한 경우 supervised learning은 데이터 불균형 문제를 맞이하게 됩니다. 또한 generalized된 판단을 수행할 수 없게 됩니다. 사실 anomaly 데이터는 우리가 관측한 케이스 외에도 다양한 형태로 존재할 수 있는데 모형은 우리가 관측한 anomaly 데이터의 분포를 바탕으로 판단을 수행하기 때문에 학습할 때 보지 않은 unseen anomaly에 대해서는 대응할 수 있는 힘이 부족하게 됩니다. 따라서 일반적으로 AD에서는 supervised learning을 사용하는 데에는 한계점이 존재하게 됩니다. 

<br> 

[2] Unsupervised Learning ( U )

그래서 일반적으로 labeled이 정의되지 않는 데이터를 바탕으로 학습을 수행합니다. 왜냐하면 이런 경우가 실제 real world 상황에 조금 더 유사하기 때문입니다. 
실제 labeled 데이터를 얻기 어려울 뿐더러 abnormal 데이터가 거의 발생하지 않는다는 것을 바탕으로 해서 가장 real world와 유사한 것이 바로 unsupervised learning이라고 할 수 있습니다. 그리고 위 Supervised Learning에서 한계점인 generalizability에 대해서도 보장이 되기 때문에 여러 단점을 보완한 방법이라고 할 수 있습니다. 

<br>
 
[3] Semi-supervised Learning ( N + A + U, N + A << U )

하지만, Unsupervised learning은 아무런 단점이 없을까요? Unsupervised의 가장 큰 문제점은 바로 anomaly data에 대한 'pre-knowledge'가 부족한 점에 있습니다. 가뜩이나 anomaly data가 적은데, 그 특징에 대한 아무런 정보가 없게 되면 모형이 온전히 anomaly의 특징을 잡는 것도 어려운 부분이 있겠죠. 그러면 labeled된 데이터를 사용하기에는 불균형 문제가 있고, 다 비지도 학습을 사용하기에는 사전 지식이 부족한 문제가 있어 둘 다 활용하자는 취지에서 나온 것이 바로 semi-supervisd learning입니다. 

이 방법의 경우 limited number of labeled data를 사전 지식으로 활용해 보다 학습을 강화하겠다는 취지에서 비롯됩니다. 

<br>

이러한 Background를 바탕으로 본 논문의 내용을 살펴보도록 하겠습니다. 

## **1. Problem Definition**  

우선 간단하게 논문의 Introduction부터 살펴보겠습니다.

AD task를 수행하는데 있어 본격적인 deep learning 방법이 적용되기 전까지 전통적인 방법은(SVM 같은) 다음과 같은 2가지 한계점에 직면하게 되었습니다.

* high dimensionality
* highly non-linear feature relation

첫 번째로 차원의 저주 문제가 있었고 두 번째로는 anomaly를 detect하기 위한 feature간의 linear하지 않은 관계로 인해 온전한 모형 설정이 어렵게 된 것입니다. 이러한 문제는 non-linear 방법을 바탕으로 한 neural net 방법이 등장하면서 위 문제를 해결하게 되었습니다. 

하지만 neural net이 적용되고 나서 봉착하게 된 문제는 크게 2가지가 있었습니다. 

* **anomaly data가 매우 적다는 것**

>_it is very difficult to obtain large-scale labeled data to train anomaly detectors due to the prohibitive cost of collecting such data in many anomaly detection application domains_

* **anomaly data 간의 유사성이 없다는 것**

>_anomalies often demonstrate different anomalous behaviors, and as a result, they are dissimilar to each other, which poses significant challenges to widely-used optimization objectives that generally assume the data objects within each class are similar to each other_

우선 labeled 된 anomaly data의 수가 매우 적다는 것과 그것들을 얻는 데에 cost가 많이 발생하게 됩니다.

또한 일반적으로 모형이 train data기반으로 학습을 수행하는데 학습한 anomaly 데이터와는 또 다른 형태의 distribution을 갖는 anomaly 데이터가 존재할 가능성이 크다는 것입니다. 

그래서 현대 딥러닝 기반의 AD 모형은 supervised 방법이 아닌 unsupervised 방법을 적용하여 이 문제를 해결하려고 하였습니다. 

바로 **Representation learning**을 활용하는 방법인데, 다음과 같은 two-step의 approach를 적용하였습니다.


1. They first learn to represent data with new-representation

즉, 데이터를 잘 표현할 수 있는 핵심적인 feature를 뽑아내는 방법을 학습하는 단계라고 할 수 있습니다.  

2. They use the learned representations to define anomaly scores using reconstruction error or distance metrics space

그리고 Representation learning이 발전하게 되면서 AD에서는 2가지 컨셉의 metric을 적용하게 되는데, 대표적으로 'Reconstruction Error'와 'Distance-based measures'을 들 수 있습니다. 

ex) Intermediate Representation in AE, Latent Space in GAN, Distance metric space in DeepSVDD

## **2. Motivation**  

**하지만 저자는 이러한 two-step의 approach가 갖는 문제점으로서, representation learning을 하는 부분과 anomaly detection을 하는 부분이 separate되어 있다는 점을 지적합니다.**

**또한 prior-knowledge의 부족으로 Unsupervised learning을 바탕으로 anomaly detection을 수행하는 경우, data noise나 uninteresting data를 anomaly 데이터로 인식하는 경우가 발생하게 됩니다.** 
이 문제에 대한 해결책으로 저자는 제한된 수의 labeled data를 활용해서 사전 
지식이 부족한 문제를 보완할 수 있다고 제안합니다.

따라서 위 방법에 대한 문제점을 극복할 수 있는 방법으로 저자는 'Anomaly Scores'를 학습하는 end-to-end learning 방법을 제안하게 됩니다. 

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FM56LS%2FbtrzadrcSgk%2F5q7PD9FVlcGeocnN8PFYSk%2Fimg.png height="500"/></center>

<br>

위 그림을 한번 살펴보겠습니다. (a)의 경우 기존 representation을 바탕으로 anomaly detection을 수행하는 방법을 도식화한 그림이고 (b)의 경우 본 논문에서 제안하는 방법의 도식도입니다. 위 그림에서 알 수 있듯 기존 방법은 데이터로부터 feature를 뽑아내어 이를 바탕으로 여러 metric을 적용 ( reconstruction error, distance metric ) 하여 detecting을 수행하게 됩니다. 하지만 저자는 이러한 부분이 **indirect**하게 모형을 optimize하는 것이라고 지적합니다. 하지만 (b)같은 경우는 데이터를 입력받으면 end-to-end로 바로 anomaly score를 도출하게 되어 detecting을 수행하게 됩니다. 즉, 모형을 **direct**하게 optimize할 수 있는 구조라고 언급합니다. 

또한 본 논문에서 제안하는 방법의 또 다른 novelty는, anomaly의 정도를 판단할 수 있는 reference score를 정의하였다는 점입니다. 즉, normal data로부터 추출된 평균적인 anomaly score와 현재 입력 간의 deviate된 정도를 바탕으로 판단을 한다는 점에서 기존 방법에서 갖는 방법과 차별된다는 점을 언급합니다.

<br> 

따라서 본 논문이 갖는 novelty를 다음과 같이 2가지로 정리할 수 있습니다.

1. **_With the original data as inputs, we directly learn and output the anomaly scores rather than the feataure representations._**

2. **_Define the mean of anomaly scores of some normal data objects based on a prior probability to serve as a reference score for guiding the subsequent anomaly score learning._** 


## **3. Method**  

자 그러면 본격적인 methodology를 살펴보도록 하겠습니다.

### **End-to-End Anomaly Score Learning**

위에서 언급한 prior knowledge가 부족한 문제를 해결하기 위해 unlabeled 데이터와 limited labeled 데이터를 혼재한 데이터셋을 사용합니다. 

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fpz3nl%2FbtryZHVg9s6%2FSW2PC0KyVud89mPvFQnu3k%2Fimg.png width="350"/></center>

N의 경우 unlabeled 데이터의 수를 의미하여 K의경우 매우 소량의 labeled된 anomaly 데이터를 의미합니다. 

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcpGeJV%2Fbtry7SVewZt%2F881Rh4aPrNNFagSXzt7N50%2Fimg.png width="350"/></center>

이렇게 데이터를 세팅하게 되면,

우리의 가장 큰 목적은 바로 anomaly score를 도출하는 이 파이 함수를 잘 학습해서 anomaly와 normal 간의 anomaly scoring 차이를 만드는 것을 목적으로 하게 됩니다. 

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F3ERIN%2FbtrzaclALeh%2FZhkwGvlYAaQ7VrMNW8KcWK%2Fimg.png width="200"/></center>
<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FUbCcl%2Fbtry7vmlCJ9%2Fb982kWn8jGAKbzGDlyXJw0%2Fimg.png width="500"/></center>

그렇다면 거시적인 Framework부터 살펴볼까요?

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBSZ9D%2Fbtry8PxBOB6%2FA8EWYWSv5qd28q89YThFGk%2Fimg.png width="400"/></center>

<br>

우선 'Anomaly Scoring Network'이라고 하는 anomaly score를 도출하는 네트워크 하나와 Reference score를 generate하는 부분으로 크게 2개의 아키텍처를 가짐을 알 수 있습니다.

Anomaly score를 도출하는 네트워크의 경우 2개의 세부적인 구조를 가짐을 알 수 있는데, input이 들어오고 나면 representation을 만드는 'Intermediate representation' layer와 이를 바탕으로 곧바로 anomaly score를 도출하는 layer로 구성이 되어 있습니다. 이에 대한 구체적인 구조는 뒤에서 자세하게 다루겠습니다.

그리고 Reference score를 도출하는 부분은 R = { x1, .., xl }을 보면 l개의 random sample을 뽑게 되는데 이 샘플들은 normal 데이터에서 뽑은 임의의 샘플들이고 이들의 평균을 계산한 µ_r 을 이용하여 이들로부터 떨어진 정도로 anomaly를 판단하는 방법으로 모형이 동작하게 됩니다. 

그러면 구체적으로 어떻게 구현되는지 살펴보겠습니다.

### **Deviation Network**

저자가 제안한 방법을 한 문장으로 요약하면 다음 문장으로 정리할 수 있을 것 같습니다.

> _"The proposed framework is instantiated into a method called **Deviation Networks (DevNet)**, which defines a **Gaussian prior** and a **Z Score-based deviation loss** to enable the direct optimization anomaly scores with an end-to-end neural anomaly score learner"_

다음 빨간색의 단어들이 핵심 아키텍처를 이루는 부분인데요. 우선, 첫번째 Deviation Network의 backborn을 이루는 Anomaly Scoring Network부터 살펴보도록 하겠습니다.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FJGHpz%2FbtrzprQH0qZ%2FmpTQRo3I7WBjBfMRKKX9rK%2Fimg.png width="300"/></center>

Anomaly Scoring Network는 ∅함수로 나타낼 수 있는데, 이는 크게 Intermediate representation space인 Q를 만드는 ψ네트워크와 anomaly score를 나타내는 η 네트워크로 구성이 됩니다.  

* Intermediate representation space (Q)
<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fdnn4IU%2FbtrzrK9BwBG%2FKpWM3ceCIbPKVKPOqKaOD0%2Fimg.png width="100"/></center>

* Total anomaly scoring network
<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F9PoPw%2Fbtrzprwoj8U%2FtWKEGfRvtREZK9re3UqA7K%2Fimg.png width="150"/></center>

이 ∅ 네트워크를 구성하는 2가지 sub network

[1] Q를 만드는 ψ 네트워크 ( feature learner )

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOsAcu%2FbtrzpV48e5d%2FMO5vdFH47lnoYDtwROkY4K%2Fimg.png width="150"/></center>

[2] Q에서 anomaly score를 도출하는 η 네트워크 ( anomaly score learner )

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F0bO5p%2Fbtrzn9cbQ0Q%2FYjMJxJhtU87hvKI0TVQgH0%2Fimg.png width="150"/></center>

그리고 ψ 네트워크는 Feature learner로서 H개의 hidden layer로 구성.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FtprzG%2FbtrzpVKN0gs%2FMc0MkOb4BNKJt6QFKgw2M0%2Fimg.png width="150"/></center>

따라서 다음과 같이 표기를 가능.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbaelvj%2FbtrzpsWq4zT%2FKgy1DJkq52dL7FBTyvvmc1%2Fimg.png width="150"/></center>

Feature learner를 구성하는 hidden layer는 들어오는 input, 수행하려는 task에 따라 다르게 구성됩니다. 가령 이미지 데이터를 feature representation을 해야 하는 경우는 CNN 네트워크를, sequence data같은 경우는 RNN 네트워크를 사용하게 됩니다.

그리고 η 네트워크의 경우 simple linear neural unit을 사용해서 스코어를 계산하도록 네트워크를 구성하였습니다. 

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FoehTT%2FbtrzpVKOfiu%2F3XPHaenP2jtYAkYiI1KarK%2Fimg.png width="400"/></center>

따라서 다음과 같이 전체 anomaly scoring network를 구성할 수 있게됩니다.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcmhY8U%2FbtrzrbGozM2%2F4i4haskM90WJez4gJKYtlK%2Fimg.png width="250"/></center>

저자는 이와 같은 방법을 통해 기존의 방법과 달리 direcly하게 data를 anomaly score로 mapping할 수 있다고 주장합니다.

<br>

저자는 여기서 추가적으로 이 score가 정말 anomaly한지 안한지를 같이 참고할 수 있는 reference score를 구축하는 방법을 제안하게 됩니다. 

기본적으로 이 reference score는 normal objects인 R에서 랜덤으로 뽑아 그 score를 기준으로 optimization에 활용하는 방식으로 동작하게 됩니다.

이 방법을 수행하는 방법으로는 2가지로 접근할 수 있습니다.

1. Data-driven approach

2. Prior-driven approach

우선 Data-driven 방법은 학습 데이터 X를 기반으로 하여 anomaly score를 도출하여 평균을 계산한 µ_r 를 도출해 활용하는 방법입니다. 하지만 이 방법은 제한이 많은데, 바로 이 µ_r 이 X 값이 바뀔 때 마다 조금씩 변하는 특징이 있기 때문입니다. 

그래서 본 논문에서는 **사전 확률 F에서 추출하는 reference score를 바탕으로 µ_r 를 계산하는 방법**을 채택하였습니다. 그 이유로 저자는 2가지 이유를 설명합니다. 

1. The chosen prior allows us to achieve good interpretability of the predicted anomaly scores

2. It can generate µ_r constantly, which is substantially more efficient than the data-driven approach

첫 번째로 다음과 같은 방법이 anomaly score를 예측할 시 good interpretability(좋은 해석?)을 갖는다고 합니다. 또한 µ_r 을 constant하게 고정할 수 있는 장점이 있습니다. 

하지만 잘 와닿지가 않죠. 사전 확률 F라는 것이 대체 무엇인지 감을 잡을 수 없으니까요. 

하지만 잘 와닿지가 않죠. 사전 확률 F라는 것이 대체 무엇인지 감을 잡을 수 없으니까요. 

가우시안 분포를 사용하면 정말 normal data의 anomaly score를 잘 대변할 수 있을까요?

저자는 다음과 같이 주장합니다.  

> _Gaussian distribution fits the anomaly scores very well in a range of data sets. This may be due to that the most general distribution for fitting values derived from Gaussian or non-Gaussian variables is the Gaussian distribution according to the central limit theorem._

즉, 앞서 AD task에서 대부분의 데이터는 normal이라는 이야기를 하였습니다. 그러면 이 normal 데이터들에 대한 anomaly score들도 어떠한 distribution을 따르게 되겠죠. 이 점수들 또한 sample들을 충분히 뽑게 되면 이 친구들도 gaussian distribution을 따르게 될 겁니다. 바로 **Central Limit Theorem** 때문이죠. 

즉, 다음과 같은 그림으로 쉽게 이해해볼 수 있습니다.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBTIGr%2FbtrzkpM7bbF%2FMM2ppNm1rXkgD1zZesLmkk%2Fimg.png width="400"/></center>

즉, normal 할수록 µ_r 에 더 가까운 점수를 형성하게 될거지만 abnormal할 수록 µ_r로부터 더 거리가 있는 점수가 나오겠죠. 이러한 deviation 정도를 사용하여 loss function을 정의하게 됩니다.

즉, normal 할수록 µ_r 에 더 가까운 점수를 형성하게 될거지만 abnormal할 수록 µ_r로부터 더 거리가 있는 점수가 나오겠죠. 이러한 deviation 정도를 사용하여 loss function을 정의하게 됩니다.

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdIpinw%2FbtrzrM0GfIk%2FI2IGMisxbwgx833klrwmh1%2Fimg.png width="400"/></center>

저 각각의 ri는 정규 분포에서 도출하며 저 ri는 랜덤한 normal 데이터 객체의 anomaly score를 의미하게 됩니다. 본 연구에서는 µ_r을 0, σ_r를 1로 설정하였으며, 랜덤 샘플은 CLT를 만족할 수 있는 충분한 양이면 전부 사용 가능하다고 명시합니다. 저자는 5000개의 샘플을 사용하였습니다.

자 그러면 이러한 reference score를 바탕으로 구체적으로 어떻게 loss function을 정의하는 지를 살펴보도록 하겠습니다. 본 논문에서는 deviation 정도를 측정하는 지표를 Z-score 방법을 차용해서 표현하고 있습니다. 아래의 loss를 저자는 contrastive loss라고 명명합니다.  

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdTpvEM%2FbtrzraOizYQ%2FUmqJrNaprfQTmBBtCwlfyK%2Fimg.png width="400"/></center>

#####  <center>Contrastive Loss</center>

<br>

예를 들어 normal 데이터면 저 ∅ 값이 µ_r이랑 근사하게 되겠고 abnormal 데이터면 그렇지 않겠죠.

그리고 위의 contrast loss를 바탕으로 최종 deviation loss는 다음과 같이 정의됩니다. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbTgy9y%2Fbtrzqz1QfLG%2FIUp2skbkoYxINpU7MFc4N0%2Fimg.png width="550"/></center>

#####  <center>Deviation Loss</center>

<br>

x가 anomaly인 경우 y = 1, x가 normal인 경우 y = 0이 됩니다. 

그리고 위 'a'의 경우 Z-score의 confidence interval paramter 가 됩니다. 

<br>

이게 무슨 말일까요? 

다시 한 번 구체적으로 term을 살펴보겠습니다. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb0i88Z%2Fbtrzn8En8Rx%2FBjaTc6qznSKzfgH0wU9GRk%2Fimg.png width="200"/></center>

> _Note that if x is an anomaly and it has a negative dev(x), the loss is particularly large, which encourages large positive derivations for all anomalies._ 

만약에 x가 anomalies인데 deviation이 음수이면 전체 loss 값은 커지게 됩니다. 따라서 모형은 이 anomaly data의 deviation이 큰 양수 값을 가지게끔 만들려고 하게 됩니다. 거의 a값에 근사해질 만큼으로 말이죠.

이 말의 의미는 다음 그림으로도 쉽게 이해할 수 있습니다.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbjVLGR%2FbtrznzoamSa%2FHasjnvVlfhx0oEwQIhnPv0%2Fimg.png width="400"/></center>

즉, dev 값이 0 근처로 오는 경우는 정상 데이터들이지만 a=5로 주었을 때는 이 경우 anomaly를 저 곳으로 근사시키는 경우로 적용시킬 수 있습니다. 본 논문에서도 a=5로 주어 사용했다고 하네요

원문은 정확히 다음과 같이 언급하고 있습니다.
> _Therefore, the deviation loss is equivalent to enforcing a statistically significant deviation of the anomaly score of all anomalies from that of normal objects in the upper tail. We use a = 5 to achieve a very high significane level for all labeled anomalies._

하지만 여기서 분명 물음표를 찍으시는 분들이 계실 겁니다. 자꾸 normal, normal하는데 애초에 우리는 limited된 anomalies를 제외하면 어떤 normal labeled 데이터를 모르는 상황인데, 왜 normal 데이터를 언급하냐는 부분이 궁금하실 겁니다. 사실 이건 다음과 같이 해결합니다.

> _We address this problem by simply treating the unlabeled training data objects in U as normal objects._

즉 그냥 normal 데이터로 간주하는 것입니다. Unlabeled 데이터가 전부 normal이라는 보장이 있는 것도 아니면서도 말이죠 ( 실제 Unlabeled 데이터에 abnormal이 들어있는 경우를 contaminated 되었다고 표현합니다 )

굉장히 이상하죠...! 왜 그렇게 할까요?

사실 공부하다보면 많은 semi-supervised learning에서는 Unlabeled된 데이터를 전부 normal이라고 치부하여 모형을 fitting하게 됩니다. 사실 그 이유는 2가지가 있는데 처음으로는 이것이 실제 real world와 굉장히 유사한 상황이기 때문입니다. 실제로도 우리는 많은 labeled 되지 않는 데이터가 존재합니다. 하지만 우리는 알듯, anomaly한 상황은 굉장히 scarce하게 발생하게 됩니다. 즉 대부분의 가지고 있는 데이터는 normal하다는 전제가 들어가는 것이죠. 따라서 이러한 real world 상황을 그대로 고려해주는 조치인 것입니다. 또한 이 매우 적은 anomaly 데이터가 실제 backpropagation을 수행할 때 SGD 기반의 optimization에 대해서는 그다지 영향력이 크지 않다는 것을 전제합니다. 즉, 모형의 성능에 그렇게 큰 영향을 주지 않을 것이라는 거죠. 따라서 거의 rule of thumb 식으로 semi-supervised learning에서는 unlabeled 데이터를 normal이라고 간주해서 사용하곤 합니다. ( 물론 이 부분에 대해서도 더 연구가 되어야할 것으로 보입니다. )

따라서 다음과 같이 DevNet 알고리즘을 슈도 코드로 정리할 수 있습니다. 

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FmYXPN%2Fbtrzra1PoFF%2F2TCqBjTtEebqG6s0NrdSB0%2Fimg.png width="600"/></center>

그렇다면 이제 loss function까지 디자인하여 training할 준비가 되었으니 다음으로 check 해봐야하는 부분은 바로, 'Interpretability'입니다. 즉, 어떤 경우일 때 normal, abnormal이라고 판단하냐는 거죠. 

본 연구는 다음 Proposition을 사용합니다. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb6pwta%2FbtrzmjzHdLP%2F89HskRtyfRlfu2ycz2dKr0%2Fimg.png width="500"/></center>

<br>

이 말의 의미가 뭘지 한번 생각해봅시다. 

일반적인 표준정규분포에서는 다음과 같은 성질이 있죠.

만일 µ_r가 0이고 σ_r가 1인 정규분포가 있고, p=0.95라고 하면, z(0.95)=1.96이 되므로=

µ_r = 0을 기준으로 ( µ_r - z ~ µ_r + z ) 구간이 결국 신뢰 구간이 됩니다.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FepoY3i%2Fbtrzprcerkk%2FmEGSMnkea8IeIzR42ambp0%2Fimg.png width="500"/></center>

<br>

하지만 만약 새로 받은 anomaly score가 이 boundary를 넘어가는 곳에 mapping이 된다면 어떨까요? 그 의미는 결국 다음과 같이 생각할 수 있습니다.

> _The object only has a probability of 0.05 generated from the same machanism as the normal data objects._

즉, normal 일 확률이 매우 낮게 되는 것입니다. 

왜 이런 form을 기준으로 anomaly를 판단하는 threshold를 설정했냐면,

> _This proposition of DevNet is due to the Gaussian prior and Z-Score-based deviation loss._

즉, Z-score 기반의 deviation loss를 정의했기 때문입니다. 

이렇게 학습이 수행되게 됩니다. 

## **4. Experiment**  

본 논문에서는 9가지의 **real-world 데이터셋**을 사용합니다. 구체적으로는 다음과 같은 데이터셋을 사용합니다. 

- Fraud Detection ( fraudulent credit card transaction )
: 신용 카드 거래 사기 탐지

- Malicious URLs in URL
: 이상 URL 탐지

- The thyroid disease detection
: 갑상선 비대증 사진 탐지

- ...

자세한 사항은 저자가 Appendix에 추가해놓은 아래 링크를 참고하면 좋을 것 같습니다.  

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fs5wZ9%2FbtrzMJXGKyh%2F6X5vfPXude8Vjjdll2tnF1%2Fimg.png width="500"/></center>

<br>

비교 집단으로 사용한 모형은 다음과 같습니다. 


[1] REPEN
<br> 

[2] adaptive DeepSVDD 
<br> 

[3] Prototypical Network (FSNet) 
<br> 

[4] iForest
<br>

REPEN의 경우 limited labeled data를 사용하는 neural net 기반의 AD network이고, FSNet의 경우 few show classification을 수행하는 네트워크입니다. 두 네트워크 모두 limited labeled 데이터를 사용하는 것이 특징입니다. DevNet과 동일한 조건이죠. 반면 Unsupervised 방법으로 AD를 수행하는 앙상블 기반의 모형 iForest도 비교집단으로 사용하였습니다. 

DeepSVDD 같은 경우는 굉장히 유명한 AD를 수행하는 알고리즘인데요, 여기서 저자는 어떠한 조작을 가해 DeepSVDD를 DevNet과 비교할 수 있는 조건으로 만들어 놓습니다. 바로 semi-supervised learning을 할 수 있는 형태로 만드는 것이죠. 

> _We modified DSVDD to fully leverage the labeled anomalies by adding an additional term into its objective function to guarantee a large margin between normal objects and anomalies in the new space while minimizing the c-based hypershere's volume._

즉, labeled된 anomalies를 사용할 수 있는 형태로 loss function을 조금 adjust했다고 언급합니다. 그리고 실제로 이러한 수정하는 과정을 거침으로서 original SVDD보다 더 성능이 잘 나왔다고 언급합니다.

아시는 분도 계시겠지만 DeepSVDD의 semi-supervised learning 버젼은 2019-2020년에 동일한 저자가 작성한 DeepSAD이라는 방법이 있습니다. 하지만 DevNet이 나올 땐 아직 DeepSAD이 나오기 전이기 때문에 당시 DevNet 저자는 이러한 heuristic을 적용했던 것 같습니다. 

이렇게 총 4개의 비교 집단을 사용하여 성능을 비교하였습니다. 

<br>

Metric은 어떻게 될까요?

일반적으로 AD에서 사용하는 metric은 AUROC와 AUC-PR을 사용합니다. 

이 둘에 대해서는 간략하게 정리해보도록 하겠습니다. 

<br>

* AUROC( Area Under Receiver Operating Characteristics )

많이들 알고 계시는 ROC curve입니다. 이 ROC curve를 이해하기 위해서는 우선 Confusion Matrix를 이해할 필요가 있습니다. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbfou5g%2FbtrzL9WAdnq%2F3PQIBWSzyR8qx6DfeSihsk%2Fimg.png width="500"/></center>

<br>

actual은 말 그대로 실제 label 값이고 pred는 prediction의 결과입니다. 다음과 같은 상황에서 우리는 True Positive, True Negative, False Positive, False Negative를 정의할 수 있고 이 수치들을 사용해서 다음과 같은 특징들을 정리할 수 있습니다. 

<br>

Sensitivity ( 민감도, True Positive Rate ) = ( True Positive ) / ( True Positive + False Negative )

Specificity ( 특이도, True Negative Rate ) = ( True Negative ) / ( False Positive + True Negative )

False Positive Rate = ( False Positive ) / ( False Positive + True Negative ) 

Precision = ( 정밀도, True Positive ) / ( True Positive + False Positive ) 

Recall = ( 민감도, True Positive ) / ( True Positive + False Negative ) 

Accuracy = ( 정확도, True Positive + True Negative ) / ( True Positive + True Negative + False Negative + False Positive )

<br>

이 개념들 중에서 TPR(True Positive Rate)과 FPR(False Positive Rate)을 사용하여 Curve를 그리면 다음과 같은 Curve를 그릴 수 있습니다. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fc19T4r%2FbtrzQidNtZp%2FddKzsh1Hj4svLdey9vrWAK%2Fimg.png width="500"/></center>

<br>

빨간 점선처럼 선이 형성되는 경우 random하게 classifier한 경우와 동일합니다. 그리고 빨간 점 아래에서 영역이 형성되는 경우는 0.5죠. 반면 보라색의 선의 경우 가장 모형이 강력한 경우, 즉 모든 경우를 맞춘 경우로 아래 면적은 1이 됩니다.

바로 이 면적의 크기에 따라 성능을 측정하는 것이 바로 AUROC입니다. 

* AUC-PR

하지만 이러한 방법은 한계가 있습니다. 바로 minor class의 error에 대한 비중이 낮게 잡기 때문인데요.

감이 잘 안오시죠?

이해를 위해 예를 살펴보겠습니다.

예를 들어 normal 데이터가 30,000개, abnormal 데이터가 100개 있는 데이터를 본다고 해봅시다.

여기서 똑같이 50개를 틀린다고 하면, normal에서는 50개 틀린 것이 상대적으로 적게 되지만, abnormal에서는 50개가 틀리면 전체 2분의 1이 틀리는 것이기 때문에 적은 수치라고 볼 수 없습니다. 즉 이 틀리는 정도를 갖게 해서는 안되는 것입니다.

이러한 부분을 잘 반영해주는 것이 바로 Precision과 Recall의 조합입니다. 

이 둘의 조금 더 명확한 이해를 위해 다음 예시를 들어보겠습니다. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd9MBAs%2FbtrzKKbmH9S%2FJeTMJZpQVte7xfDeMBgPH1%2Fimg.png width="500"/></center>

<br>

Object Detection을 수행하는 task를 예로 들어보죠. 여기서 모형이 다음과 같이 사람 2명이 있는 사진에서 한 사람을 탐지하고 이를 사람이라고 예측을 해봤다고 해봅시다. Precision 관점에서는 내가 예측을 시도한 경우가 다 잘 들어맞았다고 생각해서 100%의 정확도를 갖는다고 이야기합니다. 하지만 Recall 관점은 조금 다르죠. 사람이 실제로 한 명 더 있는데 이 사람은 맞추지를 못했으니 정확도는 50%이라고 이야기하는 겁니다.

즉 이 다른 두 관점을 적절하게 조합하여 metric을 만들면 위와 같은 문제를 극복할 수 있다고, 즉 minor한 class에 대해 더 적절한 가중치를 줄 수 있다고 판단한 것입니다. 이로부터 나온 개념이 바로 AUC-PR입니다.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F75oHD%2FbtrzNATLZdb%2FFOBpQBYbZ7kBnVnBDTktDk%2Fimg.png width="500"/></center>

<br>

자 그럼 다시 본론으로 돌아와서, 실험에 대한 구체적인 내용을 살펴보겠습니다.

우선 실험 환경 설정은 다음과 같이 설정하였습니다.

신경망의 깊이는 한 개의 hidden layer를 사용하였고 구체적인 파라미터 설정은 다음과 같이 설정하였습니다

- 20 neural units

- RMSProp Optimizer

- 20 mini batch

- ReLU

- L2 Norm Regularize

또한 network은 대부분의 데이터가 unordered multidimensional data라는 점을 고려하여 Multilayer perceptron을 사용하였습니다. 

이러한 설정을 바탕으로 모형 간 성능을 비교한 결과는 아래와 같습니다.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqsUaH%2FbtrzGhgYfLN%2FKAIoj79Hv0GgtCV5qrDyj0%2Fimg.png width="700"/></center>

<br>

census 데이터를 제외하고는 모든 경우에서 AUROC와 AUC-PR 모두 DevNet의 성능이 가장 우수함을 확인할 수 있었습니다. 

본 그림에서 Data Characteristic에서 나온 notation의 의미는 다음과 같습니다. 

* '# obj' -> 데이터의 수 

* 'D' -> 데이터의 dimension

* 'f1' -> 학습 데이터에서 anomaly 데이터 비중

* 'f2' -> 전체 데이터에서 anomaly 데이터 비중


<br> 

하지만 여기서 끝나면 뭔가 아쉽죠.

저자는 구체적인 여러 실험을 통해 DevNet의 효율성을 검증합니다.

<br>

### [1] Data Efficiency

첫 번째로 저자는 다음과 같은 궁금증을 해결하고자 하였습니다. 

> * _How data efficient are the DevNet and other deep methods?_
> * _How much improvement can the deep methods gain from the labeled anomalies compared to the unsupervisd iForest?_

**즉, 얼마나 본인들의 모형이 label이 추가되는 anomlies를 잘 활용하냐, prior knowledge를 잘 활용하냐를 측정하고자 하였습니다.**

이 때 base line으로 사용된 iForest의 경우 labeled이 없기 때문에 label을 주든 안주든 모형의 성능 차이는 없게 됩니다. 

다음 그림을 살펴보겠습니다.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Ft0saz%2FbtrzKQwytqF%2FzL4211l0OALkcWfJKrKdCK%2Fimg.png width="600"/></center>

<br>

추가되는 anomalies의 수에 대해 가장 높은 성능을 DevNet이 보임을 알 수 있습니다. 특별히 campaign, census, news20, thyroid 데이터에 대해서 추가되는 anomalies label에 대해 더 큰 폭의 성능의 향상을 보임을 확인할 수 있었습니다. 


### [2] Robustness w.r.t. Anomaly Contamination

두 번째로는 anomaly contamination에 대해 얼마나 강력한 대응을 수행하냐입니다. 

앞에서 잠깐 언급했었지만 Contamination이 정확히 뭘까요? 다음 글을 참고해봅시다.

> _To confuse data by sampling anomaly and adding it to the unreliable training data or removing some anomaly_

즉 일반적인 unlabeled 된 데이터를 normal 로 간주하는 상황에서 그 unlabeled된 데이터에 anomaly 비중이 contamination된 정도를 나타냅니다. 바로 이 양을 늘려감에 따라 모형이 얼마나 sensitive한 지를 측정한 결과입니다. 

저자는 0~20% 정도로 contamination을 주면서 성능의 변화를 측정하였습니다.

다음 그림을 참고해봅시다.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcZ21uH%2FbtrzQoSEvNJ%2FPPtgkXAIMtrm5V1DQjsgJ1%2Fimg.png width="600"/></center>

<br>

대체적으로 attack에 대해 robust한 성능을 유지함을 확인할 수 있습니다. 

다만 의문점은 'news20' 데이터셋에서는 유달리 DevNet이 drastic한 감소를 보임을 알 수 있었습니다. Text data로 구성된 news20에 대해 contamination에 대해서는 상대적으로 더 큰 감소폭을 보임을 알 수 있었습니다. 

### [3] Ablation Study

세 번째로 ablation study를 수행합니다.

저자는 DevNet의 아키텍처에 사용되는 여러 구성요소 { intermediate representation, FC layer, One hidden layer } 같은 것들이 실제 각각의 역할을 수행하는 지를 검증하였습니다. 저 요소들이 꼭 다 필요한 지를 확인하고자 한거죠.

그래서 총 3개의 비교 집단을 만들게 됩니다. 조금 더 간단하게 도식화해보면 다음과 같습니다. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcqCafv%2FbtrzKugeIL7%2FQQDGN7Uum8jv33nqwIOz40%2Fimg.png width="800"/></center>

<br>

즉, 기존의 Def가 DevNet이라면, 각각의 구성 요소를 하나씩 제거하거나 layer를 3개로 늘리는 조작을 가하게 됩니다. 

DevNet-Rep의 경우 마지막에 anomaly score를 scala 형태로 도출하는 FC layer를 제거하였습니다. 즉 20개의 dimension을 갖는 벡터를 바탕으로 성능을 도출하게 되는거죠. ( 이부분에서 어떻게 anomaly score를 도출했는 지가 명확히 언급되어 있지 않네요. ) DevNet-Linear의 경우 feature representation을 수행하는 network를 제거하고 바로 linear함수를 통해 anomaly score를 도출하게 됩니다. 마지막으로 DevNet-3HL은 3개의 hidden layer는 20개의 ReLU를 사용하는 하나의 layer가 아닌 1000 - 250 - 20 개의 ReLU를 사용하는 3개의 hiddne layer로 구성합니다. 

이에 대한 성능의 결과는 다음과 같습니다.

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FBo4XO%2FbtrzLILcDiC%2FIJt3BFq2HKebK9jV7qubkK%2Fimg.png width="600"/></center>

<br>

대부분의 데이터셋에서 AUROC는 본 DevNet의 성능이 우수하였습니다. 하지만 AUC-PR의 경우 일부분에서 Rep나 3HL이 더 우수하게 나온 경우도 존재하였음을 알 수 있습니다. 특히 census 데이터의 경우는 Rep에서 성능이 가장 우수했음도 확인할 수 있었습니다. 

하지만 대체적으로 Def의 성능이 가장 우수하였으므로 end-to-end learning을 구축하는 각각의 요소들이 전부 적절한 contribution을 갖는다고 설명합니다. 

또한 3HL 같이 더 깊은 신경망을 적용한 것이 왜 더 잘 동작하지 않는 지를 설명하는데, 이 이유로, 매우 적은 labeled anomalies를 매우 깊은 신경망을 쌓게 되면 그 특징을 놓치기 쉽다는 부분을 언급합니다. 즉, 대부분의 normal 데이터가 데이터를 이루는 상황에서 신경망을 깊게 쌓게 되면 anomalies의 특징을 잃기 때문입니다. 따라서 저자는 one-hidden layer가 가장 fit하다는 것을 설명합니다. 

### [4] Scalability

마지막으로 데이터의 size와 dimension에 따른 수행 시간이 어느정도 되는지, 즉 complexity를 확인하기 위한 실험을 추가적으로 진행하였습니다. size를 고정시켜놓고 dimension의 변화에 따른 수행 시간을 측정하였고 dimension을 고정시켜놓고 size의 변화에 따른 수행 시간을 측정하였습니다. 이에 대한 결과는 다음 그림과 같습니다. 

<br>

<center><img src=https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fctk3aV%2FbtrzMJKjFsF%2F0r6c9G6kpyhPhYz7KD3PS0%2Fimg.png width="600"/></center>

<br>

대부분의 모형에서 linear time이 소모되었지만, DevNet의 경우 data size에 대해서는 크기가 10만이 넘어가는 영역에서도 준수한 속도를 보임을 확인할 수 있었습니다. dimension에 대해서도 FSNet보다는 느리지만 다른 타 모형에 비해 더 빠른 성능을 보임을 알 수 있었습니다. 

# **5.Conclusion** 

따라서 본 논문이 제기하는 Contribution에 대해 정리하면 다음 3가지로 정리할 수 있습니다. 


> ### 1. _This paper introduces a novel framework and its instantiation DevNet for leveraging a few labeled anomalies with a prior to fulfill and end-to-end differentiable learning of anomaly scores_

대부분의 Unsupervised learning의 한계점을 극복하기 위해 적은 소량의 limited anomaly labeled 데이터를 사용했다는 점

> ### 2. _By a direct optimization of anomaly scores, DevNet can be trained much more data-efficiency, and performs significantly better in terms of both AUROC and AUC-PR compared other two-step deep anomaly detectors that focus on optimizing feature representations_

기존 representation 영역과 detecting 영역이 two-step으로 구분됨으로서 indirect하게 optimize하는 것이 아니라 direct로 optimize하는 방법을 제안

> ### 3. _Deep anomaly detectors can be well trained by randomly sampling negative examples from the anomaly contaminated unlabeled data and positive examples from the small labeled anomaly set._

또 다른 측면은 바로 normal distribution을 바탕으로 한 reference score를 사용하여 anomaly 정도를 측정하려고 한 점도 독특한 접근이라고 판단할 수 있었습니다. 

# **6.Code Review** 

코드 실습에 대해서는 포스팅으로 정리하는 것에 의미가 크지 않다고 생각하여 추가적으로 학습을 하시고자 하는 분들에게는 개인적으로 녹화해놓은 유튜브 영상의 좌표를 남깁니다. 따로 정리하지 못한 점 양해부탁드립니다..!

https://www.youtube.com/watch?v=1lEtPCn-lcY 

**코드 리뷰 : 55:30 ~ 끝까지**

지금까지 DevNet 리뷰였습니다!!

긴 글 읽어주셔서 감사합니다!!

---  
## **Author Information**  

* Yesung Cho (조예성) 
    * Knowledge Service Engineering, M.S Course, KS Lab (Prof. Mun.Y. Yi) 
    * Anomaly Detection in Computer Vision

## **Reference & Additional materials**  

### Github code

[1] 본 논문 원서의 github은 아래 링크와 같으며 'keras' 기반으로 구성되어 있습니ㅏㄷ. 

github code : 
https://github.com/GuansongPang/deviation-network

[2] 동일 저자의 논문으로 'DevNet'을 보다 더 개선한 논문이 존재합니다 (하지만 핵심적인 아키텍처는 거의 동일합니다). 아래 github 링크를 가면 DevNet 구현 코드를 볼 수 있고 'Pytorch'로 구현되어있음을 확인할 수 있습니다. 유튜브 영상의 리뷰도 해당 코드로 리뷰하였습니다.

[Pang, G., Ding, C., Shen, C., & Hengel, A. V. D. (2021). Explainable Deep Few-shot Anomaly Detection with Deviation Networks. arXiv preprint arXiv:2108.00462.](https://arxiv.org/abs/2108.00462)

github code : 
https://arxiv.org/abs/2108.00462
(this is what I reviewed in youtube)

<br>

### Other materials

본 포스팅을 위해 추가적으로 다음 reference들을 참고하였습니다. 

[3] Mohammadi, B., M., & Sabokrou, M. (2021). Image/Video Deep anomaly detection: A survey. arXiv preprint arXiv:2103.01739

[4] Ruff, L., Vandermeulen, R. A., Görnitz, N., Binder, A., Müller, E., Müller, K. R., & Kloft, M. (2019). Deep semi-supervised anomaly detection. arXiv preprint arXiv:1906.02694.

[5] Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., ... & Kloft, M. (2018, July). Deep one-class classification. In International conference on machine learning (pp. 4393-4402). PMLR.

[6] Shi, P., Li, G., Yuan, Y., & Kuang, L. (2019). Outlier Detection Using Improved Support Vector Data Description in Wireless Sensor Networks. Sensors, 19(21), 4712.

# **Feedback** 

본 논문을 리뷰한 다음 여러 한계점을 느꼈습니다. 첫 번째로 대표적인 indirect한 optimize 방법으로 소개되는 representation learning의 대표적인 방법 중 하나인 AE/GAN 기반의 모델을 SOTA 비교로 사용하지 않았다는 점입니다. 논문이 나왔던 당시 2019년도 GAN 기반의 AD 모델이 가장 많이 나오는 시점이었는데 이를 비교집단으로 활용하지 않는 것이 의문이었습니다. 또한, normal distribution에서 뽑은 anomaly score가 정말 실제 normal 데이터의 anomaly score를 계산했을 때와 유사하다는 것을 CLT를 통해서만 보장이 될지에 대한 의문도 품게 되었습니다. 

하지만 본 논문의 방향을 생각해보았을 때 참고할 만한 부분은 바로 'Reference Score'를 활용했다는 부분이었습니다. 결국 AD에서 핵심은 거의 Normal 데이터 밖에 없는 문제였는데, 이를 reference score를 바탕으로 anomaly를 구분하는 것이 research idea로서 참고를 하게 되었고 본 수업에서 저희는 이를 바탕으로 Normal 데이터를 unsupervised learning으로 충분히 학습한 어떤 모형이 Abnormal의 특징을 잘 학습한 모형이 내놓는 결과를 reference하면 더 좋을 것이다라는 아이디어를 착안해 Team Project 주제를 생각하게 되었습니다. 

그리고 본 논문 이후에도 계속해서 Unbalanced 문제를 해결하려고 하는 많은 연구들이 나오고 있는 상황입니다. 이 부분에 대해 더 공부를 해봐야겠다는 생각이 들었습니다.
