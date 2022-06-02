---
description: >-
  Robin Hesse / Fast Axiomatic Attributions for Neural Networks / 2021
  Conference on Neural Information Processing Systems (NeurIPS)
---

# XGradient

본 논문은

* 효율적인 Axiomatic Attributions의 필요성을 먼저 제시하고,
* X-DNN(nonnegatively homogeneous DNN)의 경우 기존 방법의 계산복잡도를 줄일 수 있음을 증명한다.
* 또한, nonnegatively homogeneous 하지 않은 DNN도 X-DNN으로 변환 할 수 있음을 함께 보여준다.

논문링크 https://arxiv.org/abs/2111.07668

## **1. Problem Definition**

## 1) Atrribution Method

모델의 예측결과에 대한 특정 input 값의 기여도를 의미한다.\
예를 들면, 모델이 특정 사진을 고양이로 예측했을때 그 사진에서 정확히 어떤 부분(얼굴, 귀 등)이 예측결과에 유의미한 정보를 많이 포함하고 있는지를 나타낸다. input의 영향력을 판단하는 방법으로는 크게 perturbation 방법과 backpropagation 방법이 있다. perturbation 방법은 인풋값으르 반복적으로 변경해가면서 실질적인 모델예측값의 변화를 측정하는 방법이고, backpropagation 방법은 학습과정에서 역전파된 gradient를 사용하는 방법이다. backpropagation 방법은 추가작업 필요없기 때문에 perturbation 방법보다 더 효율적인 방법이다. 가장 쉬운 backpropagation 방법으로는 학습결과 역전파된 gradient 값을 직접적인 input의 기여도로 성립시키는 방법이 있다. 더 많은 방법은 [Towards better understanding of gradient-based attribution methods for Deep Neural Networks(2018)](https://arxiv.org/abs/1711.06104)에 설명되어 있다.

## 2) Axiomatic Attributions

본 논문의 motivation이 된 [Axiomatic Attributions for Neural Networks (2017)](https://arxiv.org/abs/1703.01365)의 저자는 좋은 Attribution method가 가져야할 6가지 공리를 정의하고, 이를 모두 만족하는 Integrated Gradient(IG)를 제시했다.\
![Axioms](../../.gitbook/2022-spring-assets/junghurnkim\_1/Axioms.png)\
다양한 이유로 위의 성질을 모두 만족하는 기존의 Attribution 방법은 없으며, IG는 Gradient Saturation Effect를 해결함으로서 위의 성질을 모두 만족한다고 설명한다.

Gradient Saturation Effect란.\
우리가 흔히 알고 있는 [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing\_gradient\_problem)과 유사하게, Attribution Score를 구하는 과정에서도 모델에 사용되는 함수의 gradient가 양 끝단(특히, 마지막 부분)에서 매우 작은 값이 될 경우 문제가 된다. 실제로 input feature의 기여도는 높지만 우리가 최종적으로 확인한 gradient 값은 작기 때문에 해당 input feature 가 별로 중요하지 않은 부분이라고 볼 수 있기 때문이다.

아래 그림은 물을 뿌리고 있는 fireboat 이미지와 실험결과를 보여준다.\
(이하 그림에 대한 설명은 저자의 자세한 설명을 구할 수 없어서 주관적인 해석이 포함되어 있음)

* 첫번째 줄은 실제 input image의 픽셀값을 0에서 조금씩 증가시키는 과정이고,
* 두번째 줄은 이에 따른 각 픽셀의 gradient 값을 강조해 보여준 결과이다 ![Gradient\_Saturation\_Effect 설명](../../.gitbook/2022-spring-assets/junghurnkim\_1/Gradient\_Saturation\_Effect.png)

왼쪽에서 오른쪽으로 갈수록 사진의 밝기가 밝아지는 양상인데, 사람의 눈으로 볼때는 밝기만 다른 비슷한 사진으로 보이지만 학습결과는 다르게 나타난다. 스코어가 급격히 증가한 구간(interesting gradient)에서는 fireboat와 관련된 물줄기 부분이 잘 강조되어 보이는 반면, 유의미한 증가값이 없는 구간(Uninteresting gradient)에서는 fireboat와 관련이 없는 부분이 강조되어 있다.

이를 통해 우리는 특정 이미지에서 모델 학습에 중요한 역할을 하는 픽셀이 학습에 실질적인 영향을 주는 시점이 존재하고 특정 시점의 gradient값만으로는 기여도가 높은 feature를 포착하기 어렵다는 것을 알 수 있다.

## 3) Integrated Gradient (IG)

위의 논문에서 소개된 위에 언급된 6가지 axiom을 모두 만족하는 방법이다. Gradient Saturation Effect를 해결하기 위해 \[baseline, input feature] 직선 이동경로 구간에서 누적 gradient를 계산한다. 즉, baseline에서 원래의 input 값으로 각 픽셀의 값을 linear하게 증가시키면서 누적된 gradient값을 Attribution score로 사용하는 방법이다.

수식으로는 다음과 같다. ![IG 수식](../../.gitbook/2022-spring-assets/junghurnkim\_1/IG\_math.png)\
x와 x'은 각각 input과 baseline을 의미하며, 대부분의 image task에서는 baseline으로 black image를 사용한다.\


일반적으로 위의 수식의 input 변화값에 대한 모든 값을 적분하기에는 오버헤드가 심하기 때문에 실제 구현은 적분에 대한 리만근사값을 사용하는 것으로 대체한다. ![IG 근사 수식](../../.gitbook/2022-spring-assets/junghurnkim\_1/IG\_math\_approximation.png)\
m은 스텝사이즈(적분근사범위)를 의미하고, 실험적으로 20\~300 사이로 선택할 수 있다고 논문에 제시되어 있다.

## **2. Motivation : Efficient Attribution Method**

본 논문의 저자는 다음가 같은 두가지 이유를 통해 Attribution method의 효율성을 강조한다.

1. 기존의 IG(Integrated Gradient)는 적분 근사를 적용하더라도 근본적으로 computation overhead가 심한 방법이다.
2. Attribution 값은 예측 결과를 설명하기 위해 사용되기도 하지만 모델의 학습방향을 직접적으로 컨트롤하기도 한다.

## **1) Convergence of Integrated Gradients**

아래는 논문의 appendix에 첨부된 IG(step=300)와 IG(각 step size)를 비교해놓은 그림이다. step=300을 기준으로 했을때 어느정도 합리적인(300일때의 성능으로 수렴하는) step의 크기는 100을 넘어가는 것을 확인 할 수 있다.\
![IG costs 설명](../../.gitbook/2022-spring-assets/junghurnkim\_1/IG\_costs.png)\
따라서 저자는 기존의 IG는 최소 100회 이상의 gradient를 계산해야하는 효율적이지 않은 방법이라고 주장한다.

## **2) Attribution Priors**

저자는 Attribution method의 효율성을 개선해야 하는 또 다른 이유로 Attribution Priors를 소개한다.\
![Attribution priors 수식](../../.gitbook/2022-spring-assets/junghurnkim\_1/Attribution\_priors.png)\
Attribution Priors는 domain knowledge를 인코딩하는 방법중에 하나로, 일반적인 loss 뒤에 feature attribution _A_의 가중치를 조절하는 조건을 추가함으로서 원치 않는 feature에 패널티를 주고 모델이 bias 되는 경향을 줄여주는 방법이다. 이 방법은 모델을 컨트롤 하는 목적으로 학습 중간에 개입하기 때문에, Attribution을 계산하는 연산속도는 모델의 학습속도에 직접적인 영향을 줄 수 있다. 이 외에도 Attribution은 다양한 방법으로 모델의 구성요소가 될 수 있을 것이다.

## **3. Method**

저자는 nonnegatively homogeneous DNN을 X-DNN으로 정의하고, X-DNN의 경우 Integrated Gradient 값은 InputXGradient (한 번의 forward/backward pass로 얻은 Gradient에 input을 곱한 값)와 동일함을 보여준다. 또한 X-DNN에서 얻은 attribution 값을 XG로 정의하고, nonnegatively homogeneous DNN 아니더라도 이러한 XG를 구할 수 있는 방법을 함께 제시한다.

## **1) Nonnegatively homogeneous DNN (X-DNN) and X-Gradient (XG)**

$$F(\alpha x) = \alpha^k F(x)$$, ($$\alpha > 0$$) 를 만족하는 함수 $$F$$를 homogeneous degree가 k인 [positive homogeneous function](https://en.wikipedia.org/wiki/Homogeneous\_function)이라고 한다. 저자는 이 성질을 이용해서 positive homogeneous한 DNN에 대해서 baseline을 0벡터로하는 IG를 구한다면, 그 결과는 Gradient에 input을 곱한 값과 동일하다고 보여준다. 이 말은 positive homogeneous function에 대해서는 gradient를 누적해서 여러번 계산할 필요없이 단 한번의 forward/backward pass로 매우 빠르게 구할 수 있다는 것을 보여준다.\


| Positive homogeneous                                                                                              |
| ----------------------------------------------------------------------------------------------------------------- |
| ![positive homoegeneous](../../.gitbook/2022-spring-assets/junghurnkim\_1/positive\_homoegeneous.png) |

(2)번식은 IG정의에 의해서 모든 수식이 명백하게 이해가 가는 부분이지만, (3)식의 경우 왜 $$\beta$$를 0으로 보내는 식을 썼는지 정확한 이유는 알 수 없다. 다만 (2)번식 적분의 시작이 0인 것은 IG정의에서 비롯된 값임을 고려했을 때, (3)번식의 적분의 시작이 0인 것 은 우리가 baseline을 0으로 설정한 부분이기 때문에 이 점을 강조하기 위함이 아닐까 추측한다.

이어서 Nonnegatively homogeneous한 DNN을 X-DNN, 0 baseline에 대한 X-DNN의 attribution 값을 X-Gradient라고 정의한다.

| Nonnegative homogeneous                                                                                               |
| --------------------------------------------------------------------------------------------------------------------- |
| ![nonnegative homogeneous](../../.gitbook/2022-spring-assets/junghurnkim\_1/nonnegative\_homogeneous.png) |

한가지 본 논문의 흐름에서 어색한 점은 X-DNN을 정의하는 이 부분부터는 homogeneous degree가 1인 DNN만 다루고 있다는 것이다. Integrated Gradient와 InputXGradient의 동일성은 이미 다른 연구에서 증명된 내용이지만, 저자는 homogeneous degree가 1보다 큰 경우에 대해서도 동일함을 보여주었기 때문에 본 논문이 novelty를 가진다고 설명한다.

> _While Ancona et al. \[1] already found that Input×Gradient equals Integrated Gradients with the zero baseline for linear models or models that behave linearly for a selected task, our Proposition 3.2 is more general: We only require strictly positive homogeneity of an arbitrary order k ≥ 1. This allows us to consider a larger class of models including nonnegatively homogeneous DNNs, which generally are not linear._

하지만 이 언급 이후 논문에서는 homogeneous degree가 1보다 큰 경우에 대한 추가적인 언급이나 실험이 없었기 때문에 이 부분이 조금 아쉬운 점인 것 같다.

## **2) Constructing X-DNN**

우리가 사용하는 DNN은 내부적으로 activation function과 pooling function으로 구성되어 있다. DNN에서 주로 사용되는 activation function(ReLU, LeakyReLU, PReLU)과 pooling function(average pooling, max pooling, min pooling)은 모두 위에서 제시한 nonnegative homogeneous function이기 때문에 부분의 DNN의 경우 X-DNN으로 바로 사용가능하다. 하지만, linearly bias term이 있는 경우에는 homogeneous하지 않아서 바로 X-DNN으로 사용할 수 없고, 이를 해결하기 위해 논문에서는 bias term을 제거하고 X-DNN으로 사용할 것을 제안한다.

## **4. Experiment**

## **1) Removing the bias term in DNNs**

![regular X-DNN 비교](../../.gitbook/2022-spring-assets/junghurnkim\_1/regular\_x\_dnn.png)\
X-DNN을 구성할때 bias term을 제거할 경우 DNN 성능이 떨어질 수 있기 때문에 저자는 이에 대한 정당성을 먼저 제시한다. X-DNN의 경우 Top5 accuracy의 경우 1퍼센트 미만의 조금의 성능을 희생하고 IG와 Input×Gradient의 차이는 거의 없음을 보여주기 때문에 이점이 있다고 주장한다.

## **2) gradient-based attribution method**

다른 Gradient 기반 Attribution method와 성능비교에 대한 실험결과를 보여준다.

### **Experiment setup**

* Dataset : ImageNet (1.2 million images of 1000 different categories)
* X-AlexNet : 모든 bias term이 0으로 세팅된 AlexNet
* Evaluation Metric :\


| Metric                               | Description                                                                                                                                 |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| <p>Keep Positive Mask<br>(KPM)</p>   | <p>attribution값이 큰 feature가 모델에 더 많은 영향을 미치는 feature<br>attribution 값이 작은 feature부터 순차적으로 masking 하면서 AUC값 측정<br>높을수록 좋은 attribution</p>    |
| <p>Keep Negative Mask<br>(KNM)</p>   | <p>attribution 값이 작은 feature부터 순차적으로 masking 하면서 AUC값 측정<br>낮을수록 좋은 attribution</p>                                                         |
| <p>Keep Absolute Mask<br>(KAM)</p>   | <p>attribution 절대값이 클 수록 모델에 더 많은 영향을 미치는 feature<br>절대값이 큰 feature는 유지하고 작은 feature부터 순차적으로 masking 하면서 AUC값 측정<br>높을수록 좋은 attribution</p> |
| <p>Remove Absolute Mask<br>(RAM)</p> | <p>절대값이 큰 feature부터 순차적으로 masking 하면서 AUC값 측정<br>낮을수록 좋은 attribution</p>                                                                    |

* baseline :\


| Notation | Baseline            | Description                                                                                                                                           |
| -------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| IG       | Integrate Gradient  | baseline(**0**)\~input 직선경로의 gradient를 누적, 128번의 gradient 계산                                                                                          |
| Random   | random attributions | 랜덤값                                                                                                                                                   |
| Grad     | input gradient      | 한 번의 input gradient                                                                                                                                   |
| EG       | Expected Gradients  | <p>실제 dataset에서 랜덤하게 k개의 reference를 지정해서 baseline으로 사용<br>해당 aseline에 대해서 baseline~input 직선경로중 임의 지점에서의 gradient 계산<br>본 실험결과에서는 1개의 reference 사용</p> |

### **Result**

![Attribution method 실험결과](../../.gitbook/2022-spring-assets/junghurnkim\_1/experimental\_result.png)\
제시된 gradient 방법들중 IG와 XG만 앞서 motivation에서 제시된 Axiom을 만족하기 때문에, 실험결과에서도 뛰어난 Attribution 성능을 보여주는 것을 확인할 수 있다. 저자는 그 중에서도 XG는 단 한번의 역전파된 gradient만으로 IG 만큼의 Attribution 성능을 보여주고 있다고 강조한다.

## **3) Training with attribution priors**

* Dataset : Health survey data of the CDC of the U.S.(13,000명의 118개 의료정보)
* Task : 데이터 관측 후 10년후 사망여부
* baseline :\


| Notation | Baseline                    | Description                                             |
| -------- | --------------------------- | ------------------------------------------------------- |
| Unreg    | Unregularized model         | attribution priors loss에서 regularize term을 제거한 baseline |
| RRR      | Right for the Right Reasons | log prediction에 대한 gradient를 사용                         |

![Attribution prior 실험결과](../../.gitbook/2022-spring-assets/junghurnkim\_1/experimental\_result2.png)\
한번의 gradient를 사용하는 다른 모델과 비교했을 때 월등히 좋은 성능을 보여준다.\
(IG는 오래걸리기 때문에 Attribution Prior 모델에 적용할 수 없다는 단점이 있다.)

## **5. Conclusion**

Summary

* 저자는 Integrated Gradient은 Attribution method로서 좋은 성능을 보여주지만, 계산량이 많다는 단점이 크다고 지적한다. 본 논문은 제시된 문제점을 해결하기 위해 X-DNN이라는 개념을 정의하고 단 한번의 연산으로 Integrate Gradient와 동일한 성능을 보장한다는 것을 이론적, 실험적으로 증명했다.

Opinion

* 개인적으로 Attribution 값을 활용해서 모델성능을 높이는 방안을 연구중인데, 기존의 Attribution에 대한 논문들은 결과를 얼마나 잘 설명하는지, 얼마나 좋은 해석을 제시하는지에만 초점이 있었고 이렇게 직접개입하는 과정에서의 문제점을 언급한 부분은 부족하다고 느꼈다. 그래서 본 논문에서 모델 학습에 직접적으로 관여하는 역할을 할 경우에는 더더욱 중요한 요소가 될 수 있다는 점을 제시한 점이 좋았던 부분이라고 생각한다.

Limitation

* [3. 1) X-DNN and X-Gradient](NeurIPS-2021-XGradient.md#1-nonnegatively-homogeneous-dnn-x-dnn-and-x-gradient-xg)에서도 언급했지만 homogeneous degree가 1보다 큰 경우에 대한 실험결과가 없었던 점이 아쉬웠다. 기존의 연구와 달리 homogeneous degree가 1보다 큰 경우에 대해서도 Integrated Gradient와 InputXGradient 동일함을 보여준 것이 novelty라고 주장했기 때문에 이에 대한 추가적인 실험이나 해석을 기대했지만 이 점이 부족했다고 생각한다.

***

## **Author Information**

* 김정헌(JUNGHURN KIM): Master student, KSE, KAIST

## **6. Reference & Additional materials**
* https://visinf.github.io/fast-axiomatic-attribution/
* https://github.com/visinf/fast-axiomatic-attribution
* Integrated Gradient github https://github.com/ankurtaly/Integrated-Gradients
* Expected Gradient https://arxiv.org/abs/1906.10670
