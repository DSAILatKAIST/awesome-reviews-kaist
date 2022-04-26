---
description : Xiangning Chen et al., / When Vision Transformer Outperform ResNet Without Pre-Training Or Strong Data Augmentation / ICLR-2022  
---

# **When Vision Transformer Outperform ResNet Without Pre-Training Or Strong Data Augmentation** 

## **1. Problem Definition**  


![fig1](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fa054ce54-ccde-4c75-a438-277e63bcc76a%2FUntitled.png?table=block\&id=6b4ec770-3591-48e1-b817-9c3e62dcdea5\&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d\&width=2000\&userId=\&cache=v2)

* Convolution-free한 ViT와 MLPs만 사용한 모델의 경우 inductive bias의 부재로 인해 매우 많은 데이터를 학습시키거나 강력한 augmentation 전략을 사용해 이를 극복하고자 하였다.
* Transformer가 NLP분야에 소개된 이후 vision에서의 활용이 있었고 hand-wired feature와 inductive bias가 없이 일반화 가능한 모델을 만들기 위해 상당한 양의 데이터를 투입시키는 방법이 활용되었다. 예를들어 pre-train ViT는 google의 private dataset으로 3억장의 labeled image를 학습하였다
* ViT와 Mixer는 ResNet에 비해 기하적으로 아주 sharpe한 loss landsacpe를 가지고 있음을 알 수 있다. 이것은 학습-일반화 성능의 괴리가 생기는 원인으로 지목된다

## **2. Motivation** 
* 2. Motivation에서는 motivation이 된 아이디어와 그 근거를 객관적으로 제시한 실험들을 살펴보겠습니다.

### **2.0 Motivation**

* 학습된 landscape는 상당히 local minima에 sharpe한 모습을 보였지만 최근 제시된 sharpness-aware optimizer(SAM)을 활용한 결과 ViT와 MLP Mixer는 지도, 비지도, 적대적, 전이 등 다양한 학습 전략에서 상당한 성능 향상이 있음을 확인했다. Scratch부터 학습된다면 비슷한 size의 ViT와 ResNet에서 ViT가 ResNet의 성능을 뛰어넘을 수 있음을 확인했다
* SGD,Adam과같은 first-order optimizer는 training error를 낮추는데에는 좋은 알고리즘이다. 하지만 이것은 더 고차원적인 목표인 주변의 loss도 낮게 만듦에는 신경쓰지 않는다. 따라서 최근의 연구인 SAM에서 이런 문제의 해결을 찾게 되었다. SAM optimizer는 single poing에서의 loss 최저보다 single poing 주변 모두가 loss가 낮아지도록 설계되었다. 이렇게 향상된 일반화 성능은 강력한 augmentation과 pre-training과정없이 ViT, MLPs가 ResNet을 뛰어넘을 수 있게 하였다
* SAM을 사용한 후 model의 (특히 첫 몇 개의 레이어에서) [Hessian 고윳값](https://angeloyeo.github.io/2020/06/17/Hessian.html)이 작아지는 것을 확인했다(=볼록한 정도가 감소했다=sharpeness하지 않아졌다). Weight norm은 이것을 커지게 만들었고 이는 일반화에 자주 쓰이는 weight decay가 regularization에 크게 도움되지 않았을지 모른다는 가정을 가능하게 한다. SAM과 강력한 augmentation과의 비슷한 특성들을 살펴볼 것이다


### **2.1 ViTs and MLP-Mixers converge to extremely sharp local minima**

![fig2](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Ff9de0b6f-c4d6-476e-9bd1-17d102d2b668%2FUntitled.png?table=block\&id=cf50b7ae-7730-47e2-a373-4ec7ae48f5bc\&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d\&width=2000\&userId=\&cache=v2)

* Figure1 (a) vs (b),(c) : ResNet에 비해서 conv-free한 방법은 loss의 landscape가 sharper함을 확인할 수 있다. Table1을 보면 Hessian 행렬에서 얻은 고윳값 중 제일 큰 ![](https://latex.codecogs.com/gif.latex?%5Clambda\_%7Bmax%7D)을 확인할 수 있다. ![](https://latex.codecogs.com/gif.latex?%5Clambda\_%7Bmax%7D)는 landscape의 최대(최악) 곡률이다. ResNet과 비교하면 ViT와 특히, Mixer가 상당히 높다는 것을 알 수 있다


### **2.2 Small training errors**

![fig3](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F641cd367-d795-449e-851b-301e78ba204e%2FUntitled.png?table=block\&id=32aa79ba-a881-4f93-ae85-5c4fc26c419b\&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d\&width=2000\&userId=\&cache=v2)

* ViT와 MLP가 극소적인 부분에 수렴한다는 것은 training에 dynamics가 존재한다는 의미이다. Figure2의 좌측과 중앙을 보면 MLP는 ViT보다 낮은 training loss를 가지지만 test에서의 성능은 더 나쁘다. 이는 부분적인 극소에 수렴했다고 분석할 수 있다


### **2.3 ViTs and MLP-Mixers have worse trainability**

* 또한 ViT와 MLP가 poor한 trianability에 노출되어있다는 것을 알 수 있었다. 여기서 trainability란 경사하강법에 의해 네트워크가 최적화되는 현상의 효율성을 말한다. Xiao의 연구에 의하면 Jacobian 행렬(Jacobian 행렬은 1차 미분 행렬) ![](https://latex.codecogs.com/gif.latex?J)에 대하여 neural tangent kernel(NTK) ![](https://latex.codecogs.com/gif.latex?%5CTheta=JJ%5E%5Ctop)로 정의한다. ![](https://latex.codecogs.com/gif.latex?%5CTheta)의 고윳값 ![](https://latex.codecogs.com/gif.latex?%5Clambda\_1%5Cgeq%5Ccdots%5Cgeq%5Clambda\_m)에 대하여 ![](https://latex.codecogs.com/gif.latex?%5Ckappa=%5Clambda\_1/%5Clambda\_m)으로 정의한다. 만약 ![](https://latex.codecogs.com/gif.latex?%5Ckappa)가 지속적으로 변하면 학습이 불안정하다. Table1의 모델에 따른 ![](https://latex.codecogs.com/gif.latex?%5Ckappa)를 비교해 볼 것


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

* 이 연구는 지금까지 방대한 양의 데이터와 augmentation에 의존한 ViT, MLP의 기하적 loss 특징을 분석했다. 여기서 local minima 문제를 찾고 SAM을 도입해 flatten loss landscape로 일반화 성능의 향상을 끌어냈다. ViT는 어떠한 pre-training이 없다면 ResNet의 성능을 앞지를 수 있었다. 또한 SAM을 도입한 ViT의 attention map이 더욱 해석 가능하게 변화했다
* 개인적 의견으로, ViT가 강력한 구조임은 맞지만 data hungry한 특징이 유난히 강해 연구 이후의 실용 단계에서 어떻게 이용될 수 있을지 회의적이었다. optimizer를 바꾸는 방법으로 보통의 실험실과 PC에서도 구현이 가능한 방법을 제안했다는 것에 큰 의의가 있다고 생각한다.

---  
## **Author Information**  

* 홍성래 SungRae Hong
    * Master's Course, KAIST Knowledge Service Engineering 
    * Interested In : SSL, Vision DL, Audio DL
    * Contact : sun.hong@kaist.ac.kr

## **6. Reference & Additional materials**  

이 논문을 이해하는데 필요한 수학적 지식을 아래에서 확인할 수 있습니다.

* 출처 : 공돌이의 수학노트
    * [헤세 행렬(Hessian Matrix)의 기하학적 의미](https://angeloyeo.github.io/2020/06/17/Hessian.html)
    * [자코비안(Jacobian) 행렬의 기하학적 의미](https://angeloyeo.github.io/2020/07/24/Jacobian.html)



