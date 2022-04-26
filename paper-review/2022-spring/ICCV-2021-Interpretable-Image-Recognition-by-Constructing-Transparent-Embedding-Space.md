---
description: >-
  Jiaqi Wang / Interpretable Image Recognition by Constructing Transparent
  Embedding Space / ICCV-2021
---

# TesNet

## **1. Problem Definition**

Convolution Neural Network(CNN)의 결과 해석은 판단의 정확한 근거가 필수적인 자율 주행 자동차와 암 진단과 같은 의료 분야에서 중요한 과제입니다. 그러나 다양한 태스크에서 CNN의 성능이 비약적으로 발전한 데에 비해, 여전히 네트워크의 output을 사람이 쉽게 이해할 수 있는 의미들로 해석하는 데에는 어려움이 많습니다. 이러한 문제를 해결하기 위해 최근에 CNN 내부의 feature representation을 시각화하는 많은 interpetable한 방법들이 제안되었지만, 시각화된 네트워크 내부 feature와 의미 해석 간의 gap은 여전히 큽니다.

따라서 interpretable image classification(해석 가능한 이미지 분류)를 위해 사람들이 쉽게 그 의미를 이해할 수 있는 input image의 concepts를 추출하는 방법에 대한 연구가 이루어지고 있습니다. 그러나 기존 관련 연구들이 제안한 concepts는 서로 뒤얽혀있어 output class에 대한 각 개별 concept의 영향을 해석하기 어렵습니다.

본 논문에서는 이를 문제점으로 지적하며 output class에 대한 input image의 특징을 효과적으로 설명할 수 있으면서, 동시에 서로 얽혀있지않고 orthogonal한(직교를 이루는) concepts를 추출할 수 있는 방법론을 제안합니다.

## **2. Motivation**

그렇다면 `Interpretable Concepts` (해석이 용이한 컨셉)이란 무엇일까요? 인지적 관점에서 Interpretable Concepts는 다음의 세 가지 조건을 만족해야 합니다.

`(1) Informative`\
Input data는 basis concept들로 spanned된 vector space상에서 효율적으로 나타내져야하며, input의 essential information(중요한 정보)가 새로운 representation space에서도 보존되어야합니다.\
`(2) Diversity`\
각 데이터(ex.이미지)는 서로 중복되지 않는 소수의 basis concepts와 관련 있어야하며, 같은 class에 속하는 데이터들은 비슷한 basis concepts를 공유해야 합니다.\
`(3) Discriminative`\
Basis concepts는 (1)에서 언급한 basis concept vector space상에서도 class가 잘 분리되도록 class-aware해야 합니다. 즉, 같은 class와 연관된 basis concepts끼리는 근접하게, 다른 class의 basis concepts 간에는 멀게 embedding되어 있어야 합니다.

데이터의 concepts를 추출하기 위해 이전 연구들은 auto-encoding, prototype learning과 같이 deep neural network의 high-level feature를 이용하는 방식을 제안하였습니다. 그 중 한 방법은 U-shaped Beta Distribution을 이용하여 basis concepts의 개수를 제한함으로써 각 input data를 소수의 의미 있는 basis concept들로 나타내기도 하였습니다. 이러한 연구들은 Interpretable Concepts의 첫번째 조건을 만족하였지만, 앞서 언급하였듯이 basis concepts가 서로 얽혀있어(entangled) input과 output에 대한 개별 concept의 영향을 해석하기 어렵다는 문제점이 존재합니다.

따라서, 이 논문에서는 위의 세가지 `Interpretable Concepts` 조건을 모두 충족시키는 basis concepts를 설계하는 데에 주목하고 있습니다. 논문에서 설계한 basis concepts는 다음과 같은 특징들을 가집니다.

`(1) 각 class는 자신만의 basis concepts를 가지며 class가 다른 경우 basis concepts도 최대한 다릅니다.`\
`(2) High-level feature과 basis concepts 사이를 효과적으로 연결하는 mapping을 제공합니다.`\
`(3) Input image 상의 basis concepts는 각 class에 대한 prediction score을 계산하는 데에 도움이 됩니다.`

위의 세 가지 특징을 만족하는 basis concepts 설계를 위해, 본 논문은 기존 연구들과 다르게 `Grassmann manifold`를 도입하여 `basis concept vector space`를 정의합니다. 다음의 그림처럼, 각 class마다의 basis concepts subset이 Grassmann manifold 상의 point로 존재합니다.\
![figure1](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure1.PNG?raw=true)\
Grassmann manifold는 쉽게 말하면 linear subspaces의 set(집합)이라고 생각할 수 있습니다. 여기서 subspace란 vector space _V_ 의 subset(부분집합) _W_ 가 _V_ 로부터 물려받은 연산들로 이루어진 또 다른 하나의 vector space일 때 _W_ 를 _V_ 의 subspace라고 말합니다.

또한 projection metric을 통해 각 class의 basis concept들은 서로 orthogonal하도록, 동시에 class-aware한 basis concepts subset들은 서로 멀리 위치하도록 규제됩니다. 이 두 가지 규제를 통해 basis concepts가 서로 얽히지 않도록 함으로써 기존 연구의 한계점을 극복하고 있습니다.

논문은 이렇게 설계된 `transparent embedding space` (concept vector space)가 도입된 새로운 interpetable network, `TesNet을` 제안하고 있습니다.

## **3. Method**

### **The overview of TesNet architecture**

다음은 TesNet의 전체적인 architecture의 모습입니다.\
![figure2](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure2.PNG?raw=true)\
그림과 같이 TesNet은 convolutional layers _f_, trasparent subspace layer $s\_{b}$, 그리고 classifier _h_ 이렇게 세 가지의 핵심 요소로 이루어져 있습니다.

각 요소를 하나씩 살펴보면, 먼저 convloutional layers _f_ 는 1X1 convolutional layer들이 추가된 기본 CNN 네트워크(ex.ResNet) 입니다. s\_{b}는 feature map을 transparent embedding space에 projection시키는 subspace layer입니다. 각 class마다 subspace가 존재하여 총 class 개수만큼의 subspace가 존재합니다. 각 class의 subspace는 M개의 basis concepts로 spanned 되어있습니다. 이 M개의 within-class concepts(클래스 내부 concepts)는 서로 orthogonal하다고 가정합니다. 따라서 총 C개의 class가 있을 때, 각 class 마다 M개의 basis concepts가 존재한다고 가정하면 전체 CM개의 basis concepts가 존재합니다.

### **Embedding space learning**

그렇다면 `basis concepts는` 어떻게 정의되어 embedding space를 이루고 있는지 살펴보겠습니다.

각 basis concept은 basis vector로 표현됩니다. 이 basis vector는 다음 세 가지 조건을 만족해야합니다.\
`(1) 다른 basis vector 사이에는 의미가 중복되면 안됩니다.`\
`(2) embedding space에서도 각 class는 구분되어야 합니다.`\
`(3) basis vector들은 비슷한 high-level patch(사람들이 인식할 수 있는 level의 image)들을 군집화하고 다른 것들끼리는 분리할 수 있어야 합니다.`

이 세 가지 조건을 만족시키기 위해 전체 architecture에서 보았던 convolutional layer, basis vectors, classifier layer의 weight들이 서로 joint하게 optimize(최적화)될 수 있도록 joint optimization problem을 정의하고 있습니다. 다음은 각 weight를 최적화하기 위한 Loss와 optimization 과정입니다.

### **Orthonormality for Within-class Concepts**

조건 (1)을 만족시키기 위한 Loss는 다음과 같습니다.

![figure3](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure3.PNG?raw=true)

basis vector사이에 의미가 중복되지 않는다는 것은 같은 class에 속한 basis concepts이더라도 반드시 서로 다른 측면들을 나타내고 있어야한다는 뜻입니다. 그러기 위해선 같은 class에 속한 basis concept vectors가 서로 orthogonal해야 하므로 `각 class의 basis vectors 사이의 orthonormality를 규제하는 Loss`를 사용합니다.

Loss 식을 살펴보면 각 class의 basis vector matrix 행렬곱과 identity matrix 사이의 L2 norm을 모두 더하고 있습니다. 즉, 각 class의 basis vectors간의 correlation(상관 관계)를 최소화시키기 위한 Loss입니다. 이러한 Loss를 통해 학습된 orthonormal basis vectors가 각 class의 subspace를 span하게 됩니다.

### **Separtion for Class-aware Subsapces**

두번째로 조건 (2)를 만족시키기 위한 Loss는 다음과 같습니다.

![figure4](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure4.PNG?raw=true)\
embedding space상에서 class가 구분되기 위해서는 각 class의 subspace가 서로 멀리 위치해 있어야합니다. 즉, Grassmann manifold 상에서 class-aware subspace들의 거리가 최대한 멀어지도록 규제합니다. 각 subspace는 Grasmann manifold상에서 unique한 projection으로 존재하므로, subspace 사이의 거리를 `projection mapping`을 이용하여 수치화할 수 있습니다.

Loss 식에서 B^{c}는 class c의 orthonormal basis vectors로 이루어진 matrix를 의미하고, 이 matrix의 행렬곱이 class c와 연관된 subspace의 projection mapping입니다. 결국 Loss는 `서로 다른 class의 projection mapping 사이의 L2 norm distance들의 합을 최소화시키기 위한 Loss`로 이해할 수 있습니다.

### **High-level Patches Grouping**

마지막으로 조건 (3)을 만족시키기 위한 Loss입니다. ![figure5](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure5.PNG?raw=true)\
조건 (3)은 결국 high-level 이미지 패치들이 embedding subspace에도 잘 projection 되어야 한다는 의미입니다. 즉, 이미지 패치들이 subspace에 embedding 되었을 때 이미지가 속한 ground-truth class의 basis vectors와 근접해야합니다. 이를 위해 논문은 `Compactness Loss`와 `Separation Loss`를 정의하고 있습니다.

먼저 Compactness Loss의 식을 살펴보면, 이미지 패치와 ground-truth class의 basis vectors사이의 cosine distance(negative cosine similarity)를 최소화하고 있습니다. 이는 결국 이미지 패치와 ground-truth class의 basis vectors사이의 cosine similarity를 크게하는 것과 같습니다.

반면, Separation Loss는 이미지 패치가 ground-truth가 아닌 class의 basis vectors과는 멀어지도록 둘 사이의 cosine similarity를 최소화하고 있습니다.

이 두 Loss를 hyper-parameter _M_ 을 사용하여 더함으로써 `Compactness-Separation Loss`를 정의합니다.

### **Identification**

마지막으로 classifier layer를 optimize하기 위한 Loss로서 `Cross Entropy Loss`를 이용합니다.\
![figure6](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure6.PNG?raw=true)

최종적으로, 지금까지 정의된 loss들을 jointly optimize하기 위해 `Total Loss for Joint Optimization`을 정의합니다.\
![figure7](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure7.PNG?raw=true)

hyper-parameters를 사용하여 `classification loss(cross entropy loss)에 orthonormality loss, subspace separation loss, compactness-separation loss`를 적절한 비율로 더해줍니다. 이 total loss와 함께 convolutional layer, basis vectors가 동시에 최적화되며 `concept embedding subspace`가 학습됩니다.

### **Concept-based classification**

embedding space가 학습되고 나면, convolutional layers와 basis vectors의 parameter를 고정시킨 후, 마지막 단의 classifier를 학습시키게 됩니다. classifier는 concept-class weight _G_ 를 최적화함으로써 학습이 되는데, weight _G_ 는 _G(c,j)_ 의 값이 j번째 unit이 class c에 속하는 경우를 제외하고 모두 0인 sparse matrix입니다. 앞서 정의한 Identification Loss에 `weight`` `_`G`_` ``를 sparse하게 유지하게 하는 규제를 더하여 Loss`를 정의하고, 이 Loss를 최소화하도록 classifier가 학습됩니다.\
![figure10](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure10.PNG?raw=true)

## **4. Experiment**

본 논문에서는 다양한 CNN architecture에 대한 TesNet의 넓은 적용성을 입증하기 위해 두 가지의 case study를 진행하였습니다. 그 중 첫번째 case study인 `bird species identification`에 대해서 자세히 살펴보겠습니다.

### **Experiment setup**

* **Dataset**\
  Caltecg-USCD Birds-200-2011 dataset을 사용하여 bird species classification 실험을 진행하였습니다. dataset은 200 종(species)의 bird 이미지 5994+5794장으로 이루어졌습니다. 그 중 5994장은 training, 나머지 5794장은 test시 이용하였습니다. 각 bird class마다 30장의 이미지밖에 존재하지 않아, 논문에서는 random rotation, skew, shear, flip 등의 `augmentation`을 통해 training set의 각 class마다 1200장의 이미지가 존재하도록 데이터를 증강하였습니다.
* **baseline**\
  non-interpetable한 본래 `VGG16, VGG19, ResNet34, ResNet152, DenseNet121, DenseNet161` 네트워크들을 baseline으로 삼고, 각 네트워크에 interpetable한 `TesNet`을 적용한 경우와 비교 실험하였습니다. 또한, TesNet과 유사한 interpetable network architecture인 `ProtoPNet`을 적용한 결과도 함께 비교하였습니다.
* **Evaluation Metric**\
  실험의 성능 평가지표로 `classification accuracy`를 사용하였습니다.

### **Result**

* **Accuracy comparison with diffrent CNN architectures**\
  아래 표에서 알 수 있듯이, baseline network에 TesNet을 적용한 경우 분류 정확도가 최대 8%정도 크게 향상된 것을 볼 수 있습니다. 또한, TesNet의 Loss를 다양하게 정의하여 실험한 결과, 4가지 Loss를 모두 jointly하게 optimize하였을 때 가장 정확도가 높은 것을 확인할 수 있습니다.\
  ![figure8](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure8.PNG?raw=true)
* **The interpretable reasoning process**\
  다음 그림은 TesNet이 test image에 대하여 decision을 내리는 reasoning process를 시각화한 것입니다.\
  ![figure9](https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/TaeMiKim\_1/figure9.PNG?raw=true)

European Goldfinch라는 class의 test image가 주어졌다고 할 때, TesNet은 학습된 basis vectors를 통해 feature map을 re-represent할 수 있습니다. 각 class c에 대해서, 모델은 학습된 basis vectors를 image patch에 re-represent함으로써 그 image가 class c에 속할 score를 계산합니다.

예를 들어, 위 그림에서 모델은 European goldfinch class의 basis vector(concept)를 test image(original image)가 이 class에 속할지에 대한 증거로 활용합니다. Activation map column을 살펴보면, European goldfinch class의 첫 번째 basis vector가 의미하는 'black and yellow wing concept'이 test image 상에서 가장 두드러지게 activated(활성화) 된 것을 확인할 수 있습니다. 같은 방식으로 두 번째 basis vector가 의미하는 'head concept', 세 번째 basis vector가 의미하는 'brown fur concept'이 image상에서 크게 활성화되었습니다.

이를 바탕으로 모델은 class의 각 basis concept vector와 test image상에서 activated된 부분 사이의 similarity(유사도)를 구하고 basis concept의 중요도에 따라 가중치를 매겨 더함으로써 최종적인 European Goldfinch class에 대한 score를 구합니다. 이 score를 바탕으로 test image의 class를 예측합니다.

이러한 reasoning 과정을 통해 baseline CNN 모델들보다 높은 분류 정확도를 달성할 수 있습니다.

## **5. Conclusion**

* **Summary**\
  TesNet은 다른 CNN 모델에 plug-in되어 classifiaction 성능을 향상시킬 수 있는 적용성 높은 architecture입니다. TesNet은 class-aware concepts를 설계하고 같은 class에 속한 concepts끼리 얽히지 않도록 하며 효과적으로 prediction 성능을 향상시켰습니다. 또한, TesNet은 image의 어떤 concept이 CNN을 학습시키고 예측하는 데에 근거로 사용되는지를 설명할 수 있습니다.\
  그러나, TesNet은 basis concepts가 모두 flat하다는 전제를 하고 있어, 사람들이 실제로 사물을 분류할 때의 인지 과정과 큰 차이가 있습니다. 또한 실제로 real world에서의 concepts는 서로 계층적으로 이루어져있기 때문에, hierarchical basis concepts를 학습할 수 있는 네트워크에 대한 연구가 필요합니다.
* **Opinion**\
  CNN의 output 해석에 있어 input image의 concept이라는 개념을 잘 정의한 연구라고 생각합니다. 특히 basis vector, subspace, manifold와 같이 어렵지않은 수학적 개념들을 잘 적용하여 의미있는 결과를 도출해낸 점이 굉장히 인상깊습니다. 평소 알고만 있던 수학적 개념들을 neural network와의 연결 지점을 다시 생각해볼 수 있는 기회였고, 개인적으로 Explainable AI에 관심이 많아 흥미로웠습니다. 그러나 이런 interpretable한 network가 주로 이미지 데이터쪽에 치우쳐 있다는 점이 아쉬웠고 audio, text 등에도 general하게 쓰일 수 있는 architecture에 대한 연구의 필요성을 느꼈습니다.

***

## **Author Information**

* TaeMi, Kim
  * KAIST, Industrial and Systems Engineering
  * Computer Vision, XAI

## **6. Reference & Additional materials**

* Github Implementation\
  None
* Reference
  * Chaofan Chen et al, This looks like that: deep learning for interpretable image recognition, NeurIPS, 2019.
  * https://en.wikipedia.org/wiki/Grassmannian
