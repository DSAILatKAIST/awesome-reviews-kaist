---
description : Jiaqi Wang / Interpretable Image Recognition by Constructing Transparent Embedding Space / ICCV-2021  
---

# **Interpretable Image Recognition by Constructing Transparent Embedding Space** 

## **1. Problem Definition**  

Convolution Neural Network(CNN)의 결과 해석은 판단의 정확한 근거가 필수적인 자율 주행 자동차와 암 진단과 같은 의료 분야에서 중요한 과제입니다. 그러나 다양한 태스크에서 CNN의 성능이 비약적으로 발전한 데에 비해, 여전히 네트워크의 output을 사람이 쉽게 이해할 수 있는 의미들로 해석하는 데에는 어려움이 많습니다. 이러한 문제를 해결하기 위해 최근에 CNN 내부의 feature representation을 시각화하는 많은 interpetable한 방법들이 제안되었지만, 시각화된 네트워크 내부 feature와 의미 해석 간의 gap은 여전히 큽니다.

따라서 interpretable image classification(해석 가능한 이미지 분류)를 위해 사람들이 쉽게 그 의미를 이해할 수 있는 input image의 concept을 추출하는 방법에 대한 연구가 이루어지고 있습니다. 그러나 기존 관련 연구들이 제안한 concept들은 서로 뒤얽혀있어 output class에 대한 각 개별 concept의 영향을 해석하기 어렵습니다. 

본 논문에서는 이를 문제점으로 지적하며 output class에 대한 input image의 특징을 효과적으로 설명할 수 있으면서, 동시에 서로 얽혀있지않고 orthogonal한 (직교를 이루는)  concept들을 추출할 수 있는 방법론을 제안합니다. 

## **2. Motivation**  

그렇다면 Interpretable Concepts (해석이 용이한 컨셉)이란 무엇일까요? 인지적 관점에서 Interpretable Concepts는 다음의 세 가지 조건을 만족해야 합니다.

(1) Informative   
Input data는 basis concept들로 spanned된 vector space상에서 효율적으로 나타내져야하며, input의 essential information(중요한 정보)가 새로운 representation space에서도 보존되어야합니다.   
(2) Diversity   
각 데이터(ex.이미지)는 서로 중복되지 않는 소수의 basis concepts와 관련 있어야하며, 같은 class에 속하는 데이터들은 비슷한 basis concepts를 공유해야 합니다.   
(3) Discriminative  
Basis concepts는 (1)에서 언급한 basis concept vector space상에서도 class가 잘 분리되도록 class-aware해야 합니다. 즉, 같은 class와 연관된 basis concepts끼리는 근접하게, 다른 class의 basis concepts 간에는 멀게 embedding되어 있어야 합니다.

데이터의 concepts를 추출하기 위해 이전 연구들은 auto-encoding, prototype learning과 같이 deep neural network의 high-level feature를 이용하는 방식을 제안하였습니다. 그 중 한 방법은 U-shaped Beta Distribution을 이용하여 basis concepts의 개수를 제한함으로써 각 input data를 소수의 의미 있는 basis concept들로 나타내기도 하였습니다. 이러한 연구들은 Interpretable Concepts의 첫번째 조건을 만족하였지만, 앞서 언급하였듯이 basis concepts가 서로 얽혀있어(entangled) input과 output에 대한 개별 concept의 영향을 해석하기 어렵다는 문제점이 존재합니다.

따라서, 이 논문에서는 위의 세가지 Interpretable Concepts 조건을 모두 충족시키는 basis concepts를 설계하는 데에 주목하고 있습니다. 논문에서 설계한 basis concepts는 다음과 같은 특징들을 가집니다.   
첫번째, 각 class는 자신만의 basis concepts를 가지며 class가 다른 경우 basis concepts도 최대한 다릅니다.   
두번째, high-level feature과 basis concepts 사이를 효과적으로 연결하는 mapping을 제공합니다.   
세번째, input image 상의 basis concepts는 각 class에 대한 prediction score을 계산하는 데에 도움이 됩니다.

위의 세 가지 특징을 만족하는 basis concepts 설계를 위해, 본 논문은 기존 연구들과 다르게 Grassmann manifold를 도입하여 basis concept vector space를 정의합니다. 다음의 그림처럼, 각 class마다의 basis concepts subset이 Grassmann manifold 상의 point로 존재합니다.    
<img width="200" src="awesome-reviews-kaist/.gitbook/2022-spring-assets/Taemi Kim_1/figure1.PNG">   
![figure1]("https://github.com/TaeMiKim/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Taemi%20Kim_1/figure1.PNG")

또한 projection metric을 통해 각 class의 basis concept들은 서로 orthogonal하도록, 동시에 class-aware한 basis concepts subset들은 서로 멀리 위치하도록 규제됩니다. 이 두 가지 규제를 통해 basis concepts가 서로 얽히지 않도록 함으로써 기존 연구의 한계점을 극복하고 있습니다. 
논문은 이렇게 설계된 transparent embedding space (concept vector space)가 도입된 새로운 interpetable network, TesNet을 제안한다.

## **3. Method**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily. 

TesNet은 convolutional layers _f_, basis concepts _B_ 로 이루어진 trasparent subspace layer $$s_{b}$$, 그리고 classifier _h_ 이렇게 세 가지의 핵심 요소로 이루어져 있습니다. 

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
    * Computer Vision

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation  
* Reference  

