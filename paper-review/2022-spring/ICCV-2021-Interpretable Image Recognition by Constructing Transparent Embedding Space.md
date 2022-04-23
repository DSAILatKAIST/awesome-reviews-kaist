---
description : Jiaqi Wang / Interpretable Image Recognition by Constructing Transparent Embedding Space / ICCV-2021(description)  
---

# **Title** 

Interpretable Image Recognition by Constructing Transparent Embedding Space

## **1. Problem Definition**  

Convolution Neural Network(CNN)의 결과 해석은 판단의 근거가 필수적인 자율 주행 자동차와 암 진단과 같은 의료 분야에서 중요한 과제이다. 그러나 다양한 태스크에서 CNN의 성능이 비약적으로 발전한 데에 비해, 여전히 CNN의 결과를 사람이 쉽게 이해할 수 있는 의미들로 해석하는 데에는 어려움이 존재한다. 이러한 문제를 해결하기 위해 최근에 CNN 내부의 feature representation을 시각화하는 많은 interpetable한 방법들이 제안되었지만, 네트워크 시각화와 의미 해석 간의 gap은 여전히 크다.
따라서 interpretable image classification(해석 가능한 이미지 분류)를 위해 사람이 쉽게 의미를 이해할 수 있는 input image의 concept을 추출하는 방법에 대한 연구가 이루어지고 있다. 그러나 기존 연구들의 concept들은 서로 뒤얽혀있어 output class에 대한 각 개별 concept의 영향을 해석하기 어렵다. 이를 문제점으로 지적하며 이 논문에서는 ouput class에 대한 input image의 특징을 효과적으로 설명할 수 있으면서 서로 얽혀있지않고 orthogonal 한 concept들을 추출할 수 있는 방법론을 제안한다. 

## **2. Motivation**  

Please write the motivation of paper. The paper would tackle the limitations or challenges in each fields.

인지적 관점에서 Interpretable concepts(해석 가능한 컨셉)이란 다음의 세 가지 조건을 만족해야 한다.
(1) Informative
input data는 basis concept들로 spanned된 vector space상에서 효율적으로 나타내져야하고, essential information이 새로운 representation 공간에서도 보존되어야한다.
(2) Diversity
각 데이터(이미지)는 겹치지 않는 소수의 basis concept들과 관련 있어야하고, 같은 class에 속하는 데이터들은 비슷한 concept들을 공유해야 한다.
(3) Discriminative
basis soncpet들은 concept space 상에서도 class가 잘 분리되도록 class-aware해야 한다.

데이터의 concept들을 추출하기 위해, 기존 연구들은 auto-encoding이나 prototype learning과 같이 deep neural network의 high-level feature를 이용하는 방식을 제안하였다. U-shaped Beta distribution을 이용하여 concept의 개수를 제한함으로써 이 방식들은 input data를 몇 개의 basis concept들로 나타내어 첫번째 조건을 만족한다. 그러나, 앞어 언급하였듯이 기존 연구들의 basis concept들은 서로 얽혀있어(entangled) 각 개별 concept의 input과 output에 대한 영향을 해석하기 어렵다.
따라서, 이 논문에서는 위의 세가지 interpretable concepts 조건을 만족시키는 basis concepts를 설계하는 데에 주목하였다. 
먼저, 

After writing the motivation, please write the discriminative idea compared to existing works briefly.



## **3. Method**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily. 

TesNet은 convolutional layers _f_, Trasparent subspace layer $$s_{b}$$ 세 가지의 핵심 요소로 이루어져 있다. 

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

