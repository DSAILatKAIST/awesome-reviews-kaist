---
description : Y Rubanova / Latent ODEs for Irregularly-Sampled Time Series / NeurIPS-2019(description)  
---

# **Title** 

Latent ODEs for Irregularly-Sampled Time Series

## **1. Problem Definition**  

> 시계열 데이터를 다루는 Deep learning에 **미분방정식 (Ordinary Differential Equation)** 을 접목시키자!  

본 논문은 continuous-time dynamics를 가지는 RNN(Recurrent Neural Networks)을 ODEs(Ordinary Differential Equations)를 사용해 정의해 새로운 모델 ODE-RNN을 만들어냅니다.

또한  NeurIPS에 2018년에 publish 된 '[Neural ordinary differential equations](https://arxiv.org/abs/1806.07366)' 라는 논문에서 제시한 Latent ODE model의 recognition network을 ODE-RNN으로 대체합니다. 이를 통해 관측값 사이의 임의의 time gap을 다룰 수 있습니다.

## **2. Motivation**  

Please write the motivation of paper. The paper would tackle the limitations or challenges in each fields.

After writing the motivation, please write the discriminative idea compared to existing works briefly.
> 기존 시계열 데이터를 다루는 RNN은 **irregurlarly-sampled time series data**를 잘 fitting하지 못한다!

RNN은 high-dimensional, regularly-sampled time series data에 대해 좋은 성능을 보이나, data의 time-gap이 불규칙적인 경우 좋은 성능을 내지 못합니다.


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

