---
description : Y Rubanova / Latent ODEs for Irregularly-Sampled Time Series / NeurIPS-2019
---

# **Title** 

Latent ODEs for Irregularly-Sampled Time Series

## **1. Problem Definition**  

> 시계열 데이터를 다루는 Deep learning에 **미분방정식 (Ordinary Differential Equation)** 을 접목시키자!  

본 논문은 continuous-time dynamics를 가지는 `RNN(Recurrent Neural Networks)`을 ODEs(Ordinary Differential Equations)를 사용해 정의해 새로운 모델 `ODE-RNN`을 만들어냅니다.

또한  NeurIPS에 2018년에 publish 된 '[Neural ordinary differential equations](https://arxiv.org/abs/1806.07366)' 라는 논문에서 제시한 `Latent ODE model`의 recognition network을 `ODE-RNN`으로 대체합니다. 이를 통해 관측값 사이의 임의의 time gap을 다룰 수 있습니다.

## **2. Motivation**  
 
> 기존 시계열 데이터를 다루는 `RNN`은 **irregurlarly-sampled time series data**를 잘 fitting하지 못한다!

`RNN`은 regularly-sampled time series data에 대해 좋은 성능을 보이나, data의 time-gap이 불규칙적인 경우 좋은 성능을 내지 못합니다.  

이에 지금까지 사용하던 몇 가지 해결책이 있었는데,
* timeline을 equally-sized intervals로 나누거나,
* observation들을 평균을 사용해 impute/agrregate
하는 등의 간단한 trick을 사용했습니다.

하지만 이러한 방식은 measurement의 timing 같은 정보량을 줄이거나 왜곡하는 문제가 있었습니다.

  
<br/>
<div align="center">
  
**_이에 저자들은 모든 time point에 정의된 latent space를 가지는 continuous-time model을 정의하고자 합니다._**
  
</div>

<br/>
<br/>
<div align="center">
  
![image](https://user-images.githubusercontent.com/99710438/164024065-a992aa76-a84a-4a63-b840-a164dd414dae.png)

</div>

예를 들어, 위 사진은 `RNN`과 저자들이 제시한 `ODE-RNN`의 차이를 보여줍니다. 각 line은 hidden state의 trajectory를 나타내고 수직 점선은 observation time을 나타내는데, `RNN`은 observation이 나타날 때만 hidden state에 변화가 있어 각 observation 사이를 예측하긴 어렵습니다. 

반면에 `ODE-RNN`은 각 observation 사이에도 trajectory를 fitting하며 observation이 들어올 때 마다 값을 수정해주는 것을 확인할 수 있습니다. 이런 식으로 `ODE-RNN`은 observation이 불규칙적으로 있어도 좋은 예측 성능을 보일 수 있습니다.


## **3. Method**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  
> ### **Backgrounds**
논문에서 제안한 방법론을 이해하기 위해서는 `RNN`, `Neural Ordinary Differential Equations`, 그리고 `Variational Autoencoder`의 개념을 알고 있어야 합니다. 

본 포스팅에서는 간단하게 소개를 하겠으며, 세 가지 방법론에 대해 자세히 알고 싶으시면 각각 [여기](https://www.youtube.com/watch?v=6niqTuYFZLQ), [여기](https://www.youtube.com/watch?v=AD3K8j12EIE), 그리고 [여기](https://www.youtube.com/watch?v=9zKuYvjFFS8)를 참고하시기 바랍니다.

#### **1. RNN**

#### **2. Neural Ordinary Differential Equations**

#### **3. Variational Autoencoder**




<div align="center">  

![image](https://user-images.githubusercontent.com/99710438/164017436-f435d0f4-24f9-4d66-9fcc-87ec0c1775bf.png)

</div>  



<div align="center">  
  
![image](https://user-images.githubusercontent.com/99710438/164017499-a8fcab15-b16c-40bd-a0be-cf6d272cd574.png)
  
</div>  


<div align="center"> 
  
![image](https://user-images.githubusercontent.com/99710438/164017531-002e6512-f1c5-4430-904d-d19f82f2a9e4.png)
  
</div>  

<div align="center">  
  
![image](https://user-images.githubusercontent.com/99710438/164017572-bacb1d58-885d-4659-b6cc-4c0fd5035876.png)
  
</div>  

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
