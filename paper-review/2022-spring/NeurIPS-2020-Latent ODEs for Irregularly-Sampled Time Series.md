---
description : Y Rubanova / Latent ODEs for Irregularly-Sampled Time Series / NeurIPS-2019
---

<br/>

# **Title** 

Latent ODEs for Irregularly-Sampled Time Series

<br/>

## **1. Problem Definition**  

> 시계열 데이터를 다루는 Deep learning에 **미분방정식 (Ordinary Differential Equation)** 을 접목시키자!  

<br/>


본 논문은 continuous-time dynamics를 가지는 `RNN(Recurrent Neural Networks)`의 hidden dynamics를 ODEs(Ordinary Differential Equations)로 정의해 새로운 모델 `ODE-RNN`을 만들어냅니다.

또한  NeurIPS에 2018년에 publish 된 '[Neural ordinary differential equations](https://arxiv.org/abs/1806.07366)' 라는 논문에서 제시한 `Latent ODE model`의 recognition network을 `ODE-RNN`으로 대체합니다. 이를 통해 관측값 사이의 임의의 time gap을 다룰 수 있습니다.

<br/>

## **2. Motivation**  
 
> 기존 시계열 데이터를 다루는 `RNN`은 **irregurlarly-sampled time series data**를 잘 fitting하지 못한다!

<br/>


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

<br/>


## **3. Method**  

<br/>


> ### **Backgrounds**: What is RNN, Nerual ODE, Variational Autoencoder?

<br/>

논문에서 제안한 방법론을 이해하기 위해서는 `RNN`, `Neural Ordinary Differential Equations`, 그리고 `Variational Autoencoder`의 개념을 알고 있어야 합니다. 

본 포스팅에서는 간단하게 소개를 하겠으며, 세 가지 방법론에 대해 자세히 알고 싶으시면 각각 [여기](https://www.youtube.com/watch?v=6niqTuYFZLQ), [여기](https://www.youtube.com/watch?v=AD3K8j12EIE), 그리고 [여기](https://www.youtube.com/watch?v=9zKuYvjFFS8)를 참고하시기 바랍니다.


<br/>


#### **1. RNN**
RNN은 hiddent layer에서 나온 결과값을 output layer로도 보내면서, 다시 다음 hidden layer의 input으로도 보내는 특징을 가지고 있습니다. 

아래 그림을 보시겠습니다.
<div align="center">  
 
![image](https://user-images.githubusercontent.com/99710438/164171475-fe065e6c-5bbf-4c9f-bc59-37c954b9717e.png)

</div>  

![](https://latex.codecogs.com/gif.latex?x_{t}) 는 input layer의 input vector, ![](https://latex.codecogs.com/gif.latex?y_{t}) 는 output layer의 output vector입니다. 실제로는 bias ![](https://latex.codecogs.com/gif.latex?b) 도 존재할 수 있지만, 편의를 위해 생략합니다. 

RNN에서 hidden layer에서 activation function을 통해 결과를 내보내는 역할을 하는 node를 셀(cell)이라고 표현합니다. 이 셀은 이전 값을 기억하려는 일종의 메모리 역할을 수행하므로 이를 **메모리 셀** 또는 **RNN 셀**이라고 합니다.

이를 식으로 나타내면 다음과 같습니다.

* Hidden layer: &nbsp; ![](https://latex.codecogs.com/gif.latex?h_{t}=tanh(W_{x}x_{t}+W_{h}h_{t-1}+b))

* Output layer: &nbsp; ![](https://latex.codecogs.com/gif.latex?y_{t}=f(W_{y}h_{t}+b))


idden layer의 메모리 셀은 각각의 시점(time step)에서 바로 이전 시점에서의 메모리 셀에서 나온 값을 자신의 입력으로 사용하는 재귀적(recurrent) 활동을 하고 있습니다. 그러나 그림에서 보이듯이, RNN은 각 time step에서만 정보를 처리하므로 time step이 불규칙적이거나, 각 time step 사이의 값에 대해서는 예측 성능이 좋지 않습니다. 

<br/>

<div align="center">
 
_저자들은 이런 **discrete한 hidden layer를 ODE를 사용해서 continuous하게** 바꾸려는 겁니다._
 
</div>


<br/>



#### **2. Neural Ordinary Differential Equations**

Neural ODE는 continuous-time model의 일종으로, 지금까지 discrete하게 정의되었던 hidden state ![](https://latex.codecogs.com/gif.latex?h_{t}) 를 ODE initial-value problem의 solution으로 정의합니다. 이를 식으로 나타내면 다음과 같습니다.

<div align="center">
 
![](https://latex.codecogs.com/gif.latex?dh_{t}/dt=f_{\theta}(h(t),t)) &nbsp; _where_ &nbsp; ![](https://latex.codecogs.com/gif.latex?h(t_{0})=h_{0})
 
</div>

여기서, ![](https://latex.codecogs.com/gif.latex?f_{\theta}) 는 hidden state의 dynamics를 의미하는 neural network입니다. 
Hidden state ![](https://latex.codecogs.com/gif.latex?h(t_{0})) 는 모든 시간에 대해 정의되어있으므로, 어떠한 desired time에 대해서도 아래의 식을 통해 evaluate 될 수 있습니다.

<div align="center">

![](https://latex.codecogs.com/gif.latex?h_{0},...,h_{N}=ODESolve(f_{\theta},h_{0},(t_{0},...,t_{N})))
 
</div>

<br/>


#### **3. Variational Autoencoder**


<br/>


> ### **ODE-RNN**



<br/>



> ### **Latent ODEs**

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

Neural ODE라는 새로운 방식을 여러 방면에 접목시킨 논문들이 우후죽순 생겨나고 있습니다. 처음 시도되는 방법론이다 보니 특별한 theoretical contribution이 없어도 접목만 잘 시키면 논문이 좀 더 publish 되기가 용이한 것 같습니다. 우리도 지금 어떤 연구가 trend인지 잘 follow up하는 자세가 필요한 것 같습니다.

---  
## **Author Information**  

* Yulia Rubanova
    * University of Toronto and the Vector Institute  
    * Deep generative models, Time series modelling, Optimization over discrete objects, Real-world applications
    
* Ricky T. Q. Chen
    * University of Toronto and the Vector Institute
    * Integrating structured transformations into probabilistic modeling, Tractable optimization
    
* David Duvenaud
    * University of Toronto and the Vector Institute
    * Neural ODEs, Automatic chemical design, Gradient-based hyperparameter tuning, Structed latent-variable models, Convolutional networks on graphs

## **6. Reference & Additional materials**  

* Github Implementation  
    * None

* Reference  
    * [Recurrent Neural Networks](https://wikidocs.net/22886)
    * [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
    * [Variational Autoencoder](https://arxiv.org/abs/1312.6114)
