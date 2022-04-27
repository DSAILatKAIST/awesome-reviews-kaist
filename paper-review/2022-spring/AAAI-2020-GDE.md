---
description : Michael Poli, Stefano Massaroli / Graph Neural Ordinary Differential Equations
 / AAAI(DLGMA)-2020  
---

# **Graph Neural Ordinary Differential Equations** 

## **1. Problem Definition**  
> Appropriate Relational Inductive Bias on GNN in terms of Continuous Domain !

적절한 `Inductive Bias` 를 부여하는 것은 딥러닝 모델에서 아주 중요한 요소로 부각된다. 이때, 우리는 Inductive bias가 무엇인지 조금 더 엄밀하게 짚고 넘어가야할 필요가 있다. Inductive Bias는 모델이 unseen data에 대해서 더욱 잘 예측하기 위해 우리가 도입하는 *추가적인 가정* 으로 이해하면 좋다. 이 중에서도 Relational Inductive Bias는 이름 그대로 모델을 이루는 entities 간의 relationship 에 대해 우리가 부여하는 추가적인 가정으로 이해할 수 있다. 대표적으로, MLP와 CNN을 비교해보면, MLP는 하나의 인풋이 모든 아웃풋에 관여하지만 인풋 간의 관계성은 등한시되는 반면 CNN은 convolution filter가 window sliding을 하며 업데이트를 이어나가게 되고 이를 바탕으로 local한 인풋은 서로 weight sharing을 하는, 서로 관계가 비슷할 것이라는 가정이 들어가게 된다. 즉, Relational Inductive Biase 측면에서 MLP는 Weak한 반면 CNN은 Local내에서 Strong하다고 볼 수 있다. 이 관점에서, **Relational Inductive Bias**는 **Weight Sharing**과 대응되는 개념으로 바라볼 수 있게 된다.

그렇다면, 이제 본론으로 넘어와서 우리가 현재 포커싱하고 있는 `Graph Neural Network` 관점에서 바라보면 어떠한가? GNN은 node와 edge를 바탕으로 한 그래프 자료구조에서 딥러닝을 접목시켰기 때문에 Relation은 직관적으로 edge를 바탕으로 이뤄지고 있음을 알 수 있다. 이때, Relational Inductive Bias는 정형화되지 않은, arbitrary한 특성을 가지고 있다. 이는 GNN에서는 특정 두 노드를 기준으로 엣지가 정의되고 있는 본질적인 성질 때문이다. 예를 들어보자면, 전 세계 대학원생의 그래프를 만들어본다고 했을 때, 각 국가를 라벨로 가지고 노드는 해당 대학원생의 특성으로 정의된다고 해보자. 이때, 두 대학원생을 이어주는 엣지는 취미생활이 같은 경우(e.g., 축구)라고 하면 한국 내에서 대학원생 김 모군 - 박 모군 의 관계도 있을 것이고 미국에서 John - Andrew 의 관계도 있을 것이다. 두 Pair(김-박, John-Andrew)는 그래프 내에서 local하지는 않아도 관계는 일치하기 때문에 비슷한 weight를 share할 가능성도 크다. 이러한 경우가 바로 aribitrary한 Relational Inductive Bias를 갖는 경우로 볼 수 있다. 앞선 두 문단을 하나의 사진으로 요약하면 아래와 같다.

<img align ="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/inductivebias.png" width="1000">

우리가 오늘 특히 집중할 Inductive Bias는 시스템 내 데이터를 바탕으로 한 `Temporal Behavior`, 즉 시간에 따른 행동양상이다. 앞서 우리는 inductive bias를 *추가적인 가정* 에 대응되는 개념이라고 했다. 즉, 현재 경우에 접목시켜보면 우리의 추가적인 가정은, 시간에 변화 따라 시스템의 dynamics이 **discrete**한 지, **continuous**인지 등의 가정을 부여해줄 수 있다. 이 중, 특히 신경망을 연속적인 layer의 구조로 표현하는 관점 그리고 *상미분방정식(ODE)의 초기값 문제의 해* 를 통해 이를 업데이트 하는 과정은 최근 딥러닝 모델로 하여금 새로운 패러다임을 제안했다고 볼 수 있다.

이러한 흐름 즉, GNN관점에서 Temporal Behavior 측면 상 적절한 Inductive Bias를 부여하기 위해 해당 paper는 아래 3가지의 문제에 접근하고자 한다.

1. **Blending graphs and differential equtations**
    - 상미분방정식(ODE)을 GNN으로 parameterize하고 system-theoretic한 framework, **Graph neural Ordinary Differential Euqations, GDE**를 제안하고자 한다. 이때 GDE는 ODE를 바탕으로 한 연속적인 관점, 그리고 GNN의 본질적인 Reltaional Inductive bias를 내포한 체로 디자인된다는 특징점을 가진다. 이러한 GDE는 효과성은 semi-supervised의 node classification task와 spatio-temporal foecasting task에서 실험적으로 입증한다.

2. **Sequence of graphs**
    - GDE의 framework를 spati-temporal 세팅으로 가저간 뒤, hybrid dynamical system관점에서 일반적인 autoregressive(자가회귀모형)으로 모델링한다. 이때, autoregressive GDE는 ODE의 적분 구간을 변경해줌으로써, irregular한 데이터 관측에서도 잘 대응할 수 있게 해준다.

3. **GDEs as general-purpose models**
    - 특히, GDE는 연속적인 환경에서 GDE를 효과적이게 모델링하기 위한 별도의 assumption, 가정이 불필요하다는 장점을 가진다. 즉, 범용적인 측면에서 높은 성능을 내고 효과적임을 실험적으로 입증한다.

## **2. Motivation**  
> GNN + Neural ODE = Continuous GNN !

이 Paper에서 기존의 GNN이 가지는 discrete한 모델링의 한계를 극복하고자 한다. 즉, continuous한 `Graph Neural Network`을 모델링해보고자 함이다. 무엇을 통해? 바로 NeurIPS 2018 best paper로 선정된 **Neural Ordinary Differential Equations** 에서 제안된 신경망을 미분방정식의 해로 바라보는 관점을 통해. 앞선 두 문장에서 유추할 수 있듯, 우리는 두 가지 key concept을 먼저 이해하고 본격적인 GDE의 모델링 측면으로 넘어가고자 한다.

### (1) Graph Neural Network
- 먼저, GNN을 이해할 필요가 있는데, 이 중 가장 대표적인 `GCN(Graph Convolutional Networks)`을 간단히 설명해보고자 한다. 기존의 딥러닝 분야(e.g., CNN)과 달리 GNN은 non-euclidean space에서 정의되고 feature들이 독립적이지 않다는 속성을 key motivation으로 삼아 발전한, Graph 자료구조에 신경망을 접목시킨 모델이라고 이해할 수 있다. 그 중, GCN은 이름에서 유추할 수 있듯, CNN에서의 Convolution 연산을 그래프 자료구조에 접목시킨 대표적인 모델이다. Graph에서 Convolution을 적용하기 위한 과정으로는 퓨리에 변환(Signal을 Frequency로 변환하는 과정)이 필수적으로 수반되게 되고 이때 우리는 Signal을 node의 label, Frequency를 중심노드와 이웃노드의 차이로 대응할 수 있게 된다. 핵심은, 우리는 이러한 중심노드와 이웃노드의 차이가 적기를 바라며, 이러한 **이웃노드들로부터 자신의 노드를 업데이트하는 과정** 이 바로 GCN의 본질이라는 것이다. 이에 대한 더욱 자세한 설명은 해당 페이지에서 GCN에 대한 PDF를 참고하면 도움이 된다. [https://github.com/SukwonYun/GNN-Papers](https://github.com/SukwonYun/GNN-Papers)

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/gcn.png">

- 수식을 통해 이해해보면 아래와 같이 설명할 수 있다. 이때, ![](https://latex.codecogs.com/svg.image?\mathbf{h_v^{l}})은 ![](https://latex.codecogs.com/svg.image?l)번째 레이어에서 노드 ![](https://latex.codecogs.com/svg.image?v)의 hidden representation 이고 ![](https://latex.codecogs.com/svg.image?\mathcal{N}_v)는 노드 ![](https://latex.codecogs.com/svg.image?v)의 이웃노드들의 집합이다. 수식에서 보듯 자기자신의 representation을 업데이트하는 과정에서 이웃노드들의 representation을 바탕으로 한다는 것이 GNN의 핵심이다. GCN은 이 중에서도 Laplacian 연산(i.e.,![](https://latex.codecogs.com/svg.image?\mathbf{\hat{D}}^{-1/2}\mathbf{\hat{A}}\mathbf{\hat{D}}^{-1/2}))을 통해 자기자신과 이웃노드들의 representation 평균 합으로 AGGREGATE하는 GNN으로 이해하면 된다. 이때, ![](https://latex.codecogs.com/svg.image?\mathbf{W}^{(l-1)})은 ![](https://latex.codecogs.com/svg.image?l-1)번 째 layer에서 업데이트 대상이 되는 파라미터이다.


$$
\begin{equation}
\mathbf{h}_{v}^{(l)}=\text{COMBINE}^{(l)}\left(\mathbf{h}_{v}^{(l-1)}, \text{AGGREGATE}^{(l-1)}\left(\left\{\mathbf{h}_{u}^{(l-1)}: u \in \mathcal{N}(v)\right\}\right)\right)
\end{equation}
$$

$$
\begin{equation}
    \mathbf{H}^{(l)} = \sigma(\mathbf{\hat{D}}^{-1/2}\mathbf{\hat{A}}\mathbf{\hat{D}}^{-1/2}\mathbf{H}^{(l-1)}\mathbf{W}^{(l-1)})
\end{equation}
$$

### (2) Nerual ODE
- 2018년 발표된 Neural Ordinary Differential Equations는 `Neural Network를 Continuous Domain`에서 바라볼 수 있게 한, 새로운 패러다임을 제안한 논문으로 평가되고 있다. 사실 이 논문의 key contribution은 신경망을 미분방정식의 해로 표현하는 그 주춧돌 역할을 제안했다기 보다는, Backward pass를 adjoint sensitivity method를 도입하므로써 gradient를 아주 효과적으로 구할 수 있게 해준데 있다. 먼저, 어떻게 discrete했던 기존의 신경망을 continuous하게 바라볼 수 있게 되었는지 `ResNet`을 통해 간단히 intuition을 살펴보고자 한다.
핵심은 ResNet에서 비롯된 residual connection을 좌변으로 넘겨서 1이었던 변화량을 generalize하여 미분의 관점으로 바라보는 것이다. 이로써, discrete했던 layer의 index 혹은 timestamp, ![](https://latex.codecogs.com/svg.image?t)를 하나의 variable로 모델링할 수 있어진다. 이는 아래의 식과 같이 나타낼 수 있고 이러한 변화를 통해 Residual Network와 ODE Network의 Depth별 gradient의 흐름, hidden state의 업데이트 과정을 아래와 같이 이미지화 할 수 있다.

$$
\begin{equation}
\begin{align*}
\mathbf{h}_{t+1} = f(\mathbf{h}_{t}, \theta_t) + \mathbf{h}_{t} \\
\mathbf{h}_{t+1} - \mathbf{h}_{t} = f(\mathbf{h}_{t}, \theta_t) \\
\frac{\mathbf{h}_{t+\Delta} - \mathbf{h}_{t}}{\Delta}|_{\Delta=1} = f(\mathbf{h}_{t}, \theta_t) \\
\lim_{\Delta \rightarrow 0}  \frac{\mathbf{h}_{t+\Delta} - \mathbf{h}_{t}}{\Delta} = f(\mathbf{h}_{t}, \theta_t) \\
\frac{d\mathbf{h}(t)}{dt} =f(\mathbf{h}(t),t,\theta) \\
\end{align*}
\end{equation}
$$

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/neuralode.png" width="900">

- 이러한 intuition을 바탕으로 2018년 Neural ODE는 Backward Pass에 `Adjoint Sensitivity Method`를 접목시켜서 parameter를 업데이트 시키는 과정에서 gradient를 훨씬 효과적으로 구해낼 수 있게 하였고 이는 연구자들로 하여금 새로운 출발점을 알린 획기적인 시점이 되었다. Forward Method와 대비되는 Adjoint Sensitivity Method는 과연 무엇인지 아래 슬라이드 두개로 대체하고자 한다. 간단히 요약하자면, 초기값 문제를 풀고 Loss를 정의하여 parameter를 업데이트하는 과정에서 time dependent solution function에 대한 parameter 변화량(i.e., ![](https://latex.codecogs.com/svg.image?\frac{d\textbf{u}}{d\boldsymbol{\theta}})을 구해야하는데 이를 구하기가 상당히 수고스러운 일이었다. 이에 비해 Adjoint Sensitivity Method는 이를 직접적으로 구하지 않고 Optimization 문제로 치환하여 Lagrangian을 도입하고 앞선 변화량(i.e., ![](https://latex.codecogs.com/svg.image?\frac{d\textbf{u}}{d\boldsymbol{\theta}})의 계수들을 0으로 만드는 별도의 초기값 문제를 하나 더 제안하여, 총 2개의 ODE를 푸는 것만으로 파라미터를 업데이트 하는 방법론이다.구체화 된 과정은 아래 슬라이드와 같이 나타낼 수 있다.  
  
  
<img align="left" src="../../.gitbook/2022-spring-assets/SukwonYun_1/forward.png" width="440" height="200">  

<img align="right" src="../../.gitbook/2022-spring-assets/SukwonYun_1/asm.png" width="440" height="200">  

  

- 이러한 Neural ODE는 기존의 딥러닝 모델과 같이 gradient를 직접적으로 구하는 것이 아닌 `gradient를 mimic하는 과정`으로 볼 수 있기에 별도의 gradient를 저장할 필요가 없어진다. 따라서, **memory efficient**하다는 장점, timestamp에 종속적이었던 시간 ![](https://latex.codecogs.com/svg.image?t)를 별도의 변수로 모델링하여 **구간 내의 dynamics를 하나의 함수로 모델링** 할 수 있다는 장점, **irregular한 time에 대한 대응**, 그 간 **수리적으로 입증되었던 미분방정식 풀이법을 딥러닝에 접목** 시킬 수 있다는 장점 등 다수의 매력을 내포한 체 딥러닝에 새로운 패러다임을 제안하게 되었다.



## **3. Method**  
본격적으로 Method로 들어가고자 한다. 앞선 Motivation이 어느정도 구체적이었고 길었던 이유는 바로 오늘의 Graph neural ordinary Differential Euqations, GDE가 결과적으로 `GNN과 Neural ODE를 접목시킨 퍼스트 펭귄의 역할을 하는 paper`로 볼 수 있기 때문이다. 먼저 순서는 GDE에 대한 definition, Static Model에서의 GDE, Spatio-Temporal Model에서의 GDE순으로 이번 파트를 설명하고자 한다.

### (1) **Definition of GDE**

먼저 기존의 resiudal connection이 추가된 기존의 GNN이 업데이트되는 방식을 살펴보면 아래와 같이 나타낼 수 있다. 이때 함수 **F**는 GNN layer로 바라볼 수 있고, parameter는 layer별로 지정되는 것을 볼 수 있다. 또한 layer는 자연수의 범위 내에서 정의됨을 확인할 수 있다.

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/equation1.png" width="650">  

다음으로는 앞서 Motivation에서 살펴봤듯, residual connection을 좌변으로 넘겨서 이 term을 변화량의 관점에서 해석한 뒤, 미분방정식을 새롭게 만들어내었을 때 비로소 우리는 '**Graph Neural Ordinary Differential Equation(GDE)**'의 초기값 문제(IVP)관점에서 아래와 같이 정의할 수 있게 된다. 중요한 점은, 위의 식과 달리 우리는 초기값을 가진 미분방정식을 Formulation 했다는 점이고, layer가 자연수 범위가 아닌 **실수 범위** 에서 정의된다는 점에 주목할 필요가 있다.

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/equation2.png" width="600">  

미분방정식을 Formulation하는 것도 중요하지만 또 하나의 중요한 점은 미분방정식의 해가 존재하는지 그리고 그 해가 유일한지 `Well-posedness`를 따져볼 필요가 있다. 우리는 이때, 적분구간을 [0,1]로 설정한 뒤, Hidden state에서의 **Lipshitz Continuity**, layer의 index로 해석될 수 있는 위의 식의 s에서의 **Uniform Continuity**를 조건으로 부여해줌으로써 해당 구간 내에서의 해(hidden representation)의 유일성을 정의할 수 있게 된다. 최종적으로 위의 hidden representation을 적분함으로써 우리는 GDE의 output을 아래와 같이 나타낼 수 있게된다. 이때, 적분구간을 [0,1]로 둔다고 하면, 0에서의 적분값은 정의한 미분방정식의 초기값과 만나 상쇄되게 된다.

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/equation3.png" width="350">  



### (2) **GDE on Static Models**

우리는 GDE를 2가지 관점에서 해석할 수 있게 되는데 먼저 time에 variant하지 않은 static 모델에서 정의할 수 있다. Residual connection을 가진 GCN은 아래와 같이 나타낼 수 있게 되고 해당 식을 여태 위에서의 과정과 같이 미분 관점에서 해석하게 되면 아래 두 번째 식으로 나타낼 수 있게 된다. 이 때, L은 Laplacian Matrix를 나타내고 Hidden Representation을 나타내는 함수 **F** 는 모델링의 자유도를 가지는데 주로 Multilayer Convolution 등으로 표현할 수 있게 된다. 이 때, 우리는 GCN을 베이스로 하였기에 `Graph Convolutional Differential Equation, GCDE`로 GDE를 부를 수 있게된다.

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/equation4.png" width="500">  
<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/equation5.png" width="800">  


### (3) **GDE on Spatio-Temporal Models**

다음으로는, 시간에 따라 variant한, autoregressive 속성을 가진 Spatio-Temporal 관점에서 GDE를 정의할 수 있다. 시간의 축을 포함시켜줌으로써, 우리는 depth domain을 time domain과 동치시킬 수 있다(RNN에서 layer를 쌓는다는 것은 그 만큼 더 많은 time input을 고려해준다는 의미와 상통된다고 생각하면 된다). 따라서 특정 시간에서 시간 변화량만큼 적분을 해줌으로써 우리는 Hidden representation을 업데이트할 수 있게 된다. 이 때 역시, 미분방정식의 관점에서 이를 해석할 수 있게되고, 시간 구간 내에서의 dynamic를 parameter ![](https://latex.codecogs.com/svg.image?\boldsymbol{\theta}) 를 가진 하나의 함수로서 나타낼 수 있게된다. GDE Framework에서 아래 등장하는, F, G, K는 **GNN-operator** 혹은 GNN layer로 생각할 수 있다.

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/equation6.png" width="500">  
<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/equation7.png" width="550">  


## **4. Experiment**  
다음은 GDE의 효과성을 입증하기 위한 실험이다. 먼저 실험은 크게 3파트, `(1) Semi-supervised node classification`, `(2) Trajectory extrapolation task`, `(3) Traffic forecasting` 에서 진행되었다. 먼저 Overall하게 각 task에서 사용한 Dataset과 Baseline 그리고 Evaluation Metric을 정리하면 아래와 같이 나타낼 수 있다.

### **Experiment setup**  
* Dataset  

    - (1): Graph 분야에서 Benchmark dataset으로 꼽히는 Cora, Citeseer, Pubmed 기준으로 실험을 진행하였다. 특히 GDE의 Static 버전에서 실험을 진행하였다.
    - (2): 시간에 따른 dynamical behavior를 효과적으로 살펴보기 위해 multi-particle system을 토대로 하였고 시간의 흐름에 따라 입자의 position, velocity의 trajectory가 어떻게 그려지는지 살펴보았다.
    - (3): Undersampled 버전의 PeMS 데이터 셋을 활용하였고 이는 228개의 센서 스테이션이 5분의 주기로 교통량의 정보를 담고있는 데이터 셋이다.

* baseline  

    - (1): 기존의 GCN, GCDE가 가장 효과적이었을 때의 hyperparmeter 셋팅인 GCN* 을 사용하였고, 대표적인 Numerical Solver인 Runge-Kutta 2,4 그리고 Dormand-Prince를 비교하였다.
    - (2): 다른 모델과의 비교에 초점을 두기보다는 position, velocity에서의 trajectory를 중점적으로 살펴보았다.
    - (3): 시간에 따른 변화를 잘 모델링할 수 있는 기존의 GRU, 그리고 Graph Convolution을 접목시킨 GCGRU와 중점적으로 비교하였다.

* Evaluation Metric
    - (1): Node classification task에서 주로 활용되는 test accuracy를 기반으로 하였다.
    - (2): 업데이트 하는 과정에서 Mean Squared Error, MSE를 활용하였고, Figure로는 Trajectory를 제시하였다.
    - (3): Root Mean Squared Error (RMSE)와 Mean Absolute Percentage Error (MAPE)를 활용하였다.

### **Result**  
**(1) Semi-supervised node classification**

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/experiment1.png">  

위의 Figure는 Node embedding 의 trajectory를 2차원으로 나타낸 그림인데 색은 node의 label이고 적분 구간이 종료점에 가까워질때까지 trajectories가 divergent하고 그말인 즉슨, label별로 update가 잘 일어나고 있음을 나타낸다.

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/experiment2.png" width="900">  

기존의 GCN, 그리고 가장 좋았던 hyperparmeter 셋팅을 바탕으로 한 GCN* 와의 비교 그리고 numerical solver의 variant로 비교를 한 Test accuracy이다. 성능이 압도하진 않지만 기존의 vanilla GCN 보다는 우월함을 드러내는 Table로 해석될 수 있다.

**(2) Trajectory extrapolation task**

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/experiment5.png" width="900">  

이 task는 Extrapolation에서의 GDE의 퍼포먼스를 중점적으로 테스트한 결과인데, 변위 그리고 속도의 관점에서 입자의 trajectory를 plotting 한 결과이다. 총 10개의 particle system에서 측정하였기에 색은 총 10개의 variant한 입자를 나타내며, 각 입자별로 입자의 변위 그리고 속도 변화를 잘 표현해내고 있음을 확인할 수 있다.


**(3) Traffic forecasting**

<img align="center" src="../../.gitbook/2022-spring-assets/SukwonYun_1/experiment3.png">  

Model에서의 퍼센티지는 Undersample 관점에서 해석하여, training 으로 활용한 데이터셋의 비율이다. 상대적으로 Harsh한 Undersampling 환경에서도 GCDE-GUR의 우월함을 확인할 수 있다. 앞선 static 실험에서와 달리 성능 향상의 폭이 유의미한 것을 확인할 수 있는데, 이로 유추해보아 확실히 time domain, continuous domain에서의 GDE의 효과성을 살펴볼 수 있다.

## **5. Conclusion**  

이로써 우리는 GNN과 Neural ODE가 처음으로 접목된, GDE를 살펴볼 수 있다. 오늘의 Paper는 퍼스트펭귄의 역할로서의 유의미함이 크고, 어떻게 Graph domain에 Neural ODE를 효과적으로 접목시킬 수 있는지에 대한 고민을 바탕으로 발전된 논문이다. 세줄 요약을 해보면 다음과 같다.

- (1) GNN에 Neural ODE를 접목시킨 초창기 연구
- (2) Graph Neuarl Ordinary Differential Equation, GDE에 대한 정의
- (3) GDE에 대한 분류 - Static, Spatio-Temporal

  


그리고, 필자는 이렇게 글을 마무리 짓고 싶다.  

**`GNN with Differential Equations? Way to go!`**

## **Author Information**  

* Sukwon Yun (윤석원) 
    * Master Student in ISySE, KAIST ([DSAIL](http://dsail.kaist.ac.kr/))
    * Interested in **GNN, Differential Equations on NN, Long-Tail Problem on NN**
    * Contact: swyun@kaist.ac.kr 

## **6. Reference & Additional materials**  
- Paper: [https://arxiv.org/abs/1911.07532](https://arxiv.org/abs/1911.07532)
- Code: [https://github.com/Zymrael/gde](https://github.com/Zymrael/gde)
- Neural ODE: [https://arxiv.org/abs/1806.07366](https://arxiv.org/abs/1806.07366)
    - **< 도움이 되는 영상자료 모음 >**
    - 직관적 이해: [https://www.youtube.com/watch?v=AD3K8j12EIE](https://www.youtube.com/watch?v=AD3K8j12EIE)
    - 논문의 key point: [https://www.youtube.com/watch?v=jltgNGt8Lpg](https://www.youtube.com/watch?v=jltgNGt8Lpg)
    - 최윤재 교수님 강의자료: [https://www.youtube.com/watch?v=sIFnARdTVvE&t=3046s](https://www.youtube.com/watch?v=sIFnARdTVvE&t=3046s)
    - Adjoint Sensitivity Method: [https://www.youtube.com/watch?v=k6s2G5MZv-I&t=512s](https://www.youtube.com/watch?v=k6s2G5MZv-I&t=512s)
- GCN: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
    - PDF 자료: [https://github.com/SukwonYun/GNN-Papers](https://github.com/SukwonYun/GNN-Papers)
    - 윤훈상 연구원님 자료: [https://www.youtube.com/watch?v=F-JPKccMP7k&t=635s](https://www.youtube.com/watch?v=F-JPKccMP7k&t=635s)


