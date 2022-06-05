---
description: Qitian Wu et al. / Handling Distribution Shifts on Graphs_An Invariance Perspective / ICLR-2022
---

# Handling Distribution Shifts on Graphs: An Invariance Perspective

논문 [링크](https://arxiv.org/abs/2202.02466)

Official Github은 아직 공개되지 않았습니다.

## **1. Problem Definition**

### **1-1. Background**

#### **1-1-1. Out-of-distribution Generalization**

현실 문제에서 보지 않은 데이터를 처리하는 것에 대한 요구가 증가함에 따라 _out-of-distribution(OOD) generalization*_ 에 대한 연구가 활발해지고 있지만, 최근 연구들은 인공신경망(Neural Networks)들이 _distribution shift_ 에 민감하기 때문에 새로운 환경 내에서 만족스럽지 못한 성능을 보일 수 있음을 시사하고 있습니다.
_OOD generalization_ 에 대한 최근의 연구들은 training 데이터와 test 데이터 사이의 _distribution shift_ 의 원인을 잠재적으로 알려지지 않은 환경 변수 ![](https://latex.codecogs.com/svg.image?\mathbf{e})로 취급합니다.
머신러닝 문제를 ![](https://latex.codecogs.com/svg.image?\mathbf{x})가 주어졌을 때 ![](https://latex.codecogs.com/svg.image?\mathbf{y})를 예측하는 것으로 가정한다면, 앞서 언급한 환경 변수 ![](https://latex.codecogs.com/svg.image?\mathbf{e})는 데이터 생성 분포에 영향을 미칩니다: ![](https://latex.codecogs.com/svg.image?p(\mathbf{x},\mathbf{y}|\mathbf{e})=p(\mathbf{x}|\mathbf{e})p(\mathbf{y}|\mathbf{x},\mathbf{e})).
따라서, OOD 문제는 아래와 같이 표현할 수 있습니다.

$$\begin{equation}\min_{f}\max_{e \in \mathcal{E}} \mathbb{E}_{(\mathbf{x}, y) \sim p(\mathbf{x},\mathbf{y}|\mathbf{e}=e)}[l(f(\mathbf{x}), y)|e]\end{equation}$$

여기서, ![](https://latex.codecogs.com/svg.image?\mathcal{E})는 환경변수의 support, ![](https://latex.codecogs.com/svg.image?f)는 예측모델, ![](https://latex.codecogs.com/svg.image?l(\cdot&space;,\cdot)) 은 loss 함수를 의미합니다.

\* Out-of-Distribution Generalization

대부분의 모델들은 Training 데이터와 Test 데이터가 동일한 분포를 가진다고 가정합니다. 그러나 그림 1과 같이 모델이 예측하는 데이터가 training 데이터에 적절히 표현되어 있지 않았던 경우, 이 데이터를 OOD (out-of-distribution)라고 합니다. 모델은 각 예측마다 자신의 예측에 대한 신뢰도(또는 불확실도)를 확률값으로 출력하는데, 이러한 OOD 데이터가 모델에 입력되었을 때 그림 2와 같이 자신 있게 잘못된 예측을 하는 경우가 발생하게 됩니다.

좀 더 자세한 내용은 아래의 survey 논문과 workshop을 참고해주세요:

* [Towards Out-of-Distribution Generalization: A Survey](https://arxiv.org/abs/2108.13624)
* [NeurIPS 2021 Workshop - Out-of-distribution generalization and adaptation in natural and artificial intelligence](https://nips.cc/Conferences/2021/ScheduleMultitrack?event=21852)
* [NeurIPS 2021 Workshop - Distribution Shift: connecting method and application (DistShift)](https://neurips.cc/Conferences/2021/ScheduleMultitrack?event=21859)

<p align="center">
  <img width="50%" src="../../.gitbook/2022-spring-assets/yunhak2/ood1.png">
</p>
<div style="text-align:center;">
<p><span style="color:grey; font-size:75%";><em>그림 1 - OOD 예</em></span></p>
</div>

<p align="center">
  <img width="50%" src="../../.gitbook/2022-spring-assets/yunhak2/ood2.png">
</p>
<div style="text-align:center;">
<p><span style="color:grey; font-size:75%";><em>그림 2 - OOD 샘플에 대한 높은 신뢰도의 잘못된 예측의 예</em></span></p>
</div>

#### **1-1-2. Invariant Model**

실제로 training 데이터가 모든 환경은 포함하고 있지 않기 때문에 Eq1과 같은 OOD 문제를 푸는 것은 매우 어려운 일입니다. 다시 말하면, 실제 목표는 ![](https://latex.codecogs.com/svg.image?p(\mathbf{x},\mathbf{y}|\mathbf{e}=e_1))의 데이터로 학습된 모델을 ![](https://latex.codecogs.com/svg.image?p(\mathbf{x},\mathbf{y}|\mathbf{e}=e_2)) 데이터에서도 일반화시키는 것이 됩니다.
최근 연구<sup>[1](#fn1)</sup>는 다음과 같은 __data-generating__ 가정을 통해 domain-invariant model을 학습하는 새로운 가능성을 제시했습니다:

> 여러 다른 환경에 걸쳐서 ![](https://latex.codecogs.com/svg.image?\mathbf{y})를 예측하는데 invariant한 정보가 ![](https://latex.codecogs.com/svg.image?\mathbf{x})의 일부(portion)에 존재한다.

이를 근거로, 이들은 _equipredictive_ representation model ![](https://latex.codecogs.com/svg.image?h)를 학습하는 것을 중요한 포인트로 봅니다 (이 때, ![](https://latex.codecogs.com/svg.image?h)는 모든 환경 ![](https://latex.codecogs.com/svg.image?\mathbf{e})에 대해 같은 conditional distribution ![](https://latex.codecogs.com/svg.image?p(\mathbf{y}|h(x),\mathbf{e}=e))을 생성). 즉, 이런 representation ![](https://latex.codecogs.com/svg.image?h(x))는 임의의 환경에서 어떤 downstream classifier에 대해 같은 (최적의) 성능을 가져올 것이라는 것을 의미합니다. 이런 모델을 ![](https://latex.codecogs.com/svg.image?\hat{p}(\mathbf{y}|\mathbf{x})) invariant 모델 또는 predictor라고 부릅니다.


### **1-2. Problem Formulation**

앞서 1-1장에서 일반적인 OOD Generalization에 대한 배경지식을 살펴보았으니, 이 장에서는 범위를 좀 더 좁혀 본 논문에서 실제 풀고자 하는 OOD Generalization 문제를 좀 더 엄밀히 정의해보겠습니다.

대부분의 OOD에 대한 연구는 이미지 등과 같은 Euclidean data에 대해 탐구되었고 그래프 구조 데이터와 관련된 연구는 거의 없었습니다.
그래프 구조 데이터에 대한 많은 연구가 개별 노드들에 대한 예측 문제를 포함하는데, 이런 그래프 구조의 데이터는  Euclidean data와는 다른 두 가지 특성을 가집니다:

1. Node는 동일한 환경(즉, 하나의 그래프) 내 데이터 생성 측면에서 non-independent 및 non-identically distributed하게 상호 연결되어 있습니다게
2. Node feature 이외에도 structural 정보가 예측에 있어서 중요한 역할을 하며, 환경이 변화하는 상황에서 모델이 일반화하는 과정에 영향을 미칩니다.

__따라서, 본 논문은 그래프 구조 데이터의 node-level task에서 발생할 수 있는 OOD 문제를 distribution shift의 관점에서 해석하여 이를 해결하는 새로운 방법론을 제시하려 합니다.__


#### **1-2-1. Out-of-distribution Problem for Graph-Structured Data**

앞으로의 논의에서 사용할 표기법(notation)은 다음과 같습니다.

* Input graph: ![](https://latex.codecogs.com/svg.image?G=(A,X))
    * Random variable of input graph: ![](https://latex.codecogs.com/svg.image?\mathbf{G})
* Node set: ![](https://latex.codecogs.com/svg.image?V)
* Adjacency matrix: ![](https://latex.codecogs.com/svg.image?A=\{a_{vu}|v,u&space;\in&space;V\})
* Node feature: ![](https://latex.codecogs.com/svg.image?X=\{x_v|v\in&space;V\})
* Label: ![](https://latex.codecogs.com/svg.image?Y=\{y_v|v\in&space;V\})
    * Random variable of nodel label vector: ![](https://latex.codecogs.com/svg.image?\mathbf{Y})

그러므로 Eq.1은 다음과 같이 표현될 수 있습니다. ![](https://latex.codecogs.com/svg.image?p(\mathbf{G},\mathbf{Y}|\mathbf{e})=p(\mathbf{G}|\mathbf{e})p(\mathbf{Y}|\mathbf{G},\mathbf{e}))

다만, 위의 정의는 node-level task에는 적합하지 않습니다. 왜냐하면, 대부분의 node-level task는 다양한 노드를 포함하는 하나의 단일 그래프가 모델의 input이기 때문입니다. 따라서 본 논문은 local view를 보는 방법을 선택해 각 노드에 L-hop ego-graph(이를 필요한 모든 정보를 포함하고 있는 부분 집합인 [markov blanket](https://en.wikipedia.org/wiki/Markov_blanket)으로 볼 수 있음)를 취하는 방법으로 문제를 변형하여 정의합니다.

그렇다면, 위의 표기법들을 각각 ![](https://latex.codecogs.com/svg.image?v) 노드에 대한 ego-graph 용으로 다음과 같이 표현할 수 있습니다.

* Centor node: ![](https://latex.codecogs.com/svg.image?v)
    * Random variable of nodes: ![](https://latex.codecogs.com/svg.image?\mathbf{v})
* L-hop neighbors of node ![](https://latex.codecogs.com/svg.image?v): ![](https://latex.codecogs.com/svg.image?N_v)
* Input graph: ![](https://latex.codecogs.com/svg.image?G_v=(A_v,X_v))
    * Random variable of input graph: ![](https://latex.codecogs.com/svg.image?\mathbf{G_v})
* Adjacency matrix: ![](https://latex.codecogs.com/svg.image?A_v=\{a_{uw}|u,w&space;\in&space;N_v\})
* Node feature: ![](https://latex.codecogs.com/svg.image?X_v=\{x_u|u\in&space;N_v\})
* Label: ![](https://latex.codecogs.com/svg.image?Y=\{y_v|v\in&space;V\})
    * Random variable of nodel label vector: ![](https://latex.codecogs.com/svg.image?\mathbf{Y})

그러므로, 데이터 생성 과정 ![](https://latex.codecogs.com/svg.image?\{(G_v,&space;y_v)\}_{v&space;\in&space;V}&space;\sim&space;p(\mathbf{Y}|\mathbf{G},\mathbf{e}))은 다음과 같이 2 단계로 나타낼 수 있습니다:

1. 전체 input graph가 ![](https://latex.codecogs.com/svg.image?\{G\sim&space;p(\mathbf{G}|\mathbf{e}))를 통해 생성되고 이것은 ![](https://latex.codecogs.com/svg.image?\{G_v\}_{v&space;\in&space;V})로 분리
2. 각 노드의 label은 ![](https://latex.codecogs.com/svg.image?\{y&space;\sim&space;p(\mathbf{y}|\mathbf{G_v}=G_v,\mathbf{e}))를 통해 생성

그렇다면 Eq.1은 다음과 같이 표현될 수 있습니다.

$$\begin{equation}\min_{f}\max_{e \in \mathcal{E}} \mathbb{E}_{\cancel{G \sim p(\mathbf{G}|\mathbf{e}=e)}}\bigg[\frac{1}{|V|}\sum_{v \in V}{\mathbb{E}_{y\sim p(\mathbf{y}|\mathbf{G_v}=G_v, \mathbf{e}=e)}[l(f(G_v),y)]}\bigg]\end{equation}$$

#### 1-2-2. Invariant Features for Node-Level Prediction on Graphs

1-1-2절에서 논의한 것과 같이 Invariant model은 data-generating 과정에 대한 다음과 같은 가정을 가진다: Input의 일부의 feature(invariant feature)는 1) target에 대한 충분한 예측정보를 가지고 있고, 2) 다양한 환경에 걸쳐서 downstreaam classifier에 대한 동일한 (최적의) 성능을 가져온다.

Node-level에서 invariant feature를 정의하기 어렵기 때문에, 이들은 Weisfeiler-Lehman test를 활용해 invariant 가정에 대한 정의<sup>[2](#fn2)</sup><sup>[3](#fn3)</sup><sup>[1](#fn1)</sup>를 확장했습니다.

<p align="center">
  <img src="../../.gitbook/2022-spring-assets/yunhak2/assumption1.png">
</p>

이 정의는 각 레이어에 이웃된 노드는 다양한 환경에 걸쳐 y를 안정적으로 예측하는데 기여하는 causal feature를 일부분 가지고 있다고 해석할 수 있습니다. 이는 1) (non-linear) transformation ![](https://latex.codecogs.com/svg.image?h_{l}^{*})가 각 레이어 마다 다를 수 있다는 점과 2) 임의의 노드 u의 centor node v에 대한 causal effect는 ego-graph에서의 상대적인 위치에 따라 달라질 수 있다는 이점을 가지기 때문에, 그래프 데이터에 대해 유연한 모델을 생성할 수 있습니다.

## **2. Motivation**

앞서 1-2-2절에서 논의한 Assumption 1을 기반으로 한 간단한 실험에서 저자들은 GCN이 가지는 한계를 발견하고 이를 Motivation으로 삼습니다.

Ego-graph ![](https://latex.codecogs.com/svg.image?G_v) (와 ![](https://latex.codecogs.com/svg.image?N_v))를 1-hop으로 정의하고, 논의를 간단하게 하기 위해 ![](https://latex.codecogs.com/svg.image?h^{*})와 ![](https://latex.codecogs.com/svg.image?c^{*})를 identity mapping으로, ![](https://latex.codecogs.com/svg.image?\Gamma)를 mean pooling으로 정의해 봅시다.
그리고, 2차원의 node feature ![](https://latex.codecogs.com/svg.image?x_v=[x_v^{(1)},x_v^{(2)}])와

$$\begin{equation}y_v=\frac{1}{|N_v|}\sum_{u \in N_v}{x_u^{(1)} + n_{v}^{(1)}}, x_v^{(2)}=\sum_{u\in N_v}{y_u+n_v^{(2)}+\epsilon}\end{equation}$$

로 가정하고, 이 때 ![](https://latex.codecogs.com/svg.image?n_v^{(1)},n_v^{(2)})는 independent standard normal noise이고, ![](https://latex.codecogs.com/svg.image?\epsilon)은 평균이 0이고 분산은 0이 아닌 환경 변수 e에 종속적인 random variable 입니다.

이를 바탕으로 vanilla GCN이 예측 모델이라면 ![](https://latex.codecogs.com/svg.image?\hat{y}_v=\frac{1}{|N_v|}\sum_{u&space;\in&space;N_v}{\theta_1&space;x_{u}^{(1)}&plus;\theta_2&space;x_{u}^{(2)}}이고, solution은 ![](https://latex.codecogs.com/svg.image?[\theta_1,&space;\theta_2]=[1,0]) 입니다. 즉, GCN은 invariant feature (i.e. https://latex.codecogs.com/svg.image?x_{u}^{(1)})을 알아낼 수 있다는 것을 의미합니다. 하지만, 아래의 명제를 보다시피 __일반적인 empirical risk minimization(ERM)를 사용할 때, 우리는 ideal solution을 도출할 수 없음__ 을 확인하게 됩니다 (자세한 증명은 논문의 Appendix 참고).

<p align="center">
  <img src="../../.gitbook/2022-spring-assets/yunhak2/motivation1.png">
</p>

여기서 저자들은 명제 2를 통해 다양한 환경에 걸쳐 variance를 tackle하여 최적 해를 도출할 수 있는 새로운 objective를 제안합니다. 

<p align="center">
  <img src="../../.gitbook/2022-spring-assets/yunhak2/motivation2.png">
</p>

즉, 본 논문은 node-level task에서의 OOD 문제를 정의하고, 기존의 접근법과 달리 varience를 tackle하는 접근법을 통해 invariant 가정에 근거한 새로운 학습 방법 (_Explore-to-Extrapolate Risk Minization; EERM_)을 제시합니다.


## **3. Method**  



## **4. Experiment**

### **4-1. Datasets**

### **4-2. Handling Distribution Shifts with Artificial Transformation**

### **4-3. Generalizing to Unseen Domains**

### **4-4. Extrapolating over Dynamic Data**

## **5. Conclusion**



**Contribution**



**Limitation**


***

## **Author Information**

* [오윤학(Yunhak Oh)](https://yunhak0.github.io)
  * M.S. Student in [DSAIL](https://dsail.kaist.ac.kr) at KAIST
  * Research Topic: Artificial Intelligence, Data Mining, Graph Neural Networks

## **6. Reference & Additional materials**

<a name="fn1">1</a>: Martín Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. CoRR, abs/1907.02893, 2019.

<a name="fn2">2</a>: Mateo Rojas-Carulla, Bernhard Schölkopf, Richard E. Turner, and Jonas Peters. Invariant models for causal transfer learning. Journal of Machine Learning Research, 19:36:1–36:34, 2018.

<a name="fn3">3</a>: Mingming Gong, Kun Zhang, Tongliang Liu, Dacheng Tao, Clark Glymour, and Bernhard Schölkopf. Domain adaptation with conditional transferable components. In International Conference on Machine Learning (ICML), pp. 2839–2848, 2016.

<a name="fn4">4</a>: 

<a name="fn5">5</a>: 

<a name="fn6">6</a>: 

<a name="fn7">7</a>: 

<a name="fn8">8</a>: 

<a name="fn9">9</a>: 

<a name="fn10">10</a>: 

<a name="fn11">11</a>: 

<a name="fn12">12</a>: 

<a name="fn13">13</a>: 
