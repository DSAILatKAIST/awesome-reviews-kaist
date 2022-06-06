---
description: >-
  Liu, H. et al./ Overcoming Catastrophic Forgetting in Graph Neural Networks /
  AAAI-2021
---

# Overcoming Catastrophic Forgetting in Graph Neural Networks

  

## **1. Problem Definition**

  

> ### **과거의 정보를 유지**하면서 계속해서 들어오는 **새로운 정보를 학습**한다

본 논문은 Graph domain에서 **Catastrophic Forgetting**을 최대한 방지하는 `Continual learning` 모델을 제시합니다. 


> ### `Continual learning`이란?

과거의 정보를 최대한 유지하면서 새로운 정보를 학습하는 방법으로, `Lifelong learning`, `Incremental learning`이라고도 부릅니다. 

인간이 '강아지'라는 동물을 알고 있는 상태로, '고양이'라는 동물을 새로 학습했을 때, '강아지'를 잊지 않고 '강아지'와 '고양이'를 구별해 낼 수 있는 것처럼, 인공지능을 지속적으로 들어오는 새로운 class의 data를 학습함과 동시에 이전에 학습되었던 class들을 잊지 않고 구별할 수 있게 하는 것이 목적입니다.

이 때, 새로운 데이터가 들어옴에 따라 이전에 학습했던 data의 정보를 망각하는 과정을 **Catastrophic Forgetting**이라고 합니다. 아래 그림을 보시겠습니다. 

![Continual learning과 Catastrophic forgetting](https://user-images.githubusercontent.com/99710438/170692095-ffd4e6e3-e483-4f3a-9142-a8bc77d2a5c1.png)

그림에서 볼 수 있듯이, Task 1에서는 파란색 node들을 구별하도록 학습합니다. Task 2에서는 보라색 class의 새로운 node가 추가되면서 파란색과 보라색을 포함해 학습시키고, Task 3에서는 빨간색의 새로운 node가 추가되면서 새롭게 학습이 진행됩니다. 이 과정이 `Continual learning`입니다.  

그리고 Task가 진행됨에 따라 이전 Task에서 학습했던 node들에 대한 예측 성능이 낮아지는 것을 볼 수 있습니다. 예를 들어 Task 1에서 파란 node들은 95%의 예측성능을 보였지만, Task 2에서는 55%로 줄었고, Task 2에서 보라색 node들은 94%의 성능을 보인 반면 Task 3에서는 56%에 불과합니다. 이렇게 Task가 진행됨에 따라 앞서 학습했던 정보를 잊는 것을 **Catastrophic Forgetting**이라고 합니다.


_**저자들은 Catastrophic Forgetting을 최대한 줄이는 `Graph Continual Learning` 모델을 제시하고자 합니다.**_

  
   
   

## **2. Motivation**

  

> ### Non-grid domain에서의 `Continual learning`

지금까지 주류를 이루는 `Continual learning` 방법론은 Image 데이터에 적용되는 `CNN` based 모델들이 많습니다. 하지만 실제 세계의 데이터는 non-grid 형태가 많은데, Graph 데이터에 적용되는 모델은 많이 없기 때문에 저자들은 `GNN`에 적용될 수 있는 `Continual learning` 방법론을 소개합니다.

`CNN` 기반 모델들과 달리, 본 논문에서는 그래프의 **topological 정보**까지 고려하는 `Topology-aware Weight Preserving(TWP)` 모듈을 제시합니다.

이 모듈을 제시함으로써 parameter를 update할 때 **node-level learning** 뿐 아니라 **node 사이의 propagation**까지 고려할 수 있게 되는 것입니다. 

  

> ### Computation & memory cost!

`Continual learning`의 대표적인 방법 중 하나로 replay apporach가 있습니다. 이는 이전 task에 있었던 data를 이후 task의 data를 학습시킬 때도 사용하는 방법인데요, task가 많아짐에 따라 replay approach는 computation cost와 memory cost가 증가하게 됩니다. 

반면에, 저자들은 이전 task를 학습하는데 **중요했던 parameter들을 최대한 보존**하고, **중요하지 않은 parameter들을 이후 학습에 최대한 활용**하는 방식으로 computation & memory cost를 줄이려고 합니다.


  
  

## **3. Method**

  

> ### **Preliminaries**: `GNN`

논문에서 제안한 방법론을 이해하기 위해서는 `GNN`의 개념을 알고 있어야 합니다.

본 포스팅에서는 간단하게 소개를 하겠습니다.

$$N$$개의 노드를 가진 그래프 $$\mathcal{G}= \lbrace \mathcal{V},\mathcal{E} \rbrace $$ 가 주어지고, $$X = \lbrace x_{1}, x_{2}, ..., x_{N} \rbrace$$ 을 node feature의 집합이라고 하고, $$A$$를 node들의 관계를 표현하는 adjacency matrix라고 하겠습니다.

$$l-th$$ hidden layer에서의 $$v_{i}$$의 hidden representation을 $$h_{i}^{(l)}$$ 이라고 할 때, 이 $$h_{i}^{(l)}$$는 다음과 같이 계산됩니다:

$$h_{i}^{(l)} = \sigma(\sum_{j \subset \mathcal{N}(i)} \mathcal{A_{ij}}h_{j}^{(l-1)}W^{(l)})$$

이 때, $$\mathcal{N}(i)$$ 는 $$v_{i}$$의 neighbors를 의미하고, $$\sigma ( \bullet )$$는 activation function, $$W^{(l)}$$은 $$l-th$$ layer의 transform matrix를 나타냅니다.

$$h_{i}^{(0)}$$은 node $$v_{i}$$의 input feature를 나타내고, $$\mathcal{A}$$는 neighbors의 aggregation strategy이며, `GNN`의 핵심 중 하나입니다.

이 논문에서는 `GAT`를 기본 `GNN`모델로 채택하는데, `Attention based GNN`중 하나인 `GAT`는 이 $$\mathcal{A}$$를 다음과 같이 pair-wise attention으로 정의합니다.

$$e_{ij}^{(l)} = S_{j \subset \mathcal{N}(i)}a(h_{i}^{(l-1)}W^{(l)},h_{j}^{(l-1)}W^{(l)})$$ 

이 때, $$a$$는 neural network이고, $$S$$는 softmax normalization입니다. 

`GAT`에 관해 자세한 부분은 [원 논문](https://arxiv.org/abs/1710.10903)을 참고하시기 바랍니다. 

  

> ### Problem Formulation

연속적인 학습 과정에서, 모델은 일련의 task $$\mathcal{T} = \lbrace \mathcal{T_{1}}, \mathcal{T_{2}}, ..., \mathcal{T_{K}}  \rbrace $$ 을 받습니다.

각 task $$\mathcal{T_{k}}$$는 training node set $$\mathcal{V_{k}^{tr}}$$과 testing node set$$\mathcal{V_{k}^{te}}$$으로 구성되어 있고, 이들 각각은 feature sets $$X_{k}^{tr}$$, $$X_{k}^{te}$$를 포함하고 있습니다. 각 task의 label은 겹치지 않습니다. (다른 task에는 다른 class의 node들이 학습된다는 의미입니다.)

  

> ### Topology-aware Weight Preserving

본 논문에서 제시하는 TWP 모듈은 topology 정보를 구함과 동시에 각 task에서 학습 관련 중요한 파라미터, topology 관련 중요한 파라미터를 찾아냅니다. 

TWP 모듈은 두가지 서브 모듈로 구성됩니다. 첫 번째는 minimized loss preserving 모듈이고, 두 번째는 topological structure preserving 모듈입니다.

  

**1. Minimized Loss Preserving**

Task $$\mathcal{T_{k}}$$를 학습한 뒤에, 모델은 해당 task에서 loss를 최소화하는 optimal parameter $$W_{k}^{*}$$ 를 가지고 있습니다. 

Parameter가 아주 조금($$\Delta W = \lbrace \Delta w_m \rbrace$$) 변할 때, loss의 변화량은 다음과 같이 나타낼 수 있습니다.

$$\mathcal{L}(X_{k}^{(tr)};W+\Delta W)-\mathcal{L}(X_{k}^{tr};W) \approx \sum_{m} f_{m}(X_{k}^{tr}) \Delta w_m$$

이 때, $$f_{m}(X_{k}^{tr})$$는 $$w_m$$에 해당되는 loss의 gradient입니다. 

미래의 Task를 진행하는 동안, 이 task $$\mathcal{T_{k}}$$를 기억하기 위해서 저자들은 Minimized Loss Preserving module을 통해 해당 task를 학습하는데 중요한 parameter들을 최대한 보존하고자 합니다. 

Parameter $$w_{m}$$의 중요도는 $$f_{m}$$의 크기로 나타내며, $$\mathcal{T_{k}}$$에서의 전체 파라미터 $$W$$의 중요도는 $$I_{k}^{loss} = [\lVert f_m(X_{k}^{tr}) \rVert]$$ 로 나타냅니다. 

이는 전체 파라미터의 loss preserving importance score를 포함하는 matrix가 되는 것입니다.

  


**2. Topological Structure Preserving**

그래프 데이터에서는 topological 정보도 중요하기 때문에, topological 정보를 보존하는 모듈도 고려합니다. 

위에서 보여드린 `GAT`식에서, $$l-th$$ layer에서의 node $$v_i$$와 $$v_j$$ 사이의 attention coefficient $$e_{ij}^{(l)}$$를 다음과 같이 matrix 형태로 쓸 수 있습니다:

$$e_{ij}^{(l)}=a(H_{i,j}^{(l-1)};W^{(l)})$$

여기서 $$H_{i,j}^{(l-1)}$$ 는 $$(l-1)-th$$ layer에서의 node $$v_i$$와 $$v_j$$의 embedding feature를 포함하고 있습니다. 

앞서 보여드린 **Minimizing Loss Preserving**과 비슷하게, Parameter가 아주 조금($$\Delta W = \lbrace \Delta w_m \rbrace$$) 변할 때, $$e_{ij}^{(l)}$$의 변화량은 다음과 같이 나타낼 수 있습니다.

$$a(H_{i,j}^{(l-1)};W^{(l)}+ \Delta W^{(l)})-a(H_{i,j}^{(l-1)};W^{(l)}) \approx \sum_{m} g_{m}(H_{i,j}^{(l-1)}) \Delta w_m $$

마찬가지로, $$g_{m}(H_{(i,j)}^{(l-1)})$$는 파라미터 $$w_m$$의 attention coefficient에 대한 gradient입니다. 

Training set의 모든 node에 대한 topological loss $$g_{m}(H^{(l-1)})$$는 $$e_i^{(l)}$$의 $$l_2$$ squared norm의 gradient으로 계산합니다. 

Parameter $$w_{m}$$의 중요도는 $$g_{m}$$의 크기로 나타내며, $$\mathcal{T_{k}}$$에서의 전체 파라미터 $$W$$의 중요도는 $$I_{k}^{ts} = [\lVert g_m(H_{k}^{(l-1)}) \rVert]$$ 로 나타냅니다. 

이는 전체 파라미터의 topology preserving importance score를 포함하는 matrix가 되는 것입니다.

최종적으로 본 논문에서 사용하는 $$W$$의 importance score는 위의 두 score를 합친 $$I_k = \lambda_{l} I_k^{loss} + \lambda_{t} I_k^{ts}$$ 가 됩니다.

여기서 $$\lambda_{l}$$ 와 $$\lambda_{t}$$는 hyperparameter로, 어떤 score를 중점적으로 고려할지 사용자가 정할 수 있습니다.


  

> ### `Continual Learning` on `GNN`

위에서 구한 importance를 가지고, `Continual learning`에 어떻게 적용시킬 수 있을 지 보겠습니다. 

새로운 task $$\mathcal{T_{k+1}}$$을 학습할 때, 새로운 task의 performance를 올림과 동시에 과거의 task들을 기억해야합니다(중요한 파라미터를 최대한 변하지 않도록 유지시키면서요!)

이는 다음과 같은 loss function을 정의해서 이루어집니다.

$$\mathcal{L_{k+1}^{'}}(W) = \mathcal{L_{k+1}^{new}}(W) + \sum_{n=1}^k I_n \otimes (W-W_n^{(*)})^2$$

이 때, $$\otimes$$는 element-wise multiplication입니다. 

$$\mathcal{L_{k+1}^{new}}(W)$$는 새로운 task의 loss function이고, $$I_n$$은 old task의 parameter importance matrix입니다. $$W_n^{(*)}$$는 $$\mathcal{T_n}$$의 optimal parameter입니다. 

위 loss function을 해석해보면, 이전 task들에서 importance score가 높았던 parameter가 새로운 task에서 많이 바뀌게 될 경우 penalty를 받는 형식입니다. 

이를 통해 모델은 이전 task들에서 importance score가 높았던 parameter를 최대한 보존하면서 새로운 task를 학습하게 됩니다.

  
  

> ### Promoting Minimized Importance Scores

더 나아가, 모델의 capacity는 한정되어 있으므로, 위에서 구했던 loss function에 importance score의 $$l_{1}$$ norm을 추가시켜서 다음과 같은 최종 loss function을 얻습니다.

$$\mathcal{L_{k+1}}(W) = \mathcal{L_{k+1}^{'}}(W)+ \beta \lVert I_{k+1} \rVert_1$$

이 식을 해석해보면, loss function을 구할 때 importance score 도 어느정도 규제를 해서 importance score가 과도하게 높아지는 것을 방지하는 것입니다. 현재 task에서 parameter들의 importance score가 과도하게 높아지면, 다음 task에서는 optimize 할 수 있는(중요도가 낮은) parameter들이 적어지게 되기 때문입니다.

`Continual learning`에서 task는 지속적으로 들어오는 것을 감안했을 때 합리적인 regularization입니다. 

$$\beta$$가 높아지면 미래의 task를 위해 더 많은 learning capacity를 보존하겠다는 의미가 됩니다.

  

> ### Extension to General GNNs

지금까지는 `GAT`에 대해서만 `TWP`모듈을 적용했지만, 저자들은 다른 `GNN`모델들에 대해서도 쉽게 적용이 가능하다고 합니다. 

여기서는 topological structure를 다음과 같이 정의합니다.

$$e_{ij}^{(l)}=(h_i^{(l-1)}W^{(l)})^{T}tanh(h_j^{(l-1)}W^{(l)})$$

이를 통해서 attention weights가 node $$v_i$$와 $$v_j$$ 사이의 거리에 dependent하게 구해질 수 있습니다. 

위 식을 통해서 `TWP` 모듈을 구성하면, `GAT`뿐 아니라 임의의 `GNN`모델에 이 방법론을 적용할 수 있습니다.

  

지금까지의 설명을 바탕으로, 이 논문에서 제시한 방법론의 전체적인 개요는 다음 그림과 같습니다.


![Overview of the proposed method](https://user-images.githubusercontent.com/99710438/170720633-9cf611e6-fc8b-47ff-a46b-c268ebf7fb96.png)


  

  


## **4. Experiment**

  

> 본 논문에서 저자들은 다양한 baseline과 실험을 통해 제시한 방법론의 성능을 평가했습니다. 

`GNN`모델을 위해 만들어진 `Continual learning` 방법론이 없으므로, `CNN`을 위해 만들어진 모델들을 그래프 도메인에 적용시켜 비교했습니다. `GNN`, `GAT`, `GIN`에 여러 baseline들을 접목시켜 성능을 평가했습니다.

### **Experiment setup**

* Dataset
  * Corafull (node classification)
  * Amazon Computers (node classification)
  * Protein-Protein interaction(PPI) (node classification)
  * Reddit posts (node classification)
  * Tox21 (graph classification)
* baseline (`GAT`, `GCN`, `GIN`과 합쳐서 쓴 module들)
  * Fine-tune (Girshick et al., 2014)
  * LWF (Li and Hoiem, 2017)
  * EWC (Kirkpatrick et al., 2017)
  * MAS (Aljundi et al., 2018)
  * GEN (Lopez-Paz and Ranzato, 2017)
  * Joint train (Caruana, 1997)
  * **TWP** (Proposed)
* Evaluation Metric
  * Average performance (AP)
  * Average forgetting (AF)
  * micro F1
  * AUC score
* Other setup
  * Optimizer: Adam SGD
  * Initial learning rate: 0.005
  * Epochs: 200, 300, 400, 30, 100, respectively.
  * $$\lambda_{t}$$: 10,000
  * $$\lambda_{l}$$: 100 or 10,000 for different datasets
  * $$\beta$$: 0.1 or 0.01 for different datasets
  
  

### **Result**

> ### Performance

* Node Classification

전체적인 Node classification의 performance는 아래 table과 같습니다. 

빨간색은 best, 파란색은 second best performance를 나타내고, 위쪽 화살표는 높을수록 좋은 지표, 아래쪽 화살표는 낮을수록 좋은 지표를 의미합니다.

![Node classification performance](https://user-images.githubusercontent.com/99710438/170866564-6d37e4df-e480-407c-8daf-4d55771ef9fe.png)


Table에서 확인할 수 있듯이 저자들이 제시한 방법론은 모든 `GNN`모델, 모든 데이터셋에 대해서 best or second best performance를 보였습니다. 

  

![Evolution of performance](https://user-images.githubusercontent.com/99710438/170867125-09fe7247-e12d-4397-b8e7-465d3c341b98.png)


위 그림은 Corafull dataset에서 `GAT`를 base model로 했을 때 9개의 task동안의 training curve를 그린 것입니다. 저자들은 자신들의 모델이 topological information까지 고려하기 때문에 task가 진행되더라도 크게 이전 task의 성능이 떨어지지 않는 것이라고 주장했습니다. 

  

![Perofrmance of the first task](https://user-images.githubusercontent.com/99710438/170867937-6c8bad1f-58ca-4bfb-800f-d1ac769e637b.png)


![Average Performance](https://user-images.githubusercontent.com/99710438/170868107-024d2d65-80d9-4bd2-a262-e07208df46b2.png)

  

좀 더 자세히 보자면, 위 그림 중 첫 번째 그림은 첫 번째 task의 성능이 task가 진행됨에 따라 변화하는 모습을 보인 그림이고, 두 번째 그림은 평균적인 performance를 나타낸 그림입니다. (a)부터 (d)까지는 각각 Corafull, Amazon Computers, PPI, Reddit 데이터셋입니다.

`Joint train` 방법을 제외한다면 저자들의 방법론이 제일 적은 **Catastrophic forgetting** 현상을 보이고 있는 것을 확인할 수 있습니다. 

이 `Joint train` 방법은 task가 계속 추가됨에 따라 과거의 data까지 **전부 다** 포함하여 학습하는 방법입니다.

하지만 Computation & Memory cost 때문에 모든 data를 계속 저장하고 학습하는 것은 현실성이 떨어지므로, `Continual learning` 성능의 upper bound라고 생각하시면 되겠습니다.

이를 고려했을 때, 저자들의 방법론이 task가 진행됨에 따라 이전 task를 가장 잘 기억한다는(**Catastrophic forgetting**이 적다는) 것을 확인할 수 있습니다.

  


* Graph Classification
저자들의 방법론은 graph classification task에서도 좋은 성능을 보였습니다.

![Graph classification performance](https://user-images.githubusercontent.com/99710438/170868529-08576d78-b8ec-4ea4-bb2f-b68684278cc9.png)


위 테이블에서 확인할 수 있듯이, 저자들의 방법이 Graph classification task에서도 가장 좋은 성능을 내고 있습니다.

  

![Performances across all tasks](https://user-images.githubusercontent.com/99710438/170868472-3bf928c9-9146-45d6-a7e9-0b4f7e78256f.png)


Node classification과 마찬가지로 task가 진행됨에 따라 성능이 어떻게 변화하는지 위 그림을 통해 확인할 수 있습니다.

왼쪽 그림은 first task의 performance 변화, 오른쪽 그림은 평균 performance의 변화입니다.

  

* Ablation Study
앞서 설명드린대로 저자들의 방법론은 두 가지 모듈(Minimized Loss Preserving, Topological Structure Preserving)을 사용했는데요, 이 각각의 모듈이 과연 모델에 도움을 주는지 확인하기 위해 ablation study도 진행했습니다.

  

![Ablation study](https://user-images.githubusercontent.com/99710438/170868926-5c86f974-2001-4ed2-820d-d368b746efb7.png)


위 표에서 W/Loss는 Minimized Loss Preserving 모듈을 빼고, W/TWP는 Topological Structure Preserving 모듈을 빼고 학습을 진행한 결과를 나타내고, Full은 두 모듈 다 사용한 모델입니다. 

Corafull과 Amazon Computers dataset에서 실험을 한 결과, Full이 나머지 두 경우보다 성능이 좋은 것으로 말미암아 두 모듈 다 모델의 성능을 높이는데 기여를 한 것으로 볼 수 있습니다.

  

  

## **5. Conclusion**

> **Summary**

본 논문에서는 Graph domain에 적용될 수 있는 `Continual learning` 방법론을 제시했습니다.

파라미터가 이전 task의 성능에 미치는 중요도와 graph data에서 필요한 topological information에 미치는 중요도를 동시에 고려해, 중요한 파라미터는 최대한 보존하고 덜 중요한 파라미터가 새로운 task를 학습하는데 사용되도록 모델을 구성했습니다.

이 방법론은 어떠한 `GNN`모델과도 결합되어 사용할 수 있습니다.

또한 이 모델을 기존의 `CNN`에 적용될 수 있는 `Continual learning`방법론들과 비교했습니다. `GAT`, `GCN`, `GIN`과 결합하여 실험했을 때, 다양한 dataset에서 좋은 성능을 내는 것을 확인했습니다.

더 나아가 ablation study를 통해 각 모듈이 중요하다는 것도 증명했습니다.

  

> **내 생각...**

`Continual learning`은 domain을 막론하고 연구되어야 할 주제라고 생각합니다. 새로운 data는 항상 생겨나며 이전에 학습된 정보를 유지하는 것이 중요하기 때문입니다.

인간이 새로운 지식을 학습하지만 이전에 학습했던 지식을 잊지 않는 것 처럼, 인공지능이 나아가야 할 궁극적인 방향이라고 생각합니다.

하지만 본 방법론은 한정된 parameter(weight)들로 학습하는데 이것이 가장 큰 한계라고 생각합니다.

중요한 weight를 최대한 보존하면서 덜 중요한 weight로 학습을 진행하는 것은, task가 많아지면 많아질수록 의미없는 방식이 될 것 같습니다.(언젠가는 이전에 중요하다고 판단되었던 weight들도 전부 수정이 될 것이기에..)

`Continual learning`의 최종적인 목표는 무한히 들어오는 새 data를 학습하는 것이라고 생각하며, 이러한 setting을 염두에 둔 연구가 필요할 것 같습니다. 

***

  

  

## **Author Information**

* Wonjoong Kim
  * Affiliation: [DSAIL@KAIST](http://dsail.kaist.ac.kr)
  * Research Topic: GNN, NeuralODE, Active learning, Continual learning
  * Contact: wjkim@kaist.ac.kr

  

  


## **6. Reference & Additional materials**

* Github Implementation
  * https://github.com/hhliu79/TWP
* Reference
  * [Overcoming Catastrophic Forgetting in Graph Neural Networks with Experience Replay](https://arxiv.org/abs/2003.09908)
  * [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
