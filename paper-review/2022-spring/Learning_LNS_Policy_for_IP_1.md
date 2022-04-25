---
description : Wu, Yaoxin, et al. /Learning Large Neighborhood Search Policy for Integer Programming./NeurIPS-2021  
---

# Learning Large Neighborhood Search Policy for Integer Programming 



<br>

# **1. Problem Definition** 

조합최적화 문제는 NP-hardness로 정확한 해를 효율적으로 찾아내기가 힘들다는 특징을 가지고 있습니다. 따라서 다양한 휴리스틱 방식을 통해 풀어내는데, 여기서 휴리스틱 방식은 비슷한 구조가 반복되면서 해를 찾아나가는 특성을 가지고 있다는 것이 이러한 방식으로 자동으로 학습할 수 있는 머신러닝의 도입 계기가 됩니다. 머신러닝 기법을 이용하여 풀어내는 휴리스틱 방식을 크게 Constructive heuristics, improvement heuristics로 나뉘는데 본 논문에서 개선하고자하는 LNS방식은 풀이는 improvement heuristics 방식에 속하게 됩니다. initial solution을 기반으로 해를 조금씩 바꿔가며 구하는 improvement heuristics의 LNS 기법을 조합최적화 문제 뿐만이 아닌 보다 general한 IP문제를 풀기 위해 bipartite graph, Action factorization등의 방법을 이용하여 제한 시간동안 좋은 해를 찾는 방법을 제안합니다.


<br>

# 2. Motivation

LNS방식은 현재의 해의 일부를 파괴하고(destroy) 다시 그 부분을 더 좋은 방식으로 고치며(repair) 기존의 해를 개선하는 방법입니다. 이 중 해를 파괴하는 방식은 어떤 해를 파괴할지 고르는 정책이 필요한데, 변수의 수가 많아질수록 해를 선택할 집합은 exponentially하게 증가합니다. 따라서 파괴할 해를 몇개를 고를지에 대한 제약을 둠으로써 해결하고자 하였습니다.  본 논문에서는 destroy하는 정책을 보다 flexible하게 만들기 위해 Action factorization을 이용하였고, 이는 선택되는 해의 모든 경우의 수(![](https://latex.codecogs.com/svg.image?2^n)를 고려할 수 있도록 하였습니다.

<br>

# 3. METHOD

LNS Framework은 MDP Formulation, large scale action space에 대한 factorized representation, policy parametrize, policy network을 학습하는 actor-critic이렇게 크게 4개로 나뉘게 됩니다.

## MDP formulation

앞서 설명드렸던 것 처럼 이 논문에서는 일반적인 IP문제에 LNS를 적용하게 됩니다. 각 스텝마다 destroy하여 reoptimized할 변수들의 set을 선택하는 정책을 학습하기 위해  discrete한 sequential decision 문제를 MDP로 formulation 해야합니다.
<a href='https://ifh.cc/v-9yqN8w' target='_blank'><img src='https://ifh.cc/g/9yqN8w.png' border='0'></a>

**State** : IP 문제를 위 그림과 같이 formulation하였을 때 variable(x), matrix(A), constraint(C)와 같은 변하지 않는 정보들(static features)과 현재 해![](https://latex.codecogs.com/svg.image?x_t)현재까지 가장 좋았던 해![](https://latex.codecogs.com/svg.image?x^*)와 같은 변하는 정보들(dynamic features)에 대해 반영하고 있습니다.

**Action** : destroy하여 reoptimize를 할 variable들을 선택하는 것입니다.

**Transition** : repair operator(IP solver)가 action을 통해 선택된 variable들을 다시 reoptimize하면서 dynamic features들이 업데이트되면서 ![](https://latex.codecogs.com/svg.image?s_t&space;\to&space;&space;s_{t&plus;1})로 transition이 일어나게 됩니다. 이를 식으로 나타내면 다음과 같습니다.
$$
x_{t+1} = argmin_x\{\mu^Tx|Ax\le b;x\ge0;x\in {Z^n};x^i= x_t^i, \forall x^i \notin a^t\}
$$
(action에 의해 선택된 variable ![](https://latex.codecogs.com/svg.image?x^i)들만 새롭게 optimization문제를 풀고 나머지는 고정)

**Rewards** : ![](https://latex.codecogs.com/svg.image?r_t(s_t, a_t) = \mu^T(x_t - x_{t+1})) objection value인 ![](https://latex.codecogs.com/svg.image?\mu^Tx)가 얼마나 변했는지에 대한 것입니다.

**Policy** : GNN기반의 policy network를 가지고, ![](https://latex.codecogs.com/svg.image?\pi(a_t|s_t))는 ![](https://latex.codecogs.com/svg.image?s_t)가 given 되었을 때 선택될 variable들의 subset들의 conditional probability distribution을 의미합니다. (ie. 변수 ![](https://latex.codecogs.com/svg.image?x_1,x_2))가 있을 때, 주어진 state ![](https://latex.codecogs.com/svg.image?s_t)에서 ![](https://latex.codecogs.com/svg.image?(x_1), (x_2) ,(x_1, x_2))가 각각 destroy variable로 선택될 확률을 나타냄)

<br>

## Action factorization

Variable 수가 linear하게 늘어날 때 variable을 선택하는 action space는 exponentially하게 늘어납니다. RL알고리즘을 적용하기 위해서는 어마하게 큰 action space를 exploration하고 모든 action을 representation 해야되는 이슈가 발생을 하게 되는데 논문에서는 action factorization를 이용하여 보다 효율적으로 해결할 수 있다고 합니다. 이는 전체 variable n개에서 destroy variable을 선택할 확률을 각 variable $x_i$를 선택할지 말지에 대한 확률 n개의 곱으로 나타낸다는 것입니다.
$$
\pi(a_t|s_t) = \prod_{i=1}^n \pi^i(a_t^i|s_t)
$$
이는 실제 ![](https://latex.codecogs.com/svg.image?2^n)의 action space를 탐색해야되는 문제를 n개의 policy를 통하여 n개의 action space를 탐색하는 문제로 바꾸어 large scale 문제에서도 효율적으로 exponential하게 증가하는 action space를 탐색하도록 합니다.

<br>

## Policy parametrization


Policy network는 GNN 기반으로 모든 variable들이 같은 parameter들을 공유하고 있습니다.

<a href='https://ifh.cc/v-6HnafS' target='_blank'><img src='https://ifh.cc/g/6HnafS.png' border='0'></a>

위 그림의 예시를 보면, bipartite graph로 state를 나타낸 것을 볼 수 있습니다.

여기서 biparite graph ![](https://latex.codecogs.com/svg.image?\mathcal{G} = (\mathcal{V, C},A))로 구성되어있는데 ![](https://latex.codecogs.com/svg.image?\mathcal{V})는 변수들의 갯수만큼의 노드를 의미하고  각 노드들은 ![](https://latex.codecogs.com/svg.image?d_v)차원의 노드 features를 가지고 있습니다. ![](https://latex.codecogs.com/svg.image?\mathcal{C})는 제약식의 갯수만큼의 노드를 의미하고 각 노드들은 ![](https://latex.codecogs.com/svg.image?d_c)차원의 노드 features를 가지고 있습니다. 마지막으로 A는 노드 ![](https://latex.codecogs.com/svg.image?v_i)와 ![](https://latex.codecogs.com/svg.image?c_j)간의 weight를 나타내는 edge로 실제 IP문제에서 incidence matrix역할을 합니다. 즉 위의 예시에서는 4개의 변수와 3개의 제약식이 있으므로 ![](https://latex.codecogs.com/svg.image?A \in R^{4 \times 3}) 가 됩니다.

각 노드들의 embedding은 Graph Convolutional Network 구조를 이용하는데
$$
C^{(k+1)} = C^{(k)} + \sigma(LN(AV^{(k)}W_v^{(k)})
$$

$$
V^{(k+1)} = V^{(k)} + \sigma(LN(A^TC^{(k)}W_c^{(k)}), k = 0,...,
$$



다음과 같이 각 variable node와 constaint node들이 업데이트 됩니다. 

여기서![](https://latex.codecogs.com/svg.image?V^{(k)}, C^{(k)}) 는 각각 k번째 layer에서의 variable node들![](https://latex.codecogs.com/svg.image?([v_1^{(k)} \cdots v_n^{(k)}]) constraint node들![](https://latex.codecogs.com/svg.image?([c_1^{(k)} \cdots c_n^{(k)}])를 의미하고, LN, ![](https://latex.codecogs.com/svg.image?(\sigma(\cdot))는 layer normalization, tanh함수를 의미합니다.

이렇게 K iteration을 통해 최종 업데이트된 ![](https://latex.codecogs.com/svg.image?(V^{(k)}))에 해당되는 각 변수에 해당되는 노드들 ![](https://latex.codecogs.com/svg.image?([v_1^{(k)} \cdots v_n^{(k)}]])은 마지막에 MLP layer와 sigmoid 활성함수를 통과하여 ![](https://latex.codecogs.com/svg.image?\pi^i(a_t^i|s_t)) 즉 i번째 변수가 선택될 확률이라는 policy와 같은 의미를 갖게 됩니다. 따라서 모든 변수들에 대해 ![](https://latex.codecogs.com/svg.image?MLP(v_i))를 통과시킨 뒤 베르누이 샘플링을 통하여 destroy할 subset들을 구성하게 됩니다.



<br>

## Training algorithm

**actor-critic**

Training단계에서는 Q-actor-critic을 이용하여 policy 및 Q-value를 학습합니다. Actor와 Critic에 대한 식을 나타내보면 다음과 같습니다.
$$
Actor : L(\theta) = E_D[(Q_w(s_{t}, a_{t}) log\pi_\theta(a_t|s_t)]
$$

$$
Critic : L(w) = E_D[(\gamma Q_w(s_{t+1}, a_{t+1}) + r_t - Q_w(s_t, a_t))^2]
$$





만약 문제의 사이즈가 커진다면 action space가 매우 커질 것이고, 이는 즉 Q-network가 매우 sparse할 수 있으므로 Q-network를 바로 학습시키기에는 적절하지 않을 수 있습니다. 논문에서는 actor의 식에서의 ![](https://latex.codecogs.com/svg.image?log\pi_\theta(a_t|s_t))를 위에서 Action factorization에서 언급한대로

![](https://latex.codecogs.com/svg.image?\pi(a_t|s_t) = \prod_{i=1}^n \pi^i(a_t^i|s_t))를 이용하여 다음과 같이 나타냅니다.
$$
수정된 Actor_loss : L(\theta) = E_D[(Q_w(s_{t}, a_{t}) \sum_{i=1}^n log\pi_\theta(a_t|s_t)]
$$
**clipping & masking**

마지막 테크닉으로 보다 넓은 범위의 action space를 exploration하기 위해서 각 variable이 선택될 확률을 ![](https://latex.codecogs.com/svg.image?[\epsilon, 1-\epsilon], \epsilon < 0.5)과 같이 clipping을 합니다. 예를 들어 ![](https://latex.codecogs.com/svg.image?\epsilon = 0.2)라고 하면 모든 variable들의 선택될 확률은 [0.2, 0.8] 범위를 벗어날 수 없는 것입니다.. 이는 매우 높거나 낮게 선택되는 variable들로 인해 학습이 편향되지 않도록 합니다. 또한 sub-IP문제로 적당하지않은 모든 variable들이 선택되는경우나 아닌경우에 대해서는 masking을 통해 그러한 경우를 방지했습니다.

## **4. Experiment**  Dataset

<a href='https://ifh.cc/v-HMYPvr' target='_blank'><img src='https://ifh.cc/g/HMYPvr.png' border='0'></a> 

실험은 4개의 NP-Hard문제에 대한 dataset을 생성하여 진행되었습니다. 위의 표를 보면 각 생성한 dataset들의 Training에 해당하는 부분의 변수와 제약식의 수, 원래 training한 문제보다 더 큰 사이즈에 적용하기 위한 dataset들로 구성되어있습니다.

1. Set Covering(SC)

    특정 Set(=집합)이 존재한다고 가정했을 때, Sub sets(=부분 집합)들이 합쳐져 특정 set을 나타낼 수 있는지에 대한 문제입니다. 1000개의 column(변수)과 5000개의 row(제약식)을 생성하였습니다.

2. Maximal Independent Set(MIS)

    graph에서 서로 인접하지 않은 vertex의 집합을 Independent Set이라 부르는데 이 Independent Set들중 가장 vertex의 수가 많은 것을 찾는 문제입니다. Erdos-Rényi random graphs를 이용하여1500개의 column(변수)와 affinity number를 4로 설정하여 생성하였습니다.

3. Combinatoraial Auction(CA)

    구매자들이 원하는 상품들을 조합해서  판매자의 이익을 최대로 하는 구매하는 문제입니다.

    입찰자 4000명 상품을 2000개를 arbitary한 관계를 가지도록 생성하였습니다.

4. Maximum Cut(MC)

    부분 집합 로부터 그 여집합 까지를 횡단하는 변의 무게 의 총합이 최대가 되는 그래프의 정점 의 부분 집합 를 구하는 것입니다. 그래프는 arabasi-Albert random graph models로부터 생성하였고 평균 degree는 4, graph 노드 수는 500개로하였습니다.

### Baseline

- SCIP : IP문제를 푸는 solver로 모든 LNS기법 후 선택된 해들을 re-optimize할 solver이다
- U-LNS : re-optimize할 변수들을 uniform하게 선택하는 LNS 방식이다.
- R-LNS : 선택될 변수의 갯수를 fix하고 전체 변수들 중 그 수만큼 random하게 선택하는 LNS방식이다. (2-5사이의 변수를 선택하였고 그 중 성능이 가장 좋은 갯수를 선정하였다.)
- FT-LNS : R-LNS방식을 통해 가장 좋은 trajectory들을 모아 그 선택방식을 지도학습으로 학습하는 LNS방식이다.

### Performance metric

- objective value

  

  ### Result

<a href='https://ifh.cc/v-9h8fmT' target='_blank'><img src='https://ifh.cc/g/9h8fmT.png' border='0'></a> 

위의 표는 각 데이터셋들에 대한 퍼포먼스를 비교한것이고 각각 50번의 테스트를 통한 objective value값의 평균과 분산을 나타낸 것입니다. ![](https://latex.codecogs.com/svg.image?SCIP^*, SCIP^{**})는 각각 500초 1000초 동안 돌렸을때를 의미합니다. 나머지 방법들은 200초동안의 시간제한을 두고 평가를 하였는데, 본 논문의 방법이 모든 방법론들에 비해 좋은 성능을 보이는 것을 확인할 수 있습니다.

<a href='https://ifh.cc/v-yODhDx' target='_blank'><img src='https://ifh.cc/g/yODhDx.jpg' border='0'></a> 

위 테이블 표는 각각 원래 training했던 사이즈보다 2배, 4배의 문제를 푼 것의 퍼포먼스를 보여줍니다. 첫번째 행![](https://latex.codecogs.com/svg.image?SC_2, MIS_2, CA_2, MC_2)이 2배사이즈의 문제 두번째 행![](https://latex.codecogs.com/svg.image?SC_4, MIS_4, CA_4, MC_4)이 4배 사이즈의 문제에 대한 각 모델의 퍼포먼스를 보여줍니다. 문제의 사이즈가 커질수록 본 논문에서 제시한 방법론의 퍼포먼스가 좋으며 전체사이즈의 문제를 계속해서 1000초간 푼 ![](https://latex.codecogs.com/svg.image?SCIP^{**})보다 성능이 좋은 것을 확인할 수 있습니다.

즉 문제의 사이즈가 커지면 커질수록 작은 문제로 쪼개서 푸는 LNS방식이 제한된 시간내에 더 효율적인 것을 확인할 수 있고, 본 논문에서 제시한 방법론이 현재까지 나온 LNS 방식 중 가장 좋은 솔루션을 제시한다는 것을 확인할 수 있습니다.

<br>

## **5. Conclusion** 

본 논문에서는 제안된 시간내에 IP문제를 빠르고 좋은 성능을 내도록 하는 LNS 기반의 policy를 학습하는 RL 방법에 대해서 제안하였습니다. actor factorization이 이 논문에서 가장 중요한 아이디어라고 생각되며 이를 통해 모든 변수의 subset($2^n$)들을 고려하여 보다 general한 destroy operator를 만들 수 있었습니다. 또한 GNN기반의 policy network을 통하여 각 변수들간의 parameter를 공유함으로써 각 변수들의 policy를 보다 효율적으로 학습하고 actor-critic에서 global한 Q-network를 선택하되, action-factorization으로 표현한 policy를 통하여 광범위한 action space를 학습할 수 있었던 것 같습니다. 또한 작은 사이즈 문제로 학습한 문제가 보다 큰 문제 사이즈에서도 잘 푸는것을 보여주어 적당히 작은 시간내에 IP문제를 풀 때 적용할 여지가 충분하다는 것을 보여주었습니다.


<br>

개인적인 의견

destroy-operator를 학습시킬 때 re-optimize할 변수들의 수를 미리 정해두고 학습하여 유연하지 않았던 부분들에 대해 actor factorization 방법을 통해 보다 유연하게 destroy할 변수들을 고를 수 있도록 한 부분이 가장 흥미로웠던 것 같습니다. 또한 graph구조의 policy가 작은 문제 사이즈에 대해 학습하고 어느정도 사이즈에 대해 robust하게 잘 작동할수 있게 하도록 도움을 준다고 생각이 들었습니다. 한가지 궁금했던 점은 만약 training size를 하나의 사이즈가 아니라 여러 사이즈로 학습한다면 그래프의 크기 변화에 대해서 학습을 하여 보다 더 generalization을 할 수 있을지에 대한 궁금점이 생기게 되었습니다.

* ## **Author Information**  

  * Yaoxin Wu  
      * Nanyang Technological University, Singapore  
      * deep reinforcement learning

## **6. Reference & Additional materials**    

* [GitHub - WXY1427/Learn-LNS-policy](https://github.com/WXY1427/Learn-LNS-policy) 

   
