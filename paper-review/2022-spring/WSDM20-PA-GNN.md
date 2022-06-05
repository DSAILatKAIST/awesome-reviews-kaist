---
description : 1st author / Transferring Robustness for Graph Neural Network Against Poisoning Attacks / WSDM-2020 
---

# **Title** 

Transferring Robustness for Graph Neural Network Against Poisoning Attacks

## **1. Problem Definition**  
* Graph Neural Networks (GNN) 은 Graph 를 사용하는 다양한 도메인에서 매우 좋은 성능을 보이고 있습니다.
* 하지만 GNN 은 다른 Deep Learning model 들과 마찬가지로 adversarial attack 에 취약합니다.
* 따라서 이 논문에서는 adversarial attack 하에서도 robust 한 예측을 할 수 있는, adversarially robust GNN 의 개발을 목적으로 합니다.



## **2. Motivation**  
### Graph Neural Networks (GNN)
* GNN 은 Graph 데이터를 사용할 수 있는 다양한 application 에서 매우 좋은 성능을 보이고 있습니다.
    * 예를 들면, social network, recommender systems, 분자구조 그래프 등
* GNN 이 이러한 데이터에서 좋은 성능을 보일 수 있는 이유는 node feature 뿐만 아니라 structure 의 정보도 같이 활용할 수 있는 **message-passing process** 에 있습니다.

### Adversarial attack 
* 한편, Deep learning-based model 들은 adversarial attack 에 취약하다는 것이 많은 연구들로부터 증명되어 왔고, 그에 따라 이러한 attack 을 막아내는 defense 방법 또한 많이 연구되고 있습니다.
* 아래 그림으로부터 우리는 눈에 보이지 않는, 인식할 수 없는 변화가 classifier 의 예측 값을 완전히 뒤바꾸어 버린다는 것을 알 수 있습니다.
* 즉, adversarial attack 은 의도적으로 생성된 imperceptible small perturbation 이고, 이러한 작은 변화가 Deep Learning model 의 성능을 매우 망가트리게 됩니다.
<p align='center'><img width="700" src="./yeonjunin2/fig1.png"></p> 

### Adversarial attack on graphs

* GNN 에서도 이러한 문제는 존재했습니다. 
* 특히 GNN 은 node feature 와 structure 정보를 모두 사용하므로, 1) graph 에 있는 edge 를 지우거나 더하는 structure attack, 2) node feature 에 noise 를 더해주는 feature attack, 두 가지 종류의 attack 에 모두 취약하다는 것이 많은 연구들에 의해 증명되었습니다.
* 그 중에서도 node feature attack 보다 structure attack 이 더 강력한 attack 효과를 보이며, attack 을 받은 node 는 classification accuracy 가 매우 하락하게 됩니다. 그래서 대부분의 graph adversarial attack 방법들은 structure attack 에 집중하고 있습니다.
* 아래 그림처럼, 7번 node 가 원래는 초록색 label 을 가지는데, adversarial structgure attack 을 받으니까 파란색 label 로 잘못 예측하게 되는 것을 알 수 있습니다.

<p align='center'><img width="700" src="https://d3i71xaburhd42.cloudfront.net/346c5e5d28c45ad3460277699bf8e244cd33c2cf/2-Figure1-1.png"></p> 

### Existing adversarial defense methods on graphs
<p align='center'><img width="700" src="./yeonjunin2/fig3.png"></p> 

* 이러한 adversarial attack 하에서도 robust 한 예측을 하기 위해서, 다양한 defense 방법들이 연구되고 있습니다.
* 크게 두 갈래로 나누어보자면 다음과 같이 나눌 수가 있습니다. 
    1. robust 한 GNN encoder 를 개발, 
    2. graph structure 를 purify 하는 graph structure learning 방법을 개발
* 먼저 robust 한 GNN encoder 의 궁극적인 목적은 message passing 과정에서 attacked edge 의 영향력을 최소화시키는 것입니다. 위 그림에서 굵은 검은 선은 많은 message 를 받게 되는 non-attacked edge 이고, 얇은 검은 선은 message 를 덜 받게 되는 attacked edge 입니다.
* 그리고 graph structure learning 의 궁극적인 목적은 주어진 graph structure 를 점진적으로 학습을 해내가는 것입니다. 결과적으로는 attacked edge 는 지우고 message passing 에 도움이 되는 edge 가 추가되게 됩니다. 마찬가지로 위 그림에서 빨간 선은 attacked edge 이고 structure 를 update 하여 이러한 edge 를 지워서 clean graph 를 만들게 됩니다



### Pitfalls of existing defense methods
* 위에서 설명한 defense 방법들을 한줄로 요약해보면 다음과 같습니다.
> negative 한 영향을 주는 neighbor 와 연결된 edge 를 제외시키자! 
* 이러한 방법들은 심각한 attack이 가해진 상황에서도 adversarial robustness 를 달성하며 좋은 성능을 보였습니다.
* 하지만 아주 근본적인 한계점이 존재합니다. 학습과정이 poisoned graph 에 의존한다는 것입니다.
* 이것이 왜 문제가 되냐면, poisoned graph 에서는 무엇이 진짜 perturbation 이고 아닌지를 명확히 알 수 없기 때문에, 여기서 어떤 neighbor 가 negative 한 영향을 주는 지를 명확히 알 수가 없습니다.
* 예를 들어, clean graph 에서는 A 라는 특성을 가진 어떤 node 가 A 라는 특성을 가진 node 들과 주로 연결이 되어 있는 패턴을 보입니다. 그렇다면 B 라는 특성을 가진 node 와 연결된 edge 가 있을 때 이를 제외하면 될 것입니다.
* 하지만 attacked graph 에서는 A 라는 특성을 가진 어떤 node 가 A 라는 특성을 가진 node 와도 연결이 되어 있고, B 라는 특성을 가진 node 와도 연결이 되어 있는 패턴을 보입니다. 이렇게 되면 모델은 헷갈립니다. 
> 어? A 라는 특성을 가진 node 는 A 특성을 가진 node 와 연결되어야 하나? 아니면 B 특성을 가진 node 와 연결되어야하나?
* 즉, sub-optimal 한 GNN encoder 혹은 graph structure 가 학습되는 것입니다.
 

### Key idea of the proposed method
* 그래서 이 논문에서 제안하는 아이디어는 다음과 같습니다.
> Clean graph 에서 주로 발생하는 패턴을 학습하고, 그 지식을 poisoned graph 에 transfer 해주자! 
* 하지만 이 방법 또한 문제가 있는데, 우리는 clean graph 를 알 수가 없다는 점입니다. 
* 그래서 저자는 비슷한 domain 또는 같은 domain 의 clean graph 로부터 패턴을 학습하고 이를 transfer 해주는 방법을 제안합니다.
    * 예를 들어, Yelp 와 Foursquare 데이터는 비슷한 co-review network 를 가지고 있고, Facebook 과 Twitter 는 둘 다 social network 로 비슷한 domain 을 공유하고 있습니다.

## **3. Method** 


### Proposed Method: PA-GNN
**Penalized Aggregation Mechanism**

* 앞서 우리는 structure 에 대한 adversarial attack 이 GNN 의 성능에 치명적이라고 언급했었습니다.
* 그러므로, message passing 을 할 때 perturbed edge 의 영향력을 줄일 수 있다면, true neighbor 에만 집중하여 aggregation 을 할 수 있을 것이고, 그러면 adversarial attack 으로부터 해방될 수 있습니다.
* 따라서, 우리는 node pair 들에 대한 normalized attention coefficient score 를 GAT-style 로 구할 것입니다.
$$
\alpha^l_{ij}=\frac{\text{exp}(a^l_{ij})}{\sum_{k\in \mathcal{N}_i}\text{exp}(a^l_{ik})}
$$
* $\alpha_{ij}^l$ 은 $l$ 번째 GNN layer 에서의 node $i, j$ 간의 normalized attention weight 이고, 이 값은 다음 식으로 계산할 수 있습니다
$$
a_{ij}^l= \text{LeakyReLU}((\mathbf{a}^l)^\top[\mathbf{W^l h^l_i} \bigoplus \mathbf{W^l h^l_j}])
$$

* $\mathbf{a}^l$ 와 $\mathbf{W^l}^l$ 은 parameter 이고, $\top$ 은 transpose, $\bigoplus$ 는 vector의 concatenation 입니다.
* $v_i$ 를 예로 들면, $a_{ij}^l$ 은 node $i$ 의 neighbors $j\in \mathcal{N}_i$ 에 대한 값들입니다.

$$
\mathbf{h}_i^{l+1}=\sigma( \sum_{j \in \mathcal{N}_j} \alpha_{ij}^l \mathbf{W}^l \mathbf{h}_j^l )
$$

* node embedding $\mathbf{h}_i^{l+1}$ 는 $l+1$ 번째 layer 에서의 node $i$ 의 embedding 이고, $l$ 번째 layer 에서의 attention weight 를 이용해 aggregation 된 값이 됩니다.
* 다시 요약해보면, 우리는 각 layer 에서 node pair 들의 attention weight 를 구할 수 있고, 그 weight 를 이용하여 node embedding 을 구할 수 있습니다.
* 우리의 목표는 perturbed edge 에게 attention weight 가 매우 작게 assign 되는 것입니다. 하지만 문제는 이미 poison 된 graph 에서는 어떤 edge 가 perturbed edge 인지 알 수 없다는 점입니다. 

> 여기서 이 논문의 아이디어가 진가를 발휘하기 시작합니다. 다음과 같은 방식으로 접근할 것입니다.
> 
> 1. 같은 또는 비슷한 domain 의 clean graph 를 가지고 있습니다.
> 
> 2. 우리는 그 clean graph 에 adversarial attack 을 가할 수 있습니다. 그렇게 되면 어떤 edge 가 attacked edge 인지 supervision 을 가지게 됩니다.
>
> 3. 그리곤 그 graph 에서 perturbed edge 에는 적은 attention weight 를 부여하고, true edge 에는 많은 attention weight 가 부여되도록 neural network 를 학습합니다
>
> 4. 그리고 그 과정에서 배운 지식을 우리가 target 하고 있는 poisoned graph 에 transfer 합니다.

* 먼저 어떻게 *perturbed edge 에는 적은 attention weight 를 부여하고, true edge 에는 많은 attention weight 가 부여되도록* 학습하는지에 대해 알아봅시다.
* 일반적으로 node classification 을 하는 GNN 을 학습할 때는 그냥 cross entropy loss 를 최소화하면서 학습합니다. 우리는 여기에 regularizer 를 하나 추가합니다
$$
\min_{\theta} \mathcal{L} = \min_{\theta} (\mathcal{L}_c + \lambda \mathcal{L}_{\text{dist}})
$$

$$
\mathcal{L}_{\text{dist}} = - \min (\eta, \mathbb{E}_{e_{ij} \in \mathcal{E}\setminus \mathcal{P}} a_{ij}^l - \mathbb{E}_{e_{ij} \in \mathcal{P}} a_{ij}^l)
$$

$$ 
\mathbb{E}_{e_{ij} \in \mathcal{E}\setminus \mathcal{P}} a_{ij}^l = \frac{1}{L|\mathcal{E}\setminus \mathcal{P}|}\sum_{l=1}^{L}\sum_{e_{ij} \in \mathcal{E}\setminus \mathcal{P}} a_{ij}^l
$$
$$ 
\mathbb{E}_{e_{ij} \in \mathcal{P}} a_{ij}^l = \frac{1}{L|\mathcal{P}|}\sum_{l=1}^{L}\sum_{e_{ij} \in \mathcal{E}\setminus \mathcal{P}} a_{ij}^l
$$
* 이 regularizer 의 역할을 한줄로 요약하자면, perturbed edge 의 attention weight 의 평균은 작게, true edge 의 attention weight 는 크게 만드는 것입니다.
* 위의 1, 2, 3 번까지 진행했습니다. 3번까지 진행함으로써, 우리는 true edge 와 perturbed edge 의 분포에는 어떤 차이가 있는지 학습하면서 둘을 구별해낼 수 있는 GNN 을 학습하게 되었습니다
* 이제 이 GNN 학습한 지식을 transfer 해주는 4번 단계로 넘어가봅시다.

**Transfer with Meta-Optimization**

* 이 논문에서는 transfer 해주는 방법으로 [ICML 17] Model-Agnostic Meta Learning for Fast Adaptation of Deep Networks (MAML) 의 방법을 사용했습니다.
* 이해를 돕기위해 MAML 의 간단한 리뷰 후 방법론으로 넘어가겠습니다.

---
*Background: Model-Agnostic Meta Learning for Fast Adaptation of Deep Networks (MAML)*

<p align='center'><img width="700" src="./yeonjunin2/fig6.png"></p> 

* Meta learning 이란?
    * 우리는 굉장히 적은 경험을 통해서 어떤 기술을 쉽게 습득하기도 하고, 굉장히 적은 사진만 보고도 그 사진이 무엇인지를 쉽게 분류해낼 수 있습니다
    * 예를 들면, 우리는 고작 몇번 정도 연습을 통해 자전거 타는 법을 배우기도 하고, 피시방에서 친구가 하는 새로운 게임을 보기만 해도 바로 그 게임을 같이 즐길 수 있습니다.
    * 또한 우리는 치타와 표범을 제대로 본적도 공부한적도 없지만, 그냥 딱 보면 둘은 다르게 생겼다는 것을 바로 알 수 있습니다. 위 그림에서 치타와 표범의 무늬가 미세하게 다르고 우리는 이러한 시각적 특징을 잡아낼 수 있는 방법을 알고 있습니다.
    * 하지만 컴퓨터는 스타크래프트 1을 할 수 있도록 수 많은 게임 데이터로 학습을 시켜놓아도, 스타크래프트 2 하는 법을 가르치지 않으면 사람과 달리 게임을 엉망으로 하게 될 것이며,
    * 치타와 표범을 구분할 수 있으려면, 학습과정에서 치타와 표범을 매우 많이 봐야만 구분할 수가 있습니다. 하지만 이 세상에는 무수히 많은 존재들이 있고, 새로운 것들이 쏟아져 나오고 있습니다. 그 모든 것을 다 데이터로 수집하고 학습한다는 것은 불가능에 가깝습니다. 
    * 그러므로 우리는 컴퓨터가 인간처럼 사고하여 이미 알고 있는 지식을 활용하여 매우 적은 샘플, 경험만 주어져도 금방 그 특징을 학습하고 적응하기를 원합니다. 
    * 즉, 인공지능은 이미 가지고 있는 prior experience 와 매우 적은 양의 새로운 정보를 잘 결합하여 새로운 정보와 task 에 빠르게 대응할 수 있어야하고, 그러면서도 새로운 정보에 overfitting 되지 않아서 prior experience 도 잃지 않아야합니다.
    * 이를 위해 meta learning 이 제안되었습니다.

* MAML 은 제목 그대로 어떤 model architecture 에도 사용할 수 있는 범용적인 meta learning 방법론입니다
* Key idea 는 많은 task 에 적합한 "internal 한 representation" 을 학습해놓으면 아주 적은 gradient step 만으로도 다양한 task 에 금방 fine tuning 이 가능하다는 것입니다. 아래 그림으로 좀 더 자세히 설명해보자면, 
    * (a) 우리가 다른 task 는 신경쓰지 않고 task 1에 대한 loss 가 작아지는 방향으로만 모델을 학습해놨다고 합시다. 그럼 task 2나 task 3 에 대해 fine tuning 을 할 때, 굉장히 많은 training 이 필요할 것입니다. task 2와 3의 loss 가 작아지는 방향은 task 1의 방향과 매우 다르니 말입니다.
    * (b) 그렇다면, task 1,2,3 의 loss 가 공통적으로 작아질 수 있는 방향으로 모델을 학습하면 모든 task 에 대해 적은 training 만으로도 빠르게 fine tuning 이 가능할 것입니다
<p align='center'><img width="700" src="./yeonjunin2/fig7.png"></p> 

* 그래서 MAML 은 모든 task 에 대한 loss 를 작게 만들 수 있는 방향을 찾아 parameter 를 update 합니다. 자세한 procedure 는 다음과 같습니다
<p align='center'><img width="700" src="./yeonjunin2/fig5.png"></p> 

---

* 이제 다시 PA-GNN 으로 돌아와서 다음과 같은 방식으로 MAML 을 적용합니다
* M 개의 같은 domain 의 clean graph 와 perturbed edge 가 있고, 이에 대해 specific 한 task (여기서는 모두 node classification task)가 있습니다. 이에 대한 loss 를 $\mathcal{L}_{\mathcal{T}_i}$ 라 정의하고 Penalized Aggregation section 에서 정의한 loss function 을 사용합니다.
* 각 task 에 대해서 support node 를 사용하여 loss $\mathcal{L}_{\mathcal{T}_i}$ 를 구하고, single gradient descent updated parameter $\theta'_i$ 를 구합니다. 이 과정에서 parameter $\theta$ 는 아직 update 되지 않습니다.
* 위 과정을 모든 task 에 대해 거치고 나서, query node 에 대해서 $\theta'_i$ 를 활용하여 loss 를 구하고 모든 task 에 대한 loss 의 합이 줄어드는 방향으로 parameter $\theta$ 를 update 합니다. 
* 이렇게 해서, 다양한 clean graph 로부터 perturbed edge 를 구별해내는 공통적인 "internal parameter" 를 학습하게 됩니다.
* 이렇게 meta model 의 학습이 완료되면, $\theta$ 를 가지고 우리가 target 하는 poisoned graph $\mathcal{G}$ 에 대해 cross entropy loss 로 fine tuning 합니다. meta model 은 이미 true edge 와 attacked edge 를 구별하면서 classification 을 잘할 수 있게 하는 지식을 가지고 있는 상태기 때문에 poisoned graph 에도 쉽게 fine tuning 할 수 있습니다.
* 자세한 알고리즘은 아래 그림을 참고하시면 됩니다.

<p align='center'><img width="700" src="./yeonjunin2/fig8.png"></p> 

## **4. Experiment**  


### **Experiment setup**  
* Dataset
    * Pubmed: 논문 citation network. non-overlapping subgraph 6개를 sampling 한 후, 1개는 poisoned graph (target), 5개는 clean graph (meta task) 로 setting 하여 실험 진행.
    * Reddit: reddit.com 사이트에서 글이랑 댓글 network. non-overlapping subgraph 6개를 sampling 한 후, 1개는 poisoned graph (target), 5개는 clean graph (meta task) 로 setting 하여 실험 진행.
    * Yelp-Small: 식당 리뷰 데이터. real world setting 으로 다른 지역들의 graph 를 clean 이라 생각하고, meta learning.
    * Yelp-Large: 식당 리뷰 데이터. real world setting 으로 다른 지역들의 graph 를 clean 이라 생각하고, meta learning.
* baseline  
    * GCN
    * GAT
    * PreProcess
    * VPN 
* Evaluation Metric  
    * Node classification accuracy

### **Result**  

<p align='center'><img width="700" src="./yeonjunin2/fig9.png"></p> 

* 전체적으로 attack 의 강도가 강해질수록 더 robust 한 성능을 보입니다.
* 이것으로 penalized aggregation도 잘 되었고, meta learning 도 잘 된것으로 생각할 수 있습니다.

<p align='center'><img width="700" src="./yeonjunin2/fig10.png"></p> 

* Penalizing regularizer 가 있을 때, attacked edge 에 attention weight 가 더 작게 할당되는 것을 알 수 있습니다.


## **5. Conclusion**  

* Graph adversarial attack 하에서 robustness 를 어떻게 달성할 수 있는지에 대한 논문이었습니다.
* 기존의 defense 논문들은 attacked edge 의 영향력을 줄여서 robust 한 GNN 또는 structure 를 만들려고 시도했었지만, 이미 poisoned 상태의 graph 에서는 한계가 있었습니다.
* 그래서 이 논문에서는 그 한계점을 극복하기 위해서, 비슷한 domain 의 다른 clean graph 를 이용하여 true edge 와 attacked edge 를 구분하는 방법을 학습하고 이를 poisoned graph 에 transfer 해주어 attacked edge 의 영향력을 더 잘 줄일 수 있도록하였습니다.
* 그 결과 다양한 데이터셋과 attack 강도에서 가장 robust 한 성능을 보였습니다. 또한 attacked edge 와 true edge 의 attention weight 를 비교했을 때, true edge 의 attention weight 가 전반적으로 더 크게 assign 되는 것을 확인할 수 있었습니다.
* 아쉬운 점
    * 만약 비슷한 domain 의 graph 가 이미 attacked 상태라면 잘 안되지 않을까 하는 생각이 들었습니다.
    * 그래서 그런 상황에서도 잘 된다는 것을 보여줄 수 있는 실험이 있었다면 좋았을 것 같습니다.
---  

## **Author Information**  

* Author name: Yeonjun In  
    * Affiliation: KAIST ISysE DSAIL  
    * Research Topic: GNN, Adversarial robustness, self-supervised learning

## **6. Reference & Additional materials**  
* Finn, C., Abbeel, P., & Levine, S. (2017, July). Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning (pp. 1126-1135). PMLR.
* https://github.com/dragen1860/MAML-Pytorch
* https://github.com/tangxianfeng/PA-GNN


