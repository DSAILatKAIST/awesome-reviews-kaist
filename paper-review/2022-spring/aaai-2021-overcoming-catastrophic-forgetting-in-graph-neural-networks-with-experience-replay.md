---
description : Fan Zhou / Overcoming Catastrophic Forgetting in Graph Neural Networks with Experience Replay / AAAI-2021  
---

# **Title** 

Overcoming Catastrophic Forgetting in Graph Neural Networks with Experience Replay

## **1. Problem Definition**  

Static한 graph setting에 맞춰져 있는 현재의 Graph Neural Networks (GNNs)는 현실의 상황과는 거리가 멀다.  
* Sequence of tasks에 continuously 적용될 수 있는 GNN을 고안하는 것이 본 논문의 주된 목적이다. 
* Continual Learning에서 발생하는 주된 문제인 catastrophic forgetting 문제도 보완한다.

## **2. Motivation**  


### 2.1 Continual Learning과 Catastrophic Forgetting

Graph Neural Networks (GNNs)은 많은 관심을 받고 있는 연구 분야이며, 눈에 띌만한 성장세를 보이고 있다.
현재까지의 GNN은 static한 graph setting에 초점이 맞춰져 개발되었다. 하지만 현실에서의 setting은 graph가 고정되어 있지 않고, 새로운 node와 edge 등이 끊임없이 추가된다. 이러한 상황에서 model은 정확성을 지속적으로 유지할 수 있어야 한다. 그렇다면 이러한 setting에서 새로운 task까지 잘 해내는 모델을 학습해야 한다면 어떻게 해야할까?

당연히 모델을 retraining 시켜야한다. 
모델을 retraining 시키기 위해 아래 두 가지 방법을 쉽게 떠올려 볼 수 있다.

첫째, 기존 데이터에 새로운 데이터까지 추가해서 모델을 처음부터 다시 학습하는 방법이다. 이 방법이 직관적일 수 있지만, 새로운 데이터가 수집될 때마다 전체 데이터셋에 대하여 모델의 모든 가중치값들을 학습하는 것은 시간과 computational cost 측면에서 큰 손실이다. 

그렇다면, 모델을 새로운 데이터로만 retraining 시키면 어떻게 될까? 이전에 학습했던 데이터와 유사한 데이터셋을 학습하더라도 아래의 그림처럼 이전의 데이터셋에 대한 정보를 잊어버리게 될 것이다. 이 문제를 일컬어 **Catastrophic Forgetting** 이라고 부른다.
> Catastrophic Forgetting : Single task에 대해서 뛰어난 성능을 보인 모델을 활용하여 다른 task를 위해 학습했을 때 이전에 학습했던 task에 대한 성능이 현저하게 떨어지는 현상

<div align="center">

![CGL example](https://user-images.githubusercontent.com/89853986/171803616-6104ebdb-34e3-4cb8-903f-aa9148b5e0e8.PNG)

</div>

Catastrophic forgetting은 neural network의 더욱 general한 problem인 "stability-plasticity" dilema의 결과이다. 
이 때, stability는 previously acquired knowledge의 보존을 의미하고, plasticity는 new knowledge를 integrate하는 능력을 의미한다. 

### 2.2 Limitation

Graph domain에서는 continual learning에 대한 연구가 놀랍도록 얼마 없다.  
이는 몇가지 한계점이 존재하기 때문이다.  
1. graph (non-Euclidean data) is not independent and identically distributed data.  
2. graphs can be irregular, noisy and exhibit more complex relations among nodes.  
3. apart from the node feature information, the topological structure in graph plays a crucial role in addressing graph-related tasks.

### 2.3 Purpose

1. 새로운 task를 학습할 때 이전 task에 대한 catastrophic forgetting 방지.  
2. 새로운 task 학습을 용이하게 하기 위해 이전 task의 knowledge를 사용.  
3. Influence function을 이용, previous task에서 영향력이 높은 node들을 buffer에 저장하여 새로운 task 학습에 함께 사용하도록 하는 **"Experience Replay GNN (ER-GNN)"** method 고안.

### 2.4 Contributions

* Continual Graph Learning (CGL) paradigm을 제시하여 single task가 아닌 multiple consecutive task (continual) setting에서 node classification task를 수행할 수 있도록 함.
* Continual node classification task에 기존 GNN을 적용할 때 발생하는 catastrophic forgetting 문제를 해결함.
* 유명한 GNN model에 적용 가능한 ER-GNN model을 개발하고, 이는 buffer로 들어갈 replay node를 선정할 때 기존 방법과는 다르게 influence function을 사용함.

  
## **3. Method**  


### 3.1 Problem Definition

Continual Node Classification (task incremental learning) setting에서 등장하는 sequence of task의 notation은 다음과 같다.

$$ \mathcal T = ( {\mathcal T}_1, {\mathcal T}_2, ..., {\mathcal T}_i, ..., {\mathcal T}_M ) $$

Node classification task의 정의는 아래와 같다.

#### Definition 1 (Node Classification)
각 task ![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20T%7D%20_%20i) 마다 dataset이 training node set ( ![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20D%7D%20_%20i%20%5E%7Btr%7D) )과 testing node set (![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20D%7D%20_%20i%20%5E%7Bte%7D))로 나뉘어 있다.  
Node classification task의 목적은 ![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20D%7D%20_%20i%20%5E%7Btr%7D) 을 사용하여 task-specific classifier를 학습시킨 후 ![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20D%7D%20_%20i%20%5E%7Bte%7D) 의 각 node를 알맞은 class(![](https://latex.codecogs.com/gif.latex?y_i%5El%20%5Cin%20%5Cmathcal%20Y%20_i)) 로 분류하도록 하는 것이다. (![](https://latex.codecogs.com/gif.latex?%5Cmathcal%20Y%20_i%20%3D%20%5C%7By_i%5E1%2C%20y_i%5E2%2C%20...%2C%20y_i%5El%2C%20...%2C%20y_i%5EL%20%5C%7D))
  
  
### 3.2 Experience Node Replay

본 논문에서 제시한 ER-GNN의 outline은 아래의 Algorithm에서 확인 가능하다.

<div align="center">

![algorithm1](https://user-images.githubusercontent.com/89853986/171842818-11dcb89b-c813-4de6-b169-63bb57eaad75.PNG)

</div>

Task ![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20T%7D%20_%20i)를 학습시킬 때 experience buffer ![](https://latex.codecogs.com/gif.latex?%5Cmathbb%20B) 로 부터 example ![](https://latex.codecogs.com/gif.latex?B)를 골라 ![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20D%7D%20_%20i%20%5E%7Bte%7D)와 함께 training 시킨다.  
일반적인 node classification task에서 loss function은 아래와 같은 cross-entropy loss function을 사용한다. 

![cross_entropy](https://user-images.githubusercontent.com/89853986/171843705-4e1fda63-7512-47aa-afb7-247ab6d92775.PNG)


생각해보면 training set인 ![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20D%7D%20_%20i%20%5E%7Bte%7D)의 크기는 buffer ![](https://latex.codecogs.com/gif.latex?B)의 크기보다 훨씬 클 것이다. 따라서 model이 특정 node 집합을 선호하지 않도록 방지하기 위해 loss function에 weight factor ![](https://latex.codecogs.com/gif.latex?%5Cbeta)를 적용한다.  
![](https://latex.codecogs.com/gif.latex?%5Cbeta)의 값은 아래와 같다. 

![beta](https://user-images.githubusercontent.com/89853986/171844522-34347ab6-ad8e-403a-bbab-a9cdda5a09e3.PNG)


이러한 weight factor를 통해 재구성한 loss function은 다음과 같다. 

  
![final_loss](https://user-images.githubusercontent.com/89853986/171844972-533c52b2-1a83-49f5-867b-fb05edb7565c.PNG)


그 이후에는 다음과 같이 loss를 최소화할 수 있는 optimal parameters를 구하면 된다. 

![optimal_parameter](https://user-images.githubusercontent.com/89853986/171845732-dee5f68f-cc1b-4c73-9bf5-79fc1c30a7c7.PNG)

이렇게 parameter updating을 한 다음, ![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20D%7D%20_%20i%20%5E%7Btr%7D)에서 replay 시킬 experience nodes ![](https://latex.codecogs.com/gif.latex?%5Cepsilon)를 선정하여 buffer ![](https://latex.codecogs.com/gif.latex?%5Cmathbb%20B)에 추가한다. 

![](https://latex.codecogs.com/gif.latex?%5Cepsilon%20%3D%20Select%28%5Cmathcal%20D%20_i%20%5E%7Btr%7D%2Ce%29) 는 각 class에서 ![](https://latex.codecogs.com/gif.latex?e)개의 nodes를 뽑아서 buffer에 저장한다는 의미이다.

#### 3.2.1 Experience Selection Strategy

Replay할 node를 선정하는데 사용되는 3가지 방법을 소개하겠다.

**1. Mean of Feature (MF)**

가장 직관적인 방법이다.  
각 class에 대하여 average attribute vector 혹은 average embedding vector등의 prototype을 선정한 후 해당 prototype과 가장 가까운 ![](https://latex.codecogs.com/gif.latex?e)개의 node를 buffer에 추가하는 방법이다. 

**2. Coverage Maximization (CM)**

각 class마다 선정하는 experience node ![](https://latex.codecogs.com/gif.latex?e)개가 적을 경우 본 방법을 사용하는 것이 효과적이다.  
정해진 거리 안에 다른 label을 가진 node의 개수가 가장 적은, 즉, coverage가 가장 넓은 node를 사용하는 방법이다. 식으로 표현하면 아래와 같다.  
![Coverage_maximization](https://user-images.githubusercontent.com/89853986/171857055-2808b564-d57a-47fe-a79d-539bdde59f42.PNG)  
그 후, ![](https://latex.codecogs.com/gif.latex?%5Cleft%20%7C%20%5Cmathcal%20N%28v_i%29%20%5Cright%20%7C) 값이 가장 작은 ![](https://latex.codecogs.com/gif.latex?e)개의 node를 buffer에 추가하는 방법이다.

**3. Influence Maximization (IM)**

각 task ![](https://latex.codecogs.com/gif.latex?%7B%5Cmathcal%20T%7D%20_%20i)를 학습할 때 특정 node ![](https://latex.codecogs.com/gif.latex?v_*)를 training set에서 제거한 새로운 training set을 ![](https://latex.codecogs.com/gif.latex?%5Cmathcal%20D%20_%7Bi*%7D%20%5E%7Btr%7D)이라고 하자.  
이 때 새롭게 계산되는 optimal parameters는 아래와 같다.  

![new_optimal_parameter](https://user-images.githubusercontent.com/89853986/171860164-47a38b48-1016-44ba-80b5-4054eb56aa7f.PNG)  
그리고 이 때의 parameter change (![](https://latex.codecogs.com/gif.latex?%5Ctheta%20_*%20-%20%5Ctheta))를 관찰한다.

하지만, 모든 node를 제거해가면서 optimal parameter의 변화를 관찰하는 것은 computational cost 측면에서 매우 비효율적이다.  

이에, 저자는 model을 retraining하지 않고 parameter의 변화를 추정할 수 있는 **influence function**을 적용한다.  

![](https://latex.codecogs.com/gif.latex?v_*)를 small ![](https://latex.codecogs.com/gif.latex?%5Cepsilon)만큼 upweight 했을 때 얻게되는 new optimal parameter는 아래와 같이 정의할 수 있다.  

![epsilon_upweight](https://user-images.githubusercontent.com/89853986/171861381-5fe22588-49a0-42bd-88c4-e9019418fbd2.PNG)  

그리고 이 때, ![](https://latex.codecogs.com/gif.latex?v_*)를 upweighting한 영향력(influence)는 아래와 같이 계산된다.  

![upweighting_influence](https://user-images.githubusercontent.com/89853986/171862281-c04ea4f4-9f9c-4dcc-9283-94a5956de52b.PNG)  

Hessian matrix는 다음과 같이 계산된다.  

![hessian](https://user-images.githubusercontent.com/89853986/171862497-9a315cdd-b6d4-4c0b-9277-6cc8a17072de.PNG)

여기서, 다시 본론으로 돌아가 앞서 구하고자 했던 ![](https://latex.codecogs.com/gif.latex?v_*)를 제거했을 때의 optimal parameter 변화는 upweighting ![](https://latex.codecogs.com/gif.latex?%5Cepsilon)을 ![](https://latex.codecogs.com/gif.latex?%5Cepsilon%20%3D%20-%281/%28%5Cleft%20%7C%20%5Cmathcal%20D%20_i%20%5E%7Btr%7D%20%5Cright%20%7C%20&plus;%20%5Cleft%20%7C%20B%20%5Cright%20%7C%29%29)로 설정했을 때와 동일한 경우이다.  

그렇다면 retraining 없이 아래와 같이 optimal parameter의 변화를 계산할 수 있다.  
![without_retraining](https://user-images.githubusercontent.com/89853986/171863673-7f4f5691-6d8d-4692-82ad-1d0d56a7be3b.PNG)  

하지만, ![](https://latex.codecogs.com/gif.latex?%5Ctheta%20_*%20-%20%5Ctheta)의 Frobenius norm이 매우 작아 정확한 ![](https://latex.codecogs.com/gif.latex?%5Ctheta%20_%20*)을 찾기 어렵고, hessian matrix의 inverse를 구하는 것은 computationally expensive하므로 우리는 training node ![](https://latex.codecogs.com/gif.latex?v_*) 대신 testing node ![](https://latex.codecogs.com/gif.latex?v_%7Btest%7D)를 upweighting했을 때의 influence를 계산하도록 한다.  

<div align="center">

![testing_influence](https://user-images.githubusercontent.com/89853986/171865438-6ad2f248-3611-4b80-a608-259278cd958c.PNG)

</div>

본 process를 진행하는 과정에서 Hessian-vector products (HVPs)를 사용하여 아래의 식을 근사한다.  

<div align="center">

![HVPs](https://user-images.githubusercontent.com/89853986/171865817-82688c79-018a-4429-b027-28c1ce1aa30f.PNG)  
  
![](https://latex.codecogs.com/gif.latex?%5CDownarrow)  
  
![hvp](https://user-images.githubusercontent.com/89853986/171866522-e7af0ab6-ce3a-456d-b75f-df2292c1fa96.PNG)

</div>

이 때, Hessian matrix는 positive semi-definite이므로 아래와 같이 식이 변형되고,  

<div align="center">

![psd](https://user-images.githubusercontent.com/89853986/171866994-84c50cdf-e2b5-439a-8d9d-b1d0d0a13697.PNG)  

</div>
  
이로써 ![](https://latex.codecogs.com/gif.latex?H_%7B%5Ctheta%7D%5E%7B-1%7D)를 직접적으로 구하는 것이 아닌 ![](https://latex.codecogs.com/gif.latex?H_%7B%5Ctheta%7D%5Calpha)를 사용하여 conjugate gradient로 ![](https://latex.codecogs.com/gif.latex?%5Calpha)를 구하여 위의 식(influence)을 손쉽게 계산할 수 있게된다.


우리는 이 때, influence가 큰 ![](https://latex.codecogs.com/gif.latex?v_*)일수록 ![](https://latex.codecogs.com/gif.latex?v_*)가 더욱 representative하다고 가정하기 때문에 influence가 가장 큰 ![](https://latex.codecogs.com/gif.latex?e)개의 node를 뽑아 buffer에 추가한다.

## **4. Experiment**  

### **4.1 Experiment setup**  


#### 4.1.1 Dataset  

실험에서 사용한 dataset의 구성은 아래의 표와 같다.  

<div align="center">

![dataset](https://user-images.githubusercontent.com/89853986/172016299-e66d7d93-0c16-4498-8bed-04e971c23b89.png)

</div>

#### 4.1.2 baseline  

ER-GNN과의 비교를 위해 continual setting에서 아래의 GNN 모델들과 비교하였다.  

~~~
  - Deepwalk : random walk를 한 후 NLP에서 사용되는 skip-gram model을 학습.
  - Node2Vec : network에 있는 node의 neighborhood 정보를 가장 잘 보존하는 low-dimensional feature를 extract.
  - GCN : spectral convolution을 first-order approximation하여 효과적인 layer-wise propagation 진행.
  - GraphSAGE : node의 local neighborhood로부터 feature를 sampling / aggregating하여 embedding.
  - GAT : attention-based architecture를 사용하여 node를 embedding.
  - SGC : non-linearity를 제거하고, consecutive layer 간의 weight matrices를 조정하여 GCN을 simplify.
  - GIN : Weisfeiler-Lehman graph isomorphism test 만큼 강력하고, GNN 중에 가장 표현력이 뛰어남.
~~~

* 저자는 GNN method 중 GAT를 사용하여 ER-GNN을 구성하였다.  
* 위에서 설명한 3가지(MF, CM, IM) experience selection strategy에 대하여 모두 실험을 진행하였는데, 이는 ER-GNN 뒤에 표시되어 있다. (ex. ER-GAT-MF, ER-GAT-CM, ER-GAT-IM 등)  
* 별(\*) 표시가 붙어있는 방법론도 있을 것이다. 그러한 경우는 위에서 언급 하였듯, MF와 CM method를 사용할 때 attribute가 아닌 embedding을 기준으로 mean과 coverage maximization을 계산한 것을 의미한다.  


#### 4.1.3 Evaluation Metric  

본 논문의 주된 목적은 continual learning에서 고질적으로 발생하는 문제인 catastrophic forgetting을 줄이기 위함이므로 이에 알맞은 evaluation metric을 저자는 제안한다.  

* Performance Mean (PM) : 일반적인 accuracy value이다. 단, Reddit dataset에서는 class 간의 imbalance 문제 때문에 Micro F1 score를 사용한다.

* Forgetting Mean (FM) : 이후 task를 학습하고 난 뒤, task의 accuracy가 떨어지는 정도를 측정한 값이다.

### **4.2 Result**  


#### 4.2.1 Performance Mean

<div align="center">

![PM1](https://user-images.githubusercontent.com/89853986/172018509-e48c19b6-ccfe-4eda-8a7a-1a8a9a61ebb1.png)

</div>

* GNN model들과 다른 두 model (DeepWalk, Node2Vec) 모두 일정 수준의 catastrophic forgetting은 발생하는 것을 관찰할 수 있다.
* GNN model에 비해 DeepWalk와 Node2Vec은 PM의 관점에서 더 좋지 않은 결과를 보이지만, FM의 관점에서는 더 좋은 결과가 관찰된다. 이는 DeepWalk나 Node2Vec이 새로운 task를 학습하는 것을 희생하여 이전 task들의 학습을 기억하는 것에 더 초점을 맞추는 것으로 해석 가능하다. 
* GNN model 중 GAT는 PM과 FM의 관점 모두에서 좋은 결과를 보인다. 이는 attention mechanism이 continual graph-related task learning에서 new task 학습과 existing task 학습 내용을 기억하는데에 모두 장점이 있다는 것을 보여준다. 
* 저자가 고안한 ER-GNN의 경우 IM strategy를 사용한 경우 가장 좋은 performance가 도출되었다. Influence function이 node를 replay하는데에 효과가 있음을 입증한다. 
* MF와 CM strategy에서는 embedding space를 기준으로 한 model (명칭에 \*가 붙어있는)이 attribute space를 기준으로 한 model들보다 좋은 결과를 나타내었다.


<div align="center">

![PM2](https://user-images.githubusercontent.com/89853986/172018607-46974fef-a3b3-453b-af67-9673420fac75.png)
 
</div>

* Dataset 별 task가 진행됨에 따른 accuracy를 plot
* Figure를 보면 세가지 dataset 모두에서 catastrophic forgetting이 발생한다.
* ER-GNN model과 함께 influence function을 쓴 model이 catastrophic forgetting을 가장 잘 완화하는 결과이다.


#### 4.2.2 Forgetting Mean

<div align="center">

![FM](https://user-images.githubusercontent.com/89853986/172018571-0ccbdbc1-6642-4b39-ab4b-ab5191a2b0e9.png)

</div>

* SGC와 GIN model에 대해서 ER-GNN model을 적용하였다. 
* 위의 table과 비교해보면, ER-GNN을 적용하지 않은 natural SGC/GIN일 때보다 FM 값이 확연히 줄어든 것으로 보아 catastrophic forgetting을 줄이는데 도움을 준다는 것을 보여준다.
* 3가지 experience selection stragtegies 중에서 저자가 제안한 IM 방법이 가장 좋은 performance를 보인다.



#### 4.2.3 Influence of ![](https://latex.codecogs.com/svg.image?e)

<div align="center">
  
![e](https://user-images.githubusercontent.com/89853986/172018666-448666be-1d91-4456-b392-001558ae5348.png)

</div>

* Buffer에 들어가는 node의 개수를 지정하는 파라미터인 ![](https://latex.codecogs.com/svg.image?e)는 model의 성능과 직결된다.
* 예측한 바와 동일하게 buffer에 저장하는 node의 개수를 늘리면 catastrophic forgetting을 예방하는데에 큰 도움이 된다. ![](https://latex.codecogs.com/svg.image?e) 값이 무분별하게 늘어날 경우 computational cost가 증가하여 결국 retraining과 다를 바가 없게 될 수 있다.
* Hyperparameter tuning을 통해 catastrophic forgetting과 computational cost 간의 trade-off 관계에서 균형을 찾을 필요가 있을 것이다. 


## ** 5. Conclusion**  

Please summarize the paper.  
It is free to write all you want. e.g, your opinion, take home message(오늘의 교훈), key idea, and etc.

---  
## **Author Information**  

* Seungyoon Choi  
    * Affiliation : [DSAIL@KAIST](https://dsail.kaist.ac.kr/)
    * Research Topic : GNN, Continual Learning, Active Learning

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation  
* Reference  

