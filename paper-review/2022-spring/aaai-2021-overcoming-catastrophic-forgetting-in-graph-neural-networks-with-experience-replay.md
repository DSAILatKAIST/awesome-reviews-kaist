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

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  

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

* Seungyoon Choi  
    * Affiliation : [DSAIL@KAIST](https://dsail.kaist.ac.kr/)
    * Research Topic : GNN, Continual Learning, Active Learning

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation  
* Reference  

