---
description : Jiaxuan You / Position-aware Graph Neural Networks / ICML(2019)   
---

# **Position-aware Graph Neural Networks** 

## **1. Problem Definition**  

### **Position-aware Embeddings?**  
> A node embeddings ![image](https://user-images.githubusercontent.com/37684658/170816681-ba80b5bc-f1d6-44f0-9460-876beb04c206.png), ![image](https://user-images.githubusercontent.com/37684658/170816652-56863385-a6d2-4f5f-8232-3324fb2ac266.png) is position-aware if there exists a function ![image](https://user-images.githubusercontent.com/37684658/170816760-535d4ffb-9751-463e-b942-2bda93f053c4.png)
 such that ![image](https://user-images.githubusercontent.com/37684658/170816830-acf0d219-a7ee-46c0-b6ca-cd2835074ad6.png) is the shortest path distance in  ![image](https://user-images.githubusercontent.com/37684658/170816866-78e1693a-7494-4fd8-a461-a951ed16fc18.png)

본 논문에서는 Graph Neural Networks(이하 'GNN')에서 `Position-aware`의 성질을 반영할 수 있는 방안에 대해서 다루고 있다. Position-aware Embedding이란 위 정의처럼 Embedding Space에서도 두 Node간 Shortest Path Distance가 유지되도록 임베딩하는 것을 의미한다. 

### **Structure-aware Embeddings?**  
> A node embeddings ![image](https://user-images.githubusercontent.com/37684658/170817257-57a8be48-cdb9-4057-8dd8-58a9e4a759ed.png) is structure-aware if it is a function of up to q-hop network neighbourhood of node ![image](https://user-images.githubusercontent.com/37684658/170817276-c3f366e3-648e-496d-8741-7b0b67097a82.png)

다른 개념으로 `Structure-aware`가 있다. 기존의 GNN 모델들 중 GCN, GAT처럼 Target Node의 q-hop의 neighbor를 이용해서 message-passing하는 경우 그래프의 structure를 반영하여 임베딩할 수 있기 때문에 이를 Structure-aware Embedding이라고 한다.  

### Key Insights  
논문에서는 2가지의 Key Insights를 제시하고 있다.  
> **1. 그래프의 Position 정보를 어떻게 반영할 것인가?**  
 Anchors-sets를 이용해서 각각의 set에서 나오는 메시지를 Aggregation하여 Position 정보를 반영한다.  
> **2. Position의 기준이 되는 Anchor-sets들은 어떻게 구성할 것인가?**  
 Bourgain Theorem를 이용하여 이론적 근거를 바탕으로 Position-aware한 임베딩을 할 수 있는 Anchor Set를 구성한다.  

### Goal  
최종적인 목표는 그래프 내 노드의 global position을 반영하면서도, 노드 주변의 local structure를 반영할 수 있는 Position-aware Graph Neural Networks(이하 'PGNN')를 만드는 것이다.

## **2. Motivation**  

기존 GNN은 노드들이 symmetric이나 isomorphic한 position에 있는 경우 이 두 노드들을 구분할 수 없다는 문제점이 있다. 
<img width="410" alt="image" src="https://user-images.githubusercontent.com/37684658/170819612-0958ee85-f6f8-4b64-b188-3b85b5fc1545.png">  
이 문제를 다루기 위해서 heuristic한 방법을 쓰기도 하는데, 각 노드들마다 unique identifier를 할당시켜주거나, ![image](https://user-images.githubusercontent.com/37684658/170819647-485151fa-8061-4b9f-b6db-55fd2b8c07a1.png)와 같이 position의 기준이 되는 identifier를 정해준 뒤, transductive하게 그래프를 사전학습시키는 방법이 있다. 허나 이러한 방법들은 모두 scalable하지 못하고, 처음보는 그래프에 대해서는 기존에 정해두었던 identifier를 활용할 수가 없기 때문에 general하지 못한 단점이 있다.  
따라서 PGNN은 기존의 Structure-aware GNN의 장점을 그대로 가져가면서 positional 정보도 함께 임베딩하는 것을 목표로 한다.  


## **3. Method**  
### **Architecture**
<img width="967" alt="image" src="https://user-images.githubusercontent.com/37684658/170822199-7e47b798-1689-41b6-b33d-95b2d637f4ac.png">  
PGNN의 전체 Architecture는 위와 같다. 먼저 k개의 anchor-set를 만들고, 각각에 anchor-set에 노드들을 할당한다. k는 어떻게 정해지는 지, 그리고 노드들의 할당은 어떻게하는 지는 후술하도록 하겠다. 그리고 각 노드에 대해서 임베딩을 하게 되는데 ![image]("https://user-images.githubusercontent.com/37684658/170822561-024cc4fb-6b50-484f-9057-93b4e3b11a8f.png")  을 임베딩한다고 하면, ![image]("https://user-images.githubusercontent.com/37684658/170822308-6f990491-2ee0-44cd-aae5-e029a439ce9e.png")  과 나머지 노드들에 대해서 짝을 짓고, 각 anchor-set에 속하는 노드들의 정보만 모아서 anchor-set 개수만큼의 메시지를 만든다. 그 후 총 2개의 output이 나오게 된다.  

<img width="956" alt="image" src="https://user-images.githubusercontent.com/37684658/170822445-64e09402-036a-464a-9727-3f3c48aba3d8.png">

하나는 anchor-set 위치에 대해서 invariant한 임베딩(![image](https://user-images.githubusercontent.com/37684658/170822395-db11b231-e74a-4c55-b077-0a71bca158d6.png))이 나오게되고, 나머지 하나는 최종적으로 우리가 task를 수행하는 데 필요한 임베딩(![image](https://user-images.githubusercontent.com/37684658/170822432-2038c757-5a53-4f17-8504-b7d55ffb3536.png))이 나온다.



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

