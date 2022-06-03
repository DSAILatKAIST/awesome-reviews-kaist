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

Please write the motivation of paper. The paper would tackle the limitations or challenges in each fields.

After writing the motivation, please write the discriminative idea compared to existing works briefly.

### Continual Learning과 Catastrophic Forgetting

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

### Limitation

Graph domain에서는 continual learning에 대한 연구가 놀랍도록 얼마 없다.  
이는 몇가지 한계점이 존재하기 때문이다.  
1. graph (non-Euclidean data) is not independent and identically distributed data.  
2. graphs can be irregular, noisy and exhibit more complex relations among nodes.  
3. apart from the node feature information, the topological structure in graph plays a crucial role in addressing graph-related tasks.

### Purpose

1. 새로운 task를 학습할 때 이전 task에 대한 catastrophic forgetting 방지.  
2. 새로운 task 학습을 용이하게 하기 위해 이전 task의 knowledge를 사용.  
3. Influence function을 이용, previous task에서 영향력이 높은 node들을 buffer에 저장하여 새로운 task 학습에 함께 사용하도록 하는 **"Experience Replay GNN (ER-GNN)"** method 고안.


  
## **3. Method**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  

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

