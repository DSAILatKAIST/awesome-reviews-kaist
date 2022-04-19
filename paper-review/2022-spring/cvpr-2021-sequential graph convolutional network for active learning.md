---
description : Razvan Caramalau et al. / Sequential Graph Convolutional Network for Active Learning / CVPR-2021  
---

# **Title** 

Sequential Graph Convolutional Network for Active Learning  

## **1. Problem Definition**  


GCN의 message-passing을 활용하여 강하게 연결된 노드를 비슷하게 embedding한다.
그 후 _CorSet_,  _uncertainty-based methods_ 등의 active learning 방법론을 적용하여 labeling cost를 효과적으로 줄인다.


## **2. Motivation**  

딥러닝은 image classification, 3D Hand Pose Estimation (HPE) 등의 computer vision 분야에서 상당한 발전을 보이고 있다. 이것은 computing infrastructure의 발전과 large-scale dataset 덕분에 가능한 일이었다.
하지만, 모델을 학습시키려면 data에 라벨링을 하는 과정이 필요하다. 라벨링의 type이 다양하기 때문에 비교적 쉬운 경우도 있지만 대개 이것은 time-consuming task이고, 전문가와 cost가 요구된다. 
이러한 이슈로 인하여 효과적으로 의미있는 sample을 선정하는 Active Learning 방법론이 대두되고 있다.

일반적으로, Active Learning (AL) framework에는 세가지 구성요소가 존재하고 각각의 역할은 아래와 같다. 
- _learner_ : target task (_downstream task_)를 학습하는 모델
- _sampler_ : fixed budget 내에서 labeling을 요청할 unlabelled data를 선정
- _annotator_ : queried data를 labeling  

여기서, learner와 sampler의 관계에 따라 AL framework는 두가지 카테고리로 나뉘게 된다. 
- **task-dependent** 
sampler가 learner가 수행하는 task에 따라 design되는 경우이다. 초기의 대부분의 연구는 task-dependent 였으며 이 경우에는 task가 바뀔 때마다 새롭게 sampler를 디자인 해줘야 하는 한계가 존재한다 (scalability problem)

- **task-agnostic**
task-dependent와 반대로 task에 영향을 받지 않고, 동일한 sampler를 유지할 수 있다. VAAL과 Learning Loss와 같은 최근 연구가 task-agnostic 방법론을 제시한다.  
하지만 이 연구들은 labelled와 unlabelled images 간의 연관성을 탐색하는 mechanism이 부족하다는 단점이 존재한다. 

본 논문에서는 task-agnostic하면서 (learner와 sampler가 구분된 model-based AL method이기 때문) labelled, unlabelled 간의 연관성을 표현하지 못한다는 VAAL, Learning Loss의 문제점을 GCN을 적용하여 해결한다.


## **3. Method**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily. 

### **Pipeline**

저자가 제시한 method의 전체적인 pipeline은 아래 그림과 같다.  

![pipeline](https://user-images.githubusercontent.com/89853986/163950282-b032a56b-5577-439c-b28a-5cbb6ed1889c.PNG)


총 5 Phase로 구성되는데, 각각을 _learner, sampler, annotator_ 로 분류하여 설명하자면 아래와 같다.
1. _learner_ (Phase 1)
	>적은 수의 seed labelled data로 learner를 training 시키고, 학습된 parameter를 활용하여 labelled, unlabelled data의 feature를 추출해낸다. (Phase 1)
2. _sampler_ (Phase 2, 3, 4)
	>각 image의 feature를 node로, image 간의 similarity를 edge로 표현한 graph를 생성한다. (Phase 2)
	>
	>Phase 2에서 생성한 graph에 GCN을 적용하여 embedding을 한다. (Phase 3)
	>
	>적용할 sampling method (본 논문에서는 **UncertainCGN, CoreCGN** method가 제시된다. 아래에서 더 자세히 다루겠다.)에 따라 query를 선정한다. (Phase 4)
3. _annotator_ (Phase 5)
	>query로 선정된 data에 대해 labelling을 한다. (Phase 5)

Phase 1부터 5까지의 과정을 한번 진행하는 것이 한 cycle이다. 
다음 iteration에서는 Phase 5에서 labelling한 data를 추가하여 learner를 학습시키는 Phase 1부터 다시 cycle이 시작된다. 
Labelling을 할 수 있는 정해진 budget에 도달할 때까지 cycle을 반복한다.

### **Learner**
learner는 downstream task를 학습한다. 
본 논문에서는 classification, regression task 모두에 대해 다루었다.

1. Classification
>learner는 CNN image classifier를 사용한다. 특히, 비슷한 parameter complexity에 대해 좋은 성능을 보이는 ResNet-18을 model로 사용한다.
>Minimize 해야할 loss function은 아래와 같다. (cross-entropy 사용)  
>
>![loss_classification](https://user-images.githubusercontent.com/89853986/163951946-d4257605-91ba-401d-94ad-b66401c9dc95.PNG)
>
>$$M$$ 은 parameter $\Theta$를 갖고, input $x$를 output $y$로 매핑하는 deep model이고, $N_l$은 labelled training data의 개수, $f(x_i, y_i; \Theta)$는 model $M$의 posterior probability이다.
2. Regression
>3D HPE task를 다루기 위해서 _DeepPrior_ 모델을 사용한다.  
>위의 classification task와는 다르게 hand depth image로부터 3D hand joint의 위치를 regress해야한다.   
>Minimize 해야할 loss funcion은 아래와 같다.  
>
>![loss_regression](https://user-images.githubusercontent.com/89853986/163951987-b123ec14-511d-4735-9104-3ad6d4da32a0.PNG)
>
>$J$는 joint의 개수를 의미한다.

classification과 regression 이외의 task가 등장하더라도 전체 pipeline의 구조는 동일하게 유지한 채 learner만 바꿔주면 된다.

### **Sampler**
앞선 pipeline에서 살펴 보았듯이 Sampler는 주어진 budget 내에서 의미있는 unlabeled data를 sampling하여 annotator에게 labelling을 요청하는 model이다.

더욱 구체적인 sampler의 시나리오를 살펴보자.
unlabeled dataset $$ D_u $$ 에서 초기에 labelling할 initial batch $$ D_0 \subset D_u $$ 를 랜덤하게 골라주는 것으로 시나리오가 시작된다. 이렇게 초기 set이 확정이 되면 그 다음부터는 pipeline에 설명된 cycle을 돌면서 sampling할 unlabeled data를 고르고, labeling을 하여 새롭게 learner를 통해 training 시키는 과정을 최소한의 budget 내에서 수행한다.
이것을 수식으로 표현하면 아래와 같다.

![sampler](https://user-images.githubusercontent.com/89853986/163953401-9b324b99-0364-451c-b653-a5cfd9e271bc.PNG)

Sampling method $$A$$를 이용하여 최소한의 stage안에 최소한의 loss를 달성하는 것이 목적인 것이다. ($$D_n$$ 은 $$n$$번째 stage에서의 labeled dataset을 의미)

**Sequential GCN selection process**  
>- 저자가 제안한 pipeline에서 sampler는 GCN을 사용한다.  
>- GCN의 input은 앞선 learner에서 구해진 labelled, unlabelled image들의 feature를 node로, image간의 similarity를 edge로 표현한 graph이다. Graph를 생성하고, GCN을 사용하는 목적은 message-passing을 통해 node가 갖고 있는 uncertainty를 전파시켜 higher-order representation을 하기 위함이다.  
>- 이러한 과정을 통해 GCN은 어떤 image를 labeling 해야할지 결정하는 binary classifier의 역할을 하게된다.

**Graph Convolutional Network**  
>1. Graph Structure 구성
>>Graph는 node와 edge로 구성되며, node $$v \in \mathbb R^{(m\times N)}$$ 는 $$N$$개의 data (labelled, unlabelled 모두 포함)와 각각의 $$m$$ dimension feature로 표현된다.
>>Edge는 adjacency matrix $$A$$로 표현이 가능하다. Edge는 node간의 similarity를 나타내야하므로 다음과 같은 과정을 거쳐 adjacency matrix를 구성한다. 
>>1. learner에서 넘어온 feature를 $$l_2$$ normalize한다. 
>>2. $$S_{ij} = v_i^Tv_j, {i,j} \in N$$ (vector product를 통해 $$S_{ij}$$를 생성)
>>3. $$A = D^{-1}(S-I)+I$$ ($$S$$에서 identity matrix를 빼고, degree matrix로 normalise를 한 다음 identity matrix를 다시 더해 closest correlation을 자기 자신으로 설정)
>2. 1st layer of GCN
>>- Over-smoothing을 방지하기 위해 GCN을 2-layer로 쌓는다.  
>>- 첫번째 layer의 function을 $$f_{\mathcal G}^1$$로 표현한다.  
>>- 첫번째 layer는 ReLU를 activation function으로 사용한다.  
>3. 2nd layer of GCN
>>- 각 노드를 labelled와 unlabelled로 mapping해야하기 때문에 두번째 layer는 sigmoid를 activation function으로 사용한다.  
>>- 따라서 두번째 layer까지 거친 output은 0~1사이의 값을 가지는 길이 $$N$$의 vector이다. (0은 unlabelled, 1은 labelled를 의미)  
>
>전체적인 과정은 아래와 같은 식으로 표현된다.
>![gcn](https://user-images.githubusercontent.com/89853986/163961880-ea5a6f69-1ec4-4657-982f-f5780ee24f0d.PNG)
>
>또한 loss function은 아래와 같다. (cross-entropy를 사용하였고, $$\lambda$$는 labelled와 unlabelled cross-entropy간의 weight를 조절하는 parameter이다.)
>


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