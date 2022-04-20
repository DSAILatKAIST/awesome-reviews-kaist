---
description : Razvan Caramalau / Sequential Graph Convolutional Network for Active Learning / CVPR-2021  
---

# **Title** 

Sequential Graph Convolutional Network for Active Learning
<br/>
<br/>


## **1. Problem Definition**  

<code>Image data로 graph를 생성하여 GCN을 활용한 **model-based active learning (task-agnostic)** 방법론을 제시</code>

- GCN의 **message-passing** 특성을 활용하여 강하게 연결된 노드를 비슷하게 embedding한다.  
- 그 후 _CorSet_,  _uncertainty-based methods_ 등의 active learning 방법론을 적용하여 sampling할 data를 선정하고, 이를 통해 **labeling cost를 효과적으로 줄인다**.

<br/>

## **2. Motivation**  

딥러닝은 image classification, 3D Hand Pose Estimation (HPE) 등의 computer vision 분야에서 상당한 발전을 보이고 있다. 이것은 computing infrastructure의 발전과 large-scale dataset 덕분에 가능한 일이었다.
하지만, 모델을 학습시키려면 data에 라벨링을 하는 과정이 필요하다. 라벨링의 type이 다양하기 때문에 비교적 쉬운 경우도 있지만 대개 이것은 time-consuming task이고, 전문가와 cost가 요구된다.  
  
이러한 이슈로 인하여 효과적으로 의미있는 sample을 선정하는 _**Active Learning**_ 방법론이 대두되고 있다.
  
  
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

<br/>
<br/>

## **3. Method**  

### **3.1 Pipeline**

저자가 제시한 method의 전체적인 _**pipeline**_ 은 아래 그림과 같다.  

![pipeline](https://user-images.githubusercontent.com/89853986/163950282-b032a56b-5577-439c-b28a-5cbb6ed1889c.PNG)


총 _**5 Phase**_ 로 구성되는데, 각각을 _learner, sampler, annotator_ 로 분류하여 설명하자면 아래와 같다.
1. _learner_ (Phase 1)
	>- 적은 수의 seed labelled data로 learner를 training 시키고, 학습된 parameter를 활용하여 labelled, unlabelled data의 feature를 추출해낸다. (Phase 1)
2. _sampler_ (Phase 2, 3, 4)
	>+ 각 image의 feature를 node로, image 간의 similarity를 edge로 표현한 graph를 생성한다. (Phase 2)
	>
	>+ Phase 2에서 생성한 graph에 GCN을 적용하여 embedding을 한다. (Phase 3)
	>
	>+ 적용할 sampling method (본 논문에서는 **UncertainCGN, CoreCGN** method가 제시된다. 아래에서 더 자세히 다루겠다.)에 따라 query를 선정한다. (Phase 4)
3. _annotator_ (Phase 5)
	>+ query로 선정된 data에 대해 labelling을 한다. (Phase 5)

Phase 1부터 5까지의 과정을 한번 진행하는 것이 한 cycle이다.   
다음 iteration에서는 Phase 5에서 labelling한 data를 추가하여 learner를 학습시키는 Phase 1부터 다시 cycle이 시작된다.   
Labelling을 할 수 있는 정해진 budget에 도달할 때까지 cycle을 반복한다.  

### **3.2 Learner**
learner는 downstream task를 학습한다. 
본 논문에서는 classification, regression task 모두에 대해 다루었다.

**3.2.1 Classification**
>learner는 CNN image classifier를 사용한다. 특히, 비슷한 parameter complexity에 대해 좋은 성능을 보이는 ResNet-18을 model로 사용한다.
>Minimize 해야할 loss function은 아래와 같다. (cross-entropy 사용)  
>
>![loss_classification](https://user-images.githubusercontent.com/89853986/163951946-d4257605-91ba-401d-94ad-b66401c9dc95.PNG)
>
>![](https://latex.codecogs.com/gif.latex?M) 은 parameter ![](https://latex.codecogs.com/gif.latex?\Theta)를 갖고, input ![](https://latex.codecogs.com/gif.latex?x)를 output ![](https://latex.codecogs.com/gif.latex?y)로 매핑하는 deep model이고, ![](https://latex.codecogs.com/gif.latex?N_L)은 labelled training data의 개수, ![](https://latex.codecogs.com/gif.latex?f%28x_i%2C%20y_i%3B%20%5CTheta%29)는 model ![](https://latex.codecogs.com/gif.latex?M)의 posterior probability이다.  


**3.2.2 Regression**
>3D HPE task를 다루기 위해서 _DeepPrior_ 모델을 사용한다.  
>위의 classification task와는 다르게 hand depth image로부터 3D hand joint의 위치를 regress해야한다.   
>Minimize 해야할 loss funcion은 아래와 같다.  
>
>![loss_regression](https://user-images.githubusercontent.com/89853986/163951987-b123ec14-511d-4735-9104-3ad6d4da32a0.PNG)
>
>![](https://latex.codecogs.com/gif.latex?J)는 joint의 개수를 의미한다.

classification과 regression 이외의 task가 등장하더라도 전체 pipeline의 구조는 동일하게 유지한 채 learner만 바꿔주면 된다.

### **3.3 Sampler**
앞선 pipeline에서 살펴 보았듯이 Sampler는 주어진 budget 내에서 의미있는 unlabeled data를 sampling하여 annotator에게 labelling을 요청하는 model이다.

더욱 구체적인 sampler의 시나리오를 살펴보자.
unlabeled dataset ![](https://latex.codecogs.com/gif.latex?D_U)에서 초기에 labelling할 initial batch ![](https://latex.codecogs.com/gif.latex?D_0%20%5Csubset%20D_U)를 랜덤하게 골라주는 것으로 시나리오가 시작된다. 이렇게 초기 set이 확정이 되면 그 다음부터는 pipeline에 설명된 cycle을 돌면서 sampling할 unlabeled data를 고르고, labeling을 하여 새롭게 learner를 통해 training 시키는 과정을 최소한의 budget 내에서 수행한다.
이것을 수식으로 표현하면 아래와 같다.

![sampler](https://user-images.githubusercontent.com/89853986/163953401-9b324b99-0364-451c-b653-a5cfd9e271bc.PNG)

Sampling method ![](https://latex.codecogs.com/gif.latex?A)를 이용하여 최소한의 stage안에 최소한의 loss를 달성하는 것이 목적인 것이다. (![](https://latex.codecogs.com/gif.latex?D_n)은 ![](https://latex.codecogs.com/gif.latex?n)번째 stage에서의 labeled dataset을 의미)

**3.3.1 Sequential GCN selection process**  
>- 저자가 제안한 pipeline에서 sampler는 GCN을 사용한다.  
>- GCN의 input은 앞선 learner에서 구해진 labelled, unlabelled image들의 feature를 node로, image간의 similarity를 edge로 표현한 graph이다. Graph를 생성하고, GCN을 사용하는 목적은 message-passing을 통해 node가 갖고 있는 uncertainty를 전파시켜 higher-order representation을 하기 위함이다.  
>- 이러한 과정을 통해 GCN은 어떤 image를 labeling 해야할지 결정하는 binary classifier의 역할을 하게된다.

**3.3.2 Graph Convolutional Network**  
>1. Graph Structure 구성
>>Graph는 node와 edge로 구성되며, node ![](https://latex.codecogs.com/gif.latex?v%20%5Cin%20%5Cmathbb%20R%5E%7B%28m%5Ctimes%20N%29%7D) 는 ![](https://latex.codecogs.com/gif.latex?)개의 data (labelled, unlabelled 모두 포함)와 각각의 ![](https://latex.codecogs.com/gif.latex?m) dimension feature로 표현된다.
>>Edge는 adjacency matrix ![](https://latex.codecogs.com/gif.latex?A)로 표현이 가능하다. Edge는 node간의 similarity를 나타내야하므로 다음과 같은 과정을 거쳐 adjacency matrix를 구성한다. 
>>1. learner에서 넘어온 feature를 ![](https://latex.codecogs.com/gif.latex?l_2) normalize한다. 
>>2. ![](https://latex.codecogs.com/gif.latex?S_%7Bij%7D%20%3D%20v_i%5ETv_j%2C%20%7Bi%2Cj%7D%20%5Cin%20N) (vector product를 통해 ![](https://latex.codecogs.com/gif.latex?S_{ij})를 생성)
>>3. ![](https://latex.codecogs.com/gif.latex?A%20%3D%20D%5E%7B-1%7D%28S-I%29&plus;I) (![](https://latex.codecogs.com/gif.latex?S)에서 identity matrix를 빼고, degree matrix로 normalise를 한 다음 identity matrix를 다시 더해 closest correlation을 자기 자신으로 설정)
>2. 1st layer of GCN
>>- Over-smoothing을 방지하기 위해 GCN을 2-layer로 쌓는다.  
>>- 첫번째 layer의 function을 ![](https://latex.codecogs.com/gif.latex?f_%7B%5Cmathcal%20G%7D%5E1)로 표현한다.  
>>- 첫번째 layer는 ReLU를 activation function으로 사용한다.  
>3. 2nd layer of GCN
>>- 각 노드를 labelled와 unlabelled로 mapping해야하기 때문에 두번째 layer는 sigmoid를 activation function으로 사용한다.  
>>- 따라서 두번째 layer까지 거친 output은 0~1사이의 값을 가지는 길이 ![](https://latex.codecogs.com/gif.latex?N)의 vector이다. (0은 unlabelled, 1은 labelled를 의미)  
>
>전체적인 과정은 아래와 같은 식으로 표현된다.
>![gcn](https://user-images.githubusercontent.com/89853986/163961880-ea5a6f69-1ec4-4657-982f-f5780ee24f0d.PNG)
>
>또한 loss function은 아래와 같다.
>![gcn_loss](https://user-images.githubusercontent.com/89853986/163980668-5362fe71-d151-4810-8a65-2e254dee0912.png)
>
>cross-entropy를 사용하였고, ![](https://latex.codecogs.com/gif.latex?\lambda)는 labelled와 unlabelled cross-entropy간의 weight를 조절하는 parameter이다.

**3.3.3 UncertainGCN: Uncertainty sampling on GCN**  
>위와 같은 방법으로 GCN을 training시키고 난 후 sampling을 진행한다. 
>본 방법에서 unlabelled로 남아있는 data ![](https://latex.codecogs.com/gif.latex?D_U)에 대한 confidence score는 ![](https://latex.codecogs.com/gif.latex?f_%7B%5Cmathcal%20G%7D%28v_i%3BD_U%29)이다.
>일반적인 uncertainty sampling과 유사하게 UncertainGCN도 ![](https://latex.codecogs.com/gif.latex?s_{margin})이라는 변수와 함께 confidence를 기반으로 sampling할 unlabelled image를 고른다.
>기존의 labelled set인 ![](https://latex.codecogs.com/gif.latex?D_L)에서 고정된 ![](https://latex.codecogs.com/gif.latex?b)개를 querying하는 수식은 아래와 같다.
>![uncertaingcn](https://user-images.githubusercontent.com/89853986/163984729-6eca1d63-32a8-4be4-aae5-79d7e566716a.PNG)
>가장 uncertainty가 높은 unlabelled data를 고르려면 ![](https://latex.codecogs.com/gif.latex?s_{margin})을 0과 가깝게 설정하면 된다. (이 경우 0~1 범위의 confidence 값 중 1에 가까운 image들이 선택될 것이다.)
>이 과정이 주어진 budget 내에서 loss가 가장 작아질 때까지 반복되며, 알고리즘의 pseudo code는 아래와 같다.  

![pseudo](https://user-images.githubusercontent.com/89853986/163986800-325ea500-c8e4-41a5-91e8-bafe6ed40a48.PNG)

**3.3.4 CoreGCN: CoreSet sampling on GCN**
>CoreGCN은 ![](https://latex.codecogs.com/gif.latex?l2) distance를 기반으로 첫번째 GCN layer에서 추출된 feature간의 거리를 계산하고, 이를 통해 sampling할 data를 선정한다.  
>기존의 labelled set인 ![](https://latex.codecogs.com/gif.latex?D_L)에서 querying하는 수식은 아래와 같다.  
>![coregcn](https://user-images.githubusercontent.com/89853986/163989195-a0e9bd2f-b5b6-4cb8-939c-b4fb3354aa65.PNG)
>![](https://latex.codecogs.com/gif.latex?\delta)는 labelled node ![](https://latex.codecogs.com/gif.latex?v_i)와 unlabelled node ![](https://latex.codecogs.com/gif.latex?v_j)의 feature 간의 유클리디안 거리를 의미한다.  
>즉, 위의 수식은 labelled data의 feature와 unlabelled data의 feature 간의 가장 큰 거리를 최소로 만드는 unlabelled data point를 sampling하도록 한다.


## **4. Experiment**  

In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

본 논문에서는 크게 3가지 실험을 진행하였다.
- Image classification : RGB, grayscale의 image data 활용 
- Regression : depth image 활용
- Classification : RGB synthetic-generated image 활용

위 실험에 대해 각각 자세히 알아보도록 하자.

### **4.1 Classification**

**4.1.1 Datasets and Experimental Settings**  
* Dataset  
	+ CIFAR-10 (RGB)
	>- 10 classes  
	>- 1000개의 seed labelled datas  
	>- budget : 1000 images  

	+ CIFAR-100 (RGB)
	>- 100 classes  
	>- 2000개의 seed labelled datas  
	>- budget : 2000 images
	>- 다른 data에 비해 class가 많아서 더 많은 seed labelled data와 budget을 부여함  

	+ SVHN (RGB)
	>- 10 classes  
	>- 1000개의 seed labelled datas  
	>- budget : 1000 images  
	
	+ FashionMNIST (grayscale)
	>- 10 classes  
	>- 1000개의 seed labelled datas  
	>- budget : 1000 images  
<br/>

![dataset](https://user-images.githubusercontent.com/89853986/164029000-dfb9120b-2672-465d-9b22-6b5a20078663.PNG)

<br/>


**4.1.2 Implementation details**  
Then, show the experiment results which demonstrate the proposed method.  
You can attach the tables or figures, but you don't have to cover all the results.  
+ 모든 data에 대해 10번의 cycle만큼 실험 진행한다.  
+ Selection을 모든 unlabelled pooled-dataset에 대해 하는 것이 아닌, randomly selected subset ![](https://latex.codecogs.com/gif.latex?D_S%20%5Csubset%20D_U) 에서 진행한다. 이는 dataset에서 중복되는 부분이 여러번 등장하는 것을 피하기 위함이다.  
+ ![](https://latex.codecogs.com/gif.latex?D_S)의 크기는 모든 실험에서 10000으로 설정한다.  
<br/>

- _Learner_  
	+ ResNet-18을 classification model로 사용  
- _Sampler_  
	+ 2 layers GCN을 model로 사용  
	+ ![](https://latex.codecogs.com/gif.latex?%5Clambda%20%3D%201.2)로 설정, 상대적으로 개수가 많은 unlabelled dataset에 더욱 가중치를 주기 위함이다.  
	+ ![](https://latex.codecogs.com/gif.latex?s_%7Bmargin%7D%20%3D%200.1)로 설정  


	
  
**4.1.3 Compared Methods and Evaluation Metric**
* baseline  
	- **Random sampling** : 가장 기본적인 default baseline이다.
	- **CoreSet** : geometric technique 중 가장 좋은 퍼포먼스를 보인다.
	- **VAAL & Learning Loss** : task-agnostic framework에서 SOTA baseline이다.
	- **FeatProp** : GCN based framework에서 대표적인 baseline이다.

* Evaluation Metric
	- Test set에 대한 5번의 실험에서의 mean average accuracy를 바탕으로 evaluate한다.

**4.1.4 Quantitative Comparisons**

ResNet-18로 learner를 구성하여 전체 dataset을 사용하여 training을 시키면 (Active Learning 미사용) 각 dataset에서 아래와 같은 결과가 도출된다. 그리고 이것은 AL 성능의 upperbound일 것이다.  
	- **CIFAR-10** : 93.09%  
	- **CIFAR-100** : 73.02%  
	- **FashionMNIST** : 93.74%  
	- **SVHN** : 95.35%  

![quantitative_classification](https://user-images.githubusercontent.com/89853986/164178431-facc4a46-a3d6-409a-9c5a-5ae3922f708e.PNG)

- 위의 그래프는 각각의 dataset에서 저자가 제시한 UncertainGCN, CoreGCN과 다른 baseline method와의 성능을 비교하여 보여준다.  
- 저자가 제시한 두가지 sampling method 모두 다른 baseline method에 비해 웃도는 성능을 보이는 것을 그래프를 보면 확인할 수 있을 것이다.  
- 주목할만한 점은 CIFAR-100 dataset에서 CoreGCN method를 사용하면 20000개의 sampling으로 대략 69%의 accuracy를 낼 수 있는데, 이는 전체 training dataset을 모두 사용했을 때보다 4%만 낮은 수치이다. 적절한 sampling을 통해 적은 dataset을 가지고(cost 절약) 거의 비슷한 성능을 낼 수 있음을 의미한다.


**4.1.5 Qualitative Comparisons**

실제로 각 sampling method 들이 어떠한 unlabelled data를 sampling하는지를 t-SNE plot을 통해 직접 관찰한다. 
Stage가 진행됨에 따라 확연한 차이를 관찰하기 위해 첫번째 stage와 3단계가 더 진행된 4번째 stage를 plot하면 아래와 같다.

![qualitative_classification](https://user-images.githubusercontent.com/89853986/164183015-94483f1b-97df-4382-a54a-99a797bdb0c1.PNG)

- 첫번째 stage에서는 sampling method 간에 큰 차이가 관찰되지 않는다.  
- Figure 5는 CoreSet과 UncertainGCN을 비교해놓은 그림이다. 4번째 stage에서 select한 sample을 보면, CoreSet에 비해 UncertainGCN은 더욱 class의 경계에 위치하는 sample들(uncertainty가 높은 sample)을 select한 것을 확인 가능하다.  
- Figure 6은 CoreSet과 CoreGCN을 비교해놓은 그림이다. CoreGCN은 geometric information을 기반으로 하기 때문에 sample들이 몰려있는 것을 방지한다. 하지만 uncertain area로부터 message-passing을 받기 때문에 CoreSet처럼 class의 중앙에 위치하는 것은 아니다. CoreGCN은 geometric information과 uncertainty 간의 balance를 고려하여 sampling한다.  


### **4.2 Regression**

**4.2.1 Datasets and Experimental Settings**

* Dataset  
	+ ICVL (hand depth-images)
	>- 16004개의 training set과 1600개의 test set
	>- 매 selection stage에서 training data의 10%를 ![](https://latex.codecogs.com/gif.latex?D_S)로 설정
	>- 매 selection stage에서 100개의 unlabelled data를 select

<br/>

**4.2.2 Implementation details**

- _DeepPrior_ 를 learner로 사용  
- Sampler 등의 다른 요소들은 위의 classification task 때와 동일하게 유지  
- Detecting hands, centre, crop 그리고 image resize를 위해 U-Net을 사용하여 pre-train  

**4.2.3 Compared Methods and Evaluation Metric**

* baseline  
	- **Random sampling** : 가장 기본적인 default baseline이다.
	- **CoreSet** : geometric technique 중 가장 좋은 퍼포먼스를 보인다.

* Evaluation Metric
	- Test set에 대한 5번의 실험에서의 mean squared error의 평균과 standard deviation을 evaluation metric으로 사용하여 비교한다.

**4.2.4 Quantitative Evaluation**

![quantitative_regression](https://user-images.githubusercontent.com/89853986/164215933-ba9a9f4f-ae25-4d1b-b5cd-5820b1577c81.PNG)

- ICVL dataset을 가지고 4가지 방법으로 실험한 결과를 나타낸 그래프이다.   
- CoreGCN과 UncertainGCN이 second stage부터 다른 방법에 비해 낮은 mse를 보이며, 각각 6번째, 5번째 selection stage까지 급격히 감소하는 것을 볼 수 있다.  
- 이는 매우 제한된 budget 내에서도 저자가 제안한 두 방법이 다른 방법들에 비해 좋은 성능을 보일 수 있다는 것을 보여준다.  

### **4.3 Sub-sampling of Synthetic Data**

* Dataset  
	+ RaFD 
	>- StarGAN을 사용하여 generate한 face expression이다.
	>- GAN이 실제와 매우 유사한 generated image를 생성해주지만, 모든 generated image를 바로 train data로 사용하기에는 무리가 있다.
	>- 따라서 정확하고, 의미있는 data를 select할 필요가 있다.

* Result

![synthetic](https://user-images.githubusercontent.com/89853986/164217784-aaff2175-e1f7-4a43-a961-5e500f8ac43d.PNG)

- Random sampling에 비해 UncertainGCN이 더 작은 variance와 함께 더 좋은 accuracy를 보이고 있다. 
- Model을 train하기 위해 적은 수의 synthetic example만이 useful하다.

## **5. Conclusion**  

- GCN based의 task-agnostic한 sampling method를 제시하였다. 
- Image의 feature를 기반으로 node를, similarity를 기반으로 edge를 표현하여 graph를 생성하고, message-passing을 표현할 수 있는 GCN을 적용한다.
- 저자가 제안한 UncertainGCN, CoreGCN은 6개의 data에 대하여 SOTA 결과를 도출하였다.

* Key Idea
	learner를 통해 1차적으로 각 data의 feature를 추출한다음 각각의 similarity를 고려하여 graph domain으로 변경하여 message-passing이 가능하도록 sampler를 design했다.

---  
## **Author Information**  

* Author name : **Razvan Caramalau**
    * Affiliation  
    	Imperial College London
    * Research Topic  
	Deep Learning, Active Learning, 3D Hand Pose Estimation, Graph Neural Network


## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation  
* Reference
