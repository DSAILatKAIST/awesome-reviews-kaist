---
Zhao Tong, et al./ Data Augmentation for Graph Neural Networks / AAAI-2020
---

# **Data Augmentation for Graph Neural Networks** 

[논문링크](https://www.aaai.org/AAAI21Papers/AAAI-10012.ZhaoT.pdf)



## **1. Problem Definition**  

CV와 자연어처리에서 Data Augmentation은 매우 범용성이 넓으며 그 성능 역시 우수하여 많은 연구가 진행되고 있습니다. 그러나 논문에서 언급하듯이 Graph 분야에서의 Data Augmentation은 그것이 가지고 있는 여러가지 제약 때문에 타 데이터사이언스 분야에 비해 범용성이 낮다고 할 수 있는데요, 해당하는 제약은 다음과 같습니다.

- **Data space가 Non-Euclidean 성질을 띄기 때문에 비정형적이다**
- **Samples 사이에 의존성이 크다**

위 두가지 요인이 Graph에서의 Data Augmentation을 어렵게 만들며, 특히 첫 번째 제약에서 파생된 irregularity(비정형성)는 새로운 데이터에 대한 정의를 쉽게 이끌어내지 못합니다. 따라서 Graph에서의 Node Classification 성능 개선이라는 것을 목표로 Data Augmentation을 사용하기 위해서는 다른 데이터 셋과는 다른 접근 방법이 필요합니다.



## **2. Motivation**  

Graph 구조에서의 Data Augmentation의 가장 적합한 방법은 Node를 추가 및 제거하거나 Edge를 추가 및 제거하는 것입니다. 그러나 Node는 label이라는 특성을 가지고 있을 수 있기 때문에 그것을 추가하기 위해서는 라벨링과 피처를 설정해야 하는 과제가 있으며, 제거할 경우 Graph의 Data availability를 감소시키기에 적절하지 않습니다.

따라서 Node와 Node를 연결하는 Edge를 추가 및 제거하는 방법이 Graph Data Augmentation을 위한 가장 좋은 선택이라고 할 수 있습니다. 이때 어떠한 Edge를 선택하느냐는 또 다른 과제가 될 수 있습니다. 이를 위한 기존의 연구들은 다음과 같습니다.

- **DROPEDGE(Rong et al. 2019)**는 각 학습 epoch 이전에 Graph의 Edge를 랜덤하게 제거하는 방식으로 test-time 추론에 의의를 가지고 있으나, Edge를 추가하는 방안에서는 큰 영향을 얻지 못함
- **ADAEDGE(Chen et al. 2019)**는 Graph의 Data Augmentation에 대해서 동일한 Label이라고 높은 신뢰도를 가지고 분류되는 두 노드 사이에 반복적으로 EDGE를 추가하는 방식임. 그러나 이 방식은 오류를 전파시키기 쉽고, 학습 데이터의 크기에 영향을 받음
- **BGCN(Zhang et al. 2019b)**는 여러 개의 노이즈가 제거된 Graph를 생성하는 GCN 기반 stochastic block model을 반복적으로 훈련시키고, 이러한 GCN 결과들을 앙상블하는 방식임. 그러나 이 역시 오류 전파의 위험성을 가지고 있음

위 논문에서 제시한 핵심적인 idea는 같은 분류를 가지고 있는 Node Structure를 효과적으로 인코드하기 위해 intra-class(분류가 같은 노드 관계)의 edge들은 늘리고, inter-class(분류가 다른 노드 관계)의 edge들은 줄인다는 것입니다. 이는 직관적으로 same-class Node들끼리 임베딩되는 것을 장려하고 other-class Node에 대한 임베딩을 차별화하여 구분을 확실히 하는 효과를 가집니다.

이를 쉽게 보여주는 Graph 구조에 대한 GCN performace Figure는 다음과 같습니다.
<a href='https://ifh.cc/v-zFclGl' target='_blank'><img src='https://ifh.cc/g/zFclGl.png' border='0'></a>
![image-20220424122914040](.gitbook/2022-spring-assets/sejong/image-20220424122914040.png)

(a)는 Edge의 추가 및 제거가 없는 원본 그래프 구조이고, (b)는 랜덤하게 Edge를 변형시킨 구조입니다. (c)는 위 논문에서 제시한 idea를 기반으로 만든 GAUG 모델의 구조이며, 마지막으로 (d)는 이상적으로 Class간 구분이 확실하도록 Edge의 변형이 이루어진 그래프 구조입니다. 파란색 점선은 Edge의 제거를 의미하고 파란색 실선은 Edge의 추가를 의미하며, F1 score는 **Class의 구분이 드러나도록 Edge가 연결된 구조일수록 높아지는 것**을 확인할 수 있습니다(이 때 각각의 Method M과 O는 아래에 기술할 *Modified-Graph Setting*과 *Original-Graph Setting*을 의미).

### Theoretical reasons

위와 같이 intra-class의 Edge 생성을 장려하고 inter-class의 Edge는 제거하는 전략은 label이 완벽하다는 가정하에 GNN에서 매우 자명한 분류방법이라 할 수 있습니다. 만일 극단적으로 모든 intra-class Edge가 연결되고 inter-class Edge가 존재하지 않는다면 이는 ![](https://latex.codecogs.com/svg.image?k)개의 class를 가지고, 각 component의 모든 Node들이 같은 label을 갖는 ![](https://latex.codecogs.com/svg.image?k)-fully connected components라고 할 수 있을 것입니다. 이는 다음과 같은 Theorem을 통해 GNN이 이상적인 그래프에서 학습된 임베딩을 쉽게 분류할 수 있도록 합니다.

![image-20220424141833846](C:\Users\Sejong Lim\AppData\Roaming\Typora\typora-user-images\image-20220424141833846.png)

이 결과 _class-homophilic_(동일성) Graph ![](https://latex.codecogs.com/svg.image?G)의 경우 학습에서의 분류 문제는 매우 쉽게 변환될 수 있습니다. 그러나 너무 작위적인 그래프 구조 변형은 학습시 과적합을 유발할 수 있으며, 이를 극복하기 위해서 논문에서는 이상적인 그래프 ![](https://latex.codecogs.com/svg.image?G)와 유사하게 근사시킨 ![](https://latex.codecogs.com/svg.image?G_m)을 만들어냅니다.




## **3. Method**  

### Modified and Original Graph Settings for Graph Data Augmentation

Method에 앞서 Graph Modification의 기본 원리로 CV에서 사용되는 이미지 Data Augmentation 방법을 사용하는데요, 두가지 프로세스를 거친다고 볼 수 있습니다.

1) Applying a transformation S to T 
   $$
   f : S → T
   $$

2. Utilizing S with T for model training
   $$
   S\cup T
   $$

이를 통해 CV에서의 Data Augmentation과 마찬가지로 기본으로 주어진 데이터를 변환시키고, 원본 데이터와 변환된 데이터를 합쳐 모두 학습에 이용할 수 있게 되었습니다. 이와 같은 세팅을 거친 후에 그래프 데이터의 및 변환 함수 설정을 기준으로 다음과 같은 두 가지 방법으로 나누어서 학습을 진행할 수 있습니다.

- *Modified-Graph Setting*
- *Original-Graph Setting*

위 두가지 방법 모두 기본적인 핵심 아이디어는 그래프에 내재된 정보를 활용하여, 존재하지 않지만 존재할 Probability가 높은 Edge를 생성하고, 존재하지만 노이즈를 증가시키는 Edge를 제거한다는 기준을 가지고 성능을 향상시킨다는 것입니다. 이에 대해서 아래에서 좀 더 자세히 기술해보겠습니다.



### GAUG-M for Modified-Graph Setting

먼저 Modified-Graph Setting의 경우, Edge predictor fuction을 통해 원본 그래프 ![](https://latex.codecogs.com/svg.image?G)에 있는 모든 Edge가 예측 확률을 구할 수 있습니다. 이를 기반으로 새로운(기존의) Edge를 추가(제거)하여 수정된 그래프 ![](https://latex.codecogs.com/svg.image?G_m)을 만들고 이를 GNN node-classifier의 인풋 값으로 사용할 수 있습니다.

Edge predictor는 **![](https://latex.codecogs.com/svg.image?f_{ep}&space;:&space;A,&space;X&space;\to&space;&space;M) **와 같은 식으로 정의할 수 있는데요, input 그래프 데이터인 ![](https://latex.codecogs.com/svg.image?A)와 ![](https://latex.codecogs.com/svg.image?X)를  통해 각 Edge의 probability Matrix ![](https://latex.codecogs.com/svg.image?M)을 나타냅니다.  이때 ![](https://latex.codecogs.com/svg.image?M)의 각 원소 ![](https://latex.codecogs.com/svg.image?M_{u,v})는 Edge u와 v의 예측 확률을 나타냅니다. 수식은 아래와 같습니다.
$$
M=\sigma (ZZ^{T}))\;where\;Z=f^{(1)}_{GCL}(A,f^{(0)}_{GCL}(A,X)
$$
위 수식에서 Edge probability를 계산할 때 Graph Auto-Encoder(GAE) 방법을 사용하게 되는데요, 해당 GAE는 two layer GCN 인코더와 한 개의 inner-product 디코더로 이루어져 있습니다. 이는 수식에서 간단히 확인할 수 있으며 GCL은 Graph Convolution Layer를 나타내어 ![](https://latex.codecogs.com/svg.image?Z)를 표현하는 인코더로 작용합니다.

![image-20220424193457622](C:\Users\Sejong Lim\AppData\Roaming\Typora\typora-user-images\image-20220424193457622.png)

위 그림은 GAUG-M에서 Edge predictor를 사용하였을 때와 Random하게 학습하였을 때의 차이를 보이고 있는데요, 직관적으로 보면 Intra-class로 연결된 Edge의 개수와 F1-Score는 정비례하고 Inter-class로 연결된 Edge의 개수와 F1-Score가 반비례하는 것을 확인할 수 있습니다. 랜덤으로 진행하였을 경우 Intra-class보다 Inter-class를 많이 증가시키는 경우나 Inter-class보다 Intra-class를 더 많이 감소시키는 경우가 발생하는데, 이때의 성능은 두 경우 모두 떨어지며 특히 후자의 경우가 매우 저조한 성능을 보입니다.

GAUG-M은 이와 같이 변환 연산에 GAE 방법을 사용하여  ![](https://latex.codecogs.com/svg.image?G_m)를 만들고 이를 학습과 추론에 적용하는 방식으로 Graph Augmentation을 활용하고 모델의 성능을 끌어올립니다. 또한 GAUG-M은 이것과 연계된 GNN architecture의 소비시간과 공간 복잡도를 동일하게 공유합니다.



### GAUG-O for Original-Graph Setting

Original-Graph Setting 방식은 다음의 세 가지 구성요소로 이루어져 있습니다.

- Edge probability를 측정할 수 있는 미분가능한 Edge predictor
- 희소 그래프 변형을 생성하는 보간 및 샘플링 step
- 위 변형을 사용하여 Node 분류를 위한 임베딩을 학습하는 GNN

첫 번째 구성요소인 Edge predictor는 GAUG-M과 마찬가지로 GAE 방법을 사용하여 연산을 진행할 수 있습니다. 

두 번째 구성요소인 보간 및 샘플링 step은 Edge predictor가 원본 그래프 인근에서 벗어나는 것을 방지하기 위해서 예측된 변형 그래프 ![](https://latex.codecogs.com/svg.image?M)과 원본 그래프 ![](https://latex.codecogs.com/svg.image?A)를 ![](https://latex.codecogs.com/svg.image?\alpha&space;)와 ![](https://latex.codecogs.com/svg.image?1-\alpha&space;)로 combination 하여 인접행렬 ![](https://latex.codecogs.com/svg.image?P)를 만들고,  ![](https://latex.codecogs.com/svg.image?P)에 대한 비선형 함수로 새로운 인접행렬 ![](https://latex.codecogs.com/svg.image?A')을 생성합니다. 이를 식으로 표현하면 아래와 같습니다.
$$
A'_{ij}=[\frac{1}{1+e^{-(logP_{ij}+G)/r}}+\frac{1}{2}]
\\where\quad P_{ij}=\alpha M_{ij} + (1-\alpha)A_{ij}
$$
여기서의 ![](https://latex.codecogs.com/svg.image?r)는 Gumble-Softmax 분포의 파라미터이고, ![](https://latex.codecogs.com/svg.image?\alpha&space;)는 원본 그래프의 영향도를 추정할 수 있는 하이퍼파라미터 입니다.

마지막 구성요소인 GNN에는 위를 통해 얻은 인접행렬 ![](https://latex.codecogs.com/svg.image?A')가 통과하여 Node를 분류할 수 있도록 합니다. 이 때 역전파 알고리즘을 이용하여 Node-classification의 loss ![](https://latex.codecogs.com/svg.image?L_{nc})와 Edge-prediction의 loss ![](https://latex.codecogs.com/svg.image?L_{ep})을 구할 수 있으며, 이를 종합한 loss ![](https://latex.codecogs.com/svg.image?L) 역시 사용할 수 있습니다. 해당 식은 아래와 같습니다.
$$
L = L_{nc} + \beta L_{ep},\\where \; L_{nc} = CE(\check{y},y)\\and \; L_{ep}= BCE(\sigma (f_{ep}(A,X)),A)
$$
위 식에서의 ![](https://latex.codecogs.com/svg.image?\beta&space;)는 reconstruction(재건) loss에 할당된 하이퍼파라미터이고, ![](https://latex.codecogs.com/svg.image?\sigma)는 기본 시그모이드 함수이며, ![](https://latex.codecogs.com/svg.image?y,\check{y})는 실제 Node class label 및 예측 확률을 표현합니다. 마지막으로 ![](https://latex.codecogs.com/svg.image?BCE/CE)는 표준(이진) cross-entropy loss를 의미합니다. 

위 구성요소들을 바탕으로 GAUG-O를 단계적으로 표현하면 다음 그림과 같습니다.

![image-20220424205600551](C:\Users\Sejong Lim\AppData\Roaming\Typora\typora-user-images\image-20220424205600551.png)

**Input Graph → Neural Edge Predictor → Interpolation and Sampling → Graph Neural Network Node Classifier** 



## **4. Experiment**  

### **Experiment setup**  

위 논문에서의 GAUG-M과 GAUG-O의 성능을 평가하기 위한 Experiment Set up은 다음과 같습니다. GNN 아키텍쳐와 데이터셋, Method를 각각 나누어 성능을 평가하였습니다.

#### Dataset

논문에서는 총 6개의 데이터 셋을 사용하였으며 해당 데이터는 다음과 같습니다.

- Citation networks (**CORA**, **CITESEER** (Kipf and Welling 2016a))
- Protein-protein interactions (**PPI** (Hamilton, Ying, and Leskovec 2017))
- Social networks (**BLOGCATALOG**, **FLICKR** (Huang, Li, and Hu 2017)),
- Air traffic (**AIR-USA** (Wu, He, and Xu 2019))

해당 데이터를 split 할 때 <u>**학습/검증/테스트**</u>의 순서대로 <u>**10/20/70%**</u>의 비율로 진행하였으며, 위 데이터에 대한 구체적인 구성은 아래 표와 같습니다.

![image-20220424222550325](C:\Users\Sejong Lim\AppData\Roaming\Typora\typora-user-images\image-20220424222550325.png)

#### Baseline & Evaluation Metric

위 논문에 대한 Baseline과 Evaluation Metric 설정에 관한 내용입니다.

다양한 아키텍쳐 환경에서의 모델 평가를 위해 GAUG-M과 GAUG-O 둘 다 가장 범용성이 높은 4개의 GNN 아키텍쳐에서 평가하였으며 그것들은 다음과 같습니다. 

- GCN (Kipf and Welling 2016a)
- GSAGE (Hamilton, Ying, and Leskovec 2017)
- GAT (Velickovi ˇ c et al. 2017)
- JK-NET (Xu et al. 2018b)

또한 GAUG-M과 GAUG-O의 성능을 기존의 GNN 성능과 비교하기 위해  다음 baseline을 함께 평가하였습니다.

- **GAUG-M**
  - ADAEDGE (Chen et al. 2019) (modified-graph)
  - BGCN (Zhang et al. 2019b) (modified-graph)
- **GAUG-O**
  - DROPEDGE (Rong et al. 2019)

성능의 평가지표로는 **micro-F1 score**가 사용되었습니다.



### **Result**  

위 Baseline & Evaluation Metric 설정을 바탕으로 나타낸 평가지표 Table 입니다. 

![image-20220424224236954](C:\Users\Sejong Lim\AppData\Roaming\Typora\typora-user-images\image-20220424224236954.png)

대부분의 경우에서 GAUG-M과 GAUG-O의 성능이 Original Method보다 개선이 되었으며 평균적인 여타 Method를 상회하는 것을 알 수 있습니다.



아키텍쳐 별 개선 수치를 비교하면 다음과 같습니다.

- **GAUG-M**
  - GCN: 4.6%
  - GSAGE: 4.8%
  - GAT: 10.9%
  - JK-NET: 5.7%
- **GAUG-O**
  - GCN: 4.1%
  - GSAGE: 2.1%
  - GAT: 6.3%
  - JK-NET: 4.9%



데이터 셋 별 개선 수치를 비교하면 다음과 같습니다.

- **GAUG-M**

  - CORA: 2.4%
  - CITESEER: 1.0%
  - PPI: 3.1%
  - BLOGC: 5.5%
  - FLICKR: 19.2%
  - AIR-USA: 7.9%

- **GAUG-O**

  - CORA: 1.6%
  - CITESEER: 2.5%
  - PPI: 11.5%
  - BLOGC: 3.6%
  - FLICKR: 2.2%
  - AIR-USA: 4.7%

  


## **5. Conclusion**  

### Summary

그래프의 성능개선을 위한 Data Augmentation은 그래프 데이터의 특성상 비정형성을 포함하고 있기 때문에 Computer Vision이나 NLP 분야에 비해 범용성이 높지 않았습니다. 그러나 위 논문에서는 Edge에 대한 추가 및 삭제 행위를 통해 데이터 변환을 이끌어냈고, 해당하는 Edge의 선택을 Edge predictor라는 신경망을 사용하여 결정하였습니다. 이 때 변환함수와 데이터 셋 구조 등의 차이를 기반으로  GAUG-M과 GAUG-O 두가지 모델을 개발 및 평가하였으며 이에 대한 성능은 기존의 GNN 모델들과 비교하였을 때 높은 수치를 보였으며, 아키텍쳐에 이슈에 강건함을 보였습니다.



### Opinion

개인적으로 AutoEncoder에 대한 강의를 듣고 나서 Data Augmentation 기법을 이용해 이미지를 변환시키는 과정이 흥미로웠습니다. 이를 통해 타 분야에서 Data Augmentation은 어떻게 적용되고 있을지에 대해 궁금했고, 위 논문을 찾게 되었습니다. 어떠한 제한사항이 있고, 이를 극복하기 위해 무슨 과정을 거쳤는지 논문에서 쉽게 밝히고 있다고 생각합니다. 또한 성능적으로도 매우 좋은 Contribution을 보인 것 같다고 생각하며, 이를 다양한 아키텍쳐 환경에서 모델과 비교할 수 있어서 객관적인 수치를 이해하기 좋은 실험이었다고 생각합니다.



---
## **Author Information**  

* Sejong Lim
    * Dept. of Industrial and System Engineering, KAIST
    
    * Multi-Agent RL / Reinforcement Learning / VRP
    
      
    

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* CS224W: Machine Learning with Graphs
* [github링크](https://github.com/zhao-tong/GAug)
