# 1. Problem Definition
Deep clustering은 딥러닝을 활용한 군집화 방법으로 최근 대두되고 있다. Deep Clustering은 유용한 잠재 대표성(useful representation)을 군집화에 활용할 수 있다는 장점이 있지만, 아쉽게도 데이터 샘플들의 관계성에 대한 정보를 활용하지는 못한다. 이를 해결하고자 데이터의 구조 정보를 Deep Clustering에 결합한 Structural Deep Clustering Network(SDCN)을 소개한다.
  
# 2. Motivation
- 딥러닝의 뛰어난 대표성(representation)을 활용한 군집화(clustering)가 활용되고 있다. 
- 하지만, 이들은 데이터의 구조(=관계성)를 고려한 군집을 하지는 않는다. 
- 데이터의 구조 정보를 고려하기 위해 GCN(Graph Convolutional Network)을 활용한다. 
- Deep clustering 과 GCN을 결합한 SDCN에 대해 소개한다. 

# 3. Method
비교적 익숙한 GCN에 비해 Deep clustering은 익숙하지 않은 주제이기에 본격적인 Method에 대한 설명에 앞서 Deep clustering에 대해 설명한다.

## 3-1. Deep Clustering

\
![](https://github.com/Mos-start2092/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Seongbin_1/fig1.png?raw=true)\



$$
\text{<Figure 1>}
$$




Figure1에서 볼 수 있듯 Deep Clustering의 최종 목적은 딥러닝으로 추출한 잠재변수를 활용하여 군집화를 진행하는 것이다. 그렇다면 어떤 방식으로 군집화에 최적화된 잠재변수를 생성할 수 있을까? 

잠재변수를 업데이트 방식은 다음과 같다. 

- Auto Encoder를 활용하여 잠재변수 h를 추출한다. 
- h를 활용하여 K-means를 실시한다.
- i 번째 데이터가 j 번째 군집에 들어갈 확률을 활용하여 t분포를 그린다.
  (이때 t분포를 그리는 수식 q는 t-SNE를 그리는 수식과 동일하다. )
- q를 데이터로 부터 얻은 분포라고 생각하면 True distribution으로 생각되는 p를 생성한다.
- q와 p를 활용한 KL-divergence를 구하고 Loss of clustering으로 사용한다.
- 재생성된 데이터와 기존의 데이터를 기반으로한 Loss of reconstruction을 구한다. 
- Loss of reconstruction 과 Loss of clustering을 결합하여 오차 역전차에 사용할 최종 Loss를 구한다.
-  Loss를 minimization하는 방향으로 오차역전파를 사용하여 z를 생성하는데에 사용된 가중치와 편향을 조정한다.
-  위의 과정을 반복하여 군집화(k-means)에 최적화된 잠재변수를 생성한다.

위의 과정은 K-means가 distance를 기반으로하는 군집화 방법임과 중심점들(centroids)을 평균으로 가지는 Gaussian Mixture 모델로 나타낼 수 있음을 활용한 방법으로 정확한 내용은 [1]을 참고하기 바람.
$$
L_{clu}= KL(P\mid \mid Q)=\sum_{i}\sum_{j} p_{ij} log \cfrac{p_{ij}}{q_{ij}} \\
\\
$$
$$
,\;q_{ij}=\cfrac{(1+(h_{i}-\mu_{j})^2)^{-1}}{ \sum_{j'}(1+(h_{i}-\mu_{j'})^2)^{-1}}\;,\;\;\mu_i\;:\;\text{centroid of j-th cluster} \\
$$
$$
,\; p_{ij}=\cfrac{q_{ij}/f_j}{\sum_{j'}q_{ij}^2/f_{j'}} \;\;,\;f_j=\sum_{i}q_{ij} 
$$
- q : i번째 (잠재변수 형태)데이터가 j번째 군집에 들어갈 확률을 나타낸것으로 t분포를 따른다. 
- p : true districution으로 자세한 내용은 [1]을 참고 바람.
- L_clu : p, q 의 KL-divergence를 활용한 K-means에 대한 Loss

### 잠재변수
Auto Encoder를 활용하여 잠재변수를 생성하는 이유는 잠재변수가 데이터에 대한 대표성을 나타내야하기 때문이다. 이는 다양한 다른 모델을 활용하여 생성할 수도 있지만, 해당 논문에서는 일반화를 위하여 Basic AutoEncoder를 활용하였다. 

#
## 3-2. KNN Graph
군집화에 데이터의 구조를 포함하기 위해서는 데이터를 그래프 형태로 나타낼 필요가 있다. 이를 위해 KNN Graph 방법을 이용한다. 

KNN 방법에 기반을 둔 그래프를 형성하는 방법으로 가장 비슷한 데이터들을 노드로 하여 연결하여 그래프를 생성하는 방법이다. 이때 데이터간의 간의 유사성(similarity)는 다음과 같은 두가지 방법으로 측정한다. 

1) Heat Kernel
$$
S_{ij}=e^{-\cfrac{\mid\mid x_i-x_j \mid \mid ^2 }{t}}\;,\;\;t=\text{time parameter}
$$
2) Dot-product
$$
S_{ij}=x_{j}^Tx_{i}
$$

위의 방식으로 유사성 행렬(similarity matrix) S응 생성하고 각 샘플에서 가장 가까운 K 개의 데이터를 선정하여 K-nearest neighbor graph를 생성한다. 이러한 방식으로 우리는 인접행렬(Adjacency matrix) A를 생성할 수 있다. 
$$
\\
$$



## Deep clustering과 KNN Graph를 생성하는 방법을 설명했으나 다음과 같은 두가지 문제가 남아있다. 



 ### 1. 어떠한 구조에 대한 정보를 고려해야하는가?
일반적으로 데이터의 구조는 매우 복잡하다. 직접적인 관계뿐만 아니라 고차원적인 구조(high-order structure) 존재한다. 어떻게 효율적으로 다양한 구조를 고려할지를 해결할 필요가 있다. 

### 2. Deep clustering과 구조의 정보 사이의 관계가 무엇인가?
Deep clustering은 각 층(layer)마다 다른 잠재 정보를 가진다. 어떻게 데이터의 구조를 Deep clustering의 구조에 결합시킬지는 중요한 문제이다. 
#
## 3-3. GCN Module
이 section에서는 어떻게 GCN을 Deep clustering에 결합하여 사용할 수 있는지 살펴본다.
<p align="center">
 <img src="/Users/anseongbin/Desktop/수업자료/fig2.png">
 </p>

$$
\text{<Figure 2>}
$$
우선 Figure2을 통해서 전체적인 framework 를 보자. H 는 Deep clustering에서 각층에서의 잠재 정보를 의미하며 Z는 각 층에 해당하는 구조적 정보(Structure information)을 나타낸다. H는 다음과 같은 식으로 이루어진다. 
$$
H^{(l)}=\phi(W_{e}^{(l)}H^{(l-1)}\;+\;b_{e}^{(l)}), \;\; \phi :\;\text{activation function},\;W_e\;:\;weight of encoder 
$$
기존의 GCN은 다음과 같은 식을 통해 형성된다. 
$$
Z^{(l)}=\phi (\; \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}Z^{(l-1)}W^{(l-1)}\;),\\
$$
$$
,\tilde{A}=A+I\;,\;I:\text{identity matrix}\;\rightarrow\; \text{self-loop in each node}\\ 
$$
$$
,\tilde{D}_{ii}=\sum_{j}\tilde{A}_{ij} \rightarrow normalize
$$
하지만 현재의 GCN 과정은 H에 대한 정보도 포함해야하기에 다음과 같은 형태의 식으로 변형된다. 
$$
Z^{(l)}=\phi (\; \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}\tilde{Z}^{(l-1)}W^{(l-1)}\;),\\
$$
$$
\tilde{Z}^{(l-1)}=(1-\epsilon)Z^{(l-1)}\;+\;\epsilon H^{(l-1)} \rightarrow \text{combine} \; H^{(l-1)} \;\&\; Z^{(l-1)}

$$

GCN module의 마지막 층은 softmax function을 사용한다. 
$$
Z=softmax(\; \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}Z^{(L)}W^{(L)}\;)
$$

#
## 3-4. Dual Self-Supervised Module
Deep clustering이 학습하는 방식에 대해서는 3-1 에서 설명하였다. 하지만 Deep clustering에 결합된 GCN의 가중치 역시 군집화의 목적에 맞게 학습될 필요가 있기에 이를 Dual self-Supervise Module을 활용하여 학습한다. 이때 GCN에 관한 Loss 는 다음과 같다. 
$$
L_{gcn}=KL(P\mid\mid Z)=\sum_{i}\sum_{j} p_{ij} log\cfrac{p_{ij}}{z_{ij}}\\
z_{ij}\;:\; \text{probability data i to be assigned cluster j}
$$ 

또한 3-1에서 설명한 것과 달리 해당 논문에서는 Auto-Encoder를 초기 잠재변수에 더해 전체적인 과정에서 사용하므로 reconstruction error 역시 오차역전파로 사용하여 decoder의 가중치를 구한다. 
$$
L_{res}=\cfrac{1}{2N}\mid\mid X-\hat{X}\mid\mid _{F}^2\;,\; \hat{X} \;:\;\text{data}\;i\;\text{reconstructed by} \; w_d\; \& \;b_d 
$$

위의 Loss들을 사용하여 다음과 같은 과정으로 가중치 및 편차를 업데이트한다. 

<p align="center">
<img src="/Users/anseongbin/Desktop/수업자료/fig3.png">
</p>

위의 최종 Loss를 구하는 과정에서의 하이퍼 파라미터는 다음과 같은 조건을 가진다. 
$$
\alpha>0,\;\; \beta>0 \;\;\rightarrow \text{coefficient that controls the disturbance of GCN to the embedding space}
$$

# 4. Experiment
## 4-1. Experiment set up
- Dataset
   
   총 6개의 데이터를 사용하였고 데이터에 관한 내용은 다음 표와 같다. 

<p align="center">   
<img src="/Users/anseongbin/Desktop/수업자료/fig4.png">
</p>
  
  데이터에 대한 자세한 내용은 논문을 참고 바람.
- Baseline
  
  다음의 8가지 모델을 활용하여 군집화를 실시하였고 성능을 비교하였다. 
  
  - K-means : raw data에 전통적인 k-means 방법을 활용한 것이다. 
  - AE : Auto-Encoder의 결과 잠재변수에 k-means를 적용한 방법이다. 
  - Dec[1] : 3-1 에서 Loss of clustering 만을 활용한 Deep Clustering방법이다. 첫 잠재변수만 AE를 통해서 추출하고 그 이후론 encoder 층만 DNN처럼 오차역전파로 학습한 방법이다.
  - IDEC[2] : DEC에 Loss of reconstruction을 추가한 방법으로 SDCN에서 Graph를 제외한 방법이다. 
  - GAE & VGAE[3] : GCN만을 군집화에 활용한 방법으로 데이터의 구조만을 활용한다. 
  - DAEGC[4]: Attention Network 를 사용하였으며 graph clustering을 위해 Loss of clustering을 활용한 방법이다.
  - SDCN_Q : Loss of clustering 만을 활용한 SDCN방법이다.
  - SDCN : 위의 알고리즘에서 제시된 방법이다. 
  
  
   모델별로 세부적인 파라미터 조정이 다 다르기에 논문 참고를 바람.

- Evaluation Metric
  
  다음의 4가지 Metric을 활용하여 모델을 비교 및 평가한다. 
  - Accuracy(ACC)
  - Normalized Mutual Information(NMI)
  - Average Rand Index(ARI)
  - F1-score(F1)
  
  4가지 방법 전부 값이 클수록 좋은 성능임을 나타낸다. 
 
<p align="center">
<img src="/Users/anseongbin/Desktop/수업자료/fig5.png">
</p> 

## 4-2. Result
위의 결과표를 해석한 주요내용은 다음과 같다. 

- 대부분의 모델에서 SDCN과 SDCN_Q의 성능이 좋음을 볼 수 있다. 
- 특히, SDCN이 SDCN_Q보다 일반적으로 좋은 성능을 보인다. 
- Reuters 데이터에서는 SDCN_Q의 성능이 좋으나 KNN Graph 자체의 문제로 여겨진다. 
- Graph 형태의 데이터에서는 Graph 기반의 군집화가 일반 Deep Clustering 방법보다 좋은 성능을 보인다. 
  
실험을 해보니 모델은 Deep Clustering(DC)과 GCN을 결합한 모델이다 보니 다음과 같은 사항에 따라 결과가 달라짐을 볼 수 있었다. 

- Layer의 수 : 잠재변수를 생성하기까지의 층의 수에 따라 결과가 다르게 나타났다. 

<p align="center">
<img src="/Users/anseongbin/Desktop/수업자료/fig6.png">
</p>

- DC와 Graph 의 반영비율 : 3-3 에서 GCN 과정내에서 epsilon으로 H와 Z의 비율을 조정하는 것을 보았다. 그 비율에 따라 값이 상당히 다름을 아래 그래프에 나타냈다. 0일경우 graph 정보만을 사용하는 것이기에 압도적으로 낮은 성능이 확인된다. 

<p align="center">
<img src="/Users/anseongbin/Desktop/수업자료/fig7.png">
</p>

- K의 수 : KNN Graph를 그리는 과정에서 K의 개수가 군집화에 영향을 줌을 알 수 있고 아래 그래프에서 확인가능한다. 

<p align="center">
<img src="/Users/anseongbin/Desktop/수업자료/fig8.png">
</p>

# 5. Conclusion
Deep learning을 활용한 clustering방법에 데이터의 구조적 정보를 추가하여 더 뛰어난 군집화 방법을 보여준 논문이다. Auto Encoder를 기본적으로 사용하며 각 층마다의 데이터 구조를 KNN Graph 방식으로 생성한뒤 GCN방식으로 학습하였다. 이 과정에서 dual self-supervised module을 사용하여 DNN과 GCN 두가지 과정을 효율적으로 결합시켰다. 

통계쪽으로 연구를 하고 있기에 항상 independent 조건 등을 고려하고 자기상관성 등의 문제를 해결하기 위해 노력했어야 했는데 해당 논문을 보며 그러한 이러한 문제에 대한 새로운 관점을 가지게 되었으며 실제 연구에도 활용해볼 수 있을 것으로 기대된다. 

# Author Infomation
- 안성빈
  - KAIST ISYSE
  - Statistics, Data Science

# 6. Reference & Additional Materials
[1] http://proceedings.mlr.press/v48/xieb16.html

[2] https://www.researchgate.net/profile/Xifeng-Guo/publication/317095655_Improved_Deep_Embedded_Clustering_with_Local_Structure_Preservation/links/59263224458515e3d4537edc/Improved-Deep-Embedded-Clustering-with-Local-Structure-Preservation.pdf

[3] https://arxiv.org/abs/1611.07308

[4] https://arxiv.org/abs/1906.06532
