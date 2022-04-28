---
description : Victor Garcia Satorras / E(n) Equivariant Graph Neural Networks / ICML-2021
---

# E(n) Equivariant Graph Neural Networks

이 논문에서는 rotations, translations, reflections and permutations에 대해서 equivariant한 graph neural network (GNN)을 학습할 수 있는 모델을 제안한다. 기존의 방법론들은 위의 property를 얻어내기 위해서, intermediate layer에서 복잡한 계산이 필요했지만, 이 논문에서는 그런 것 없이 비슷하거나 더 좋은 성능을 얻어내었다고 한다. 또한, 3차원 까지의 공간에서만 위의 equivariant를 보장할 수 있었던 기존 방법과는 다르게 higher-dimension으로 쉽게 scale될 수 있다고 한다.

## **1. Problem Definition**

이 논문에서 풀고자 하는 문제는 translations, rotations, reflections and permutations equivariant 한 GNN을 디자인 하는 것이다. 핵심적으로 쓰이는 개념들인 Equivariance 와 Graph Neural Network에 대한 간단한 소개를 하도록 하겠다.

### 1. Equivariance

Equivariant 에 대한 정의는 다음과 같다.

> Let $$T_g:X→X$$ be a set of transformations on $$X$$ for the abstract group $$g\in G$$. We say a function $$\phi : X → Y$$ is equivariant to $$g$$ if there exists an equivalent transformation on its output space $$S_g:Y→Y$$ such that: $$\phi(T_g(x))=S_g(\phi(x))$$
> 

조금 더 쉽게, translation equivariant을 예시로 설명해보겠다. 다음과 같은 상황을 생각해 보자.

- $$\phi (\cdot)$$, non-linear function (ex. MLP, GNN, ...)
- $$x=(x_1,...,x_M) \in \mathbb{R} ^{M \times n}$$, n-dimensional 공간에 있는 M개의 input data
- $$T_g$$, input set에 대한 translation, $$T_g(x)=x+g$$
- $$S_g$$, output set에 대한 equivalent translation, $$S_g(y)=y+g$$

만약 우리의 transformation $$\phi: X→Y$$ 가 translation equivariant라고 한다면, original input, $$x$$에 translation을 적용해서 transformation을 적용한 값과, transformation을 적용한 값에 translation을 적용한 값이 동일할 것이다. 수식으로 나타내면 다음과 같다.

- $$\phi(x+g)=\phi(x)+g$$

즉, 다시한번 정리하면, 어떤 function이 translation equivariant하다면, 위의 property를 가질 수 있다는 것이다. 이러한 property를 가지면 어떤 좋은 점이 있는지는 추후에 설명하도록 하겠다.

이 논문에서는 set of particles $$x$$에 적용되는, 다음 3가지 종류의 equivariance에 대해서 탐구한다.

- **Translation equivariance.** Translating the input by $$g \in \mathbb{R} ^{n}$$ results in an equivalent translation of the output. Let $$x+g$$ be shorthand for $$(x_1+g, ..., x_M+g)$$. Then $$y+g=\phi(x+g)$$.
- **Rotation (and reflection) equivariance.** For any orthogonal matrix $$Q \in \mathbb{R} ^{n \times n}$$, let $$Qx$$ be shorthand for $$(Qx_1, ..., Qx_M)$$. Then rotating the input results in an equivalent rotation of the output $$Qy=\phi(Qx)$$.
- **Permutation equivariance.** Permuting the input results in the same permutation of the output $$P(y)=\phi(P(x))$$ where $$P$$ is a permutation on the row indexes.

그래프에 대해서 Rotation equivariance한 graph neural network 예시를 보여주는 그림이 있다.

- 왼쪽위, 오른쪽위, 왼쪽아래, 오른쪽아래 그림을 각각 그림 1, 2, 3, 4라고 하겠다.
- x, v는 rotation equivariant한 feature, h는 rotation invariant한 feature 이다.
- GNN은 x, v, h 를 이용해서 이들을 업데이트 한다.
- 1번 그림의 그래프를 rotation transformation을 통해 얻은 결과는 2번 그림이 그래프가 된다.
- 1번 그림의 그래프를 GNN을 통해 얻은 결과는 3번 그림의 그래프가 된다.
- 2번 그림의 그래프를 GNN을 통해 얻은 결과는 4번 그림의 그래프가 된다.
- 3번 그림의 그래프를 rotation transformation을 통해 얻은 결과는 4번 그림이 그래프가 된다.

<!-- ![Untitled](E(n)%20Equivariant%20Graph%20Neural%20Networks%202887f247cf004d9584e18f81cac4ca0e/Untitled.png) -->
<!-- <img src=".gitbook/2022-spring-assets/KanghoonLee_1/image1.png">   -->

![1](/.gitbook/2022-spring-assets/KanghoonLee_1/image1.png)


### 2. Graph Neural Network (GNN)

Graph Neural Network는 Graph를 input으로 받아서 Graph를 output으로 내놓는 Neural network이다. Graph는 노드와 엣지로 정의되는데, 이때 노드나 엣지마다 대응되는 feature가 있어서, 연결관계에 따른 pre-defined된 계산 과정을 통해 이 feature들이 업데이트 된다.

저자들은 graph convolutional layer를 다음과 같이 정의하였다.

> Given a graph $$\mathcal{G}=(\mathcal{V},\mathcal{E})$$ with nodes $$v_i \in \mathcal{V}$$ and edges $$e_{ij} \in \mathcal{E}$$, we define a graph layer convolutional layer following notation from (Glimer et al., 2017) as:
- $$\bold{m}_{ij}=\phi_e(\bold{h}_i^l,\bold{h}_j^l,a_{ij})$$
- $$\bold{m}_i=\sum_{j\in \mathcal{N}(i)} \bold{m}_{ij}$$
- $$\bold{h}_i^{l+1}=\phi_h(\bold{h}_i^l,\bold{m}_i)$$

각 element에 대한 설명은 다음과 같다.

- $$\bold{h}_i^l \in \mathbb{R} ^{\text{nf}}$$ is the nf-dimensional embedding of node $$v_i$$ at layer $$l$$.
- $$a_{ij}$$ are the edge attribute.
- $$\mathcal{N}(i)$$ represents the set of neighbors of node $$v_i$$.
- $$\phi_e$$ and $$\phi_h$$ are the edge and node operations (ex. MLP)

3가지의 과정을 edge

## **2. Motivation**  

딥러닝을 더욱 발전시킨 방법들 중 많은 것들은 inductive bias와 관련이 깊다. 뉴럴넷에 inductive bias를 주는 대표적인 방법으로 ‘translation equivariance가 있는 CNN’ 그리고 ‘permutation equivariance가 있는 GNN’ 이 있다. 이를 잘 정리한 내용을 ‘Relational inductive biases, deep learning, and graph networks’ 라는 paper에서 가져와 보았다. 자세한 것은 해당 논문을 보면 좋을 것 같다.

<!-- ![Untitled](E(n)%20Equivariant%20Graph%20Neural%20Networks%202887f247cf004d9584e18f81cac4ca0e/Untitled%201.png) -->
<!-- <img src=".gitbook/2022-spring-assets/KanghoonLee_1/image2.png">   -->
![2](/.gitbook/2022-spring-assets/KanghoonLee_1/image2.png)

문제의 이러한 특성을 이용해서 뉴럴넷을 디자인하는 것은 매우 효과적이다. 왜냐하면, 우리가 탐헌해야 할 함수를 매우 큰 범위에서 작은 범위로 축소시킬 수 있기 때문이다. (이에 대한 더 좋은 설명은 [https://youtu.be/VN2biLjqJXc?t=742](https://youtu.be/VN2biLjqJXc?t=742) 이 링크를 참조하면 좋을 것 같다.)

많은 문제들은 3D translation and rotation symmetry한 성격을 띠고 있다. Point clouds (Uy et al., 2019), 3D molecular structures (Ramakrishnan et al., 2014), or N-body particle simulations (Kipf et al., 2018) 등의 문제들이 그렇다. 이러한 symmetric에 대응되는 그룹을 Euclidean group: SE(3) 라고 부르고, reflection까지 추가될 경우 E(3) 라고 부른다. 이러한 task에서의 prediction같은 것들은 E(3) 변환에 대해서 equivariant하거나 invariant한 것이 바람직할 것이다.

예를 들어보면, 우리가 이산화탄소의 분자모형을 그래프로 표현해서 이를 input으로 사용하는 GNN을 통한 어떤 예측을 한다고 생각해보자. 이 분자모형의 위치가 전체적으로 오른쪽으로 조금 움직이거나, 특정 방향으로 조금 회전을 한다고 해도, 이의 본질적인 성격(독성, ph농도) 등은 변하지 않을 것이다. 만약 우리가 input이 translation 되거나 rotation 된 데이터가 들어왔을때, 이에 상관없이 결과가 똑같거나, 똑같이 translation 되거나 rotation된 결과가 나오는 함수가 있다면, 이러한 문제에서 매우 효과적일 것이다. (inductive bias를 통해서 함수에 이미 그러한 property가 반영되어 있기 때문에, 굳이 우리가 input data에 대한 property를 학습을 통해 배울 필요가 없기 때문이다.)

최근들어 E(3) 또는 SE(3) equivariance한 성질을 가진 다양한 방법들이 제안되었다. 대부분의 방법들은 intermediate network layer에서 higher-order representation로 변환하는 과정을 통해서 이루어져 왔다. 그러나, 이들이 취하는 이 방식들은 계산하는데 비용이 많이 드는 방식을 필요로 한다. 또한, 대부분은 input data, output data dimension이 매우 제한적이다. (최대 3차원)

그렇기에, 이 논문에서는 N-dimension에서 translation, rotation, reflection equivariant 하고(E(n)), input set of point에 대해서 permutation equivariant 한 architecture를 제안하였다. 뿐만 아니라, 다른 모델과 달리 매우 복잡한 계산을 필요로 하지 않으면서, 비슷하거나 더 좋은 성능을 이끌어 내었다고 주장한다.

## **3. Method**  

이들이 제안한 Equivariant Graph Neural Network (EGNN)의 각 Equivariant Graph Convolutional Layer (EGCL)는 다음의 내용들을 input으로 output을 만든다.

- (Input) $$h^l=\{{h_0^l, ..., h_{M-1}^l}\}$$, set of node embedding.
- (input) $$x^l=\{x_0^l,...,x_{M-1}^l\}$$, coordinate embedding.
- (input) $$\mathcal{E}=(e_{ij})$$, edge information
- (output) $$h^{l+1},x^{l+1}$$.

이를 간단히, $$h^{l+1}, x^{l+1}=\text{EGCL}[h^l,x^l,\mathcal{E}]$$ 로 나타낼 수 있다. 이 과정은 다음과 같다. 

- $$m_{ij}=\phi_e(h_i^l,h_j^l,||x_i^l-x_j^l||^2,a_{ij})$$
- $$x_i^{l+1}=x_i^l+C \sum_{j\neq i}(x_i^l-x_j^l)\phi_x(m_{ij})$$
- $$m_i=\sum_{j \neq i} m_{ij}$$
- $$h_i^{l+1}=\phi_h(h_i^l,m_i)$$


사실 기존의 GNN과 달라지는 부분은 첫 번째와 두 번째 수식밖에 없다. 오로지 이 과정을 통해서 위에서 언급한 2개의 equivariant한 성질을 얻어낼 수 있다고 이야기한다.

EGCL의 첫 번째 식을 살펴보면, 기존의 GNN과 비슷하게 각 node embedding과 노드를 연결하는 edge embedding이 input으로 들어간다. 여기서 추가로 node의 distance가 input으로 사용된다.

두 번째 식을 살펴보면, 첫 번째 식에서 계산된 $$m_{ij}$$가 $$\phi_x$$를 통해서 임베딩 되어 weight를 만들어 낸다. 그리고 이 값은, node간의 relative difference weighted sum의 weight이 된다.  그리고 이 값이 더해져서 coordinate embedding에 더해지게 된다. $$\phi_x$$, output function은 weight을 만들어내는 함수이기 때문에 $$\phi_x:\mathbb{R}^\text{nf}→\mathbb{R}^1$$ 임을 주의해야 한다.

### Analysis on E(n) equivariance

이 subsection에서는 제안한 model이 equivariance임을 증명해준다. 자세한 증명은 Appendix에 나와 있지만, 간단한 proof sketch만 해보도록 하겠다.

우선 기본적으로 GNN은 permutation equivariance property를 가지고 있다. 그러므로, 우리는 translation and rotation에 대한 증명만 완료하면 된다. 또한, translation and rotation equivariance property는 function composition에 대해서 보존되는 것이 trivial하기 때문에 각 layer에 대한 증명으로 충분하다. 이 논문에서는 다음의 식을 증명하는것으로 EGNN이 E(n) equivariance임을 보인다. 

증명 과정이 생각보다 간단하니 눈으로 봐도 쉽게 따라갈 수 있다. 간단히 이야기하면, translation 부분은 서로 소거되고, rotation 부분은 두개가 곱해져서 identity matrix가 되어서 사라지는 방식으로 증명이 이루어진다.

- $$Qx^{l+1}+g,h^{l+1}=EGCL(Qx^l+g,h^l)$$
, where $$g\in \mathbb{R}^n$$ is a translation vector and $$Q \in \mathbb{R}^{n\times n}$$ is a orthogonal matrix.
 

### Extending EGNNs for vector type representations

이 subsection에서는 약간의 수정을 통해서 particle의 momentum을 explicit하게 계속 track할 수 있는 방법을 제안한다. 이 방법은 particle의 estimate velocity를 매 layer에서 얻는 것 뿐만 아니라, particle의 초기 속도를 매 layer에 제공할 수 있다는 장점이 있다고 한다. (equivariant property를 유지한 채 velocity information을 제공할 수 있다는 것이 핵심인 것 같다.) 다음과 같은 modification을 통해서 momentum 정보를 포함할 수 있다고 한다. ($$v^{\text{init}}=0$$인 경우는 일반적인 EGNN과 똑같은 것을 알 수 있다.)

- $$v_i^{l+1}=\phi_v(h_i^l)v_i^{\text{init}}+C\sum_{j \neq i}(x_i^l-x_j^l)\phi_x(m_{ij})$$
- $$x_i^{l+1}=x_i^l+v_i^{l+1}$$


### Inferring the edge

만약 point cloud 또는 set of node만 주어진 상태에서, 우리는 각 노드간 어떤 연결관계가 형성되어있는지 모르는 경우도 있다. (=Adjacency matrix가 주어지지 않는 경우도 있다.) 이런 경우, 우리는 fully connected graph를 가정하여 GNN을 적용할 수 있지만, 이러한 fully connected graph는 large scale input에 대해서 잘 scale되지 않는다.

이런 이슈를 해결하기 위해서 (Serviansky et al., 2020; Kipf et al., 2018) 과 비슷하게, explicit하게 제공되지 않은 edge connectivity를 임의로 계산하여 GNN을 사용할 수 있다. Aggregation operation 부분을 다음과 같이 수정하면 된다. 

> $$m_i=\sum_{j\in \mathcal{N}(i)}m_{ij}=\sum_{j\neq i}e_{ij}m_{ij}$$

, where $$e_{ij}\approx \phi_{inf}(m_{ij})$$ and $$\phi_{inf}:\mathbb{R}^{nf} \rightarrow [0,1]^1$$
> 

message에 대한 부분은 애초에 equivariant property를 생각하지 않았으므로, 이 계산 과정은 EGNN이 equivariant한 성질에 영향을 미치지 않는다. (Equivariant함을 계속 유지시켜 준다.)

## **4. Experiment**  

### Modeling a dynamical system - N-body system

[ 실험 설명 ]

- Charged Particles N-body experiment to a 3 dimensional space
- 이 시스템에서는 5개의 positive or negative charge된 particle이 3차원 공간에서 움직인다.
- charge 에 따라서 서로 끌어당기거나 밀어낸다.
- 이 상황에서 1초 뒤의 각 particle의 위치를 estimate해야 한다.
- 이는 rotation and translation equivariant 한 task이다.

[ 데이터 ]

- 3000 training set, 2000 validation set, 2000 testing set.
- (input) $$p^{(0)}=\{p_1^{(0)},...,p_5^{(0)}\}$$, particle position
- (input) $$v^{(0)}=\{v_1^{(0)},...,v_5^{(0)}\}$$, initial velocity
- (input) $$c=\{c_1,...,c_5\}$$, charge
- Mean squared error를 통해서 optimize

[ 결과 ]

<!-- ![Untitled](E(n)%20Equivariant%20Graph%20Neural%20Networks%202887f247cf004d9584e18f81cac4ca0e/Untitled%202.png) -->
<!-- <img src=".gitbook/2022-spring-assets/KanghoonLee_1/image3.png">   -->
![3](/.gitbook/2022-spring-assets/KanghoonLee_1/image3.png)
- 기존 모델들 중 가장 좋은 성능을 보임.
- Forward time도 매우 작은 것을 확인할 수 있었음.

저자들은 여기서 training sample의 수에 따른 실험도 하였다. inductive bias를 가한 network의 특징은 해당 bias가 유효한 문제에 대해서 빠르게 generalize된다는 점이다. 즉, 적은 데이터로도 잘 학습될 수 있을것이란 결과가 예상된다. 하지만 그것 말고도 재미있는 점이 있는데, 다음의 그래프를 보면 알 수 있다.

<!-- ![Untitled](E(n)%20Equivariant%20Graph%20Neural%20Networks%202887f247cf004d9584e18f81cac4ca0e/Untitled%203.png) -->
<!-- <img src=".gitbook/2022-spring-assets/KanghoonLee_1/image4.png">   -->
![4](/.gitbook/2022-spring-assets/KanghoonLee_1/image4.png)
E(n)-equivariant한 Radial Field 방법과, 일반적인 GNN, 그리고 여기서 제안한 모델인 EGNN 세가지 모델을 학습 데이터 수에 대한 MSE를 나타낸 그래프이다. 예상과 비슷하게, EGNN과 Radial Field는 적은 데이터 샘플로도 잘 generalize하는 것을 보였다. 하지만, Radial Field의 경우 EGNN과 다르게 많은 데이터가 주어졌음에도 성능이 더욱 개선되지 않는 점을 보였다. EGNN과 GNN은 학습 데이터가 많아짐에 따라서 성능 개선이 이루어지는 것을 확인할 수 있었다.

저자들은 이를, Radial Field 방법이 너무 모델에 대한 bias가 크기 때문에, 데이터 안에서 미세하게 변하는 부분을 학습하기 힘들다고 주장한다. 즉 정리하면, EGNN은 E(n) 의 high bias를 취하면서 동시에 일반적인 GNN이 갖고 있는 flexibility도 가지고 있다고 주장한다.

### Graph Autoencoder

[ 실험 설명 ]

- Graph에서 edge coneectivity를 reconstruct하는 task를 진행하였음.
- 이 task가 간단해 보이지만, node feature가 없는 graph에 대해서는 꽤 어려운 문제가 된다.
- 왜냐하면, node feature가 없으면 GNN의 output은 오직 연결되어있는 topology에만 dependent하게 되기 때문에 connectivity를 예측하기 힘들다.
- 이를 해결하기 위해 간단한 방법은 node feature에 gaussian nosie를 input으로 주어 graph의 symmetry를 제거하는 것이다.
- 이 방법을 noise-GNN 이라는 이름으로 모델 비교를 하였다.

[ 데이터 ]

- Community Small : (You et al., 2018) 의 코드를 이용해서 데이터 셋을 생성함.
    - 그래프 사이즈는 12개~ 20개의 노드
- Erdos&Renyi : (Bollobas & B ´ ela ´ , 2001) 의 생성모델을 이용해서도 데이터 셋을 생성함.
    - 그래프 사이즈는 7개~ 16개의 노드
- 5000 training set, 500 validation set, 500 testing set.

[ 결과 ]

<!-- ![Untitled](E(n)%20Equivariant%20Graph%20Neural%20Networks%202887f247cf004d9584e18f81cac4ca0e/Untitled%204.png) -->
<!-- <img src=".gitbook/2022-spring-assets/KanghoonLee_1/image5.png">   -->
![5](/.gitbook/2022-spring-assets/KanghoonLee_1/image5.png)
- EGNN이 두개의 데이터셋 모두에서 가장 좋은 성능을 보였다.
- 위에서 언급한대로 noise를 추가한 noise-GNN은 GNN보다 좋은 성능을 보였다.

## **5. Conclusion**  

본 논문에서는 구현이 쉽고, 성능도 좋은 E(n) Equivariant한 GNN을 제안하였다. 저자들은 이 연구가 drug discovery, protein folding, design of new materials, 그리고 3D computer vision에 적용될 것으로 기대한다고 한다.

[ 개인적인 의견 ]

Equivariant, Invariance한 성질은 우리가 알게 모르게 많이 사용하고 있었던 성질 같다. 나는 이 논문을 추천 받기 전에 / 읽기 전에 이러한 성질에 대해서 모호하게만 인지하고 있었지만, 정확한 정의에 대해서는 생각해 본적이 없었다. 이 논문을 읽고 나서 inductive bias에 대해서 조금 더 이해하게 되는 좋은 계기가 된 것 같다.

주어진 데이터, 또는 문제 자체가 가지고 있는 속성(symmetry, ...)들이 있어서 이를 exploit하는 방식으로 network 를 디자인 하는 방식 또한 기존에 많이 알게모르게 사용되는 방식이었지만 그 중요성을 다시 깨닫게 되었다. 

Supervised learning이나 Reinforcement learning에서 네트워크를 잘 generalize 시키기 위해서 우리는 data augmentation을 하는 경우가 있다. 간단한 예를 들면 이미지 분류 문제에서 고양이 이미지를 뒤집어서 학습을 하거나, RL에서 cartpole 에서 state, action을 뒤집어서도 학습을 하는 방식이 있을 것이다. 이러한 방식이 작동하는 이유는 우리는 이 문제가 각각 rotational invariance하고, reflection invariance한 성질임을 알고 있기 때문이다. 우리가 이러한 문제에 대한 bias를 명확하게 알고 있다면, 굳이 이러한 식으로 학습을 하는 것보다는 network에 특정 property를 부여하는 방식이 더욱 효율적일 것이다.

이 논문은 유명한 Computer Scientist인 Max Welling 교수님의 연구실에서 쓰여졌다. 2021년 2월에 아카이브에 올라왔는데 벌써 인용수가 80이다 (2022년 4월 24일 기준). 비슷한 방법론들이 많지만, 인용수가 이렇게 높은 이유는 구현이 매우 간단하기 때문인 것 같다. 간단하면서 유용한 property를 만족한다는 것이 인상적이다. GNN을 쓰는 사람이라면 performance를 올리는 데에 쉽게 적용해 볼 수 있을 것 같다. 물론 task가 이러한 property를 가지고 있다면..

단점으로는, physical한 coordinate정보가 포함되어 있고, 이 정보가 해당 task에서 꽤 중요한 역할을 하는 경우에만 좋은 효과를 볼 것 같다. (당연한 얘기 같긴 하지만..) 실제로 그러한 케이스가 얼마나 많을지는.. 잘 모르겠다. 막상 떠오르는 것은 Trajectory Prediction, (Data driven) Control in physical system 정도 이다. 내 연구에도 쓰일 여지가 충분히 있을 것 같아서, 나중에 꼭 시도해보려고 한다.

## **Author Information**  

- **Victor Garcia Satorras**
    - Affiliation: University of Amsterdam, Netherlands.
    - Research Topics: AI, ML, Deep learning, Statistics
- **Emiel Hoogeboom**
    - Affiliation: University of Amsterdam, Netherlands.
    - Research Topics: Generative Modelling, Bayesian Inference, Artificial Intelligence
- **Max Welling**
    - Affiliation: University of Amsterdam, Netherlands.
    - Research Topics: ML, AI, Statistics
    - [https://scholar.google.com/citations?user=8200InoAAAAJ&hl=en](https://scholar.google.com/citations?user=8200InoAAAAJ&hl=en)
    

## **6. Reference & Additional materials**  

- BATTAGLIA, Peter W., et al. Relational inductive biases, deep learning, and graph networks. *arXiv preprint arXiv:1806.01261*, 2018.
- [https://www.youtube.com/watch?v=hUrbS1BhBWc](https://www.youtube.com/watch?v=hUrbS1BhBWc)
- [https://www.youtube.com/watch?v=VN2biLjqJXc](https://www.youtube.com/watch?v=VN2biLjqJXc)
- [https://dmol.pub/dl/Equivariant.html](https://dmol.pub/dl/Equivariant.html)
