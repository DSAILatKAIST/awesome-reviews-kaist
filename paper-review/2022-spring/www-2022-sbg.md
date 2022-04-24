---
description : Fan Lu / Modeling User Behavior with Graph Convolution for Personalized Product Search / WWW-2022
---

# **SBG(Successive Behavior Graph)**

# 1. Problem Definition

본 논문에서는 Latent space 기반 개인화 상품검색을 구현하기 위해 사용되는 유저 특성 모델링이 가지는 한계를 보완하기 위해 유저의 짧은 행동을 하나의 세션으로 구성하는, Successive Behavior Graph(이하 SBG)를 제안한다.

또한 이렇게 구성한 그래프에 GCN을 통해 얻은 Graph-enriched embedding 을 우수한 성능을 보이는 것으로 이미 잘 알려진 ZAM([A Zero Attention Model for Personalized Product Search](https://arxiv.org/pdf/1908.11322.pdf) / CIKM-2019) 에 적용시킴으로서 보다 좋은 성능을 가진 사용자 특성 모델링을 구현하였다.

# 2. Motivation

이커머스의 발전에 따라 상품 검색모듈의 중요성이 커지고 있다. 이때 상품 검색(Product search)은 기존의 웹 검색(Web search)과 명확히 다른 특성을 가지고 있다

- 웹페이지와 달리 제목, 리뷰와 같은 짧은 텍스트 정보만을 이용하는 점
- Ontology, Spec sheet, Figures 와 같은 다양한 형식의 데이터들을 포함한다는 점
- 웹과 달리 사용자와 물품간에 검색, 클릭, 리뷰, 구매 와 같이 다양한 relation이 존재하는점(풍부한 정보)

### Limitation

이러한 풍부한 정보들을 다루기 위해 다양한 방법들이 제안되었으며 중 이 중 사용자와 아이템, 쿼리를 같은 공간상에 올려놓는 Latent space based model들이 좋은 결과를 보였다. 하지만 이전 방법론들이 해결하지 못한 몇가지 논의점이 있는데, 이는 다음과 같다.

1. 유저의 life time behavior를 하나의 preference로 담기에는 유저의 관심사 변화나, 비의도적 행동들에 의한 노이즈로 명확한 모델링이 되지 않는다.
2. 위 문제를 피하기 위해 유저의 최근 행동만을 사용하는 경우가 많은데 이는 많은 정보를 낭비하게 되고, 충분한 특성을 모델링 할 수 없다.

### Propose

기존 방법론이 가지는 한계점들을 극복하기 위해 본 논문에서는 다음과 같은 방법을 제안한다.

- SBG를 통한 상품간 연관관계 확장
  - 단기적인 연속행동들을 하나의 세션으로 삼아, 글로벌 행동그래프 생성.
  - 개개인의 행동이 아닌, 유저 전체의 글로벌 행동 그래프를 통해 잠재된 아이템 간 연관 관계를 유저 행동을 매개로 연결가능하도록 구성
- GCN을 통한 아이템 모델링 강화
  - Graph convolution(GCN)을 통해 연관된 아이템들을 잠재 공간상에 더 비슷하게 위치시킴으로서 풍부한 상품 표현 임베딩을 만듦
  - **GCN II** 에서 영감을 받아, Jumping connection 을 가지는 efficient graph convolution를 사용

### Contributions

1. 상품 검색에 grah convolution을 적용시킨 첫번째 사례이다.
2. SBG 를 통해 지엽적인 관계 뿐만 아니라 global behavior pattern도 발견 가능하다.
3. 8개의 amazon benchmark를 통한 검증하였다.

# 3. Method

## 3.0 Background - Latent space 기반 모델

💡**검색의 목적 : User가 Query를 입력할때 구매할 확률이 가장 높은 Item을 보여주는것**

$$q$$ : 유저가 입력한 query

$$u$$ : 유저

$$i$$ : 아이템(상품)

$$
P(i \mid u, q)=\frac{\exp \left(\boldsymbol{i} \cdot \boldsymbol{M}_{\boldsymbol{u q}}\right)}{\sum_{i^{\prime} \in I_{q}} \exp \left(\boldsymbol{i}^{\prime} \cdot \boldsymbol{M}_{\boldsymbol{u q}}\right)}
$$

> $$i \in \mathbb{R}^{\alpha}$$ is the embedding representation of item , $M_{uq}$ is a joint model of user-query pair (u,q)
> **Probability** of whether $i$ would be purchased by $u$ given $q$

---

### 3.0.1 QEM(Query Embedding Model)

💡**non-personalized product search**

**개인화 까지는 아니고, query 와 item embedding 을 맞추자.**

### `Query`

$$
M_{uq} = q
$$

(User embedding 이 없음을 볼 수 있다)

> $q \in \mathbb{R}^{\alpha}$ is the embedding representation of the query q.

query는 검색 단계에서 입력되기 때문에, request time 내에 계산되어야 함

q의 계산은 다음과 같이 이뤄진다.

$$
\boldsymbol{q}=\phi\left(\left\{w_{q} \mid w_{q} \in q\right\}\right)=\tanh \left(\boldsymbol{W}_{\phi} \cdot \frac{\sum_{w_{q} \in q} \boldsymbol{w}_{\boldsymbol{q}}}{|q|}+\boldsymbol{b}_{\phi}\right)
$$

> where $w_{q} \in \mathbb{R}^{\alpha}$ is the embedding of a word $w_{q}$ in $q,|q|$ is the length of the query, and $\boldsymbol{W}{\phi} \in \mathbb{R}^{\alpha \times \alpha}$ _and_ $\boldsymbol{b}{\phi} \in \mathbb{R}^{\alpha}$ are two parameters learned in the training process.

### `Item`

Item embedding 은 Paragraph vector(doc2vec)에서 insight를 얻어서 사용했다.

![doc2vec이 information retrival task에서 적은 error를 보였기 때문](<".gitbook/2022-spring-assets/KimDaehee_1/SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled.png">)

doc2vec이 information retrival task에서 적은 error를 보였기 때문

$$
P\left(T_{i} \mid i\right)=\prod_{\boldsymbol{w} \in T_{i}} \frac{\exp (\boldsymbol{w} \cdot \boldsymbol{i})}{\sum_{w^{\prime} \in V} \exp \left(\boldsymbol{w}^{\prime} \cdot \boldsymbol{i}\right)}
$$

> $T_i$ be a set of words associated with an item $i$ (i번째 item 에 대한 단어들)
> $w \in \mathbb{R}^{\alpha}$ is the embedding of a word and $V$ is the vocabulary of all possible words.

---

### 3.0.2 HEM(Hierarchical embedding model)

[Learning a hierarchical embedding model for personalized product search(SIGIR-2017)](https://scholar.google.com/scholar_url?url=https://dl.acm.org/doi/pdf/10.1145/3077136.3080813&hl=ko&sa=T&oi=gsb-gga&ct=res&cd=0&d=15736059002742053164&ei=pstjYtiTHYySyASZk6HgCA&scisig=AAGBfm17kfq7KLvb8_VrBirjKb8qxDT-7w)

<aside>
💡 **QEM+User embedding**
개인화된 추천을 위해 User embedding 을 사용하자.

</aside>

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%201.png>)

$$
M_{uq} = q+u
$$

**user 와 query는 독립이라는 가정하에 모델링**

### `User`

$$
P\left(T_{u} \mid u\right)=\prod_{\boldsymbol{w} \in T_{u}} \frac{\exp (\boldsymbol{w} \cdot \boldsymbol{u})}{\sum_{w^{\prime} \in V} \exp \left(\boldsymbol{w}^{\prime} \cdot \boldsymbol{u}\right)}
$$

> $T_u$ could be any text written or associated to u, such as product reviews or the descriptions of items that the user has purchased.

item과 같이 user’s associated text 를 이용해 embedding을 얻어서 $M_{uq}$에 반영함.

---

### 3.0.3.AEM(Attention Embedding model)

<aside>
💡 **User preferences are not independent of query intents**
어탠션을 통해 사용자의 구매 행동을 추가해보자!

</aside>

$I_u$ 가 user $u$ 가 구매한 item set 일때, user embedding u는:

$$
\boldsymbol{u}=\sum_{i \in I_{u}} \frac{\exp (f(q, i))}{\sum_{i^{\prime} \in I_{u}} \exp \left(f\left(q, i^{\prime}\right)\right)} \boldsymbol{i}
$$

$f (q,i)$는 $I_u$의 각 item i 들이 현재 query $q$ 에 대한 attention function

$$
f(q, i)=\left(i \cdot \tanh \left(\boldsymbol{W}_{f} \cdot q+b_{f}\right)\right) \cdot \boldsymbol{W}_{h}
$$

$\boldsymbol{W}_{h} \in \mathbb{R}^{\beta}, \boldsymbol{W}_{f} \in \mathbb{R}^{\alpha \times \beta \times \alpha}, \boldsymbol{b}_{f} \in \mathbb{R}^{\alpha \times \beta}$, and $\beta$ is a hyperparameter that controls the number of hidden units in the attention network.

$$
M_{uq} = q+u
$$

---

### 3.0.4 ZAM(Zero Attention Model)

[A Zero Attention Model for Personalized Product Search(CIKM ’19)](https://arxiv.org/pdf/1908.11322.pdf)

<aside>
💡 **Advanced AEM**
Zero ****attention strategy를 통해 성능 개선

</aside>

User가 해당 query에 관련된 구매기록이 전혀 없는 경우나(cold start), 성향과 무관하게 쿼리에 따른 결과가 정해져 있는 경우(dominant brand)의 성능 하락을 개선하고자 Zero attentntion strategy 적용

> The main **difference** between ZAM and AEM is that, instead of attending to the user’s previous purchases only, ZAM **allows the attention network to attend to a Zero Vector.**

$$
\boldsymbol{u}=\sum_{i \in I_{u}} \frac{\exp (f(q, i))}{\exp (f(q, \mathbf{0}))+\sum_{i^{\prime} \in I_{u}} \exp \left(f\left(q, i^{\prime}\right)\right)} \boldsymbol{i}
$$

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%202.png>)

## 3.1. Search Framework

### 3.1.1 **Product retrieval task**

<aside>
💡 **검색의 목적 : User가 Query를 입력할때 구매할 확률이 가장 높은 Item을 보여주는것**

Rank Item by probability \*\*\*\*of whether $i$ would be purchased by $u$ given $q$아

</aside>

기본적으로 Latent space based 모델링을 차용

$$
P(i \mid u, q)=\frac{\exp \left(f\left(\boldsymbol{i}, \boldsymbol{M}_{u q}\right)\right)}{\sum_{i^{\prime} \in C} \exp \left(f\left(\boldsymbol{i}^{\prime}, \boldsymbol{M}_{u q}\right)\right)}
$$

> $i \in \mathbb{R}^{\alpha}$ is the embedding representation of item
>
> $M_{uq}$ is a joint model of user-query pair (u,q)
>
> $f$ is similarity measure function. (논문에선 cosine similarity)

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%203.png>)

### 3.1.2 **Language Modeling Task**

$i, u, q$ 중 이번 논문의 핵심이 되는 SBG, GCN을 이용하는것은 $u$ embedding
$i,q$는 이전과 크게 다르지 않음

$$
P\left(T_{i} \mid i\right)=\prod_{w \in T_{i}} \frac{\exp (\tau(w, i))}{\sum_{w^{\prime} \in V} \exp \left(\tau\left(w^{\prime}, i\right)\right)}
$$

maximize likelihood 하는 과정에서 생기는 item embedding

$$
\boldsymbol{q}=\phi\left(\left\{w_{q} \mid w_{q} \in q\right\}\right)=\tanh \left(\boldsymbol{W}_{\phi} \cdot \frac{\sum_{w_{q} \in q} \boldsymbol{w}_{\boldsymbol{q}}}{|q|}+\boldsymbol{b}_{\phi}\right)
$$

$\phi$는 여타 non-linear sequential encoder(LSTM,Transformer 등)이 될 수 있지만, 보통 query는 짧고, 단어 순서또한 그리 중요치 않아 저자는 기존의 average와 같은것을 사용

## 3.2 Efficient Graph Convolution with Jumping Conection

### 3.2.1 **Efficient Graph Convolution**

- Vanila GCN

  가장 기본적인 GCN구조

  $$
  \boldsymbol{H}^{(l)}=\sigma\left({A} \boldsymbol{H}^{(l-1)} W^{(l)}\right)
  $$

- Common GCN(self loop, normalized)
  학습에 사용되는 GCN 구조, 일반적으로 GCN 구조를 말한다면 이것을 의미한다.

      $$
      \boldsymbol{H}^{(l)}=\sigma\left(\hat{A} \boldsymbol{H}^{(l-1)} W^{(l)}\right)
      $$

      $\hat{A}=I+D^{-1} A$ is the ([normalized](https://woosikyang.github.io/Graph-Convolutional-Network.html)) adjacency matrix with self-loops,

      $D$ is the degree matrix.

      $H^{(l)}$ is the node embeddings produced by layer $l$.

      $W^{(l)}$ denotes trainable parameters

      $\sigma$ is a non-linear function such as $\operatorname{ReLU}(\cdot)$

- Efficient Graph Concovoluition ~~Network~~
  저자들의 경험적 연구에 따라 일반적인 GCN을 사용하는 것 보다 efficient graph convolution을 사용하는 것이 더 성능이 좋았다고 언급함.
  > _However, as observed from our empirical study, the projection layers may **distort the semantic product representations** learned by **language modeling** in methods such as ZAM or HEM._
  $$
  \boldsymbol{H}^{(l)}=\left(\omega \boldsymbol{I}+(1-\omega) \boldsymbol{D}^{-1} \boldsymbol{A}\right) \boldsymbol{H}^{(l-1)}
  $$
  $\omega$ 는 자기 자신의 노드를 전파하는 정보를 조절하는 하이퍼 파라미터

### 3.2.2 **Jumping Graph Convolution Layer**

GNN 구조에서 고차원의 정보(High order information)을 활용하기 위해서는 많은 층을 쌓아서 네트워크를 학습 시켜야 한다. 단 그래프를 높이 쌓을 수록 임베딩이 한 지점으로 모이는 Over smoothing problem 이 일어난다.

[Simple and Deep Graph Convolutional Networks](https://scholar.google.co.kr/scholar_url?url=http://proceedings.mlr.press/v119/chen20v/chen20v.pdf&hl=ko&sa=X&ei=hjEyYvn_AYjwyAS2o4mYAw&scisig=AAGBfm2_hCGohcaQRDILYCo-3dLq2_o3hQ&oi=scholarr)에서 제안한 **GCN II**의 idea 에 따라residual term을 추가하여 다음과 같이 $H^0$를 feeding 해준다.

$$
\tilde{\boldsymbol{H}}^{(l)}=\left(\omega \boldsymbol{I}+(1-\omega) \boldsymbol{D}^{-1} \boldsymbol{A}\right)\left(\beta \boldsymbol{H}^{(0)}+(1-\beta) \tilde{\boldsymbol{H}}^{(l-1)}\right)
$$

𝛽 는 $H^{0}$의 feeding을 조절하는 하이퍼 파라미터

## 3.3 Modeling User Behavior with Graph Convolution(SBG)

### 3.3.1 **Graph Construction**

Successive behavior graph를 구성하기 위해, successive가 뭔지 정의해야 한다.

해당 논문에서는 특정한 길이 R을 한 세션으로 설정했음

> \*If the time interval between **two consecutive actions is within a period 𝑅** (e.g., a day, a week, or a month), the **two actions are considered as successive** and will be placed in the **same successive behavior sequence\***

$𝐺_{𝑆𝐵}$ 는 이렇게 구성된 시퀀스와 상품간의 이분그래프
$G_{SB}$의 edge 는 상품 i 가 시퀀스 S 에 있을 시 $𝐺_{𝑆𝐵}(𝑖, 𝑆) = 1$로 표현

### 3.3.2 **Enriching Product Representations with Graph Convolution**

Jumping network를 위해 첫 layer를 feeding 해야 하는데, 해당 논문에서는 product i 에 대한 embedding $h_i^{(0)}$을 사용했다.

$𝐿$ efficient jumping graph convolution layers를 거친 뒤 얻게되는 각 item i에 대한 **graph-enriched product embedding**을 $\tilde{h}_i^{(L)}$ 이라 한다.

### 3.3.3 **Using Graph-enriched Product Representations for User Preference Modeling**

구성된 SBG가 GCN을 거쳐 만들어진 임베딩을 ZAM에 반영하여 Enriched product representation을 반영한 user embedding을 만들어 냄.

- **Vanila ZAM**

$$
\boldsymbol{u}=\sum_{i \in I_{u}} \frac{\exp (f(q, i))}{\exp (f(q, \mathbf{0}))+\sum_{i^{\prime} \in I_{u}} \exp \left(f\left(q, i^{\prime}\right)\right)} \boldsymbol{i}
$$

$$
s(q, i)=\left(\boldsymbol{i}^{\top} \tanh \left(\boldsymbol{W}_{f}^{\top} \boldsymbol{q}+\boldsymbol{b}_{f}\right)\right)^{\top} \boldsymbol{W}_{h}
$$

- **Graph-enriched product representations for user preference modeling**

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%204.png>)

$$
\begin{aligned}
&\boldsymbol{u}=\sum_{i \in I_{u} \cup 0} \frac{\exp (s(q, i))}{\exp (s(q, \boldsymbol{0}))+\sum_{i^{\prime} \in I_{u}} \exp \left(s\left(q, i^{\prime}\right)\right)} \tilde{\boldsymbol{h}}_{i}^{(L)} \\
&s(q, i)=\left(\tilde{\boldsymbol{h}}_{i}^{\top} \tanh \left(\boldsymbol{W}_{f}^{\top} \boldsymbol{q}+\boldsymbol{b}_{f}\right)\right)^{\top} \boldsymbol{W}_{h}
\end{aligned}
$$

> _Where $\boldsymbol{W}{h} \in \mathbb{R}^{d{a}}, \boldsymbol{W}{f} \in \mathbb{R}^{d \times d{a} \times d}, \boldsymbol{b}{f} \in \mathbb{R}^{d \times d{a}}$ are the trainable parameters, and $d_{a}$ is the hidden dimension of the user-product attention network. In particular, $\exp (s(q, 0))$ is calculated by Eq. (12) with $i$ as a learnable inquiry vector $0^{\prime} \in \mathbb{R}^{d}$.\_

## 3.4 Model Optimization

상품 검색에서와 언어모델 두가지의 Loss function을 결합하여 사용

- Product retrieval loss
  $$
  L_{P R}=-\sum_{(u, i, q)} \log P(i \mid u, q)=-\sum_{(u, i, q)} \log \frac{\exp \left(f\left(\boldsymbol{i}, \boldsymbol{M}_{u q}\right)\right)}{\sum_{i^{\prime} \in C} \exp \left(f\left(\boldsymbol{i}^{\prime}, \boldsymbol{M}_{u q}\right)\right)}
  $$
- Language model loss
  $$
  L_{L M}=-\sum_{i} \log P\left(T_{i} \mid i\right)=-\sum_{i} \sum_{w \in T_{i}} \log \frac{\exp (\tau(w, i))}{\sum_{w^{\prime} \in V} \exp \left(\tau\left(w^{\prime}, i\right)\right)}
  $$
- Total loss

$$
L_{\text {total }}=L_{P R}+L_{L M}=-\sum_{(u, i, q)} \log P(i \mid u, q)-\sum_{i} \log P\left(T_{i} \mid i\right)
$$

> Remark. It is worth noting that we **only use the graph-enriched product embeddings to represent users** but do not use them to represent products themselves in the product retrieval task or the language modeling task, because the **mixed representations may make products lose their uniqueness** and hurt performance, which is verified by our empirical study.

# 4. Experiments

## 4.1 Research Questions

총 네가지의 RQ에 대해 검증하고자 실험 진행

1. How is the performances of SBG compared to the base model ZAM?
2. How does SBG perform compared to state-of-the-art methods for personalized product search?
3. How useful is graph convolution? Can the proposed jumping connection alleviate over-smoothing?
4. What is the effect of time interval 𝑅 on the constructed successive behavior graph 𝐺𝑆𝐵?

## 4.2 Dataset

[Amazon review dataset](http://jmcauley.ucsd.edu/data/amazon)

아마존에서 발생한 사용자-아이템 간 관계를 가진 데이터셋, 상품검색/추천 시스템 연구에서 널리 사용됨

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%205.png>)

> _Product reviews are generally used as text corpus for representing products or users, and **product categories are used as queries** to simulate a search scenario_

논문에서는 5개 이상의 행동을 가지는 5-core data를 사용했다.

> If you're using this data for a class project (or similar) please consider using one of these smaller datasets below before requesting the larger files.
>
> **K-cores** (i.e., dense subsets): These data have been reduced to extract the [k-core](<https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)>), such that each of the remaining users and items have k reviews each.

## 4.2 Baseline model

**HEM, ZAM / DREM, GraphSRRL**

앞의 두 모델은 background에서 다룬 latent space 기반 모델

뒤의 두 모델은 KG,graph를 통해 product search 를 다뤘으며 가장 좋은 성능을 보여왔기 때문에 선정

## 4.3. **Evaluation**

기본적으로 train/val/test split을 진행.

각 User에 따라 시간순으로 리뷰를 정렬하고, 마지막에서 두번째 시퀀스를 Validation, 마지막 시퀀스를 Test로사용.

**Measurement Metric**

**Hit rate(HR@K) :**

검색결과가 hit 했는지에 대한 단순한 rate

![img1.daumcdn.png](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/img1.daumcdn.png>)

**Normalized discounted cumulative gain (NDCG@K)**

이상적인 컨텐츠 순서와 실제 순서 간의 차이를 점수화

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%206.png>)

**Mean reciprocal rank (MRR)**

컨텐츠 순서에 가중치를 부여

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%207.png>)

## 4.4 Implementation detail

> For all methods, the batch size is set to 1024, and the ADAM optimizer is used with an initial learning rate of 0.001. All the **entity embeddings are initialized randomly with dimension 64**. #item, user, query

For our SBG, we set the **attention dimension $[𝑑_𝑎$ to 8]**, and the **user-query balancing** parameter **[𝜆 to 0.5]**.

We employ **4 layers of jumping graph convolution,** and the weight of self-loop is set to 0.1. The strength of **jumping connection [𝛽] is also set to 0.1.**
The negative sampling rate for each word is set to 5, and that for each item is set to 2.

>

## 4.5 Result

**RQ1&2 : 성능향상**

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%208.png>)

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%209.png>)

- Latent space 기반 모델 중 가장 좋은 성능을 보였던 ZAM에 비해 모든 실험에서 유의한 성능향상(RQ1)
- SOTA 였던 DREM에 비해 거의 모든 실험에서 유의한 성능향상(RQ2)
- ZAM 에 비해 DREM 이 실패하던 분야에서도 적지만 성능향상 →graph based method 의 영향이 domain에 따라 달라짐

**RQ3: Graph convolution의 효용**

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%2010.png>)

**RQ4 : 적절한 R의 선택**

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%2011.png>)

Magazine을 제외하고 day, week를 넘어가면 성능하락이 일어남

단, Magazine은 데이터셋 크기도 작고, 정보가 충분하지 않아서 R이 길어질수록 충분한 정보가 제공되며 발생한 결과로 추정

# 5. Conclusion

![Untitled](<SBG(for%20gitbook)%204f0e1d73bec545f7937734eec9902841/Untitled%204.png>)

본 연구는 SBG를 통해 user embedding 을 풍부화 하는 방법을 연구하였다. 이는 기본적인 잠재 공간 방법론의 철학을 유지하면서, 약간의 변화를 통해 성능향상을 가져왔기 때문에 추후 방법론들이 고도화 되어도 쉽게 적용시켜서 성능향상을 도모할 수 있어 보인다.(plug and play module)

단, 본 연구는 GCN 기반의 static graph 방법론을 채택하였기 때문에, 실제 문제에서 적용하기 힘든 면이 있을 것이라 판단된다, 저자들 또한 dynamic behavior graph를 다음 연구 방향으로 제시하고 있다.

관련된 연구들을 살펴보며 실제로 상당수 연구가 Amazon, Alibaba 와 같은 커머스 회사들의 펀딩으로 진행되었음을 관찰할 수 있었다. 실제 산업에서 관심있게 바라보고 있는 주제라고 판단되었다.

궁금했던 점은 Query를 어떻게 구현했는지 좀 더 구체적인 설명이 있으면 좋았을텐데 이에 대한 정확한 언급이 없았다. Github page가 공개되어 있지만, 내용이 없다.

SBG 는 결국 bipartite graph에 edge 구성도 단순했는데, 논문의 intro 에서 주장하듯 좀 더 rich 한 정보를 담기위해 더 복잡한 graph를 구성해 볼 수 있지 않을까 싶은 생각이 들었다.

# \***\*Author Information\*\***

김대희(Kim Daehee) is M.S student in the Graduate school of Knowledge Service Engineering of the Korea Advanced Institute of Science and Technology(KAIST). He has double B.S degrees in System Management Engineering and Computer Science in Sungkyunkwan University(SKKU). His research interest is applying graph neural network to product search and recommendation.He currently works at Knowledge Innovation Research Center, of the KAIST

# 6. Reference

- https://github.com/floatSDSDS/SBG
- [Learning a hierarchical embedding model for personalized product search(SIGIR-2017)](https://scholar.google.com/scholar_url?url=https://dl.acm.org/doi/pdf/10.1145/3077136.3080813&hl=ko&sa=T&oi=gsb-gga&ct=res&cd=0&d=15736059002742053164&ei=pstjYtiTHYySyASZk6HgCA&scisig=AAGBfm17kfq7KLvb8_VrBirjKb8qxDT-7w)
- [A Zero Attention Model for Personalized Product Search](https://arxiv.org/pdf/1908.11322.pdf) (CIKM-2019)
- [Distributed Representations of Sentences and Documents](https://proceedings.mlr.press/v32/le14.pdf) (PMLR-2014)
- [Simple and Deep Graph Convolutional Networks](https://scholar.google.co.kr/scholar_url?url=http://proceedings.mlr.press/v119/chen20v/chen20v.pdf&hl=ko&sa=X&ei=hjEyYvn_AYjwyAS2o4mYAw&scisig=AAGBfm2_hCGohcaQRDILYCo-3dLq2_o3hQ&oi=scholarr) (PMLR-2020)
- [http://jmcauley.ucsd.edu/data/amazon/](http://jmcauley.ucsd.edu/data/amazon/)
- [https://woosikyang.github.io/Graph-Convolutional-Network.html](https://woosikyang.github.io/Graph-Convolutional-Network.html)
- [https://lamttic.github.io/2020/03/20/01.html](https://lamttic.github.io/2020/03/20/01.html)
