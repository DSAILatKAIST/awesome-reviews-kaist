---
description : Dupty et al. / PF-GNN- Differentiable particle filtering based approximation of universal graph representations / ICLR-2022  
---

# **PF-GNN: Differentiable particle filtering based approximation of universal graph representations** 

[comment]: <> (PF-GNN: Differentiable particle filtering based approximation of universal graph representations)

## **1. Problem Definition**  

Graph neural network (GNN)은 이웃한 노드의 정보를 반복적으로 업데이트하면서 그래프의 구조적 정보를 배우는 방법이다. 이러한 GNN의 메시지 전달 과정은 그래프 동형 (graph isomorphism) 여부를 판단하는 노드 color-refinement 과정 (또는 1차원 Weisfeiler-Lehman (WL) 과정)과 동일하다. 따라서 GNN은 1-WL color-refinement 가 겪는 표현력의 한계를 동일하게 가진다.


## **2. Motivation**  

1-WL color-refinement  각 스텝( = GNN 메시지 전달 hop)에서는 자신과 이웃한 노드의 정보를 모으고, 이 정보가 서로 다르다면 자신의 색(label, embedding)도 이웃 노드와 다르게 선택하는 과정을 수행한다 (자세한 내용은 [이곳](https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/)을 참조). 
![enter image description here](https://www.researchgate.net/publication/338590038/figure/fig1/AS:847460675100673@1579061778427/Weisfeiler-Lehman-WL-relabeling-Two-iterations-of-Weisfeiler-Lehman-vertex-relabeling.png)
그림출처: Graph kernels: A survey

이 과정을 어느정도 반복하게 되면 더 이상 색이 변하지 않게 된다.  중요한 점은 그래프 구조의 symmety 때문에 1-WL 수행 후 동일한 결과가 나왔다고 해서 동형 그래프임을 보장하지 않는다 (**표현력의 한계**). 이를 극복하기 위해서 높은 차원의 WL (또는 k-GNN)이 제안되었지만 계산량이 지수적으로 증가하기 때문에 k 값이 크면 실용적이지 못하다. 논 연구에서는 1-WL의 표현력을 높이기 위해 정확한 동형 솔버 (exact isomorphism solvers)를 사용하고 샘플링 기법을 통해 계산량을 선형으로 유지한다. 본 논문에서는 주어진 그래프의 neural representation 이 그래프를 유일하게 구분할 수 있도록 하는 GNN 을 제안하고자 한다.



## **3. Method**  

### Preliminary
#### 1-WL with individualization and refinement (IR)
Individualization 과정에서는 노드 하나의 색(임베딩)을 의도적으로 변경함으로써 aymmetricity를 부여한다. 이 후 refinement 과정은 1-WL을 그대로 따르며, 의도적으로 도입한 정보를 전파시킨다.

![.](/.gitbook/2022-spring-assets/HyeonahKim_2/fig1.png) 

이 과정에서 어떤 노드를 individulaize 할 것이고, 어떤 순서로 refinement 할 것인지에 따라 다양한 경로가 발생한다. 이를 컬러링 탐색 트리로 표현 가능하다. 여기서 permutation-invariance를 유지하기 위해서 동일한 색을 가지는 노드 집합에 대해서만 IR 을 수행한다. 이 과정을 통해 얻은 탐색 트리는 그래프 동형 집합에서 유일하게 존재한다. 즉, 동형 그래프가 아닌 모든 그래프에 대해서는 구분되는 탐색 트리를 가지게 된다.

![.](/.gitbook/2022-spring-assets/HyeonahKim_2/fig2.png)


#### Particle Filtering (Refer Appendix A)

파티클 필터링은 비선형 다이나믹 시스템에서 제한적인 관찰이 이루어질 때 스테이트의 사후 분포를 sequentially 추정하는 몬테카를로 방법이다. (논문과 아래 그림의 노테이션이 다른 경우 논문, 그림 순으로 표기)
$$x_t: state$$ $$o_t, y_t: observation $$ $$w_t, \omega_t: weight$$ $$ p(x_{t+1}|x_t): transition$$ $$p(o_t|x_t) : observation\_model$$ $$b_t(\mathcal{G})= \langle x_t^k, w_t^k \rangle, \{x_t^{(i)}, \omega_t^{(i)}\}, \left( \sum_k w^k_t=1 \right) : belief$$ $$p(x_T|o_{t=1:T}) : posterior\_distribution$$
이를 통해 K개의 파티클, 가중치로 이루어진 스테이트 공간에 대한 belief를 통해 사후 분보포 (가우시안이 아니어도 가능)를 추산하는 것을 목적으로 한다.
![enter image description here](https://d3i71xaburhd42.cloudfront.net/006e8089dad4183deeeda2e1f5038d7a3663e614/3-Figure1-1.png)
그림출처: Two Stage Particle Filter for Nonlinear Bayesian Estimation

매 스텝마다, 각 파티클들은 새로운 스테이트로 전이되고 이에 대한 새로운 가중치를 얻는다 (eta 는 normalizing factor). $$x^k_{t+1} \sim p(x^k_{t+1}|x^k_t)$$ $$ w^k_{t+1} = \eta \cdot p(o_{t+1}|x^k_{t+1})$$

이 때 가중치가 degenerate 되지 않게 하기 위해 resampling 단계를 거쳐 각 파티클들이 동일한 가중치를 가지도록 재설정 해준다.


### Particle Filter Graph Nerual Network (PF-GNN)

위에서 설명한 두 가지 컨셉을 GNN representation 학습에 적용한다.

#### Initialization
먼저 노드 특성으로 (노드 특성이 없다면 특정 상수로) 노드 임베딩을 초기화하고 1-WL GNN을 정해진 수만큼 수행한다.  1/K 가중치로 K 개 임베딩으로 belief 형성한다 (n: 그래프 노드 수, d: latent dim ?). 
$$b_1= \langle (\mathcal{G}, H^K_1), w^k_1 \rangle_{k=1:K}, H^k_1= node\_embedding (n\times d)$$

#### Transition step
각 파티클 k에 대하 하나의 노드를 골라 individualize, refinement 수행한다. 이를 위해 policy 함수를 배우는 데, 이는 노드 임베딩 H가 주어졌을 때 노드 셋에 대한 이산 확률에 해당하며, 각 노드에 대한 양수의 스코어를 제공한다. $$P(\mathcal{V}|H^k_t;\theta)$$ 이 점수를 normalize 해서 노드를 샘플링함으로써 individualize 할 노드를 고르는 것과 새로운 임베딩을 할당하는 것 (recoloring) 역시 데이터를 통해 학습한다.

![.](/.gitbook/2022-spring-assets/HyeonahKim_2/eq4_5.png)
![.](/.gitbook/2022-spring-assets/HyeonahKim_2/eq6.png)

여기서 M은 1로 이루어진 mask matrix 이다. 논문에서는 자세하게 기술되지 않았지만 코드 상으로 봤을 때, individualizing  노드의 임베딩을 통해 새로운 임베딩을 얻어내는데, 이 때 나머지 노드들은 그대로 유지하게 하기 위한 장치로 보인다.
새롭에 얻은 노드 임베딩을 GNN을 통해 다시 한번 업데이트 한다. GNN은 각 스텝마다 다른 파라미터를 사용해도 되지만 파티클 k에 대해서는 동일한 파라미터를 쉐어링한다. 따라서 파티클 수의 증가가 학습 파라미터의 증가로 이어지지 않는다.

#### Particle weights update step
매 transition 마다 observation 을 받고, 관찰 함수 f_obs 를 통해 이 관측에 대한 likelihood 를 추정한다. PF-GNN의 경우 새로운 정보는 refinement 에 포함되기 때문에 observation을 refined coloring에 conditioned latent 변수로 모델링하였다. 학습 가능한 관찰 함수는 노드 임베딩에 대한 점수를 나타내는 set function approximator 이고 가중치 업데이트에 사용된다.
$$f_{obs}(H^k_{t+1}; \theta_o) : observation\_function,$$ $$w^k_{t+1} = \frac{f_{obs}(H^k_{t+1}; \theta_o) \cdot w^k_t}{\sum_k f_{obs}(H^k_{t+1}; \theta_o) \cdot w^k_t} $$
이 관찰 함수는 particle filter 에서의 observation model p(o|x) 에 해당한다.

#### Resampling step
가중치의 degeneracy를 완화하기 위해서 이산 분포 $$\langle w^k_{t+1} \rangle_{k=1:K}$$에서 K개의 파티클을 resampling 하고 이들의 가중치를 1/K로 재설정하는 과정이 필수적이다. 그러나 이는 미분 가능한 operation이 아니기 때문에 이 논문에서는 Karkus et al. (2018)에서 제안된 soft-resapmling 전략을 사용하였다. 이 새로운 가중치는 importance sampling 을 통해 계산 가능하다. 
$$q_t(k) = \alpha w^k_{t} + (1-\alpha) 1/K, \alpha \in [0, 1]$$ $${w'^k_t} = \frac{p_t(k)}{q_t(w)} = \frac{w^k_t}{\alpha w^k_{t} + (1-\alpha) 1/K} $$

#### Readout
T번의 IR 과정 수행 후 얻어진 노드 임베딩에 평균으로 readout을 수행한다.
$$\sum_{k=1:K} w^k_T H^k_T$$
여기까지 기술된 알고리즘의은 다음과 같다.
![.](/.gitbook/2022-spring-assets/HyeonahKim_2/algorithm1.png)


#### Training
I를 individualized 된 노드들의 시퀀스라고 할때, 탐색 트리 상에서 leaf node에 해당하는 그래프들에서 얻어낸 예측치 y hat 과 타겟 y의 loss를 최소화하도록 학습한다 (REINFORCE방식).

![.](/.gitbook/2022-spring-assets/HyeonahKim_2/eq8_9.png)

기본적으로는 imitation learning이고, PF-GNN으로 얻어낸 학습 샘플이 color-refinment 탐색 트리의 leaf-node 들 중 K개에 해당하는 것으로 볼 수 있을 것 같다. Leaf-node까지 도달하는 과정에서 파티클 선택, IR, 가중치 업데이트, resampling 이 수행되며 모두 미분가능하기 때문에 마지막 loss gradient 를 사용하여 end-to-end로 학습 가능하다.


## **4. Experiment**  

본 논문에서는 총 3가지 실험 (그래프 동형 판단, 그래프 특성 detection, 실제 환경 데이터셋)을 진행하였다.

### Graph Isomorphism detection
SR5, EXP, CSL 3가지 데이터셋에 대해 학습/테스트 수행하였다. 앞에 두가지 데이터셋은 주어진 그래프 페어에 대해 동형 여부를 판단하는 테스크 (table1)이고 CSL은 주어진 그래프들을 동형 그래프 집합으로 분류하는 테스크 (table2) 이다.

![.](/.gitbook/2022-spring-assets/HyeonahKim_2/table1_2.png)

PF-GNN의 가장 큰 장점은 다양한 GNN 구조에 쉽게(?) 적용 가능하다는 것이다 (이 부분은 GNN 구조의 변경 없이 추가적인 코딩으로 구현할 수 있다는 뜻으로 해석된다). SR5 와 EXP 셋에 대해서는 각 network에 PF-GNN을 추가 구현했을 때 성능 차이를 비교하였고, CSL 에서는 GIN에 PF-GNN 추가한 것과 다른 GNN을 비교한 것으로 보인다.
Table 1에서 보면 PPGN과 GNNML3가 3-WL과 동등한 expression power를 가진 모델인데 이보다 1-WL GNN 구조에 PF-GNN을 추가 구현한 결과가 월등히 좋은 것을 실험적으로 보였다.

### Graph properties
데이터셋 LCC, TRIANGLES 은 1-WL GNN 구조들이 실패하는 테스크들이다.
* LCC: 노드 클러스터링 co-efficeint 찾기
* TRIANGLES: 그래프에서 triangle 형성하는 노드 집합의 수 예측

![.](/.gitbook/2022-spring-assets/HyeonahKim_2/table3.png)

두 데이터셋 모두 일반버전과 large 버전이 있는데 학습은 작은 데이터셋 (노드 25개 이하)에서 학습해서 큰 그래프 (대략 100개 노드)에서도 evaluation 수행했다.
RNI (Abboud et al., 2020)은 1-WL GNN이지만 초기 노드 임베딩에 randomness를 부여하여 universal approximate 가능하게 한 모델인데 randomness 다양하게 비교해도 PF-GNN의 성능이 더 좋았다.

### Real-world benchmarks
Torch geometry 에서 제공하는 벤치마크 데이터셋에 대해 PF-GNN 적용 (자세한 설명은 torch geometry 사이트 참조).

![.](/.gitbook/2022-spring-assets/HyeonahKim_2/table4_7.png)

### Ablation studies
여기서는 파티클 필터링 과정에서 파티클 수와 refinment 스텝에 따른 성능 및 런타임을 비교하였다.  흥미로운 점은 T가 일정 수준을 넘어가면 오히려 성능이 떨어지는 현상이 나타난다. 개인적으로 이런 이유에서 T와 K를 튜닝하는데 어려움이 있을 것 같다. Resampling 과정에서 사용하는 알파에 대해서는 성능 비교가 없었다.

![.](/.gitbook/2022-spring-assets/HyeonahKim_2/ablation.png)


## **5. Conclusion**  

본 논문에서는 1-WL GNN의 표현력을 높이기 위한 방법으로 graph isomorphism solver와 particle filter를 활용한 기법을 제안한다. 각 GNN 구조는 유지하되 K개의 파티클을 사용하여 individualziation 을 통해 그래프 symetricity 를 깨고 다시 refinement 하도록하는 과정을 통해 노드 임베딩이 더 많은 정보를 가지도록 학습한다. 또한 이 모든 과정이 미분 가능하여 end-to-end 학습이 가능하도록 디자인하였다. 

다만 논문에서 식을 전개하거나 과정을 기술하면서 정의되지 않은 notation의 사용이나 설명의 부족하다. Particle filter나 color-refinement에 대해 이 논문을 통해 처음 접했는데 이해를 위해 다른 자료를 많이 참고해야 해서 논문을 읽는 것이 어려웠다. 특히 일부 수식은 오퍼레이션 설명이 부족해서 코드를 통해 이해해야 했고 증명을 따라가기 힘들다. 공개한 코드의 경우 모든 실험이 구현되어 있지는 않았다. 논문의 모티베이션이나 하고자 하는 바의 아이디어는 좋았으나 이를 설명하고 실험을 수행하는 과정에서 추가적인 지식 없이 논문을 이해하는 것이 어렵다는 점이 가장 큰 단점으로 보인다.

테스크와 사용하는 GNN 구조에 상관없이 추가적인 IR 과정을 통해 GNN의 표현력을 높인다는 점에서 매우 흥미로웠다. GNN을 사용하는 강화학습에서 이와 같은 방법을 통해 표현력을 높일 수 있을지 고민해볼 필요가 있을 것 같다. 또한 k-WL GNN 보다는 연산이 작다고 했으나 만약 테스크가 iterative 하게 그래프를 인코딩, 디코딩해야 한다면 속도 측면에서 얼마나 영향이 있을지 생각해 볼 필요가 있는 것 같다 (파티클 k와 refinement step 만큼 학습 과정이 늘어남 - 병렬처리 감안해도 반복적인 인코딩이 필요한 경우 영향을 무시하기 어려울듯).

---  
## **Author Information**  

* 김현아
    * SILAB., KAIST
    * Graph Representation Learning, Deep Learning for Combinatorial Optimization
	
  
## **6. Reference & Additional materials**  

* Github Implementation
	* https://github.com/pfgnn/PF-GNN
* Reference  
	* Nikolentzos, G., Siglidis, G., & Vazirgiannis, M. (2019). Graph kernels: A survey. _arXiv preprint arXiv:1904.12218_
	* Wang, F., Zhang, J., Lin, B., & Li, X. (2018). Two stage particle filter for nonlinear Bayesian estimation. _IEEE Access_, _6_, 13803-13809.
	* Karkus, P., Hsu, D., & Lee, W. S. (2018, October). Particle filter networks with application to visual localization. In _Conference on robot learning_ (pp. 169-178). PMLR.

* Wikipedia / blog
	* 1-WL: https://davidbieber.com/post/2019-05-10-weisfeiler-lehman-isomorphism-test/
	* torch geometry: https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html


