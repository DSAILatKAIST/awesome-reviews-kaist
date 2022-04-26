---
description : Fu et al. / Towards Effective Context for Meta-Reinforcement Learning: an Approach based on Contrastive Learning / AAAI-2021  
---

# **Towards Effective Context for Meta-Reinforcement Learning: an Approach based on Contrastive Learning** 

[comment]: <> (Towards Effective Context for Meta-Reinforcement Learning: an Approach based on Contrastive Learning)

## **1. Problem Definition**  

비슷한 구조를 가지는 여러 task들이 주어진 상황에서 meta-RL은 과거 학습 경험을 바탕으로 common knowledge를 인식하고, 적은 양의 상호작용을 통해 이를 새로운 task에 적용. Context-based meta-RL에서 latent context의 퀄리티는 중요한 요소록 작용하며 알고리즘의 성능에 큰 영향을 미침. Context encoder 성능을 향상시키기 위해 contrastive learning 방법(Contrastive learning augmented Context-based Meta-RL; CCM)을 도입하고, 유용한 정보를 제공하는 trajectory 탐색 기법을 제안.


## **2. Motivation**  

Context-based meta-RL에서 latent context는 task들의 분포를 잘 파악하고 새로운 task를 잘 추론하는 것이 중요. 기존 방식들(Rakelly et al., 2019; Lee et al., 2020)은 불필요한 상관관계를 포착하거나 task 특유의 정보를 무시하는 경향을 보임. 또한 학습과정에 서로 구별되는 context 생성하기 위한 탐색(exploration)에 대한 중요성을 무시하는 경향이 있음. 본 논문에서는 context encoder 성능을 향상시키기 위해 contrastive learning 방법(Contrastive learning augmented Context-based Meta-RL; CCM)을 도입하고, 유용한 정보를 제공하는 trajectory 탐색 기법을 제안.


## **3. Method**  

### Preliminary
#### Meta-RL
Meta-RL에서는 여러 task들이 존재 - task 분포 $p(\mu)$ 가정하고, 각 task $\mu \sim p(\mu)$는 비슷한 구조를 공유하지만 서로 다른 MDP, $M_{\mu} = \{S, A, T_{\mu}, R_{\mu}\}$에 대응. 동일한 MDP 에 대해 $N$번 시도하는데, 처음 $K$ 에피소드 동안은 탐색을 수행하고 남은 $N-K$ 에피소드는 탐색을 통해 수집한 데이터를 이용하여 execution 수행. Context-based meth-RL에서는 에이전트의 정책함수 $\pi_{exe}$가 과거 모든 트랜지션 $\tau_{1:\Tau} = \{ (s_1, a_1, r_1, r'_1), \ldots,  (s_{\Tau}, a_{\Tau}, r_{\Tau}, r'_{\Tau}) \}$를 기반으로 결정 됨. 에이전트는 수집된 trajectories를 인풋으로 contex encoder $q(z|\tau_{1:\Tau}$)를 통해 latent context $z$ 생성. 이를 통해 기대되는 리턴 값을 최대화하려고 함.

![1](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_expected_return.png) 

#### Contrastive Learning

본 논문에서는 대체로 van den Oord, Li, and Vinyals(2018)의 방법을 사용하였음. Contrastive learning의 기본적인 아이디어는 의미상 비슷한 데이터(positive key )들이 latent 공간에서도 가깝게 위치하고, 의미상 비슷하지 않은 데이터들(negative keys)과는 latent 공간에서 멀도록 representation 함수를 학습하는 것. 총 $K$ 개의 negative keys와 하나의 positive key 가 존재할 때 InfoNCE (noise contrastive estimation) loss는 다음과 같이 계산 ($f$는 유사도 측정 함수로, 보통 $q^TWk$ 사용).

![2](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_NCE_loss.png) 

이 NCE loss 를 최소화하는 것은 $q$와 $k$의 mutual information 의 하한값을 최대화하는 것과 동일 ($K$가 클수록 하한이 tight해짐).

![3](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_mutual_info.png) 

### CCM Algorithm

Contrastive Learning 을 meta-RL 에 적용하기 위해, 현재 정책함수를 사용하영 각 task에 대한 trajectory 생성 및 저장.  $n$번째 학습에서, $p(\mu)$ 분포로 부터  task $\mu_n$ 샘플링하고 $\mu_n$ task로 부터 생성된 서로 다른 트랜지션 배치 $b^q_n, b^{k}_n$를 독립적으로 샘플링. $b^q_n$은 contrastive learning의 query 가 되고, $b^{k+}_n$ 는 positive key로 사용됨. 이제 $\mu_n$이 아닌 task로 부터 생성된 트랜지션에서 남은 $M-1$개 만큼 negative keys $\{b^{k}_j\}_{j=1}^M$ 샘플링. 샘플링 된 query, key 배치들을 각각 context encoding 하여 얻은 latent context를 $z_q, z_k$라고 할때, contrastive loss 는 InfoNCE loss를 사용하여 다음과 같이 계산.

![4](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_contrastive_loss.png) 


이 과정을 그림으로 나타내면 다음과 같음. 여기서 momentum encoder는 context encoder 의 momentum averged version (context encoder 파라미터로 일정 비율만큼 조금씩 업데이트 하는 방식, DDQN target network 업데이트 방식을 생각하면 됨).

![5](/.gitbook/2022-spring-assets/HyeonahKim_1/fig_encoder.png)

### Information-gain-based Exploration 

Contrastive learning 으로 context encoder를 잘 학습하기 위해서는 여기에 사용되는 trajectory들이 다양하고 유용한 정보를 가지고 있어야 함. 앞서 motivation에서 언급했듯이 기존 context meta-RL 연구들에서는 이러한 탐색의 중요성이 부각되지 않았음. 대부분 Thomson sampling 등의 방식을 사용하였는데 이러한 탐색 기법은 행동 정책(executive policy)과 탐색 정책(exploration policy)이 동일하기 때문에 이전에 학습된 task 를 푸는데 유리한 방식으로만 탐색하게 되는 경향을 보임. 이러한 경향은 새로운 task에 대한 적응력을 저해할 수 있음.

본 논문에서는 탐색 정책과 행동 정책을 분리하고, 유용한 정보를 담고 있는 트랜지션을 수집하도록 탐색 정책을 학습함. 새로운 트랜지션의 정보적 유용성을 판단하는 지표로 정보 이론의 *information gain* 을 사용.

![6](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_info_gain.png)

위 식을 다음과 같이 바꿔쓸 수 있음 (여기서 mutual information $I(z;\tau_{1:i}) = H(z) - H(z|\tau_{1:i})$).

![7](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_td_ig.png)

즉, 새로운 트랜지션의 정보적 유용성은 mutual information의 temporal difference로 정의할 수 있음. 여기서 context encoder $e$가 충분하다면 (인코딩 과정에서 mutual information 이 보존된다면), 하기와 같이 근사할 수 있음 ($z \approx e(b_{pos})=c_{pos}$).

![8](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_approx_ig.png)

$I(C_{pos};c_{1:i})$의 하한과 $I(C_{pos};c_{1:i})$의 상한을 사용하면 $I(z|\tau_{1:i-1};\tau_{i})$의 하한값 추산 가능. Preliminary 단계에서 언급되었던 하한식을 사용하면 다음과 같이 $I(C_{pos};c_{1:i})$의 하한을 얻게 됨.

![9](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_lb.png)

$W$는 tasks의 수를 나타내며, $C=C_{pos} \cup C_{neg}$. 비슷한 전개를 통해 $I(C_{pos};c_{1:i})$의 상한을 얻을 수 있으며 논문에 자세히 증명되어 있음.

![10](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_ub.png) 

위 두식을 (6)에 대입하면 다음과 같은 식을 얻을 수 있고(추가: 이 과정에서 (8)번식은 $C$에 대한 기대값인 반면, (12)에서는 $C_{pos}$에 대한 기대값으로 변경되었으나 자세한 기술은 찾을 수 없었음), $I(z|\tau_{1:i-1};\tau_{i})$의 하한값인 $L_{upper} -L_{lower}$를 전체적인 RL 리워드에 추가시켜 학습 진행. 

![11](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_12.png) 
![12](/.gitbook/2022-spring-assets/HyeonahKim_1/eq_reward.png)

전반적인 학습 알고리즘은 다음과 같음.

![13](/.gitbook/2022-spring-assets/HyeonahKim_1/algo_meta_training.png)


## **4. Experiment**  

CCM이 기존 방법들과 결합되었을 때의 효과.
In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

### **Experiment setup**  
* 환경 - MuJoCo (Todorov et al., 2012)
	* image
	* humanoid-dir, cheetah-mass, cheetah-mass-OOD, ant-mass, cheetah-vel-OOD, cheetah-sparse, walker-sparse, hard-point-robot (자세한 설명은 아카이브 버전 Appendix 참고)
	* Out of distribution (OOD) 버전  - 학습 환경과 다른 분포에서 생성된 버전
	* Sparse: 목표에 도달했을때만 reward 발생
	
	![14](/.gitbook/2022-spring-assets/HyeonahKim_1/fig_tasks.png)
  
* Baseline  
	* Recovering value-function (RV) - REARL (Rakelly et al., 2019)
	* Dynamic prediction (DP) - CaDM (Lee et al., 2020) 
	* MAML (Finn, Abbeel, and Levine, 2017), PEARL, ProMP (Rothfuss et al., 2019), vraibad (Zintgraf et al., 2020),
* Evaluation Metric  
	* Average return

### **Result**  
**CCM이 기존 방법들과 결합되었을 때의 효과**

기존 방법들에 CCM을 결합함으로써 기존보다 성능 향상이 이루어짐을 확임. 공평한 비교를 위해 모두 동일한 네트워크 구조(actor-critic, context encoder)를 사용하였으며 PEARL과 동일한 평가 방식 사용.

![15](/.gitbook/2022-spring-assets/HyeonahKim_1/exp_1.png)

* 특히 DP 방식의 경우 cheeta-vel-OOD에서 CCM을 통해 학습 효과가 크게 증가
* CCM이 OOD 버전에서 적응력이 향상되었으나 여전히 학습 데이터와 분포가 일치할때 훨씬 잘함 (cheetah-mass에서는 1500 넘었으나 OOD에서는 1000을 약간 넘는 수준).
* 학습 환경과 분포가 일치할 때는 CCM-DP가, OOD 버전에서는 CCM-RV가 높은 성능을 보임.

**다른 SOTA meta-RL 과의 비교**
순서대로 왼쪽부터 walker-sparse, cheetah-sparse, hard-point-robot task.

![16](/.gitbook/2022-spring-assets/HyeonahKim_1/exp_2.png)

* Off-policy 방법인 CCM 과 PEARL이 on-policy 방법인 MAML, ProMP, varibad 보다 학습이 잘되는 경향을 보임
* 여기서의 PEARL은 contrastive learning 이 추가된 버전인 PEARL-CL 로 추측됨 (논문의 기술이 명확하지 않음) -> CCM과 PEARL 의 성능 차이가 information-gain-based 탐색의 효과로써 기술 되어 있음.

**Regularization Term 효과**

식 (12)을 다음과 같이 쓸 수 있으며, 이때 두번째 부분은 regularization term 으로 해석가능. 실제로 이 부분을 고려하지 않으면 고려할 때보다 학습이 불안정함을 실험으로 확인.

![17](/.gitbook/2022-spring-assets/HyeonahKim_1/exp_3.png)

**추가 실험 (아카이브 버전 Appendix)**
Contex encoder의 업데이트를 traing step 단위로 할 것인지 에피소드 단위로 할 것인지에 따라 성능 차이를 보이며, reward에 information-gain term scale 에 대한 성능 차이를 실험으로 보임. 하이퍼파라미터에 따른 성능 차이 존재.

![18](/.gitbook/2022-spring-assets/HyeonahKim_1/exp_4.png)


## **5. Conclusion**  

본 논문에서는 context-based meta-RL에서 1) task 관련 정보를 잘 추출하도록 context encoder를 학습할 것인가, 2) 어떻게 유용한 정보를 가진 trajectories를 생성할 것인가라는 질문에 대해 각각 contrastive learning과 information-gain-based 탐색을 제안하고 있음. Vision 쪽에서 이미지에 대해 적용되던 contrastive learning 을 meta-RL의 task에 맞게 적용했다고 생각 함. 또한 그동안 meta-RL 중요하게 다뤄지지 않았던 탐색 기법에 대해 새로운 제안을 하였고 두 가지 모두 기존 meta-RL 프레임워크에 쉽게 적용가능하기 때문에 유용한 것 같음.

다만 논문에서 식을 전개하거나 실험에 대해 해석하는 부분의 기술이 모호한 점이 아쉬움. CCM의 성능을 보이는 과정에서 다른 방법론과 결합한 결과만 비교되어서 CCM only vs. RV(PEARL) vs. DP(CaDM) 의 성능 차이를 알 수 없었음(아카이브 버전 논문 appedix에 해당 결과가 있었으나 일부 task에서만 수행했으며 결과에 대한 기술이 부족함). Information-gain-based 탐색의 효과를 검증할 수 있는 실험이 추가되면 좋을 것 같음. 코드 공개가 되어있지 않아 디테일한 구현에 대해 확인할 수 없어 아쉬웠음.

---  
## **Author Information**  

* 김현아
    * SILAB., KAIST
    * Graph Representation Learning, Deep Learning for Combinatorial Optimization
	
  
## **6. Reference & Additional materials**  

* Github Implementation  (X)
* Reference  
	* Finn, C., Abbeel, P., & Levine, S. (2017, July). Model-agnostic meta-learning for fast adaptation of deep networks. In _International conference on machine learning_ (pp. 1126-1135). PMLR.
	* Lee, K., Seo, Y., Lee, S., Lee, H., & Shin, J. (2020, November). Context-aware dynamics model for generalization in model-based reinforcement learning. In _International Conference on Machine Learning_ (pp. 5757-5766). PMLR.
	* Rakelly, K., Zhou, A., Finn, C., Levine, S., & Quillen, D. (2019, May). Efficient off-policy meta-reinforcement learning via probabilistic context variables. In _International conference on machine learning_ (pp. 5331-5340). PMLR.
	* Rothfuss, J., Lee, D., Clavera, I., Asfour, T., & Abbeel, P. (2018). Promp: Proximal meta-policy search. _arXiv preprint arXiv:1810.06784_.
	* Todorov, E., Erez, T., & Tassa, Y. (2012, October). Mujoco: A physics engine for model-based control. In _2012 IEEE/RSJ international conference on intelligent robots and systems_ (pp. 5026-5033). IEEE.
	* Van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. _arXiv e-prints_, arXiv-1807.
	* Zintgraf, L., Shiarlis, K., Igl, M., Schulze, S., Gal, Y., Hofmann, K., & Whiteson, S. (2019). Varibad: A very good method for bayes-adaptive deep rl via meta-learning. _arXiv preprint arXiv:1910.08348_.

* Wikipedia
	* https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
	* https://en.wikipedia.org/wiki/Mutual_information
