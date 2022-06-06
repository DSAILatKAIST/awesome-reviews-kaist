---
description: >-
  Kimin Lee / Context-aware Dynamics Model for Generalization in Model-Based
  Reinforcement Learning / ICML-2020
---

# CaDM

## **1. Problem Definition**

다양한 dyanmic에서 동시에 잘 작동하는 global model을 만드는 것은 매우 어렵다. (2개의 환경(A, B)가 존재한다고 하자. 일반적으로 환경A에서만 학습시킨 모델1, 환경 A와 B에서 모두 학습시킨 모델2를 비교하면, 환경A에서는 모델1이 더 높은 performance를 보일 확률이 높다.)

논문에서는 이와 같은 차이를 context latent vector를 통해서 해결하려 한다. local dynamic을 잘 표현할 수 있으며, next state를 예측하는 데에 있어 주요한 역할을 한다.

이러한 context vector를 generate 하기 위해서 novel loss function과 학습 방식을 소개한다.

## **2. Motivation**

model-based Reinforcement Learning(MBRL)은 경험해보지 못한 새로운 dynamic에서의 적응을 상당히 힘들어 한다. 실제로 cartpole 환경에서 무게를 약간 바꾼 경우, 상당히 부정확한 next-step prediction을 보여준다. 이는 곧 MBRL에서 transition dynamic을 온전히 알아야하는 부담을 준다.(비현실적이라고 볼 수 있는 점이다)

이러한 문제를 해결하기 위해 meta-learning, graph network 등의 방법이 시도되었다. 특히나 meta-learning의 경우에는 recent trajectory에 적응하여(적은 수의 gradient update를 통해)로 부터 hidden state를 RNN과 같은 방식으로 update하고 이를 model의 input으로 사용한다. 하지만 저자는 이와 같은 방식의 적은 수의 gradient update는 rich contextual information를 추출하기 어렵다고 주장한다.

따라서 context encoding (i.e., capturing the contextual information) and transition inference (i.e., predicting the next state conditioned on the captured information)를 분리하고, 보다 효과적으로 environment dynamic을 학습할 수 있는 방법을 제시한다.

이 방법을 저자는 Context-aware Dynamics Model(CaDM)라고 명명했다.

## **3. Method**

학습에 필요한 sample은 매 iteration마다 for loop를 돌면서 Batch에 저장된다.

> **context encoder**

context encoder g parameterized by ![img\_3.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_3.png)

which produces a latent vector ![img\_5.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_5.png) given K past transitions ![img\_6.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_6.png)

recent experience로 부터 정확한 context를 추출할 수 있을 것이라는 직관에서 다음과 같이 encoder를 구성하였다. 다음 encoder는 2가지 방법을 학습된다. 첫 번째는 forward dynamic model, 두 번째는 backward dynamic model이다.

> **forward dynamic**

![img.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_0.png)

forward dynamic model f는 위의 그림과 같이 timestep (t-K) 부터 (t-1) 까지의 recent experience로부터 encoding된 context, current state s\_t, current action a\_t를 input으로 하는 모델이다. model은 주어진 input을 기반으로 environment의 next state s\_(t+1)을 예측한다.

위 언급 과정을 timestep t 뿐만이 아닌, 정해진 hyperparameter M에 따라 t+M-1까지 진행한다.

stochastic한 model을 기반으로 하여서, MSE와 같은 raw next state와의 직접 비교가 아닌, ![img\_7.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_7.png) 로 model의 transition probability를 높이는 방향으로 학습을 진행한다.

> **backward dynamic**

![img\_1.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_1.png)

backward dynamic model b는 위의 그림과 같이 timestep (t-K) 부터 (t-1) 까지의 recent experience로부터 encoding된 context, next state s\_t+1, current action a\_t를 input으로 하는 모델이다. model은 주어진 input을 기반으로 environment의 current state s\_t를 예측한다.

위 언급 과정을 timestep t 뿐만이 아닌, 정해진 hyperparameter M에 따라 t+M-1까지 진행한다.

stochastic한 model을 기반으로 하여서, MSE와 같은 raw next state와의 직접 비교가 아닌, ![img\_8.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_8.png) 로 model의 transition probability를 높이는 방향으로 학습을 진행한다.

forward와 backward 양 쪽을 모두 predidct한 다음, backward loss에 penalty parameter beta만큼 weight를 준 loss\_prediction을 아래 사진과 같이 정의한다. 아래 사진의 loss 값을 기준으로 context encoder, forward dynamic model, backward dynamic model을 모두 학습한다. ![img\_9.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_9.png)

이와 같이 생각한 이유는 backward dynamic을 예측하는 것이 환경에 대해 보다 많은 정보를 제공받을 수 있는 방법이라고 생각하였기 때문이다. (추후에 결과에서 과연 정말로 그런지 언급할 것이다)

> **additional training detail**

raw state를 input으로 넣어주는 것이 아닌, state의 difference(s\_(t+1) - s\_t)를 context encoder의 input으로 제공하였다.

data collecting을 실행할 때 있어서 Model Predictive Control(MPC)를 사용하여서 최고의 action을 매 step마다 고르는 방식으로 진행하였다. 현재 state가 주어졌을 때, sequential한 action을 distribution으로 부터 random하게 N개 뽑는다. 뽑힌 N 개의 action sequence들 중에서 best performing인 action을 base로 action distribution을 조정하고 조정된 distribution의 mean 값을 current state의 action으로 사용한다.

> **combination with model-free RL**

context encoder, forward and backward dynamic model을 통해서 환경에 대하여 학습을 완료했다고 가정하였다. state만 input으로 하는 것이 아닌, context 또한 함께 input으로 한 context-conditional policy를 구현할 수 있다.(policy의 additional input이 context가 되는 구조이다) ![img\_10.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_10.png)

> **Training Algorithm**

algorithm을 다음 사진과 같이 정리하였다. ![img\_2.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_2.png)

## **4. Experiment**

simulated robots (i.e., HalfCheetah, Ant, CrippledHalfCheetah, and SlimHumanoid)에서 proposed된 모델의 성능을 검사합니다.

자세한 디테일은 논문의 supplementary material을 참조하시길 바랍니다.

![img\_22.png](pictures\_20213592\_1/img\_22.png)

### **Experiment setup**

#### **experiment1: comparison with the Model-based Rl Methods**

* compared models
  * Vanilla dynamics model (Vanilla DM): Dynamics model trained to minimize the standard one-step forward prediction loss)
  * Stacked dynamics model (Stacked DM): Vanilla dynamics model which takes the past K transitions as an additional input)
  * Gradient-Based Adaptive Learner (GrBAL; Nagabandi et al. 2019a): Model-based meta-RL method which trains a dynamics model by optimizing an adaptation meta-objective
  * Recurrence-Based Adaptive Learner (ReBAL; Nagabandi et al. 2019a): Model-based meta-RL method similar to GrBAL
  * Probabilistic ensemble dynamics model (PE-TS; Chua et al. 2018): An ensemble of probabilistic dynamics models designed to incorporate both environment stochasticity and subjective uncertainty into the model
  * combine CaDM with Vanilla Dm
  * combine CaDM with PE-TS
* Evaluation Metric
  * average reward

#### **experiment2: comparison with the Model-free Rl Methods**

* compared models
  * Proximal Policy Optimization (PPO; Schulman et al. 2017)
  * Stacked PPO, which takes the past K transitions as an additional input
  * PPO with probabilistic context (PPO + PC), which learns context variable by maximizing the expected returns (Rakelly et al., 2019).
  * PPO with environment probing policy (PPO + EP) that takes embeddings extracted from initial interaction with an environment as an additional input(Zhou et al., 2019)
  * PPO with CaDM
* Evaluation Metric
  * average reward

#### **experiment3: effects of prediction loss**

* compared models
  * one-step forward prediction loss
  * future-step forward prediction loss
  * future-step forward and backward prediction loss
* Evaluation Metric
  * average reward

### **Result**

#### **result1: comparison with the Model-based Rl Methods**

![img\_13.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_13.png) 위의 표는 훈련 및 테스트 환경에서 다양한 MBRL 방법의 성능을 보여준다. CaDM은 특히나 더 복잡한 환경(halfCheetah, Ant, Humanoid)에서 성능을 크게 향상시킨다. 게다가 stacked DM의 경우 때때로 Vanilla DM보다 성능이 저하되는 경우가 존재한다. 이는 과거 history를 stack하는 것보다 context 추론을 통해서 환경의 정보를 근사하는 것이 보다 효과적이라고 볼 수 있다.

#### **result2: comparison with the Model-free Rl Methods**

![img\_12.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_12.png) 학습된 context latent vector가 MFRL에서도 유용하다. 특히나 더 복잡한 환경(halfCheetah, Ant, Humanoid)에서 context-conditioned policy with CaDM이 이전의 conditioned method들 보다 더 나은 성능을 보여준다. 다만 CartPole과 Pendulum같은 단순한 환경에서는 context의 영향이 미미하다.

#### **result3: effects of prediction loss**

![img\_11.png](../../.gitbook/2022-spring-assets/HaewonJung2/img\_11.png) forward step과 backward step의 loss function을 모두 사용한 model이 최상의 성능을 달성한다. 즉, context encoder를 학습시킬 때, 과거와 미래 예측 model의 영향을 모두 받는 것이 환경의 상태를 나타내는 context를 구현하는데 도움이 된다.

## **5. Conclusion**

최근에 context based meta RL 관심이 생겨 이 논문을 선택했다. 점차 RL자체의 기법을 향상 시키기 어렵다고 판단하였는지, RL training method가 발전하기 보다는 다양한 환경의 정보를 어떻게 하면 조금 더 잘 capture할 수 있는지에 중점을 두는 방식으로 전환되고 있는 듯 싶다.

특히나, 이 경우에는 MDP의 history를 사용하여 backward prediction도 context의 생성에 기여하는 점이 환경에 대한 이해도를 높이는데 엄청난 도움이 되었다고 판단한다.(section 5.4)

어쩌면 state만을 예측하는 것이 아닌, done, reward, action까지 모두 예측하고, context가 모든 예측에 영향을 받는 모델을(관련된 논문으로 decision transformer(https://arxiv.org/pdf/2106.01345.pdf)가 있었다) 이와 같은 방식으로 만드는 것도 새로운 가능성을 줄 수 있지 않을까 싶다. (물론 sparse reward의 경우에는 긍정적이지는 않을 듯 싶다)

***

## **Author Information**

* Haewon Jung
  * Affiliation: second-year Master student in ISysE at KAIST
  * Research Topic: reinforcement learning, meta-learning, latent representation

## **6. Reference & Additional materials**

* [paper](https://arxiv.org/pdf/2005.06800.pdf)
* [code of paper](https://github.com/younggyoseo/CaDM)
* [code of implementation environment](https://github.com/iclavera/learning\_to\_adapt)
