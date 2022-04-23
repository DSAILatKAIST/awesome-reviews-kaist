---
description : Michael Dennis / Emergent Complextity and Zero-shot Transfer via Unsupervised Environment Design / NeurIPS-2020  
---

# **Title**

Emergent Complextity and Zero-shot Transfer via Unsupervised Environment Design

## **1. Problem Definition**

강화학습은 agent가 주어진 environment에서 observation data를 수집하여, 주어진 task를 수행하기에 가장 정확하고 빠른 학습이 가능한 최적의 알고리즘을 통해(이를테면 TRPO, PPO 등) policy를 학습하는 흐름으로 진행된다. 이를 위해서 연구자가 environment 또는 task의 distribution을 올바르게 특정하는 과정이 필수적인데, 많은 연구에서 이러한 distribution을 특정하는 일은 오류가 발생하기 쉬우며 상당한 노력과 시간이 소요된다.

이해를 돕기 위해서 본 논문의 introduction에 수록된 예시를 빌려오자면, 물류 창고 내의 쓰레기통을 찾아다니며 작업을 수행하는 로봇을 학습시키고자 하는 RL의 경우, 실제 환경(즉 물류 창고 내의 주요 지점들의 좌표, 좌표 간의 이동거리 등)의 분포를 묘사하는데에 큰 어려움이 없을 것이다. 하지만 문제가 복잡해지고 상황 정의가 복잡해질수록 real world environment에서 분포 추정은 급격하게 어려워질 것이다.

본론으로 돌아와, 본 논문에서는 상술한 어려움을 해결하고자 RL-based framework를 개발하고 그것의 성능을 소개하는데 초점을 두었다. 

## **2. Motivation**

Problem Definition에서 소개한 문제를 해결할 수 있는 방법론으로, 연구자가 매번 상당한 시간과 노력을 들여 하이퍼파라미터를 튜닝하고 문제 상황을 면밀하게 조절하는 수고를 덜어낸 UED(Unsupervised Environment Design)가 있다. 본 논문에서 새로운 방법론과 대비를 위해 2가지 정도의 선행 연구가 이루어진 UED 테크닉을 살펴보았는데, 각각의 테크닉에는 다음과 같은 문제가 있었다: 'Domain randomization' 기법은 environment를 제대로 만들지 못했고 이로 하여금 agent의 학습에 적당한 난이도의 어려움을 부여하지 못하는 문제가 발생하였으며 'Minimax adversarial training' 기법에서는 agent가 문제를 아예 풀 수 없는 지나치게 어려운 environment를 생성하여 agent의 학습을 저해하는 문제가 있었다.

직관적으로 설명하자면 전자의 알고리즘은 agent에게 과하게 쉬운 문제를 풀게 하였고 후자의 알고리즘은 agent에게 지나치게 어려운 문제를 풀게 하였다고 볼 수 있다. 하단의 그림을 보면 푸른 화살표 agent가 미로를 풀어서 녹색의 goal에 도달하는 task를 각각의 알고리즘으로 설계했을 때, Domain Randomization 기법은 미로의 벽을 제대로 생성하지 못하여 지나치게 쉬운 environment를 설계했고, Minimax Adversarial 기법은 goal position이 미로의 벽에 가로막혀 agent가 아예 도달할 수도 없는 과하게 어려운 environment를 생성한 것을 볼 수 있다. 즉 두 알고리즘 모두 environment의 분포를 잘 추정하지 못했다.

<img src='.gitbook/2022-spring-assets/1.png'>
![1](https://user-images.githubusercontent.com/99944639/164896696-a8b3b220-6b69-4109-acec-b1a718b42969.png)

## **3. Method**

본 논문에서 상술한 문제점을 어떻게 해결하였으며, 해결 방안으로 제시한 새로운 RL-based framework는 어떤 방식으로 설계되었는지 살펴보겠다. 먼저 지금까지 설명한대로, 본 연구의 주요 관심사는 효과적인 "UED(Unsupervised Environment Design)" 테크닉 개발이다.

먼저 UED를 'underspecified environment'를 이용해 'fully specified environments'에 부합하는 distribution을 생성하는 문제로 정의했다. 이를 위해 fully specified environments와 underspecified environment를 각각 POMDP(Partially Observable Markov Dscision Process) & UPOMDP(Underspcified Partially Observable Markov Dscision Process)로 모델링 했다.

POMDP는 tuple $\lang A,O,S,T,I,R,\gamma \rang$ 로 정의한다: $A$는 set of actions, $O$는 set of obervations, $S$는 set of states, $T:S \times A \to \Delta (S)$ 는 transition function, $I:S\to O$ 는 observation function, $R$은 set of rewards, $\gamma$는 discount factor.

UPOMDP는 tuple $M=\lang A,O,\Theta,S^M,T^M,I^M,R^M,\gamma \rang$ 로 정의한다: 대부분의 정의는 상술한 POMDP와 동일하나, 모델링에 free parameter of environment를 의미하는 집합 $\Theta$가 추가된 점이 다르다. Free parameter of environment $\Theta$는 학습의 매 타임스텝마다 정해질 수 있고, $T^M:S\times A\times\Theta\to\Delta(S)$ 와 같이 transition function을 구하는데에 사용된다. 또한 environment parameter $\overrightarrow{\theta}$ 의 trajectory를 통해 environment setting을 표현할 수 있고, 이렇게 구한 setting of environment $\overrightarrow{\theta}$를 underspecified environment $M$에 대입해서 $M_{\overrightarrow{\theta}}$를 얻게 된다.

<img src='.gitbook/2022-spring-assets/2.png'>
![2](https://user-images.githubusercontent.com/99944639/164896724-706679b5-b4ce-4882-9ac0-e595346d1ada.png)


그리고 $\Pi$ : set of possible policies, $\Theta^T$: set of possible sequences of environment parameter를 이용해 environment policy $\Lambda:\Pi\to\Delta(\Theta^T)$ 를 얻을 수 있다. 다른 두 UED 기법도 상술한 흐름으로 environment policy를 구하지만, 본 논문에서 제시하는 새로운 UED framework인 PAIRED 알고리즘은 environment policy를 set of possible policies $\Pi$의 regret을 최대화 하는 $\bar{\theta}$를 이용해서 구한다. Minimax Regret decision rule을 사용했을 때 더 좋은 policy가 얻어지는 사실은 아래의 Theorem을 통해 증명할 수 있다. 

<img src='.gitbook/2022-spring-assets/3.png'>
![3](https://user-images.githubusercontent.com/99944639/164896731-4ab65011-4e05-44b8-adeb-bd43844203da.png)


마지막으로 PAIRED 알고리즘이 set of possible policies $\Pi$의 regret을 최대화 하는 $\bar{\theta}$를 통해  environment policy $\Lambda^{MR}(\pi)$를 구체적으로 어떻게 구하는지 살펴보겠다. PAIRED(Protagonist Antagonist Induced Regret Environment Design) 알고리즘은 문제를 푸는 protagonist와 antagonist, 문제를 출제하는 environment adversary(이하 adversary)로 구성된다. Adversary는 antagonist에게 유리하면서 동시에 protagonist에게 불리한 environment를 training 동안 생성한다. 이러한 편파적인 문제 생성은 앞서 설명한 decision rule: Minimax Regret을 통해서 가능하다. 알고리즘 전체는 아래의 sudo code를 통해 살펴 볼 수 있다.

$REGRET^{\overrightarrow{\theta}}=U^{\overrightarrow{\theta}}(\pi^A)-U^{\overrightarrow{\theta}}(\pi^P): \\ difference~between~the~reward~obtained~by~the~antagonist~and~the~protagonist$

<img src='.gitbook/2022-spring-assets/4.png'>
![4](https://user-images.githubusercontent.com/99944639/164896737-5c11d2ff-9fa8-4203-b76a-445e33aa4e39.png)


## **4. Experiment**

실험을 통해 PAIRED 알고리즘을 통한 학습이 agent들의 행동 양식의 복잡도 향상을 이끌어냈는지, 그리고 UED 방법론의 취지에 부합하도록 아예 새로운 environment가 주어진 상황에서도 better robust performance를 보였는지를 알아보고자 한다.

### **Experiment setup**

실험은 앞서 직관적인 설명을 위해 예시로 든 미로찾기 task에서 진행하였으며 PAIRED 알고리즘의 성능과 비교를 위한 baseline algorithm으로 Domain Randomization & Minimax Adversary algorithm을 사용했다. Agent가 task를 얼마나 잘 수행했는가를 알기 위해 생성된 environment의 상황을 {Graph of: number of blocks, distance to goal, passable path length, **solved path length**}로 파악하고, {Type of task: Empty, 50 blocks, 4 Rooms, 16 Rooms, Labyrinth, **Maze**}와 같은 다양한 조건에서 solved percent를 측정하였다.

### **Result**

<img src='.gitbook/2022-spring-assets/5.png'>
![5](https://user-images.githubusercontent.com/99944639/164896743-7efc1b80-ffde-4545-87dc-4d5b5f06fe9b.png)


<img src='.gitbook/2022-spring-assets/6.png'>
![6](https://user-images.githubusercontent.com/99944639/164896745-87888e12-45f1-44f0-aab0-6bd9a31ffefd.png)


여러 지표들 중에서 특히 중요한 Solved path length: agent가 주어진 environment에서 찾은 올바른 길의 길이, Maze: 가장 복잡한 종류의 task 에서 baseline algorithm 대비 본 논문에서 제시하는 PAIRED algorithm이 모두 우월한 성과를 보였다.

## **5. Conclusion**

본 논문은 UED 방법론의 새로운 framework로써 PAIRED 알고리즘을 소개하며, 해당 알고리즘이 어떻게 novel environment에서의 generalized improvement on the performance를 도출하는지 설명하였다.

이전까지의 연구에서 고안된 UED 방법론의 문제를 해결하여, 개인적으로 최근에 RL training 방법들 중 관심 있는 curriculum learning에서 좋은 성과를 보여준 것 같아서 흥미롭게 읽은 논문이었다.

---  

## **Author Information**

* Author name: Michael Dennis*, Natasha Jaques**, Eugene Vinitsky, Alexandre Bayen, Stuart Russell, Andrew Critch, Sergey Levine
* Affiliation: University of California Berkeley AI Research (BAIR), Google Research
* Research Topic: Multi Agent Reinforcement Learning, AI Safety

## **6. Reference & Additional materials**

* Github Implementation:
  https://github.com/google-research/google-research/tree/master/social_rl/
  
  
* Experiment result:
  https://www.youtube.com/channel/UCI6dkF8eNrCz6XiBJlV9fmw/videos
  
  
* Reference:
  
  [1] OpenAI: Marcin Andrychowicz, Bowen Baker, Maciek Chociej, Rafal Jozefowicz, Bob McGrew, Jakub Pachocki, Arthur Petron, Matthias Plappert, Glenn Powell, Alex Ray, et al. Learning dexterous in-hand manipulation. The International Journal of Robotics Research, 39(1):3–20, 2020.
  
  [2] Rika Antonova, Silvia Cruciani, Christian Smith, and Danica Kragic. Reinforcement learning for pivoting task. arXiv preprint arXiv:1703.00472, 2017.
  
  [3] Karl Johan Åström. Theory and applications of adaptive control—a survey. Automatica, 19(5):471–486, 1983.
  
  [4] J Andrew Bagnell, Andrew Y Ng, and Jeff G Schneider. Solving uncertain markov decision processes. 2001.
