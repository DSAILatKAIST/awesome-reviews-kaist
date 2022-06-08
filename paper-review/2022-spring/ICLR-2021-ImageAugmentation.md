| description                                                  |
| ------------------------------------------------------------ |
| Ilya Kostrikov  / Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels / ICLR-2021 |

# Image Augmentation Is All You Need : Regularizing Deep Reinforcement Learning from Pixels

------

## **1. Problem Definition**

![augmentation](C:\Users\user\Documents\카카오톡 받은 파일\images\images\augmentation.png)

위 figure과 같이 Image에 대한 Augmentation을 수행하고, 이를 추가적인 학습 데이터로 활용함으로써 모델의 over-fitting을 방지하고 성능을 높이는 방법은 Computer Vision에서 활발하게 사용되는 테크닉이며, 딥러닝에 관심이 있는 분들에게는 대부분 익숙한 테크닉일 것이라 생각합니다.

본 논문에서는 이러한 Image Augmentation 테크닉을 supervised learning뿐만 아니라, Reinforcement learning framework에도 적용할 수 있도록 하는 간단하지만 유용한 테크닉에 대해서 소개하고 있습니다. 

이 테크닉은 Image Augmentation을 수행하는 것과 동시에, Augmented된 image에 대하여 Q-value regularization까지 수행하여 기존 Image Augmentation 테크닉을 강화 학습 framework에서도 사용 가능하게 만듭니다. 이에 대한 자세한 내용은 3. Method에 정리되어 있습니다.

이 테크닉은 강화 학습 알고리즘이나 모델에 대한 수정이 필요하지 않고, auxiliary loss등과 같은 추가적인 loss function 또한 필요로 하지 않습니다. 이 때문에 다양한 강화학습 framework에 해당 테크닉을 적용할 수 있으며, 알고리즘 또한 간단하여 쉽게 구현할 수 있다는 장점이 있습니다.

본 논문에서는 Soft Actor-Critic 강화학습 모델을 사용하여 해당 테크닉을 적용하였고, DeepMind control suite에서 성능 실험을 진행하였습니다. 실험 결과 기존의 model-based RL approach의 성능을 능가하는 것 뿐만 아니라,  최근 제안된 constrastive learning 기반 모델의 성능을 능가하는 등, 기존 SOTA 모델들의 성능을 능가하는 결과를 보였습니다.

## **2. Motivation**

Reinforcement Learning에서 Agent는 특정 state에서 action을 취하고, 이에 대한 reward를 받으며 다음 state로 이동하게 됩니다. 이러한 과정을 반복하면서 Agent는 특정 state에서 최적의 action을 선택하도록 점차 발전하게 됩니다. 이 때, Agent에게 state를 알려주는 방법은 다양한 방법이 있습니다. Agent가 상황을 파악하기에 필요하다고 판단되는 정보들을 설계자가 직접 모아서(이를 hand-crafted features라고 합니다) 이를 state로 정의할 수도 있을 것이고, 다른 방법으로는 해당 상황에 대한 이미지가 있다면, 이 Image 자체를 state로 활용할 수도 있습니다.

논문의 제목부터 이미 눈치채셨겠지만, 본 논문에서는 Image 자체를 state로 활용하는 Reinforcement Learning 알고리즘에 대해서 다루고 있습니다. Image Pixel 자체를 input으로 활용하여 직접 학습하는 RL 알고리즘은 robotics분야를 포함하여 다양한 control 분야에 쉽게 적용할 수 있다는 장점이 있으며, 꾸준하게 발전하고 있습니다.

한편, Reinforcement Learning에서 agent가 주어진 state에서 어떠한 action을 취할 것인지에 대한 정보를 Policy, 특정 state에서 특정 action을 취한 것이(즉, 특정 state-action pair)얼마나 좋은지에 대한 가치를 Q-Value라 합니다. 본 논문에서는 다루는 문제의 특성 상 Policy에 대한 학습이 필수적입니다. 

그런데 최근 Reinforcement Learning에서는 Policy뿐만 아니라 Q-value에 대한 학습 또한 같이 활용한 모델이 우수한 성능으로 많은 각광을 받고 있으며, 본 논문에서도 이러한 모델 중 하나인 Soft Actor-Critic모델을 사용하고 있습니다.

![network](C:\Users\user\Documents\카카오톡 받은 파일\images\images\network.png)

하지만, Soft Actor-Critic과 같이 policy와 Q-value에 대한 학습을 수행하는 RL 알고리즘에서는 image를 사용한 학습이 효과적으로 이루어지기 힘들다는 한계가 있습니다. 그 이유는 image를 사용하여 학습을 수행하기 위해서는 image encoder에 대한 학습이 필요하고, 이와 더불어 Q-value와 Policy에 대한 학습까지 필요하기 때문입니다. 위 figure을 통해 다시 정리하자면, 파란색 동그라미로 표시된 image encoder에 대한 학습, 초록색으로 표시된 Q-value network에 대한 학습, 빨간색으로 표시된 policy network에 대해 모두 학습이 이루어져야 합니다. 이를 위해서는 상당히 많은 양의 데이터가 필요할 것이고, 다시 말해 충분한 양의 데이터가 주어지지 않는다면 학습이 제대로 이루어지지 않을 것입니다.

그런데 문제는 강화학습 framework에서는 agent와 environment의 상호작용으로 인해 학습 데이터가 만들어지는데, 이러한 상호작용의 경우의 수는 어느 정도의 한계가 있기 때문에 supervised learning framework처럼 수많은 학습 데이터를 만들 수가 없다는 것입니다.

이러한 데이터 부족 문제를 *Limited Supervision*이라고 하는데, 이 *Limited Supervision*은 ML의 domain을 막론하고 중요한 문제이며, 이러한 문제를 해결하기 위하여 다양한 노력들이 이루어지고 있습니다. 대표적으로 세 가지를 꼽자면

1.  pre-training with self-supervised learning(SSL), followed by standard supervised learning
2. supervised learning with an additional auxiliary loss
3. **supervised learning with data augmentation**

와 같습니다. 

강화학습 framework에서도 위 approach들에 대해서 활발한 연구가 이루어지고 있습니다. 다만 Self-supervised Learning 의 경우 data가 풍부한 상황에서 굉장히 효과적이며, 이 때문에 실제로 Computer Vision, Natural Language Processing과 같이 이미 충분한 데이터가 존재하는 domain에서 활발하게 사용되고 있습니다.

본 논문에서는 위 3가지 approach 중 세 번째에 집중하여 문제를 해결하고자 하였고, 이를 해결하기 위하여 다음과 같은 Augmentation&Regularization 테크닉을 소개합니다.

## **3. Method**

**Reinforcement Learning from Images**

image를 사용하여 학습하는 강화학습에서, POMDP(partially observable Markov decision process)는 다음과 같이 정의합니다. 

- ![](https://latex.codecogs.com/svg.image?O&space;) : high-dimensional observation space (one single image)

- ![](https://latex.codecogs.com/svg.image?A) : the action space

- transition dynamics ![](https://latex.codecogs.com/svg.image?p=Pr(o_{<=t},a_t))

  probability distribution over the next observation ![](https://latex.codecogs.com/svg.image?o_t') given the history of previous observations ![](https://latex.codecogs.com/svg.image?o_{<=t}) and current action ![](https://latex.codecogs.com/svg.image?a_t)

- ![](https://latex.codecogs.com/svg.image?r:O*A\to&space;R)

  the reward function that **maps the current observation and action to a reward** ![](https://latex.codecogs.com/svg.image?r_t&space;=&space;r(o_{<=t},a_t))

- ![](https://latex.codecogs.com/svg.image?\gamma) : discount factor [0, 1)



위와 같은 POMDP에서 *Partially Observable*은 agent가 현재 state에 대한 정보를 완전히 관찰하지 못한다는 의미이며, MDP로 볼 수 없습니다. Agent가 action을 선택하기 위해 현재 state에 대한 정보를 완전히 파악하기 위해서 다음과 같은 작업이 이루어집니다.

- ![](https://latex.codecogs.com/svg.image?s_t&space;=&space;\{o_t,&space;o_{t-1},&space;o_{t-2},...\})


즉, 기존의 이미지 1장이 observation이었던 POMDP(partially observable Markov decision process)는 image set ![](https://latex.codecogs.com/svg.image?\{o_t,&space;o_{t-1},&space;o_{t-2},...\})을 state ![](https://latex.codecogs.com/svg.image?s_t)로 사용함으로써 MDP로 정의됩니다. 이 과정을 통해 transition dynamics와 reward function또한 다음과 같이 변하게 됩니다.

- ![](https://latex.codecogs.com/svg.image?p=Pr(s_t'|s_t,a_t))
- ![](https://latex.codecogs.com/svg.image?r_t=r(s_t,a_t))



이 과정에서 과거 몇 장의 image를 쌓아서 state로 활용했는지에 대해서는 논문에서 명시하고 있지 않습니다.



**Soft Actor-Critic**

SAC는 state-action value Q,  stochastic policy function ![](https://latex.codecogs.com/svg.image?\pi), temperature ![](https://latex.codecogs.com/svg.image?\alpha)에 대한 학습을 수행합니다. 이 때 *Actor*와 *Critic*은 각각 ![](https://latex.codecogs.com/svg.image?\pi)와 Q를 나타내며, temperarue ![](https://latex.codecogs.com/svg.image?\alpha)는 학습 Objective를 위한 일종의 weight입니다.

Policy Evaluation step에서 Critic Q에 대한 학습이 이루어지는데, 이 때 *soft Bellman Residual*을 통해 학습이 이루어집니다. 

Policy Improvement step에서는 현재 학습된 Q값을 이용하여 policy ![](https://latex.codecogs.com/svg.image?\pi)를 학습하게 되는데, 이 때 *maximum-entropy objective*를 통해 학습이 이루어집니다.

본 논문에서 제안한 알고리즘은 Regularization에 대한 부분이기 때문에, Soft Actor-Critic의 자세한 학습 수식은 생략하도록 하겠습니다. 이에 대한 자세한 내용은 논문의 Appendix를 보시면 참고하실 수 있습니다.

Soft Actor-Critic의 학습 과정은 쉽게 말해 *주어진 state에서 특정 action을 취하는 것*이 얼마나 가치있는지를 학습함과 동시에, 이 가치를 이용하여 *특정 상태에서 어떠한 action을 취할 것인지* policy를 학습합니다. 이러한 학습이 반복해서 이루어짐으로써, 결과적으로 Agent는 주어진 state에서 가장 좋은 action을 취하는 방향으로 학습하게 됩니다. 전체적인 network 구조를 그림으로 나타내면 다음과 같습니다.

![SAC](C:\Users\user\Documents\카카오톡 받은 파일\images\images\SAC.png)



**Image Augmentation**

당연하게도, Image Augmentation을 Reinforcement Learning에 적용하는 것은 기존의 Computer Vision분야에 적용하는 것과 차이가 존재합니다. 그 이유는 Label의 변화 유무입니다. Image Recognition을 생각했을 때, 아래 figure에서 모든 사진은 똑같이 "cat"이라는 label을 같습니다. 

![augmentation](C:\Users\user\Documents\카카오톡 받은 파일\images\images\augmentation.png)

하지만 강화학습에서는 상황이 다릅니다. agent가 image를 하나의 state라고 생각한다면, 상하좌우로 조금씩 shift된 image, 약간 회전된 image는 agent에게 분명히 다른 state가 될 것이며 이에 대해 똑같은 action을 취하더라도 agent가 받는 reward는 달라질 수 있을 것입니다. 이러한 문제를 해결하기 위하여, 본 논문에서는 다음과 같은 Image Transformation을 사용합니다.



**Optimality Invariant State Transformation**

![eqn1](C:\Users\user\Documents\카카오톡 받은 파일\images\images\eqn1.png)

위 식에서 f는 image transformation 함수이며, v는 f의 parameter 입니다. 본 논문에서는 이러한 이미지 변환 함수로 4 pixel-random-shift를 사용하였습니다. 즉, 주어진 image에 대하여 상하좌우로 random하게 4 pixel을 shift하였으며, 논문에서는 이러한 transformation이 가장 성능이 좋았기 때문에 사용했다고 이야기하고 있습니다.

위와 같은 transformation을 통해서 augmented된 images(즉, augmented state)는 원래의 image(즉, 원래의 state)에서 같은 action을 택했을 때 동일한 Q-value 값을 갖습니다. 이러한 성질을 이용하여, 여러 개의 augmented state을 이용하여 Q-value를 계산하면 optimal Q-value는 변하지 않으면서도 Q-value 계산의 variance는 줄여서 더욱 빠르고 정확하게 Q-value를 계산할 수 있습니다.

본 테크닉에서는 보다 빠른 학습 속도와 높은 성능을 위하여, Q-target value와 Q-value function 을 계산할 때 모두 위와 같은 **Optimality Invariant State Transformation**을 수행합니다. 여기서 Q-target value란 해당 step의 Q-value를 업데이트할 때 사용하는 다음 step의 Q-value를 말하고, Q-value function은 말 그대로 Q-value를 계산하는 함수 자체를 말합니다.

이 때, 여러 개의 augmented state를 이용하여 Q-value를 계산하는 것이 곧 논문에서 이야기하는 Regularization이 됩니다. 이러한 Regularization 과정은 다음과 같은 수식으로 더욱 명확하게 이해할 수 있습니다.

![eqn2](C:\Users\user\Documents\카카오톡 받은 파일\images\images\eqn2.png)

위 Figure에서 (1)식은 Q-target에 대한 Regularization, (2)식은 Q-function에 대한 Regularization, (3)식은 이 두 식을 합한 식을 나타냅니다. 

이러한 Regularization 알고리즘을 Soft Actor-Critic 모델에 적용시킨 전체적인 강화학습 모델에 대한 수도 코드는 다음과 같습니다. Orange, Green, Blue로 표현된 부분은 각각 Image Transformation, target Q-value를 regularization하기 위한 augmentation, Q-function 자체를 regularization하기 위한 augmentation을 나타냅니다.

![algorithm](C:\Users\user\Documents\카카오톡 받은 파일\images\images\algorithm.png)

정리하자면 Agent가 environment와 상호작용하는 매 step마다 replay buffer에서 mini-batch size 만큼의 sample을 뽑고, **하나의 sample을 이용하여 Q-value를 update할 때** target Q에 대해 K개의 augmentation과 Q-value function자체에 대해 M개의 augmentation을 수행하여 Q-value를 학습하게 됩니다. 즉, replay buffer로부터 얻어진 하나의 sample data를 계산할 때 K*M개의 Q-value data를 사용하여 계산을 수행한다는 의미가 됩니다.

이 때 K와 M은 hyper-parameter이며 K와 M이 각각 1이면 해당 regularization을 수행하지 않는다는 의미가 됩니다.

본 논문에서는 제안한 모델을 ***DrQ(Data-regularized Q)***라 칭하고 있습니다. 

## **4. Experiment**

본 논문에서는 

1. Image Augmentation 자체의 성능을 보여주는 experiment
2. Regularization의 성능을 보여주는 experiment
3. Image Augmentation & Regularization 모두를 수행한 모델 ***DrQ[K=2, M=2]***의 성능을 보여주는 experiment

를 수행합니다.

### **Experiment setup**

- Dataset

  본 논문에서는 DeepMind control suite를 활용하여 실험을 진행합니다. 이 suite에서는 Image를 input으로 사용하여 학습시키는 RL agent가 다양한 게임을 얼마나 잘 수행하는지를 테스트할 수 있습니다. 각 Experiments에서는 모두 같은 common settings를 사용하는데 해당 settings는 다음과 같습니다.

  ![setting](C:\Users\user\Documents\카카오톡 받은 파일\images\images\setting.png)

### **Results**

**Experiment 1**

![experiment1](C:\Users\user\Documents\카카오톡 받은 파일\images\images\experiment1.png)

해당 실험에서는 Image Transformation을 해서 다양한 sample 데이터를 확보했을 때의 성능과 그렇지 않을 때의 성능을 비교합니다.

위쪽 (a)가 Image Transformation을 통한 Sample Data 증가 없이 soft Actor-Critic을 사용한 결과이고 아래쪽 (b)가 Data augmentation을 수행하여 soft Actor-Critic을 사용한 결과입니다. 

baseline들은 모두 SAC framework를 사용하였는데, 각각의 model은 다른 image encoder를 사용하였고 이 image encoder들의 network 크기는 모두 제각각입니다.

 (a)의 경우 image encoder의 network가 클수록 높은 성능을 보이고, image encoder의 network가 작아짐에 따라 성능이 급감하는 모습을 보입니다. 이를 통해 data가 충분하지 않을 때 over-fitting문제가 발생하는 것을 확인할 수 있습니다. 또한 (a)의 경우, 비교적 어려운 게임인 *Walker Walk*에 대해서는 image encoder가 큰 모델조차 좋은 성능을 보이지 못하는 것을 확인할 수 있습니다.

반면 (b)의 경우 모든 모델에 대하여 동등한 성능을 보임을 통해 over-fitting문제를 해결하고, *Walker Walk*게임에 대하여도 우수한 성능을 보임을 확인할 수 있습니다. 또한, 학습 속도 또한 모든 모델에서 (a)보다 훨씬 빠르게 수렴하는 것을 알 수 있습니다.



**Experiment 2**

![Experiment2](C:\Users\user\Documents\카카오톡 받은 파일\images\images\Experiment2.png)

해당 실험은 Regularization의 성능을 나타낸 것입니다. 

- 파란색 그래프 : Regularization을 수행하지 않은 경우
- 빨간색 그래프 : [K=2, M=1]. 즉, target Q에 대해서만 Regularization을 수행한 경우
- 보라색 그래프 : [K=2, M=2]. 즉, target Q와 Q-function 모두 Regularization을 수행한 경우

세 가지 경우를 비교한 결과 보라색 그래프가 전체적으로 가장 우수한 성능을 보였고, 이를 통해 본 논문에서 제안한 Regularization을 이용한 학습이 실제로 학습에 도움이 된다는 것을 알 수 있습니다.

이 때 K=2, M=2를 선택한 이유는 computation time과의 trade-off를 고려하였을 때 가장 효율적인 hyper-parameter였다고 이야기하고 있습니다.



**Experiment 3**

![experiment3_table](C:\Users\user\Documents\카카오톡 받은 파일\images\images\experiment3_table.png)

![experiment3](C:\Users\user\Documents\카카오톡 받은 파일\images\images\experiment3.png)

해당 실험은 ***PlaNet*** Benchmark에서 proposed model과 다른 model들을 비교한 결과입니다. 

*PlaNet* Benchmark란 *PlaNet*이라는 모델을 제안한 논문에서 사용한 testbed를 말하며, 위와 같이 6개의 게임에 대한 performance를 측정합니다.



초록색 그래프가 proposed model을 나타내고 나머지 그래프들은 PlaNet Benchmark에서 실험을 수행한 다른 baseline 모델들을 나타냅니다. 이 때, 검은 점선은 image가 아닌 input state를 사용했을 때 Vanilla Soft Actor-Critic model의 upper bound 성능을 나타냅니다.

각 모델마다 총 10개의 different seed를 가지고 training을 수행한 뒤, 각 seed 마다 10번의 episode를 수행하여 10000번째 step에서의 return값을 평균내어 performance를 측정하였습니다.



실험 결과 proposed model DrQ가 모든 게임에서 SOTA 모델을 능가하는 가장 우수한 성능을 보였고, 대부분의 게임에서 vanilla SAC model의 upper bound에 달하거나 능가하는 성능을 보였습니다.

또한, DrQ모델은 auxiliary loss와 같은 추가적인 model 구조를 사용하지 않기에, ***wall clock time***또한 다른 모델들에 비해 굉장히 빠른 모습을 보였습니다.

이에 따라 DrQ모델이 ***Data-efficient***적으로도, ***asymptotic performance*** 적으로도 매우 우수한 모델이라는 것을 보였습니다.



**Experiment 4**

![experiment4](C:\Users\user\Documents\카카오톡 받은 파일\images\images\experiment4.png)

![experiment4_2](C:\Users\user\Documents\카카오톡 받은 파일\images\images\experiment4_2.png)

해당 실험은 ***Dreamer*** Benchmark에서  *Dreamer*모델과 proposed model을 비교한 결과입니다. 

*Dreamer* benchmark는 *Dreamer* model을 제안한 논문에서 사용한 testbed이며, *PlaNet* benchmark에 비해 더 어려운 setting을 요구하고, 게임 종류 또한 15가지로 더욱 많고 어려운 게임들을 많이 포함하고 있는 testbed입니다.



이전 실험에서와 마찬가지로 초록색 그래프가 proposed model을 나타내고 파란색 그래프는 Dreamer 모델을 나타냅니다. 검은 점선은 image가 아닌 input state를 사용했을 때 Vanilla Soft Actor-Critic model의 upper bound 성능을 나타냅니다. 성능 측정 방법 또한 같습니다.



실험 결과 proposed model DrQ가 15개 중 3개 게임을 제외한 12개 게임에서 *Dreamer* 모델을 능가하여 SOTA 성능을 보였습니다.

이전 실험에서와 마찬가지로 대부분의 게임에서 vanilla SAC model의 upper bound에 달하거나 능가하는 성능을 보였고, ***wall clock time***또한 굉장히 빠른 모습을 보였습니다.

이에 따라 DrQ모델이 ***Data-efficient***적으로도, ***asymptotic performance*** 적으로도 매우 우수한 모델이라는 것을 한 번 더 확인할 수 있었습니다.



**Experiment 5**

![experiment5](C:\Users\user\Documents\카카오톡 받은 파일\images\images\experiment5.png)

Experiment 5는 Soft Actor-Critic 대신 Efficient DQN에 테크닉을 적용하였습니다.

***Atari 100k***라는 많은 게임이 있는 testbed를 사용하였으며, 이는 continuous task가 아닌 discrete task에 대한 performance 측정 testbed입니다.

DrQ 알고리즘을 적용한 Efficient DQN 모델과, 다양한 DQN 기반 모델들을 비교한 실험입니다.



Performance 측정은 5개의 random seed를 사용하여 진행하였고, return 값을 normalize하여 그 결과를 측정하였습니다. 해당 task에 대해서는 K=1, M=1을 사용하여 Image Transformation만 사용하였습니다.

위와 같이 Q-value Regularization을 수행하지 않고 Image Transformation만 수행했지만 더욱 복잡한 다른 model들을 능가하는 성능을 보이는 것을 확인할 수 있습니다.

이를 통해 제안한 알고리즘이 discrete/continuous task에 모두 적용 가능하며, 우수한 성능을 보인다는 것을 알 수 있습니다.



## **5. Conclusion**

본 논문의 Contribution은 다음과 같습니다.

1. 간단한 Image Augmentation 메커니즘을 통해, RL 알고리즘에 대한 복잡한 수정 없이도 over-fitting을 얼마나 크게 감소시키는지 보여준다.

2. MDP 구조를 활용하여, model-free RL 의 맥락에서 간단하게 적용해볼 수 있는 쉽지만 강력한 메커니즘을 소개한다.

3. 바닐라 SAC model에 제안한 알고리즘을 사용하여, DeepMind control suite에서 기존 SOTA 성능을 능가하는 성능을 보임을 확인하였다.

4. DQN 에이전트와 결합하여, discrete한 control에 대하여도 Atari-100k 벤치마크에서 기존 SOTA보다 우수한 성능을 보임을 확인하였다.

5. Self Supervised Learning이나 auxiliary loss와 같은 복잡한 메커니즘을 사용하지 않고, 복잡한 image encoder를 사용하지 않고도 우수하고 강건한 성능을 확인하였다. 또한 이러한 간단한 구조 덕분에 학습 속도 또한 굉장히 빠르다는 것을 확인하였다.

   

또한 이 논문을 읽고 제가 개인적으로 느낀 생각은 다음과 같습니다.

1. 이미지를 input으로 활용한다는 특정한 조건이 있지만, discrete / continuous control 모두에 적용할 수 있다는 관점에서 볼 때 다양한 RL-based research에 적용할 수 있다.
2. 모델의 간단한 구조와 빠른 학습 속도 덕분에, 적용부터 결과 확인까지 어렵지 않게 수행할 수 있을 것으로 보인다.
3. auxiliary loss, SSL 과 같은 방법론과 함께 결합하여 사용한다면 성능이 얼마나 개선될지에 대한 추가적인 연구도 가능할 것이라 생각한다.
4. vanilla SAC의 upper bound 성능은 어떻게 구한 것인지, 몇 장의 image를 합쳐서 state로 만든 것인지에 대한 명확한 설명이 논문에 나와 있지 않은 부분이 약간 아쉬웠다.

------

## **Author Information**

- Author name : 신동휘
  - Affiliation : Industrial and System Engineering, KAIST
  - Research Topic : AMHS Design and Operation