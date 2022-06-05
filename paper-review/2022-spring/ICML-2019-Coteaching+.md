# How does Disagreement Help Generalization against Label Corruption?

: No
Tag: noiselabel
URL: https://arxiv.org/pdf/1901.04215.pdf
작성일시: 2022년 6월 1일 오후 1:24

Xingrui Yu et al./ How does Disagreement Help Generalization against Label Corruption? / ICML(2019)

# 1. Problem Definition

본 논문은 Noisy labels(잘못된 레이블링)이 존재하는 데이터를 기반으로 deep neural network를 학습하는 방법에 관한 논문입니다. 동일 저자의 이전 논문[1]에서 제시된 “Co-teaching”이라는 방법론에서 “학습 epoch이 증가할수록 두 네트워크가 consensus(일치)하게 된다.”는 문제점을 보완한 후속 논문입니다.

논문 링크 [https://arxiv.org/pdf/1901.04215.pdf](https://arxiv.org/pdf/1901.04215.pdf)

---

# 2. Motivation

논문에서 제시한 방법론인 Co-teaching+의 이해를 돕기 위해 이전 논문을 간략히 요약하겠습니다.

### 2.1 **Co-teaching: Robust Training of Deep Neural Network with Extremely Noisy Labels [1]**

Noisy labels을 고려한 deep neural network 학습을 위하여 memorization effect과 Decoupling, Mentornet에 영감을 받아 제시된 방법론입니다. 알고리즘을 간략히 정리하자면 아래와 같습니다.

1. *미니배치에서 상대적으로 loss가 큰 데이터를 일정 비율(R(t))만큼 배제하고 loss가 적은 데이터(small loss instances)를 이용하여 가중치 업데이트 (R(t)는 epoch가 늘어날수록 점차 증가)*
2. *parameter initialization이 다른 두 모델이 서로 small loss instances를 교환하며 학습(cross update)*

- memorization effect
    
    deep network이 초기 epoch에서는 clean하고 쉬운 패턴을 학습하고 이후에는 세부 패턴을 학습하는 현상을 의미합니다. 이를 noisy label을 포함한 데이터로의 학습에 적용해보면 초기 epoch에서는 clean data를 학습하고, epoch이 증가할수록 noisy data에 overfit하게 될 것입니다.
    
    이 현상의 직관적인 이해를 돕는 논문[2]이 있어 소개하자면
    
    ![Untitled](Untitled.png)
    
    experiment setting: 4 layer neural network(2 CNN lavers, 2 fully-connected layers, 4.8M parameters),  MINST(train(noise 포함) 50000, val 10000, test(only clean) 10000)
    
    위의 그림의 (a)는 200 epoch만큼 모델 학습을 진행했을때 noise data에 대해 overfitting함을 나타냅니다. x축은 데이터셋의 nosie label 비율을 의미하며 비율이 증가할 경우 모델은 noise data에 대해 높은 acc를 보이며(blue), 일반화 성능(yellow)과 true label에 대한 모델의 성능은 감소함을 보여줍니다.  
    
    (b)는 동일 조건의 실험이지만 early stop했을 경우이며 모델이 noise label에 보다 robust함을 보여줍니다. 데이터셋의 nosie label 비율이 증가할 수록 train acc는 감소하며(blue), test acc와 true label에 대한 모델의 train acc은 높은 수치를 유지하고 있습니다. 
    
    → 즉, deep network는 **학습 초기에 noise label이 아닌 clean data에 대해서 fitting**하며 **많은 iteration만큼 학습하면 noise label에 보다 fitting** 하는 특성을 가지고 있습니다.
    
    - small loss trick
        
        모델에게는 clean data가 noisy label data 보다 학습하기 쉬운 데이터 일텐데 (데이터와 레이블링 간의 일관성을 보이므로) 이 때문에 clean data가 noisy label data보다 training loss가 상대적으로 빠르게 감소합니다. 그렇기에 mini-batch에서 loss가 큰 일정 비율(R(t))의 데이터를 noisy data로 볼 수 있고 이를 제외하여 학습하는 것은 noisy data를 배제한 학습을 의미하게 됩니다. 이를 small loss trick이라 칭하며 관련된 다수의 연구가 존재합니다.
        
        → Co-teaching 학습 과정에서 epoch이 커질수록 R(t)(배제할 데이터의 비율)을 증가시키는 이유가 바로 위에서 언급한 “memorization effect”에서 기인한 것입니다! 
        
- Influenced studies

![Untitled](Untitled%201.png)

- MentorNet(M-net)
    
    noisy label 문제 해결을 위한 방법론으로 추가 네트워크 (Mentornet)를 clean data만으로 pretrain하고 특정 threshold 이하의 small loss를 가지는 clean istances를 선택하여 Studentnet을 학습시키는 방식입니다. sample-selection bias로 인해 축적되는 error가 문제점으로 지적됩니다.
    
    → 단점) 단일 모델만을 이용하여 noisy label을 판단하기에 라벨이 올바르더라도 한번 학습에서 제외한 데이터는 추후 학습에서도 제외됨 
    
    ![Untitled](Untitled%202.png)
    

- Decoupling
    
    noisy label 문제 해결을 위한 방법론으로 두 네트워크를 동시에 학습시키고 서로 다른 prediction을 가지는 instances만을 사용하여 모델을 업데이트 하는 방식입니다. noisy label을 보다 명확하게 다룰 수 없다는 것이 단점입니다.
    
    → 단점) 정확히 어떤 instance가 noisy label인지 파악 불가(설명력 부족)
    
    ![Untitled](Untitled%203.png)
    

Co-teaching은 기존 방법론의 단점을 보완하기 위해

1. 두개의 모델을 **cross-update**합니다. 
    
    → 2개의 서로 다른 분류기는 다른 decision boundary를 만들어내고 하나의 decision boundary가 정확하지 않더라도 다른 decision boundary가 이를 보완해주기 때문입니다. 
    
2. 두 모델은 **서로 loss가 적은 instance 정보를 교환하여 학습**합니다.
    
    → 특정 데이터에 overfit되지 않도록 하기 위함입니다. 
    

이렇게 제시된 Co-teaching은 **보다 정확하고 직관적이게 데이터에서 noisy label을 제외해나가며 학습할 수 있는 방법론**입니다.

### 2.2 **Co-teaching+**

- Consensus issue in Co-teaching

![Untitled](Untitled%204.png)

Co-teaching의 학습과정을 구현해보면 초기 epoch에는 두 네트워크가 서로 다른 에러를 filter할 수 있는 상이한 학습능력을 가지고 있으나 epoch이 증가할수록 두 네트워크가 점차 일치하게 수렴해버리는 현상이 발생합니다(Mentornet 방식과 같아져버림). 이는 모델의 성능 저하에 큰 영향을 미치게 됩니다.

이 이슈를 다루기 위해 저자는 **training epochs 동안 두 네트워크가 계속해서 diverge 하도록 혹은 consensus 하는 속도를 늦출 수 있는 방법**을 고안해야 했음을 언급하고 있습니다.

- Update by Disagreement
    
    두 분류기의 예측이 일치하지 않은 경우의 data로만 update 하여 두 네트워크가 계속해서 서로 다른 (diverge) 상태를 유지하도록 하는 방법론으로 저자는 본 논문으로부터 영감을 받아 **Co-teaching에 Disagreement strategy를 추가한 Co-teaching+를 제안**합니다.  
    
    ![Untitled](Untitled%205.png)
    

---

# 3. Method

Co-teaching+은 크게 4단계로 구성됩니다.

1. *서로 다른 파라미터 초기값을 가지는 두개의 deep neural network  학습*
2. *mini-batch에 대해 두 network는 각각의 예측 수행*
3. *두 예측이 상이한 데이터(disagreement data)들로부터 각 네트워크는 small-loss data 선택 (**Disagreement-update step**)*
4. *서로의 small-loss data를 cross하여 파라미터 업데이트 (**Cross-update step**)*

아래에는 알고리즘에 대한 구체적인 설명을 기술하겠습니다.

### 3.1 Algorithm

![Untitled](Untitled%206.png)

- Step 4 (**Disagreement-update step**)
    
    두 네트워크는 학습 과정에서 같은 mini-batch data $\bar{\mathcal{D}}=\{(x_1, y_1), (x_2, y_2), ... (x_B, y_B)\}$에 대한 예측을 수행합니다. 그리고 예측이 일치하지 않은 disagreement data $\bar{\mathcal{D^\prime}}$을 keep 합니다.
    
    $w^{(1)}$에 의한 예측을 $\{\bar{y_1}^{(1)}, \bar{y_2}^{(1)}, ...,\bar{y_3}^{(1)}\}$, $w^{(2)}$에 의한 예측을 $\{\bar{y_1}^{(2)}, \bar{y_2}^{(2)}, ...,\bar{y_3}^{(2)}\}$이라 할 때 Eq.1 은 $\bar{\mathcal{D^\prime}} = \{(x_i, y_i): \bar{y_i}^{(1)} {\neq \bar{y_i}^{(2)}}\}$, $i \in \{1,...,B\}$ 을 나타냅니다.
    
    저자는 본 단계가 Co-training에서 아이디어를 얻었음을 명시하고 있고 두 분류기가 diverge 상태를 유지하면서 더 나은 ensemble effect를 얻을 수 있다 언급하고 있습니다.
    
- Step 5-8 (**Cross-update step**)
    
    $\bar{\mathcal{D^\prime}}$로부터 각 네트워크($w^{(1)}, w^{(2)}$)는 small-loss data($\bar{\mathcal{D}^{(1)}}, \bar{\mathcal{D}^{(2)}}$)를 선택하고 $\bar{\mathcal{D}^{(1)}}$은 $w^{(2)}$의 update를 위한 backprop을 $\bar{\mathcal{D}^{(2)}}$은 $w^{(1)}$의 update를 위한 backprop을 수행합니다. 
    
    저자는 본 단계가 인간 뇌는 다른 인간으로부터 생성된 신호로부터 학습이 이루어질때 더욱 학습을 잘 할 수있다는 내용의 “Culture Evolving Hypothesis”에서 영감을 얻었다고 합니다. 
    
- Step 9 (**update** $\lambda(e)$)
    
    $\lambda(e)$는 각 training epoch에서 사용된 small-loss data의 수를 컨트롤하는 파라미터입니다. 강조하자면 memorization effect에 의해 deep network는 먼저 clean data에 fit하고 점차 noisy data에 overfit하는 특성을 가지고 있습니다. 
    
    그러기에 초기 training에서는 각 mini-batch에서 더 많은 small-loss data(large $\lambda(e)$, dropping less data)를 사용하고 점차 적은 수의 small-loss data(small $\lambda(e)$. dropping more data)를 학습에 사용하도록 조절하는데에 본 파라미터를 사용하고 있습니다. 
    

---

# 4. Experiment

### 4.1 Experiment setup

- Dataset
    - Vision datasets
        
        MNIST, CIFAR-10 and CIFAR-100
        
    - Text datasets
        
        NEWS
        
    - Larger and harder dataset
        
        Tiny-ImageNet
        
    
    데이터셋들은 clean(noisy label이 존재하지 않는)한 데이터이므로 메뉴얼하게 label noise를 추가하였습니다(Pair-45%, Symmetry-50%, Symmetry-20%). 이를 이용해 모델을 학습하고 clean label(올바른 레이블)만 존재하는 Test-set에서의 분류성능을 도출합니다.
    
    이때, 노이즈의 가정 중 Pair이란 혼동되는 Class간의 label이 섞이는 경우를 나타내고, Symmetry란 uniform/random noise와 같이 label이 임의로 섞이는 경우를 나타냅니다.
    
    ![Untitled](Untitled%207.png)
    
- Baselines
    
    성능이 우수한 SOTA 방법론(아래)와 Simple baseline을 Co-teaching+와 비교하였으며 모든 구현은 Pytorch의 default parameters로 이루어졌습니다. 각 방법론별 비교는 아래의 표와 같습니다.
    
    - MentorNet
        
        추가적인 teacher network를 pretrain하고 noisy instance를 filter out하기 위해 이용한다.  이렇게 얻어진 clean data을 student network에서 classification을 수행하기 위한 학습 데이터로 사용하는 방법론이다.
        
    - Co-teaching
        
        두 네트워크를 동시에 학습하고 각 peer network에 파라미터를 cross update하는 방법론이다. 많은 수의 classes에도 잘 적용되며 극단적으로 많은 noisy label 환경에서도 robust한 한습이 가능하다.
        
    - Decoupling
        
        동일 데이터에 대해 두 분류기의 예측이 다른 instance만으로 파라미터 update를 수행하는 방법론이다. 
        
    - F-correction
        
        label transition matrix에 의해 prediction을 교정하는 방법론이다.  본 논문에서는 transition matrix  $Q$를 추정하기 위해 standard network를 학습했다고 한다.
        
    - Simple baseline
        
        noisy dataset을 그대로 학습시킨 standard deep network이다.
        
        ![Untitled](Untitled%208.png)
        
- Network structure
    
    MNIST에는 2-layer MLP, CIFAR-10에는 2-layer CNN과 3-layer MLP, CIFAR-100에는 7-layer network architecture을 사용하였습니다.
    
    NEWS에는 GloVe로 부터 pre-trained work embedding을 추출하였고 Softsignactive function과 함께 3-layer MLP를 이용하였습니다.
    
    Tiny-ImageNet에는 Resnet18을 이용했고 자세한 정보는 아래의 표와 같습니다.
    
    ![Untitled](Untitled%209.png)
    

- Optimizer
    
    학습에 관한 세부 사항은 아래와 같습니다.
    
    - Adam optimizer(mometum=0.9)
    - initial learning rate = 0.001 (80 epoch에서 200 epoch까지 0으로 선형 감소하도록 스케줄링)
    - batch size = 128
    - epochs = 200
    - 두 네트워크는 같은 architecture이지만 다른 파라미터 초기값을 갖도록 설정
    
     
    
- Initialization
    
    Co-teaching과 Co-teaching+는 noise rate $\tau$를 알고 있다고 가정하고 있는데 사전에 $\tau$를 알지 못한다면 validation set을 가지고 추론하는 과정이 필요하다고 합니다.
    
    저자는 $\lambda(e)$이 특정 데이터셋이 아닌 memorization effect에 의존함을 강조하며 공정한 비교를 위해 Co-teaching과 동일한 업데이트 룰 $\lambda(e)= 1-min{e \over E_k}\tau, \tau\}$, $E_k=10$ 을 적용했습니다.
    
- Evaluation Metric
    
    test accuracy=(# of correct predictions)/(# of test dataset)을 활용했습니다. 직관적으로 더 높은 test accuracy는 label noise에 더 robust함을 나타냅니다.
    

### 4.2 Result

![Untitled](Untitled%2010.png)

![Untitled](Untitled%2011.png)

![Untitled](Untitled%2012.png)

![Untitled](Untitled%2013.png)

![Untitled](Untitled%2014.png)

요약하자면 대부분의 실험에서 Co-teaching+가 높은 분류 성능을 보였습니다. cross-update와 small-loss trick을 이용하여 두 네트워크를 diverge 상태를 유지하는 것이 보다 noise robust한 학습에 분명하게 도움이 됨을 보여주고 있습니다. 

---

# 5. Conclusion

---

본 논문은 noisy 분류 상황에서의 deep neural network의 학습 방안인 Co-teaching+를 제안하였습니다. 핵심 아이디어는 두 네트워크를 동시에 유지하면서 disagreement data를 찾고 small loss data 만으로 cross-update 하는 것입니다.

실험을 통해 Co-teaching+이 기존 Co-teaching, MentorNet 보다 극단적인 noisy label 환경(45%, 50%, 20%)에서 noise label robust한 training이 가능함을 보였습니다. 

또한 Open-sets noisy data(40%)를 활용하여 close-set noisy data(잘못된 label이 true class 존재하는 경우)가 아닌 training data에 존재하지 않는 labeling이 이루어진 경우를 고려한 실험 결과 역시 제시하고 있습니다.

하지만 개인적인 의견으로는 실제 application 단에서 대부분의 training data의 noise label 비율은 실험과 같이 많은 양의 noise label을 포함하지는 않을 것 같습니다. 데이터의 종류나 label의 수 등에 따라 매우 다를 것 같아 이를 고려한 보다 현실적인 실험 결과가 궁금해졌습니다. 예를들어 Pair noise, Symmetry noise, Open-set noise에 관하여 학습 데이터에서 1%, 5%, 10% 정도의 noise labeling 비율을 가지는 경우의 실험 결과도 제시했으면 Co-teaching+을 활용할 때 보다 도움이 될 것 같다는 생각을 했습니다.

# Author Information

---

- 이 솔 (LEE SOL)
    - Affiliation
        
        KAIST, Industrial & System Engineering(Graduate School of Knowledge Service Engineering)
        
    - Research Topic
        
        Vision DL, Noise Label
        

---

# 6. Reference & Additional materials

- Github Imaplementaion
    - https://github.com/xingruiyu/coteaching_plus
- Reference
    
    [1] Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang,I., and Sugiyama, M. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In NeurIPS, 2018b
    
    [2] Li, Mingchen et al. “Gradient Descent with Early Stopping is Provably Robust to Label Noise for Overparameterized Neural Networks.” *ArXiv* abs/1903.11680 (2020): n. pag.