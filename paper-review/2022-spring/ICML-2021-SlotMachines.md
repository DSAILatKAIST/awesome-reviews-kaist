# **Slot Machines**

Maxwell Mbabilla Aladago / Slot Machines: Discovering Winning Combinations of Random Weights in Neural Networks / ICML-2021

---



## **1. Problem Definition**  

본 논문에서는 특정한 딥러닝 모델에 대해서 다루지 않고, 딥러닝의 가중치 학습 과정과 관련하여 완전히 새로운 방법론을 제시합니다.

본 논문에서는 Back-propagation을 통해 가중치 값을 지속적으로 업데이트하는 기존의 방법론과 다르게, 고정된 가중치 값을 선택하는 방법론을 제안합니다.

제안된 방법론을 사용하여 선택한 weight값은 학습을 통해 값이 업데이트되지 않음에도 불구하고, 기존 제안된 가중치 학습 모델과 유사한 성능을 보이거나, 더욱 좋은 성능을 보입니다.


## **2. Motivation**  

2019년 발표된 한 논문에서, *"무작위로 초기화된 신경망은 그와 비슷한 성능을 낼 수 있는 더 작은 규모의 sub-network들을 가지고 있다"*고 발표했습니다.  이러한 결과에 영감을 받아, weight training을 수행하지 않고도 training을 수행한 것처럼 성능을 낼 수 있는 sub-network를 찾는 방법에 대한 연구가 진행되었습니다.

실제로 이후에 진행된 연구에서는 train된 ResNet-34와 유사한 성능을 내는 ResNet-50의 sub-networks를 찾아내었습니다. 앞선 이론적 설명과, 실제로 진행된 연구를 통해 다음과 같은 conjecture가 제안되었습니다.

*무작위로 초기화된 over-parameterized된 신경망은, 상대적으로 적은 parameter를 가진 신경망이 전통적으로 훈련된 것과 유사한 성능을 가지는 sub-networks를 가진다.*

이를 통해 학습이 없이도 학습된 신경망과 유사한 성능을 낼 수 있도록 sub-network를 찾는 연구들이 수행되었고, 그 중 대표적인 것이 "가지치기"를 활용한 sub-network를 찾는 방법론입니다.



![image-20220424225705322](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220424225705322.png)

해당 방법론은 위 그림에서 오른쪽 그림에 해당됩니다. 복잡한 신경망에서 시작하여, 빨간색 점선에 대응되는 edge와 네모에 대응되는 node들을 모두 "가지치기"하고 난 후에는, 왼쪽 그림에서 빨간 점선을 제외한 것과 같은 비교적 간단한 형태의 sub-network만이 남게 됩니다. 이렇게 남은 sub-network는 모든 edge마다 특정한 weight를 가지고 있고, 이 weight는 업데이트되지 않습니다. 만약 "가지치기"를 하는 알고리즘이 적절하게 설계되었다면, sub-network는 weight optimization이 수행된 신경망과 유사한 성능을 보입니다.



본 논문에서 제안한 Slot Machines 알고리즘은 이와 공통점을 갖지만, 조금 다른 방법을 사용하여 weight가 고정된 network를 찾습니다. 위 그림에서 왼쪽 그림이 본 논문에서 제안하는 방법론에 해당됩니다.




## **3. Method**  

앞서 이야기했듯이, 본 논문에서는 Back-propagation을 통해 weight 값을 지속적으로 업데이트하는 기존의 방법론과 다르게, 고정된 weight 값을 선택하는 방법론을 제안합니다. 그렇다면 이러한 고정된 weight값은 어디서, 어떻게 선택하게 될까요?

제안한 알고리즘에서는 신경망의 모든 connection마다 |S| = K 인 weight set S를 생성하고, 그 set S에서 해당 connection에 대한 weight value를 선택합니다. 이 때 주의할 점은 모든 connection이 각각의 독립적인 weight set S를 가지고 있다는 것입니다. 논문의 제목이 Slot Machine인 이유는, 이처럼 connection마다 weight 값을 선택하고, 모든 connection에 대해 이러한 weight 선택이 독립적으로 이루어지는 과정을 Slot machine의 각 Slot에서 reels를 선정하는 모습에 비유했기 때문입니다.

하지만 weight를 선택하는 과정을 조금 더 자세히 살펴보면, 실제 Slot Machine과는 다른 것을 알 수 있습니다. 실제 Slot Machine에서는 각 Slot에서 무작위로 reels를 선정하지만, 제안한 알고리즘에서는 score를 기반으로 weight를 선정하기 때문입니다. 

이렇게 score를 기반으로 선택된 weight로 모든 connection이 구성되므로, 적절한 score를 선택하는 것은 신경망의 성능을 결정짓는 중요한 작업입니다. Slot Machines에서는 weight 값 자체를 학습하는 대신, 이 score값을 학습함으로써 최종적으로 적절한 weight를 선택하게 됩니다.

![image-20220426212914760](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220426212914760.png)

Slot Machines가 학습을 반복하는 과정은 기존 신경망의 학습 과정과 마찬가지로 Forward Pass 와 Backward Pass가 반복됩니다. 위 그림에서 볼 수 있듯이, Forward Pass에서는 score값을 기반으로 weight를 선택하고, Backward Pass에서는 모든 score 값에 대한 update를 실시합니다.

Backward Pass 와 Forward Pass는 다음과 같이 이루어집니다.

- Forward Pass

![](https://latex.codecogs.com/svg.image?\{W_{ij1},&space;W_{ij2},...,W_{ijK}\}&space;) : 임의의 layer ![](https://latex.codecogs.com/svg.image?l)에 있는 neuron ![](https://latex.codecogs.com/svg.image?i)와 layer ![](https://latex.codecogs.com/svg.image?(l-1))에 있는 neuron ![](https://latex.codecogs.com/svg.image?j)를 연결하는 connection(![](https://latex.codecogs.com/svg.image?i),![](https://latex.codecogs.com/svg.image?j))의 weight 집합 (원소 하나가 하나의 	weight 값. K개 존재)

![](https://latex.codecogs.com/svg.image?\{s_{ij1},&space;s_{ij2},...,s_{ijK}\}&space;{}) : 위 weight set에 해당하는 score set. (원소 하나가 하나의 weight에 대한 score 점수)

![](https://latex.codecogs.com/svg.image?\rho&space;(s_{ij1},s_{ij2},...,s_{ijK})) : score 기반 weight 선택 함수.  함수 ρ에 따라 선택하는 weight의 index ![](https://latex.codecogs.com/svg.image?k^*&space;)가 결정된다. ![](https://latex.codecogs.com/svg.image?k^*&space;)=![](https://latex.codecogs.com/svg.image?\rho&space;(s_{ij1},s_{ij2},...,s_{ijK}))



Slot Machines 알고리즘에서는 weight를 선택하기 위한 ρ 함수로 2가지 함수를 사용하고 있습니다.

1. Greedy Selection (GS) : ![](https://latex.codecogs.com/svg.image?\rho&space;=&space;argmax\underset{k}\{s_{ij1},...,s_{ijK}\})

2. Probabilistic Sampling (PS) : ![](https://latex.codecogs.com/svg.image?\rho&space;\sim&space;Mult(\frac{e^{s_{ij1}}}{\sum_{k=1}^{K}e^{s_{ijk}}},...,\frac{e^{s_{ijK}}}{\sum_{k=1}^{K}e^{s_{ijk}}}))

​	

- Backward Pass

Backward Pass에서는 모든 score 값이 _straight-through gradient estimation_을 통해 update됩니다. 그 이유는 score 기반 weight 선택 함수 ρ가 대부분 0 gradient를 갖기 때문입니다. _straight-through gradient estimator_는 Backward Pass에서 ![](https://latex.codecogs.com/svg.image?s_{ijk})의 loss에 대한 gradient를 다음 식과 같이 설정함으로써,  ρ함수를 기본적으로 identity function으로 취급합니다.

​	![](https://latex.codecogs.com/svg.image?\bigtriangledown&space;s_{ijk}&space;\leftarrow&space;\frac{\partial&space;L}{\partial&space;a(x)_i^{l}}h(x)_j^{l-1}W_{ijk}^l)

![](https://latex.codecogs.com/svg.image?L) : objective function

![](https://latex.codecogs.com/svg.image?a(x)_i^l&space;) : pre-activation of neuron ![](https://latex.codecogs.com/svg.image?i) in layer ![](https://latex.codecogs.com/svg.image?l)

이 때 learning rate가 ![](https://latex.codecogs.com/svg.image?\alpha)이면, score 값은 SGD(Stochastic Gradient Descent)를 사용하여 다음과 같이 update됩니다.

 ![](https://latex.codecogs.com/svg.image?\widetilde{s}_{ijk}=s_{ijk}-\alpha\bigtriangledown&space;s_{ijk})



위처럼 Forward Pass와 Backward Pass를 반복하여 최종적으로 학습된 신경망은, 모든 connection에 대하여 가능한 weight set 중 가장 높은 score값을 가진, 즉 가장 _**바람직한**_  weight를 선택하게 됩니다.(이는 GS와 PS 방법 모두 마찬가지입니다)

이렇게 선택된 weight는 고정되어 변하지 않으며, inference 진행 시 다른 weight들에 대해서는 아무런 연산도 필요하지 않습니다. 즉, Inference Time은 기존의 Weight Optimization을 통해 학습한 신경망과 동일한 수준을 가집니다.





앞서 Slot Machines는 모든 connection마다 크기가 K인 weight set과 score set을 가진다고 하였습니다. 이 weight set 내에서 score 값을 통해 가장 적합한 weight를 판단하고, 최종적인 weight를 결정하게 됩니다. 

Slot Machines는  초기에 주어진 weight set 내에서 어떠한 weight 값들을 선택할지 결정하지만, 이 weight값 자체는 결코 변하지 않습니다. 따라서, 이 weight 값을 처음에 잘 설정하는 것이 모델의 성능에 크나큰 영향을 미칠 것으로 보입니다.

그렇다면 각 connection에 대한 weight set과 score set은 어떻게 초기화될까요? 



놀랍게도, Slot Machines의 weight set은 Glorot Uniform distribution ![](https://latex.codecogs.com/svg.image?U(-\sigma&space;_x,\sigma_x)) 으로부터 random sampling 됩니다.

이와 유사하게 score set은 uniform distribution ![](https://latex.codecogs.com/svg.image?U(0,\lambda\sigma_x)) 으로부터 random sampling 됩니다.

이 때 ![](https://latex.codecogs.com/svg.image?\lambda) 는 small constant입니다. Slot Machines는 CNN architecture를 사용하여 실험을 진행하였는데, ![](https://latex.codecogs.com/svg.image?\lambda)값은 다음과 같이 사용하였습니다.

Convolutional layer : ![](https://latex.codecogs.com/svg.image?\lambda)= 1 

Fully Connected layer : ![](https://latex.codecogs.com/svg.image?\lambda)= 0.1



이어지는 4.Expriment에서는 이러한 무작위 초기화에도 불구하고, Slot Machines 모델이 기존의 Weight Optimization을 통해 학습된 model과 비슷한 성능을 나타내거나, 더욱 뛰어난 성능을 나타냄을 보여줍니다.


## **4. Experiment**  

앞서 설명한 알고리즘은 굉장히 간단하며, 특별한 알고리즘도 없이 무작위로 추출한 weight값들을 이용하여 신경망이 세팅됩니다.

하지만 이러한 간단한 알고리즘에도 불구하고 Slot Machines는 다양한 dataset과 model에 대해 꽤나 인상적인 성능을 보여줍니다.

### **Experiment setup**  
* Dataset  

​		Dataset은 MNIST와 CIFAR-10 Dataset을 사용하였습니다.

​		MNIST Dataset에 대해서는 전체 dataset의 15%를, CIFAR-10 Dataset에 대해서는 10%를 validation set으로 사용하였습니다.

​		두 Dataset모두 test set은 따로 분리해서 실험을 진행하였습니다.

​		

​		모든 model은 동일한 optimization method와 동일한 hyper-parameter, 동일한 추가 기법(data augmentation, dropout 등)을 사용하였습니다.

​		주목할만한 점은 기존의 weight학습시 사용하는 learning rate와 다르게, Slot Machine 알고리즘에서 score를 학습시키는 learning rate는 0.1, 0.2, 25 등 큰 		값을 사용했다는 것입니다.

* baseline

  baseline model로는 Renet-300-100 architecture와 VGG-19 architecture, 추가적으로 VGG-19와 유사한 3가지 architecture(CONV-2, CONV-4, CONV-6)를 사용하였습니다.

  MNIST Dataset에는 Renet-300-100 architecture 사용하였고, CIFAR-10 Dataset에는 VGG-19와, 이와 유사한 3개의 architecture(CONV-2, CONV-4, CONV-6)를 사용하였습니다.

  

  모든 model은 동일한 optimization method와 동일한 hyper-parameter, 동일한 추가 기법(data augmentation, dropout 등)을 사용하였습니다.

  주목할만한 점은 기존의 weight학습시 사용하는 learning rate와 다르게, Slot Machine 알고리즘에서 score를 학습시키는 learning rate는 0.1, 0.2, 25 등 큰 값을 사용했다는 것입니다.

  

  각 architecture에 대한 정보는 다음과 같습니다.

  ![image-20220426224722277](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220426224722277.png)

  

  

* Evaluation Metric  

  다음과 같은 평가를 실시하였습니다.

  1. K값에 따른 test accuracy 비교
  2. Baseline(weight optimization)과의 비교
     - K값에 따른 test accuracy
     - training cost VS test accuracy
  3. Slot Machines를 이용한 Fine tuning 시 성능 향상 비교
  4. 기타 Slot Machines에 대한 Experiment

### **Result**  
1. K값에 따른 test accuracy 비교

![image-20220426230627896](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220426230627896.png)

- 이는 baseline과의 직접적인 비교는 아니고, Slot Machines의 성능을 보여주기 위해 K=1일 때와 K=2일 때를 비교한 결과

- 여기서 K값은 connection하나마다 존재하는 weight set의 크기를 의미

- 위와 같이, K=1 일 때에 비해, K=2일 때 Slot Machines의 test accuracy가 모든 dataset에 대하여 급격하게 증가

- 즉, 선택 가능한 weight option이 하나만 추가되어도, 가능한 weight combination은 기하급수적으로 늘어나기에, 무작위로 초기화된 weight 값들이라 할지라도 적절한 조합을 찾아내었을 때는 꽤 훌륭한 accuracy를 보임

  

2. Baseline과의 비교

   a. K값에 따른 test accuracy

​	![image-20220426230552349](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220426230552349.png)

- Slot Machines에서 K의 값을 증가시키면서 실험을 진행
- PS보다 성능이 우수한 GS forward pass 사용
- K 값이 증가함에 따라서, test accuracy가 증가하는 것을 볼 수 있음.
  - K=8~16일 때, 전반적으로 최고 test accuracy를 보임
  - CONV-6모델을 사용한 경우에는 기존의 weight 학습 모델보다 더 높은 성능을 보임
  - 모든 weight값이 무작위로 초기화되었음에도, 최적의 조합을 찾아내기만 한다면 높은 성능을 보임을 확인
- 또한, 전반적인 training cost는 Slot machine과 기존의 traditional optimization 방법론이 비슷한 정도인 것을 확인



​		b. training cost VS test accuracy

![image-20220426230850965](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220426230850965.png)

- 전반적인 training cost는 Slot machine과 기존의 traditional optimization 방법론이 비슷한 정도이거나, 기존의 traditional optimization 방법론이 조금 더 효율적인 것을 확인



3. Slot Machines를 이용한 Fine tuning 시 성능 향상 비교

   ![image-20220426231101304](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220426231101304.png)

   ![image-20220426231235236](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220426231235236.png)



- Slot Machines 알고리즘은 기존 traditional optimization 방법론의 초기 weight 설정을 위해서도 사용 가능
- 동일한 Total Training Cost를 사용하는 조건에서, Slot Machines를 활용하여 초기 weight를 찾고 이에 대한 weight 학습을 진행한 case에서 가장 높은 test accuracy를 관찰
- 하지만 (위 그림에는 없지만) VGG-19의 경우에는 fine-tuning 시에 기존의 Slot Machines 알고리즘보다 좋은 정확도를 달성했지만, 처음부터 weight를 학습하는 방법보다는 약간 낮은 정확도를 달성
- 실제로 Slot Machines알고리즘을 활용한 Fine-tuning이 효과가 있는지를 검증하기 위해, Slot Machines 학습의 epochs를 변화하면서 최종 test accuracy를 관찰한 결과, Slot Machines의 학습 epochs이 증가함에 따라 최종 test accruacy도 증가함을 확인
  - 이 때 총 train epochs(epochs for Slot Machines + epochs for weight optimization)는 동일하다.
  - 아래 Figure 참고



4. 기타 Slot Machines에 대한 Experiment

   ![image-20220426231352167](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220426231352167.png)

   

- GS VS PS 비교

  - GS 알고리즘이 모든 K값에 대해서 더 좋은 성능을 보임 (위 Figure 참고)

  - PS 알고리즘은 Probabilistic 한 특성상 계속해서 weight가 변화하는 connection이 많고, 이에 따라 안정적인 학습이 어려움

  - 즉, 과도한 exploration에 의해 안정적인 학습이 이루어지지 않는다.

    

- Sharing Random Weights

  - 본 논문에서는 Slot Machines 알고리즘에 대한 두 가지 새로운 setting을 고려
    1. 같은 Layer내에 존재하는 connection들은 같은 random weight set을 공유
    2. 모든 connection들은 같은 random weight set을 공유
  - 두 경우 모두 K값이 큰 경우에 대해서는 학습이 적절하게 이루어졌지만, K값이 작아질수록 학습이 어려운 경우가 많음
  - Slot Machines는 weight set과 더불어 score set에 대한 메모리까지 신경써야하는 단점이 존재
  - weight를 적절하게 공유함으로써, 이러한 memory 부담을 줄일 수 있음

  


## **5. Conclusion**  

Slot Machines는 각 연결에 여러 가중치 옵션이 주어지고 이에 대한 좋은 선택 전략이 사용되는 경우, 무작위 가중치를 가진 신경망이 훌륭한 성능을 낼 수 있다는 것을 보여주었습니다.

실제로 본 연구에서 사용한 가중치 선택 전략은 간단하지만 훌륭한 성능을 나타내었고, 또한 이렇게 선택된 가중치 조합이, weight optimization을 위한 fine-tuning 방법으로도 사용될 수 있다는 것을 보여주었습니다.



다만 Slot Machines의 단점이라 한다면, 현재 알고리즘에서는 복잡한 모델일수록, K값이 클수록 weight개수가 기하급수적으로 증가하며, 모든 weight에 대해 score를 가지고 있어야 하기 때문에 상당히 많은 memory 가 필요하다는 것입니다. 이러한 memory requirement 문제를 어떻게 해결할 수 있을지 많은 고민이 필요해보이며 이러한 고민과 더불어 아직까지 많은 연구가 이루어지지 않은 분야인만큼, 관련된 다양한 Mechanism이 등장할 수도 있을 것이라 생각합니다.

Slot Machines의 아이디어와 연결된 다양한 mechanism이 등장한다면, 기존의 weight optimization에 더해 또 다른 옵션으로 충분히 고려해볼만할 중요한 방법론이 될 수 있을 것이라 생각합니다.



다양하게 고민해보아야 할 부분 중 하나를 예로 들자면, weight set에서 최종적으로 선택된 weight와 그렇지 않은 weight의 유의미한 차이를 구분할 수 있다면, 초기에 무작위로 weight를 생성할 때 유의미한 weight값들을 많이 생성할 수 있을 것이고, 학습 속도와 최종 성능에 큰 영향을 미칠 수 있을 것이라 생각합니다.



마지막으로, Slot Machines를 Fine tuning으로 활용함으로써 기존 Optimization model보다 더욱 뛰어난 성능을 보인 것은 굉장히 의미있는 성과이며, 다양한 model에도 이러한 방향을 적용해볼 수 있을 것이라 생각합니다.

---
## **Author Information**  

* Author name  : 신동휘
    * Affiliation  :  Industrial and System Engineering, KAIST
    * Research Topic : AMHS Design and Operation

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Reference  
  * [straight-through gradient estimation](https://arxiv.org/abs/1903.05662)
  * [Neural Network Pruning](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://proceedings.neurips.cc/paper/2020/file/46a4378f835dc8040c8057beb6a2da52-Paper.pdf)
  * [Optimal Lottery Tickets](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://papers.nips.cc/paper/2020/file/1b742ae215adf18b75449c6e272fd92d-Paper.pdf)
  * [supermasks](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://proceedings.neurips.cc/paper/2019/file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf)