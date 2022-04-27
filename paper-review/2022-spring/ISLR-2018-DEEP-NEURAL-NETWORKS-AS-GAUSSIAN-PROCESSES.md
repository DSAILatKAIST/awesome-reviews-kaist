# DEEP NEURAL NETWORKS AS GAUSSIAN PROCESSES

## 1. Problem Definition

 해당 논문에서는 infinitely wide deep networks 와 Gaussian processes의 동일성을 설명한다.



## 2. Motivation

 단층구조의 신경망의 모수(parameters)에 independent & identical & distributed (이하 i.i.d) 사전분포(prior)를 정의해주면 Gaussian process(GP)와 동일하다는 것은 알려져있다. 이에 더해 해당 논문은 다층구조의 신경망과 Gaussian process의 관련성을 설명한다. 



## 3. Method

 단층구조의 신경망과 Gaussian process의 연관성을 설명하고 귀납적으로 다층 구조의 신경망과 Gaussian process가 연결됨을 설명한다. 이에 앞서 Notation을 정의한다. 

L-hidden-layer-fully-connected neural network 에서
$$
N_l \ \
,\phi \ \
,x_{i}^l \ \
, z_{i}^l\ \
,W_{ij}^l\ \
,b_{i}^l
$$
를 각각 ㅣ- layer 의 width, nonlinear 활성화 함수, l - layer의 post-activation, l - layer의 post-affine transformation, 가중치와 편향을 나타낸다. 관계를 수식으로 정리하면 다음과 같이 된다. 
$$
z_{i}^l(x_{input})\ =b_i^l\ + \ \sum_{j=1}^{N_{l}}W_{ij}^l\ x_j^l(x_{input}) \,\,\ , \\x_{j}^l=\phi(z_{j}^{l-1}(x_{input})) \\,
\text{where}\ x_{input}\ \text{is input data}
$$
단순 구조의 신경망의 경우에는 다음과 같이 정의한다.
$$
z_{i}^1(x_{input})\ =b_i^1\ + \ \sum_{j=1}^{N_{1}}W_{ij}^1\ x_j^1(x_{input}) \,\,\ , \\x_{j}^l=\phi(b_i^0\ + \ \sum_{k=1}^{d_{in}}W_{jk}^0\ x_{input(k)}),\ where \\ 
x_{input(k)}\ \text{is \ k-th component \ of}\  x_{input} \ \& \\ z_{i}^1(x_{input})\ \text{is \ i-th \ of\ output}
$$
또한 Gaussian process와 연결시키기 위해서 가중치와 편향을 deterministic한 경우가 아니라 정규분포를 따른다고 가정한다. 이때 각각의 가중치와 편향은 i.i.d 조건을 만족시킨다.
$$
W_{ij}^l \sim N(0,\sigma_w^2 /N_{l})\\
b_{i}^l \sim N(0,\sigma_b^2)
$$
가중치와 편향이 i.i.d이기 떄문에 post-activation 역시 i.i.d 조건을 따르며, post-affine transformation은 i.i.d terms 의 합으로 정의된다. 이에 layer의 width를 증가시킴에 따라 Central Limit Theorem을 적용시키면 post-affine transformation은 정규분포를 따르게 된다. 
$$
W \, ,b : i.i.d \ \rightarrow x_{j}^l : i.i.d \\
\Rightarrow \ z_{i}^l\ =b_i^l\ + \ \sum_{j=1}^{N_{l}}W_{ij}^l\ x_j^l \ \sim \ Normal\ Distribution
$$
  이를 귀납적으로 적용시키면 다층구조 신경망의 모든 layer의 post-affine transformation과 post-activation은 정규분포를 따르게 되고 나아가 output역시 정규분포를 따르게 된다. 각각의 observation의 post-affine transformation이 정규분포를 따르게 되고 parameter(가중치, 편향) 역시 정규분포를 따르고 있으므로 이는 Gaussian process와 동일한 의미를 가진다. 
$$
\{ z_{i}^l(x^{\alpha = 1}), z_{i}^l(x^{\alpha = 2}), z_{i}^l(x^{\alpha = 3}),,,, z_{i}^l(x^{\alpha = k})\} \sim Joint \,Normal \, distribution\,\,,\forall i,l \\
where,\ x^{\alpha = k} \ \text{means k-th observation}\\
\Rightarrow z_{i}^l \sim GP(0,K^l)
$$
GP( , )는 Gaussian process 를 나타낸다. prior 들이 전부 평균을 0으로 하는 분포로 설정이 되었기에, GP의 평균역시 0를 가지게 되며, 공분산 함수는 kernel의 형태로 정의되며 다음과 같이 정의된다.
$$
K^l(x,x')=E[z_i^l(x)\ z_i^l(x')]=\sigma_b^2 + \sigma_w^2 E_{z_{i}^{l-1}\sim GP(0,K^{l-1})}[\phi(z_{i}^{l-1}(x))\ \phi(z_{i}^{l-1}(x'))]
$$
위의 기댓값은 joint distribution을 적분하는 것과 동일하기에 공분산 함수는 다음과 같이 전환되어 표현될 수 있다. 
$$
K^l(x,x')=\sigma_b^2 + \sigma_w^2F_{\phi}(K^{l-1}(x,x'),K^{l-1}(x,x'),K^{l-1}(x,x'))
$$
또한 첫 layer에서의 계산의 정의를 위하여 첫 kernel 함수를 다음과 같이 정의된다. 
$$
K^0(x,x')=E[z_{j}^0(x) z_{j}^0(x')]=\sigma_{b}^2 +\sigma_{w}^2({x \cdot x'}/d_{in})
$$
DNN과 Gaussian process의 연관성은 베이지안 추론(Bayesian inference)과도 연결이 된다. Gaussian prior(정규사전분포)를 사용하여 GP의 kernel 함수를 결정하며 이는 가중치, 편향, activation function, depth 등에 의해서 결정되므로 베이지안 추론(Bayesian inference)을 통한 추정으로 DNN의 hyperparameter tunning이 가능하다. 

## 4. Experiment

Gaussian process 의 베이지안 추론을 통해 생성한 Neural Network (이하 NNGP)와 SGD로 훈련된 neural networks 를 MNIST 데이터와 CIFAR-10 데이터를 통해 비교하였다. MSE(Mean Squared Error)를 성능지표로 사용하였다.

\
![](../../.gitbook/2022-spring-assets/NabilahMuallifah\_1/0.png)
![](https://github.com/Mos-start2092/ML_class_2022/tree/main/Seongbin_1/1.png)

위의 실험은 두가지 입장에서 흥미로웠다.

1. NN은 flexiblity로 인해서 powerful한 모델로 생각되는 반면, GP 방법의 경우 고정된 basis functions을 사용하여 학습을 하였다.  그럼에도 두가지 모델에서 현저한 차이가 나지 않았다는 것이다. 
2. SGD와 NNGP가 특정 가정이 있으면 서로 근접할 수 있다는 것이다. 

또한 GP의 이점을 볼 수 있는데 불확실성(uncertainty)이 예측오차(prediction error)와 상관되어있다는 것이다. GP 는 베이지안 적인 특성으로 인해 모든 예측에서 불확실성을 가진다. 즉, NNGP에서는 모든 test 는 예측 분산에 관한 추정치를 찾을 수 있다는 것이다. 위의 실험에서 이러한 불확실성(예측 분산에 대한 추정)이 예측오차와 강한 상관성이 있음을 찾아냈고 다음과 같은 형태의 그래프를 보여준다.

![image-20220424165815919](C:\Users\sungb\AppData\Roaming\Typora\typora-user-images\image-20220424165815919.png)



## 5. Conclusion

DNN을 Gaussian process와 연관하여 설명하였다. 이러한 접근은 베이지안적 모수 추정이 가능하게 하였고 gradient-based의 방법 없이도 예측값을 얻어낼 수 있었다. 실험에서 기존의 DNN과 크게 성능이 다르지 않음을 확인하였으며 실제 DNN의 width가 높아짐에 따라 (:CLT의 수렴성이 높아짐에 따라) NNGP에 접근함도 확인하였다. 또한 NNGP는 예측 오차와 관련된 불확실성을 제공한다. 그렇기에 모델의 실패를 예측하거나 label을 조정하기 위한 최적의 데이터 포인트를 식별하는데에 사용하기 유용할것이다.



### Author Information

- 안성빈
  - KAIST ISYSE
  - Statistics, Data Sceince

## 6. Reference & Additional materials

- KAIST 문일철 교수님의 Gaussian Process 강의

  https://www.youtube.com/channel/UC9caTTXVw19PtY07es58NDg
