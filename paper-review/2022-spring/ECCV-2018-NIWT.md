---
description: Ramprasaath R. Selvaraju, Prithvijit et al./ Choose Your Neuron- Incorporating Domain Knowledge through Neuron-Importance / ECCV(2018)   
---  

# Choose Your Neuron: Incorporating Domain Knowledge through Neuron-Importance

# 1. Problem Definition

본 논문은 class-specific domain knowledge를 neuron importance 와 매핑하여 zero-shot prediction을 수행함과 동시에 interpretable explanation를 제공하는 Neuron Importance-based Weight Transfer(NIWT)을 제안한다.

---

# 2. Motivation

> **“how to leverage this neuron-level descriptive supervision to train novel classifiers?”**
> 

neuron-level description을 zero-shot learning classifier에도 적용하기 위한 고민이 본 논문의 핵심이다.

먼저, zero-shot learning이란 deep classifier가 학습 시 보지 못한 unseen class data를 분류하도록 하는 방법이며 massive labeled datasets 없이도 모델 학습의 일반화 성능을 개선할 수 있다는 점에서 주목받고 있다. 다양한 방법론(attribute-based, Text-based)이 존재하지만 unseen class data에 대한  network decision의 interpretability를 제공하기 위한 연구가 부족하며, 관련 연구가 필요함을 주장한다.

저자는 external domain knowledge(text based or otherwise)를 neuron과의 직접 mapping을 통해 zero-shot learning과 동시에 interpretable explanation을 제공하는 방법론을 제시한다.


![fig0.png](https://github.com/LOVELYLEESOL/awesome-reviews-kaist/blob/patch-5/.gitbook/2022-spring-assets/LEESOL_1/fig0.png)


---

# 3. Method

NIWT는 크게 3가지 단계로 구성된다.

1. seen class 기반으로 학습한 네트워크의 fixed layer에서 neuron-importance 계산
2. domain knowledge와 Neuron-importance의 mapping 학습
3. unseen class 기반으로 예측된 neuron-importance를 바탕으로 clssifier weights 최적화

아래는 각 단계에 대한 구체적 설명이다.

### 3.1 Preliminaries: Generalized Zero-Shot Learning

Generalized Zero-Shot Learning의 목표는 ![](https://latex.codecogs.com/gif.latex?f:\mathcal{X}\rightarrow\mathcal{S}\cup\mathcal{U}) 를 학습하는 것이다. 

dataset : ![](https://latex.codecogs.com/gif.latex?\mathcal{D}=\{(x_i,y_i)\}_i^N)

seen classes : ![](https://latex.codecogs.com/gif.latex?\mathcal{S}=\{1,...,s\})

unseen classes : ![](https://latex.codecogs.com/gif.latex?\mathcal{U}=\{s+1,...,s+u\})

domain knowledges : ![](https://latex.codecogs.com/gif.latex?\mathcal{K}=\{k_1,...,k_{s+u}\})

### 3.2 Class-dependent Neuron Importance

![](https://latex.codecogs.com/gif.latex?\mathrm{NET}_\mathcal{s}(.))은 seen class 예측을 위해 학습한 네트워크를 의미하며(![](https://latex.codecogs.com/gif.latex?{o_c|c\in\mathcal{S}\})) ![](https://latex.codecogs.com/gif.latex?o_c)의 ![](https://latex.codecogs.com/gif.latex?a^n_{i,j})에 대한 gradient를 구한 후 global average pooling을 통해 class dependent neuron importance를 도출할 수 있다.
 
![fig1.png](https://github.com/LOVELYLEESOL/awesome-reviews-kaist/blob/patch-5/.gitbook/2022-spring-assets/LEESOL_1/fig1.png)


![](https://latex.codecogs.com/gif.latex?n): Channel dimesion

![](https://latex.codecogs.com/gif.latex?a^n_{i,j}): The activation of neuron n at spatial position i, j

![](https://latex.codecogs.com/gif.latex?o_c): Prediction score

*자세한 설명은 Selvaraju, R.R., Das, A., Vedantam, R., Cogswell, M., Parikh, D., Batra, D.:
Grad-CAM: Why did you say that? Visual Explanations from Deep Networks via
Gradient-based Localization. ICCV (2017) 참고*

### 3.3 Mapping Domain Knowledge to Neurons

 ![](https://latex.codecogs.com/gif.latex?\mathrm{NET}_\mathcal{S}(.))의 한 layer을 L이라 하고 seen classes instances을  ![](https://latex.codecogs.com/gif.latex?(x_i,y_i)\in\mathcal{D_s})이라 할때, ![](https://latex.codecogs.com/gif.latex?a_c=\{a^n_c|n\in{L}\})는 L layer로부터 계산된 class c에 대한 neuron importance vectors이다.

Domain knowledge와 neuron importance vector를 linear mapping(transformation)하기 위해 먼저, importance vectors ![](https://latex.codecogs.com/gif.latex?a_{y_i})를 계산한 후 해당 class와 관련있는 domain knowledge(![](https://latex.codecogs.com/gif.latex?k_{y_i}))를 매칭한다(![](https://latex.codecogs.com/gif.latex?a_{y_i}),![](https://latex.codecogs.com/gif.latex?k_{y_i})). ![](https://latex.codecogs.com/svg.image?W_{\mathcal{K}\rightarrow&space;a})를 추정하기 위해 cosine distance를 이용하여 loss를 정의하고 gradient를 이용하여 이를 minimize한다. 

![fig2.png](https://github.com/LOVELYLEESOL/awesome-reviews-kaist/blob/patch-5/.gitbook/2022-spring-assets/LEESOL_1/fig2.png)


### 3.4 Neuron Importance to Classifier Weights

unseen class 예측을 할 수 있는 classifier를 학습하기 위해서 predicted importance를 사용한다.

1. Seen class를 기반으로 학습한 network ![](https://latex.codecogs.com/gif.latex?\mathrm{NET}_\mathcal{S})의 output space에 unseen class를 포함시키기 위해 마지막 fully connected layer에 unseen classes weight vectors ![](https://latex.codecogs.com/svg.image?\mathrm{w}^1,...,\mathrm{w}^u)을 추가하여 output scores를 ![](https://latex.codecogs.com/gif.latex?\{o_c|c\in\mathcal{U}\})로 확장시킨다(![](https://latex.codecogs.com/gif.latex?\mathrm{NET}_{\mathcal{S}\cup\mathcal{U}})). 
    
    이때, unseen classes의 초기 weight vector은 multivariate normal distribution으로부터 랜덤하게 샘플링 한 것이며, 이로부터 도출된 output score은 uninformative한 상태이다.
    
2. 3.3에서 도출한 ![](https://latex.codecogs.com/svg.image?W_{\mathcal{K}\rightarrow&space;a})과 unseen class domain knowledge ![](https://latex.codecogs.com/gif.latex?\mathcal{K}_\mathcal{U})을 바탕으로 unseen class importance ![](https://latex.codecogs.com/gif.latex?A_\mathcal{U}=\{a_1,...,a_u})를 예측한다. ![](https://latex.codecogs.com/svg.image?a_c=W_{\mathcal{K}\rightarrow&space;a}k_c)(unseen class c). 
4. ![](https://latex.codecogs.com/gif.latex?\mathrm{NET}_{\mathcal{S}\cup\mathcal{U}})으로 부터 unseen class c에 대한 importance vector을 계산하고 (![](https://latex.codecogs.com/gif.latex?\hat{a}^c))  weight parameter ![](https://latex.codecogs.com/gif.latex?w^c)를 gradient descent를 통해 optimize한다. (predicted importance vector(![](https://latex.codecogs.com/gif.latex?a_c)), observed importance vector(![](https://latex.codecogs.com/gif.latex?\hat{a}^c)) 사이의 cosine distance를 minimize)
5. Cosine distance는 scale을 고려하지 않으며 regularization가 없으면 seen class weight나 unseen class weight 한쪽으로의 bias을 초래할 수 있다. 이러한 문제점을 해결하기 위해 unseen weight를 seen weight의 평균(![](https://latex.codecogs.com/gif.latex?\bar{\mathrm{w}}_\mathcal{S}))과 유사한 scale로 학습할 수 있도록 하는 L2 regualization term을 추가했다(![](https://latex.codecogs.com/gif.latex?\Lambda)는 regulization의 정도를 control).

![fig3.png](https://github.com/LOVELYLEESOL/awesome-reviews-kaist/blob/patch-5/.gitbook/2022-spring-assets/LEESOL_1/fig3.png)


정리하면, ![](https://latex.codecogs.com/gif.latex?a^c)는 network gradient를 통해 계산할 수 있고, weight는 위의 loss를 이용하여 update하는 방식으로 학습을 진행한다.

---

# 4. Experiment

### Experiment setup

- Dataset
    - Animals with Attributes2 (AWA2)
        
        50가지의 동물 종으로 구성된 37,322개의 이미지 데이터셋이다. 각 클래스에는 85개의 binary와 continuous attribute가 labeled 되어있다.
        
    - Caltech-UCSD Birds 200 (CUB)
        
        200가지의 새의 종으로 구성된 11,788개의 이미지 데이터셋이다. 각 클래스에는 312개의 binary와 continuous attribute가 labeled 되어있다. 이 attribute에는 새의 특징 (색, 몸통의 생김새 등)을 포함하고 있으며 각 이미지에는 10개의 human captions이 있다.
        
    
- Evaluation Metric
    
    Generalized zero-shot learning (GZSL)에 대한 성능평가를 진행했으며 seen class와 unseen class 모두에 대한 accuracy를 도출했다. 
    
    - Unseen accuracy: ![](https://latex.codecogs.com/gif.latex?\mathrm{Acc}_\mathcal{U})
    - Seen accuracy: ![](https://latex.codecogs.com/gif.latex?\mathrm{Acc}_\mathcal{S})
    - Harmonic mean between both: ![](https://latex.codecogs.com/gif.latex?\mathrm{H})

- Model
    
    ImageNet에 대해 pretrain된 ResNet101, VGG16을 기반으로 seen class에 대해 finetuning이 이루어졌다. 각 모델에  1) 모든 layer에 대한 finetuning (FT), 2) 마지막 classification weights undate (Fixed) 총 2가지의 학습을 진행했다.
    
    Resnet의 경우 FT에 경우 accuracy가 현저히 작은 결과를 보였으나(60.6% finetuned vs 28.26% fixed for CUB and 90.10% vs 70.7% for AWA2), VGG의 경우 FT와 Fixed setting의 정확도가 유사했다(74.84% finetuned vs 66.8% fixed for CUB and 92.32% vs 91.44% for AWA2). 
    
- NIWT Settings
    
    Domain knowledge를 neuron importance와 mapping하는 학습을 위해 홀드아웃 검증을 진행하였고 observed importance와 predicted importance의 rank correlation이 최대일때 optimization을 멈추도록 설계했다. 
    
    Attribute vector로는 각 클래스의 class level attribute를 사용했고 CUB의 captions은 word2vec embadding의 클래스별 평균을 사용했다. weight optimizing시 loss가 40 iteration을 거치는 동안 1% 이상의 개선이 없다면 학습을 중단했다. 
    
    - Hyper parameter
        
        ![](https://latex.codecogs.com/gif.latex?\Lambda)와 learning rate는 ![](https://latex.codecogs.com/gif.latex?1e^{-5})와 ![](https://latex.codecogs.com/gif.latex?1e^{-2})사이에서 설정했고 batch size는 ![](https://latex.codecogs.com/gif.latex?\mathrm{H})기반 grid search({16,32,64})를 진행했다.
        
    
- Baselines
    
    성능이 우수한 zero-shot learning 방식으로 알려진 ALE와 Deep Embadding을 baseline으로 선정했다.
    
    - ALE(Accumulated Local Effects)
        
        Ranking loss를 사용하여 class label과 visual feature사이의 compatibility function을 학습하는 방법.  (Xian, Y., Schiele, B., Akata, Z.: Zero-shot learning - the good, the bad and the ugly. In: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (July 2017) 참고)
        
    - Deep Embadding
        
        Deep network를 이용하여 domain knowledge와 deep feature을 end-to-end로 학습하는 방법. (Zhang, L., Xiang, T., Gong, S.: Learning a deep embedding model for zero-shot learning. In: CVPR (2017) 참고) 
        

### Result
![fig4.png](https://github.com/LOVELYLEESOL/awesome-reviews-kaist/blob/patch-5/.gitbook/2022-spring-assets/LEESOL_1/fig4.png)


실험 결과에서 도출할 수 있는 NIWT의 Contribution은 아래와 같다.

1. **Generalized zero-shot learning에서 NIWT이 SOTA를 보였다.**
    
    두 데이터셋 모두에서 VGG16 기반의 NIWT-Attributes 방식이 harmonic mean에 대한 SOTA를 보였다.(48.1% for AWA2 and 37.0% for CUB). AWA2에 대해서는 deep feature embedding 기반의 이전 SOTA에서 약 10%나 개선되었다.
    
2. **Seen class finetuning(FT)방식이 harmonic mean $\mathrm{H}$의 개선에 기여한다.**
    
    두 데이터셋 모두에서 seen class image에 대해 finetuning된 VGG network 기반의 NIWT가 높은 ![](https://latex.codecogs.com/gif.latex?\mathrm{H})을 보였다 (36.1%→48.1% on AWA2 and 26.7%→37.0% on CUB H respectively). ResNet 역시 유사한 양상을 보였다(27.5%→40.5 %H on AWA2 and 17.3%→27.7% H on CUB). 이러한 경향은 다른 method에서는 볼 수 없다는 점에서 주목할만 하다.
    
3. **NIWT는 attributes와 free-form language 모두에서 효과적이다.**
    
    Attributes와 caption 모두에 대해 NIWT의 성능이 뛰어남을 확인했다 (27.7% and 23.8% H for ResNet and 37.0% and 23.6% H for VGG). 본 논문에서 Caption을 word2vec embadding의 클래스별 평균으로 representation 하였는데 이러한 다소 단순한 처리가 성능 감소 요인으로 작용했을 것으로 예상된다.
    

---

# 5. Analysis & Explaining NIWT

- Regularization Coefficient $\Lambda$의 영향
    
    Regularizer term의 영향을 실험하기 위해 0에서 ![](https://latex.codecogs.com/gif.latex?\1e^{-2})의 ![](https://latex.codecogs.com/gif.latex?\Lambda) 범위에서 AWA2 데이터셋을 기반으로 seen class accurancy와 unseen class accuracy를 도출했다.
    
    Regulation이 없을 경우 (![](https://latex.codecogs.com/gif.latex?\Lambda)=0) unseen accuracy는 약 33.9%이다. ![](https://latex.codecogs.com/gif.latex?\Lambda)의 값이 증가할수록 unseen accuracy는 증가하며 ![](https://latex.codecogs.com/gif.latex?\1e^{-5})일때 가장 최고치의 accuracy(41.3%)를 보인다. 이는 Regulation이 없을 경우보다 있는 것이 성능 개선에 도움이 된다는 것을 보여준다.
    
    이러한 unseen class accuracy는 seen class accuracy와 같은 interval [![](https://latex.codecogs.com/gif.latex?\1e^{-5}), ![](https://latex.codecogs.com/gif.latex?\1e^{-4})]에서 약 3% 정도의 trade-off가 존재했다. 또한 $\Lambda$>$1e^-4$의 경우 regulation이 매우 크기 때문에 NIWT가 unseen class에 대한 학습에 어려움이 있었다고 해석할 수 있다.
    

![fig5.png](https://github.com/LOVELYLEESOL/awesome-reviews-kaist/blob/patch-5/.gitbook/2022-spring-assets/LEESOL_1/fig5.png)


- Explaining NIWT
    - Visual Explanation
        
        NIWT에서 unseen class classifier를 학습하는 것은 기존의 seen class 기반의 deep network를 확장하는 방식이기에(3.4, 1. 참고) unseen class에 대해서도 end-to-end pipe line을 유지한다. 이러한 특성으로 인해 기존 deep network interpretability mechanism을 적용하는데 제한이 없다.
        
        본 논문은 NIWT 기반의 network의 decision에 관한 정보를 시각화하기 위해 unseen class instance에 Grad-CAM을 사용했다.
        
    - Textual Explanation
        
        3.3에서 external domain knowledge(attribute or caption)과 neuron과의 mapping(![](https://latex.codecogs.com/svg.image?W_{\mathcal{K}\rightarrow&space;a}))을 학습했다. 이와 유사하게 neuron importance에서 attribute or caption과의 inverse mapping을 통해 모델의 decision에 있어서 text explannation을 제공할 수 있다 (inverse mapping(![](https://latex.codecogs.com/gif.latex?W_{a\rightarrow\mathcal{K}}))에서  ![](https://latex.codecogs.com/gif.latex?a_c) (unseen class neuron importance)가 주어졌을때 가장 높은 score의 ![](https://latex.codecogs.com/gif.latex?k_c)(attribute) 도출).
        
![fig6.png](https://github.com/LOVELYLEESOL/awesome-reviews-kaist/blob/patch-5/.gitbook/2022-spring-assets/LEESOL_1/fig6.png)

        

# 6. Conclusion

---

본 논문에서는 unseen class에 대한 domain knowledge를 network neuron importance와 접목하여 classifier weight에 직접 mapping하여 학습하는 Neuron Importance-aware Weight Transfer (NIWT)을 제안한다. 

실험을 통해 NIWT의 weight optimization 방식은 unseen class prediction에 대해 기존 방식보다 뛰어난 성능을 보임을 확인하였고, neuron을 semantic 개념과 연결하여 시각, 텍스트 설명을 제공할 수 있음을 보였다.

# Author Information

---

- 이 솔 (LEE SOL)
    - Affiliation
        
        KAIST, Industrial & System Engineering(Graduate School of Knowledge Service Engineering)
        
    - Research Topic
        
        Computer Vision
        

---

# 6. Reference & Additional materials

- Github Imaplementaion
    - https://github.com/ramprs/neuron-importance-zsl
- Reference
    - [https://arxiv.org/abs/1808.02861](https://arxiv.org/abs/1808.02861)
