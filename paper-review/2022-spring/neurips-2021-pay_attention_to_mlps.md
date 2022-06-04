---
description : Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le / Pay Attention to MLPs / NeurIPS - 2021
---

# Pay Attention to MLPs


## 1. Problem Definition

- gating과 MLP 구조를 사용한 간단한 네트워크를 통해 Transformer와 비슷한 성능을 보이는 gMLP 제안

## 2. Motivation
-   _**Motivation**_
    -   Transformer는 두 가지 중요한 부분을 가지고 있음
        -   각 토큰이 병렬로 처리 될 수 있는 recurrent-free 한 아키텍쳐
        -   multi-head self attention을 통해, 토큰들 간의 정보를 통합
    -   여기서 attention 구조는 인풋 간의 관계에 따라 파라미터가 달라지면서 토큰들의 관계를 판단하는 inductive bias 를 가지고 있음
    -   그런데, 과연 self-attention 구조가 Transformer의 성능에 중요한 역할을 했을까?
        -   self-attention이 정말 필요할지에 대한 의문에서 연구가 시작
 - _**Idea**_ 
    -   self-attention 구조가 없는 MLP 기반 모델 제안 (basic MLP layers with gating)

## 3. Method
>📌 **관련 용어**
>-  **Identity Mapping**  
>    - 입력으로 들어간 값 x가 어떠한 함수를 통과하더라도 다시 x가 나와야 한다는 것

  ### 3.1 Model description
  - 동일한 크기와 구조인 L개의 blocks으로 이루어져 있음    
-   각 블록은 다음과 같이 정의됨
- ![](https://latex.codecogs.com/svg.image?Z&space;=&space;\sigma&space;(XU),&space;\widetilde{Z}&space;=&space;s(Z),&space;Y&space;=&space;\widetilde{Z}V)
- ![](https://latex.codecogs.com/svg.image?X&space;\in&space;\mathbb{R}^{n\times&space;d})은  시퀀스 길이가 n이고 차원이 d인 token representations
-  ![](https://latex.codecogs.com/svg.image?\sigma&space;)은 activation function (such as GeLU)
-  U와 V는 channel dimension상에서의 linear projection
- s( )는 spatial interaction을 캐치할 수 있는 레이어
- s가 indentity mapping일 때, s를 통과하면 regular Feed forward neural network로 변함
  - 여기서 각 token이 어떤 상호작용 없이 독립적으로 처리됨 (<->  transformer 와 달리)
 - 본 연구에서 주요 focus 중 하나는 바로 충분히 복잡한 spatial interaction을 포착할 수 있는 좋은 s를 디자인 하는 것! 
 - 전반적인 블록 레이아웃은 inverted bottlenecks에서 영감을 얻어 spatial depthwise convolution으로 구상
 - **Transformer와 달리, 본 연구에서 제시한 모델은 position embedding을 요구하지 않음**
    - 왜냐하면, 이런 정보가 s( )에서 포착될 것이기 때문
 - 본 연구에서의 모델은 BERT / ViT와 동일한 input, output format을 사용함
    - 예시) language task를 할 때, multiple text segments를 concat하고 prediction이 마지막 레이어 representation을 통해 도출됨 (이러한 구조가 동일하게 본 연구에서의 모델에도 쓰인다는 의미)

### 3.2 Spatial Gating Unit
- ![](https://latex.codecogs.com/svg.image?s(Z)&space;=&space;Z&space;\odot&space;f_{W,b}(Z))
  - elementwise multiplication
   - 이때, f(Z)는 단순한 linear projection으로 다음과 같음 
     -  ![](https://latex.codecogs.com/svg.image?f_{W,b}(Z)&space;=&space;WZ&space;&plus;&space;b)
     - W는 n * n 행렬 (n: 시퀀스 길이)
- 학습의 안정성을 위해, W를 0으로, b를 1로 initialize 해주는 것 중요함
  - ![](https://latex.codecogs.com/svg.image?f_{W,b}(Z)&space;\approx&space;1) , ![](https://latex.codecogs.com/svg.image?s(Z)&space;\approx&space;Z) 
- 이런 초기화가 모델의 각 블럭이 학습 초기 단계에서 regular FFN처럼 행동하도록 함
- 또, s(Z)를 연산할 때, Z를 나누어 연산하는 것이 더 효율적이라고 함
	- ![](https://latex.codecogs.com/svg.image?s(Z)&space;=&space;Z_1&space;\odot&space;f_{W,b}(Z_2))
- gMLP Overview ---> 그림 넣기

## 4. Experiment
- 본 논문에서는 크게 2가지 분야에서 gMLP 검증
	- Image classification
	- Language Modeling

### 4.1 Image classification
- gMLP 모델과 attentive model 들 사이 비교
	- Vision Transformer (ViT)
	- DeiT (ViT with improved regularization)
	- several other representative convolutional networks
- Architecture specifications of gMLP models for vision -> 사진 넣기
- **Results**
	![](https://blog.kakaocdn.net/dn/bicySA/btq6k7MqyPc/ZMSXD6336qnrTgtUfPsoy1/img.png)
	- 위 결과를 통해 gMLPs가 DeiT와 견줄만 하다는 것을 보임
		- 즉, self-attention이 없는 모델도 Transformer만큼 efficient 할 수 있다는 것을 알 수 있음
		
### 4.2   Masked Language Modeling with BERT
- masked language modeling (MLM) task 실험 진행
- input/output 형식은 BERT를 따름
- Transformer와 다른 점은 positional embedding을 사용하지 않는다는 것
- ablation & case study에서 모델은 batch size 2048, max length 128 for 125K steps로 학습되었음
- main experiments는 batch size 256, max length 512 로 학습됨

#### 4.2.1 Ablation:  The Importance of Gating in gMLP for BERT’s Pretraining
- gMLP 여러개의 버전과 baselines 비교
- 사용한 baselines
	- 기존 BERT 
	- 기존 BERT  +  relative position biases
	- (기존 BERT  +  relative position biases) - softmax에서 모든 content-dependent terms 제거 
		-  variant of Transformers without self-attention라고 볼 수 있음
	- Tranformer에서의 multi-head attention을 대체하는 MLP-Mixer 모델
	- Metric으로 언어모델의 성능을 판단할 수 있는 지표인 perplexity를 사용
		- 값이 낮을 수록 모델이 잘 학습되었다는 것을 의미
- **Results**![](https://blog.kakaocdn.net/dn/Kv8U6/btq6kGIDnAE/fmMkEESonK1UBXsAeWFuMK/img.png)
	- gMLP with SGU가 Transformer 만큼의 perplexity를 얻음

#### 4.2.2  Case Study: The Behavior of gMLP as Model Size Increases
- 모델의 크기가 커질 때마다 성능이 어떻게 변하는지 확인하였음
- **Results**![](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FxbTHF%2Fbtq6hUHp0Yx%2FAGS2oh5zNLovKErlex6Y9K%2Fimg.png)
  - gMLP 모델이 deep 해질 수록 Transformer와 성능이 비슷해지며, 심지어는 outperform 하는 경우도 존재함
 - 이미지 추가
	 -  SST-2, 즉 sentiment analysis task 측면에서도 같은 개수의 파라미터일 때, gMLP 모델이 Transformer 보다 뛰어나다는 것을 보임
	 - 하지만, natural language inference task 측면에서는 Transformer가 더 좋은 성능을 보였음
		 - 왜냐하면, 이 task에서는 모델이 2 문장을 다루어야 하는데, 여기서 self-attention 이 유용한 역할을 했기 때문 -> self-attention을 가지고 있는 Transformer 가 훨씬 유리



#### 4.2.3 Ablation: The Usefulness of Tiny Attention in BERT’s Finetuning

- 위의 MNLI-m 결과에서 gMLP가 Transformer 모델보다 성능이 낮았던 것을 개선하기 위해 tiny self-attention block을 모델에 추가하였음
	- 이미 gMLP가 spatial 관계를 파악할 수 있기 때문에 self-attention 모듈이 heavy 할 필요는 없다고 생각
	- 사이즈가 64인 single head -> "aMLP" 라고 함
- 사진 추가 ...
	- aMLP가 모두 Transformer 보다 좋은 성능을 보임



#### 4.2.4 Main Results for MLM in the BERT Setup

- full BERT setup에서 실험을 진행함
- 공정한 비교를 위해 gMLP의 depth와 width를 조정해줌
- 이미지 추가 (모델 specification)
- 실험 결과 이미지 추가
	- gMLP가 Transformer와 견줄만 하다는 것을 알 수 있음
	- 앞서 계속 언급했지만, gMLP의 크기가 커질 수록 Transformer와의 성능 갭이 줄어드는 것을 확인 할 수 있음
	- 또, tiny single-head self-attention을 사용하기만 해도 Transformer보다 좋은 성능이 나올 수 있다는 것을 알 수 있음



## Conclusion

- 본 연구는 Transformer의 self-attention이 중요한 역할을 하는가에서 시작하여 연구를 진행함
- 따라서 본 연구에서는 multi-head self-attention layer를 대신할 수 있는 간단한 방법을 제시
	- gMLPs, a simple variant of MLPs with gating
- gMLP는 특정 분야에서 Transformer보다 더 좋은 성능을 낼 수 있음을 보임
- 또, Transformer의 multi-head self-attention이 문장과의 관계를 고려하는 task에서 유용함을 알 수 있었음
- 모델 사이즈를 늘리는 것보다 small single-head self attention 을 추가하는 것이 gMLP가 더 좋은 성능을 가지게 한다는 것을 보임



