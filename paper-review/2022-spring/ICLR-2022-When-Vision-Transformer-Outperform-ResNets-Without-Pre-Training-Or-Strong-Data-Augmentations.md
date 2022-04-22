# When Vision Transformer Outperform ResNets Without Pre-Training Or Strong Data Augmentations


# Abstract

- Convolution-free한 ViT와 MLPs만 사용한 모델의 경우 inductive bias의 부재로 인해 매우 많은 데이터를 학습시키거나 강력한 augmentation 전략을 사용해 이를 극복하고자 하였다. 그럼에도 SOTA성능은 아니었다
- 학습된 landscape는 상당히 local minima에 sharpe한 모습을 보였지만 최근 제시된 sharpness-aware optimizer(SAM)을 활용한 결과 ViT와 MLP Mixer는 지도, 비지도, 적대적, 전이 등 다양한 학습 전략에서 상당한 성능 향상이 있음을 확인했다
- Scratch부터 학습된다면 비슷한 size의 ViT와 ResNet에서 ViT가 ResNet의 성능을 뛰어넘을 수 있음을 확인했다

# 1. Introduction
![fig1](https://user-images.githubusercontent.com/99710438/164282561-92a1143f-2469-4b8a-aad7-435c7b6bd50f.PNG)
![fig1](awesome-reviews-kaist/.gitbook/2022-spring-assets/ICLR-2022-When-Vision-Transformer-Outperform-ResNets-Without-Pre-Training-Or-Strong-Data-Augmentations/Untitled%1.png)

- Transformer가 NLP분야에 소개된 이후 vision에서의 활용이 있었고 hand-wired feature와 inductive bias가 없이 일반화 가능한 모델을 만들기 위해 상당한 양의 데이터를 투입시키는 방법이 활용되었다. 예를들어 pre-train ViT는 google의 private dataset으로 3억장의 labeled image를 학습하였다
- ViT와 Mixer는 ResNet에 비해 기하적으로 아주 sharpe한 loss landsacpe를 가지고 있음을 알 수 있다. 이것은 학습-일반화 성능의 괴리가 생기는 원인으로 지목된다
- SGD,Adam과같은 first-order optimizer는 training error를 낮추는데에는 좋은 알고리즘이다. 하지만 이것은 더 고차원적인 목표인 주변의 loss도 낮게 만듦에는 신경쓰지 않는다. 따라서 최근의 연구인 SAM에서 이런 문제의 해결을 찾게 되었다. SAM optimizer는 single poing에서의 loss 최저보다 single poing 주변 모두가 loss가 낮아지도록 설계되었다. 이렇게 향상된 일반화 성능은 강력한 augmentation과 pre-training과정없이 ViT, MLPs가 ResNet을 뛰어넘을 수 있게 하였다
- SAM을 사용한 후 model의 (특히 첫 몇 개의 레이어에서) Hessian 고윳값이 작아지는 것을 확인했다(=볼록한 정도가 감소했다=sharpeness하지 않아졌다). Weight norm은 이것을 커지게 만들었고 이는 일반화에 자주 쓰이는 weight decay가 regularization에 크게 도움되지 않았을지 모른다는 가정을 가능하게 한다. SAM과 강력한 augmentation과의 비슷한 특성들을 살펴볼 것이다

### Hessian matrix

[헤세 행렬(Hessian Matrix)의 기하학적 의미](https://angeloyeo.github.io/2020/06/17/Hessian.html)

# 2. Background And Related Work

- (똑같은 이야기) ViT는 매우 큰 데이터셋으로 pre-training. CNN과같은 locality나 translation equivariance를 가지지 않는다. NLP의 transformer처럼 대용량의 dataset과 strong data augmentation이 필요함
- MLP-Mixer는 ViT처럼 patch 단위로 겹치지 않게 이미지를 잘라 입력으로 받음
- ViT S : small , B : baseline, 숫자는 patch의 크기를 뜻함

# 3. ViTs And MLP-Mixers Converge To Sharp Local Minima

- ViT와 MLP mixer는 아주 극소한 부분으로 수렴하는 경향이 있다. 일반화 성능에 나쁜 현상이다

### ViTs and MLP-Mixers converge to extremely sharp local minima

![Untitled](When%20Vision%20Transformer%20Outperform%20ResNets%20Without%206b4ec770359148e1b8179c3e62dcdea5/Untitled%201.png)

- (a) vs (b),(c) : ResNet에 비해서 conv-free한 방법은 loss의 landscape가 sharper함을 확인할 수 있다. Table1을 보면 Hessian 행렬에서 얻은 고윳값 중 제일 큰 $\lambda_{max}$을 확인할 수 있다. $\lambda_{max}$는 landscape의 최대(최악) 곡률이다. ResNet과 비교하면 ViT와 특히, Mixer가 상당히 높다는 것을 알 수 있다

### Small training errors

![Untitled](When%20Vision%20Transformer%20Outperform%20ResNets%20Without%206b4ec770359148e1b8179c3e62dcdea5/Untitled%202.png)

- ViT와 MLP가 극소적인 부분에 수렴한다는 것은 training에 dynamics가 존재한다는 의미이다. Figure2의 좌측과 중앙을 보면 MLP는 ViT보다 낮은 training loss를 가지지만 test에서의 성능은 더 나쁘다. 이는 부분적인 극소에 수렴했다고 분석할 수 있다

### ViTs and MLP-Mixers have worse trainability

- 또한 ViT와 MLP가 poor한 trianability에 노출되어있다는 것을 알 수 있었다. 여기서 trainability란 경사하강법에 의해 네트워크가 최적화되는 현상의 효율성을 말한다. Xiao의 연구에 의하면 Jacobian 행렬(Jacobian 행렬은 1차 미분 행렬) $J$에 대하여 neural tangent kernel(NTK) $\Theta=JJ^\top$로 정의한다. $\Theta$의 고윳값 $\lambda_1\geq\cdots\geq \lambda_m$ 에 대하여 $\kappa=\lambda_1/\lambda_m$으로 정의한다. 만약 $\kappa$가 지속적으로 변하면 학습이 불안정하다. Table1의 모델에 따른 $\kappa$를 비교해 볼 것

[자코비안(Jacobian) 행렬의 기하학적 의미](https://angeloyeo.github.io/2020/07/24/Jacobian.html)

# 4. A Principled Optimizer For Convolution-Free Architectures

- 가장 많이 사용되는 first-order optimizer들은 $L_{train}$만을 낮추게 설계되어 있어서 더 넓은 관점에서의 일반화를 위한 곡률, 상관같은 properties는 고려하지 않는다. 하지만 DNNs는 non-convex하기 때문에 training error는 0에 수렴하나 test에서는 그러지 못한다. 심지어 ViT와 MLP는 inductive bias의 부재로 sharp loss landscape를 가지는 현상이 두드러진다

## 4.1 SAM : Overview

$$
(1)\space\min_w \max_{||\epsilon||_2\leq\rho}L_{train}(w+\epsilon)\\
(2)\space \hat{\epsilon}(w)=\argmax_{||\epsilon||_2\leq\rho}L_{train}(w)+\epsilon^T\bigtriangledown _wL_{train}(w)\\=\rho\bigtriangledown _w L_{train}(w)/||\bigtriangledown _w L_{train}(w)||_2
$$

- 직관적으로 SAM은 주변의 이웃의 $L_{train}$까지 낮아지게 하는 weight $w$를 찾는 알고리즘이다. 여기서 $\rho$는 얼마나 멀리까지를 고려하는지 고려하는 hyper-parameter이다
- (1)수식은 $L_{train}$ 을 최소화하는 weight $w$를 찾음과 동시에 $L_{train}(w+\epsilon)$을 최대화하는 $\epsilon$을 찾는 것이 목표이다. 이따 $\epsilon$은 L-2 norm이 $\rho$보다 작아야한다. $\min_w$는 sharp하게 수렴하려하고 $\max_{||\epsilon||_2\leq \rho}$은 $w$주변의 Loss를 키워 sharpness를 완화시킨다
- (2)수식을 보면 어떻게 가장 적절한 $\epsilon$을 찾는지 알 수 있다. $\rho\bigtriangledown_w L_{train}(w)/||\bigtriangledown_w L_{train}(w)||_2$을 보면 $\bigtriangledown_w L_{train}(w)/||\bigtriangledown _w L_{train}(w)||_2$ 는 L-2 표준화된 $w$에 대한 Loss의 gradient다. 즉 Loss의 방향은 그대로 가지고 여기에 $\rho$를 곱해서 그 방향의 주변만큼 스텝을 옮긴 지점을 $\hat \epsilon(w)$로 인정하는 것이다

## 4.2 Sharpness-Aware Optimization Improves ViTs And MLP-Mizers

- ViT의 hyper-parameter의 수정없이 scratch부터 ImageNet을 학습시킴. Inception style processing을 입력단에 추가함. MLP에는 입력단에 강력한 augmentation이 포함되어 있는데 공정한 비교를 위해서 Inception style processing로 대체함. learning에 필요한 hyper-parameter는 gready search를 통해서 찾았다
    
    ### Smoother regions around the local minima
    
    - Figure 1의 (d),(e)처럼 smoother regions에 수렴한 것을 확인할 수 있다. 또한 $\lambda_{max}$역시 하락했다
    
    ### Higher accuracy
    
    ![Untitled](When%20Vision%20Transformer%20Outperform%20ResNets%20Without%206b4ec770359148e1b8179c3e62dcdea5/Untitled%203.png)
    
    ### Better robustness
    
    - Table2의 ImageNet-R : Train set과 Test set의 class는 같지만 distribution이 다름
    - ImageNet-C : Test set에 강력한 noise가 있음

## 4.3 ViTs Outperform ResNets Without Pre-Training Or Strong Augmentations

```bash
When trained from scratch on ImageNet with SAM, ViTs outperform ResNets of similar and 
greater sizes (also comparable throughput at inference) regarding both clean accuracy.
```

- 비슷한 #params를 놓고 비교하면 ViT가 ResNet의 성능을 상회했다

## 4.4 Intrinsic Changes After SAM

### Smoother loss landscapes for every network component

![Untitled](When%20Vision%20Transformer%20Outperform%20ResNets%20Without%206b4ec770359148e1b8179c3e62dcdea5/Untitled%204.png)

- SAM 이후 곡률의 정도인 $\lambda_{max}$가 작아진 것을 모델의 레이어별로 수치화하였다

$$
(3)\space H_k=(a_{k-1}a_{k-1}^T)\otimes\mathcal H_k, \space \mathcal H_k =B_kW^T_{k+1}\mathcal H_{k+1}W_{k+1}B_k +D_k\\
(4) B_k=\mathrm{diag}(f'_k(h_k)),\space D_k=\mathrm{diag}(f''_k(h_k)\frac{\partial L}{\partial a_k})
$$

> $f(\cdot)$ : activation , GELU
$W_k$ : k번째 layer의 params
$h_k$ : $W_k a_{k-1}$, activation을 거치기 전의 k번째 layer의 output
$a_k$ : $f_k(h_k)$
$\otimes$ : Kronecker product
$\mathcal H_k$ : layer k의 activation을 거치기 전의 Hessian 행렬
$H_k$ : $W_k$의 Hessian 행렬
> 
- Table 3을 보면 앞단의 레이어가 더 높은 $\lambda_{max}$를 가짐을 알 수 있다. 이는 (3) 수식에서 분석가능한데, Hessian norm은 역전파에 의하여 뒤에서 앞으로 누적이 되기 때문이다

### Greater weight norms

- Table3를 보면 SAM을 사용한 후 $||\cdot||$ 으로 나타난 norm결과를 보면 오히려 상승한 것을 알 수 있다. 이는 weight decay가 ViT와 MLP를 규제하는데는 그다지 효율적이지 않았음을 뜻한다

### Sparser active neurons in MLP-Mixers

- 수식 (3)과 (4)를 보면 $B_k$는 $f_k$가 GELU이기 때문에 0보다 큰 값으로 결정된다. GELU의 1차도 함수는 입력 값이 0보다 작다면 급격히 작아진다(사실상0). 따라서 active된 GELU의 수가 Hessian norm으로 직결 되는 것이다
- Figure2의 맨 오른쪽을 보면 SAM을 사용하면 활성화된 뉴런의 수가 더 적어졌음을 알 수 있다. 이것은 image patch가 어떤 반복성을 가지고 있었을지 모른다는 가정을 가능하게 한다(=반복적인 정보가 있으니 sparse해도 감지가 가능했다?)

### ViT's active neurons are highly sparse

- Transformer의 첫 레이어에서 active neuron(value가 0보다 크다)의 수가 10%정도였고 이는 50%이상이 active한 ResNet과 비교하면 매우 낮은 수치이다
- 따라서 pruning의 여지가 있음을 확인했고 왜 multi modality data가 transformer 구조에 잘 적합되는지도 설명할 수 있다

### More perceptive attention maps in ViTs

![Untitled](When%20Vision%20Transformer%20Outperform%20ResNets%20Without%206b4ec770359148e1b8179c3e62dcdea5/Untitled%205.png)

## 4.5 SAM vs. Strong Augmentations

![Untitled](When%20Vision%20Transformer%20Outperform%20ResNets%20Without%206b4ec770359148e1b8179c3e62dcdea5/Untitled%206.png)

- 비교를 위해 strong augmentation은 mixup과 RandAugment를 사용하였다
    
    ### Generalization
    
    - SAM의 경우 ViT, Mixer에서 augmentation보다 좋은 성능을 내었고 특히 dataset이 작은 경우 이런 현상이 두드러졌다(i1k). 흥미로운 관찰로, Figure2의 중간을 보면 Aug와 SAM이 training error를 오히려 높혔다. 이는 regularization의 효과로 보인다. 둘의 차이가 있다면 Aug쪽이 training loss가 요동치는 모습을 볼 수 있다
    
    ### Sharpness at convergence
    
    ![Untitled](When%20Vision%20Transformer%20Outperform%20ResNets%20Without%206b4ec770359148e1b8179c3e62dcdea5/Untitled%207.png)
    
    - Augmentation이 SAM과같이 기하적으로 loss를 smoothe 할 수 있을까?
    - 우선 시각화하여 확인하였다. AUG가 $\lambda_{max}$를 크게했지만 average flatness를 알 수 있는 Gaussian 섭동 loss $L^{\mathcal N}_{train}=\mathbb E_{\epsilon\sim\mathcal N}[L_{train}(w+\epsilon)]$ 를 ViT-B에서보다 작게 만들었다($w$주변에서의 로스가 작았다는 것은 최적으로 인정된 $w$주변에서의 로스가 작았다는 뜻이고 이는 곧 $w$주변으로 flatness하다는 뜻이다)
    - 이것은 SAM과 augmentation모두가 loss landscape를 flat하게 만들었다는 뜻이다
    
    <aside>
    ❓ 가우시안 섭동이 작은거는 알겠습니다. 하지만 $\lambda_{max}$가 1659.3으로 비교적 큰 값이 나왔는데?
    - 이것은 SAM과 Augmentation의 flaten 전략이 다르기 때문이다. SAM은 minmax를 사용해 전체적인 Loss에서의 landscape를 강제하지만 Augmentation같은 경우 최악의 case는 무시해버리고 augmentation으로부터 알 수 있는 inductive bias 방향으로의 landscape flaten을 하기 때문이다
    
    </aside>
    

# 5. Ablation Studies

## 5.1 When Scaling The Training Set Size

![Untitled](When%20Vision%20Transformer%20Outperform%20ResNets%20Without%206b4ec770359148e1b8179c3e62dcdea5/Untitled%208.png)

## 5.2 When SAM Meets Contrastive Learning

- AUG, 많은 데이터와 함께 일반화에 사용되는 contrastive learning도 유효한지 적용해본다. 350eph 사전학습 후 하위 task에 fine-tuning하였다. ViT-S/16 : 77%→78.1%, ViT-B/16 : 77.4%→80.0%

# 6. Conclusion

- 이 연구는 지금까지 방대한 양의 데이터와 augmentation에 의존한 ViT, MLP의 기하적 loss 특징을 분석했다. 여기서 local minima 문제를 찾고 SAM을 도입해 flatten loss landscape로 일반화 성능의 향상을 끌어냈다. ViT는 어떠한 pre-training이 없다면 ResNet의 성능을 앞지를 수 있었다. 또한 SAM을 도입한 ViT의 attention map이 더욱 해석 가능하게 변화했다
