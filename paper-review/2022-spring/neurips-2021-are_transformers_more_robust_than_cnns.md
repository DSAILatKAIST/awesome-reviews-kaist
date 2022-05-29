---
description : Y Bai et al., / Are Transformers More Robust Than CNNs? / Neurips-2021
---

# **Are Transformers More Robust Than CNNs?** 


## **1. Problem Definition**  

- Vision Transformer(ViT) Network는 CNN보다 강력하고 robust하다고 알려져있다.
- 하지만 이 연구에서는 몇가지 실험을 통해서 기존의 이런 믿음에 의문을 제기하고 공정하게 설계된 실험조건에서 강건성을 다시 조사한다.
- 결론적으로 adversarial attack에 CNN도 충분히 강건할 수 있음을 확인했다
- 강건성에대한 실험 도중에, 방대한 양의 데이터를 사용한 pre-training이 transformer가 CNN의 성능을 넘는데 꼭 필요한 것은 아님도 부가적으로 확인했다.


## **2. Motivation**  

- Pure-attention based model인 transformer가 inductive bias없이 CNN의 성능을 뛰어넘었고 Detection, instance segmentation, sementic segmentation에서도 연구되고있다
- 또한 최근 연구들에서 Transformer는 OOD와 적대적 공격에 CNN보다 강건함이 밝혀졌다
    - *하지만*, 저자는 이런 결과가 unfair한 환경에서 도출되었다고 주장한다
    - #params가 Transformer쪽이 많았고 training dataset, epochs and augmentation 전략 등이 동일하게 맞춰지지 않았다(뒤에 실험에서 확인할 수 있듯이 ViT에게 유리한 조건이 다수 있다)
- 이 연구에서 공정한 비교를 통해 적대적 공격과 OOD에 대한 강건성을 확인할 것이다
    - CNN이 Transformer의 training recipes를 따른다면 perturbation과 patch에 기반한 attack에 더 강건함을 발견했다
    - 여전히 Transformer가 OOD에 강건함을 발견했고 이는 pre-training이 없어도 가능했다. Ablation study에서 self-attention이 이런 현상의 이유임을 발견했다

<aside>

💡  이 연구가 다른 Architecture끼리의 강건성을 비교하는 표준이 되길 바란다고 저자는 밝히고 있습니다

</aside>



## **3. Method**  
- 이 챕터에서는 다음과 같은 내용을 다룬다. 모두 실험에서 자주 등장할 내용이므로 주의깊게 숙지하길 바랍니다.
1. CNN과 ViT의 학습조건 비교
2. 다양한 Attack과 OOD Dataset

## 3.1 Training CNNs and Transformer

- 학습 후 CNN와 ViT의 Top-1 Acc는 76.8, 76.9로 매우 비슷한 성능을 냄

### CNN

- ResNet-50이 ViT와 비슷한 #params를 가지므로 채택
- ImageNet에 학습
- 기타 학습 디테일(SGD-momentum, 100eph, L2규제)

### ViT

- 외부 데이터없이 좋은 성능을 낸 DeiT의 recipe를 따라서 DeiT-S(#params가 ResNet50과 비슷)를 default ViT로 채택함
- AdamW, 3개의 Aug(Rand, Cut, MixUp)
- ResNet과 학습 환경을 맞추기위해 Erasing, Stochastic Depth, Repeated Aug를 사용하지 않음. DeiT는 300eph학습되지만 같은 이유로 100eph만 학습

## 3.2 Robustness Evaluations

### 3.2.1 Adversarial Attack
#### PGD
- PGD(Projected Gradient Descent) : 사람은 확인하기 어렵지만 기계를 속일 수 있는 섭동

![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F42d2c8ef-1b52-4718-a081-9f6d2426de53%2FUntitled.png?table=block&id=c8d0616a-d5f1-492b-8c77-a31b94d5b362&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=2000&userId=&cache=v2)

#### TPA
- TPA : texture가 있는 patch를 붙여 네트워크를 속이는 attack


![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fc37dcc0d-a43c-4f71-a9c6-cf7f25bf73e8%2FUntitled.png?table=block&id=0da98098-a42c-45d4-98ee-f9187cb9e2cd&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=1030&userId=&cache=v2)
        
![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F6764d1d3-4b81-4d2b-9bb1-87e945c2d3c4%2FUntitled.png?table=block&id=2687d828-3a4f-4982-804a-c8119aa82f0f&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=1610&userId=&cache=v2)
        
### 3.2.2 OOD 
- 논문과 PaperWithCode(PWC)에 있는 설명이 조금 다른데 PWC를 기준으로 적었다
    
    - *mageNet-A* : ResNet model이 강한 확신으로 틀린 이미지셋. 기계학습 모델이 어려워하는 즉 학습 분포랑은 좀 다른 이미지들의 모임이다. 실제 이미지를 보면 왜 그런 틀린 답을 냈는지 알 것도 같다
      ![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fc7c1a4b6-25a3-4a24-bb46-5ffb43f1f7f2%2FUntitled.png?table=block&id=4b49d8f7-468b-4c9c-a6c4-5b7c5056c74e&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=2000&userId=&cache=v2)
        
    - *ImageNet-C* : 이미지에 다양한 Augmentation이 적용된 이미지셋
        
      ![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F3802b66c-15d1-47f4-8702-7160fbb557c2%2FUntitled.png?table=block&id=c9fb7059-3a0e-4dd5-a9d2-6b413ed3c73b&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=2000&userId=&cache=v2)
        
    - *Stylized ImageNet* :  이미지당 다양한 texture를 입한 데이터셋
        
      ![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F6ebce612-8a69-4031-8c37-1a34b32c60b9%2FUntitled.png?table=block&id=ce5c92ce-3ea5-4672-b32a-1f55c8b11ec9&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=2000&userId=&cache=v2)
        






## **4. Experiment**  
- 실험은 크게 두 개의 파트로 구성되어있습니다.
1. 적대적 공격에 대한 강건성
2. OOD Sample에 대한 강건성

### **4.1 Adversarial Robustness**  



- 5000장의 ImageNet 검증데이터를 사용하였음
    
### 4.1.1 Robustness to Perturnation-Based Attacks

![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fba30da01-1ae5-4c23-8e2f-b7978ba7c328%2FUntitled.png?table=block&id=6b36f65d-49dc-43db-80c8-9066b77e8310&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=1720&userId=&cache=v2)

- AutoAttack의 섭동을 높이니 완전히 fooled
- 그러나 두 모델이 전혀 Adversarial training되지 않았음을 기억하자

    #### Adversarial Training

    ![](https://latex.codecogs.com/gif.latex?\theta)
    
    \argmin_{\theta}\mathbb E_{(x,y)\sim\mathbb D}[\max_{\epsilon\in\mathbb S}L(\theta,x+\epsilon,y)]
    

    | ![](https://latex.codecogs.com/gif.latex?\theta) | parameters | ⁍ | max ![](https://latex.codecogs.com/gif.latex?\epsilon) |
    | --- | --- | --- | --- |
    | ⁍ | expectation | ⁍ | perturbation |
    | ⁍ | data | ⁍ | dataset |
    - 섭동을 주어서 Loss를 최대화하는 sample $x+\epsilon$에서의 최적 parameter를 찾으라는 내용의 수식이다
    - 정확히는 PGD가 사용되었는데 반복적인 step을 통해서 최적 공격지점을 찾는 방법이라 이해하면 되겠다

    #### Adversarial Training on Transformers

    - CNN은 문제 없었으나 Transformer는 강한 Augmentation이 PGD와 함께 적용되니 collapse되어버리는 문제가 있었다
    - 따라서 Augmentation을 eph증가에 따라 점점 강도를 높여가며 학습한 결과 44%의 robustness를 얻었다

    #### Transformers with CNNs’ Training Recipes

    - CNN에서 사용된 학습조건(M-SGD, 강한 Augmentation 배제)을 Transformer에 사용했더니 학습이 안정되긴 했지만 clean data에 대한 성능과 PGD-100에 대한 방어율이 하락했다
    - 이러한 현상이 나타난 이유는 강한 Augmentation을 규제해 overfitting이 쉽게 일어났기 때문이고 이전 연구에서 밝혀졌듯이 Transformer 자체가 SGD와같은 optimizer에서 최적점을 잘 찾지 못하기 때문이다

    #### CNNs with Transformers’ Training Recipes

![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F2582dadc-09e2-4efd-a35f-122ab9f221a0%2FUntitled.png?table=block&id=8ff952c2-ab75-4383-900e-8222603c5c14&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=2000&userId=&cache=v2)

    - ResNet-50 + ReLU의 결과를 보면 ViT보다 덜 강건하다. 이런 실험결과에 끝나지 않고 저자들은 새로운 실험을 해볼 motivation을 얻었다고한다. Transformer의 recipes를 CNN에 적용해 비교해보는 것이다
    - Transformer가 쓰는 optimizer와 strong regularization는 별 효과가 없거나 학습에서 collapse를 일으켰다
    - non-smooth한 특성을 가진 ReLU를 transoformer가 쓰는 GELU로 대체했다. ReLU는 적대적 공격에 취약한 activation임이 알려져있다
    - **그 결과 ResNet-50 + GELU는 DeiT에 필적하는 적대적 공격에대한 성능을 내었으며 이는 기존 연구의 결론을 반박하는 것이다**


### 4.1.2 Robustness to Patch-Based Attacks

![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F2dfb3c60-828a-471a-b938-f698c9661cc8%2FUntitled.png?table=block&id=b2385299-43ed-440d-94ee-e2a9ef8e8a02&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=2000&userId=&cache=v2)

- default로 4개의 patch로 대상 이미지의 전체 면적에 10%안쪽이 되게 attack했다. 두 모델 모두 TPA에 대한 적대적 학습은 하지 않았다. 그 이유가 좀 헷갈리는데 적대적 학습시에 non-trivial 그러니까, 성능이 너무 좋아져서 비교가 어렵다는 취지로 해석했다
- Table 3의 결과를 보면 CNN은 Transformer의 강건성에 미치지 못하고 기존 연구들의 주장이 맞아보인다
- 하지만 저자들은 TPA의 특성에 주목하여 새로운 지적을 한다. TPA는 이미지위에 인위적인 patch가 붙는 형태이다. 이는 patch를 잘라 붙이거나 삭제하는 CutMix와 유사하며 CutMix는 ViT에만 적용되었기때문에 ViT에게 TPA가 당연히 유리한 task라는 것이다

![Untitled](https://erratic-tailor-f01.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Ffa8c3d16-f52a-4edc-af2b-262d2b981013%2FUntitled.png?table=block&id=12f54bf2-8f4f-4a25-8ce2-9f1546fe3dec&spaceId=ad2a71b5-1b0d-4734-bbc4-60a807442e5d&width=2000&userId=&cache=v2)

- 그에대한 증명으로 ViT에 적용되었던 3개의 strong augmentation을 적용해 ResNet-50을 학습시켜 TPA에대한 성능을 살폈더니 table 4와 같았다
    - 가설대로 CutMix의 유무가 성능을 크게 좌우했다
    - **RandAug+CutMix에서 DeiT의 TPA에대한 강건성보다 높은 성능을 보였고 이는 기존 연구들이 주장한 patch-based 공격에대한 transformer의 강건성이 CNN보다 좋다는 주장을 반박한다**






### **4.2 **  
Then, show the experiment results which demonstrate the proposed method.  
You can attach the tables or figures, but you don't have to cover all the results.  
  



## **5. Conclusion**  

- unfair한 조건에서 실행되던 실험을 적절한 조치를 통해 비교하니 Transformer는 적대적 공격에서 CNN보다 강건하지 않았다
- 또한 OOD에서의 Transformer성능은 self-attention과 관련이 있음을 확인했다
- 이 연구로 transformer에 대한 이해가 향상되고 transformer과 CNN사이 공정한 비교가 가능해지길 바란다

### 개인적 의견으로..
- ViT의 등장은 많은 이슈를 낳았습니다. 처음 CNN이후 Image분류를 위한 근원적인 새로운 방법론 제시였고 무엇보다 성능이 좋았습니다. 심지어 최근 연구들에서는 ViT가 CNN보다 강건하기까지 하다는 결과를 도출하면서 Vision의 영역은 이제 (엄청난 pretrain dataset을 가진 사업체가 학습한) ViT가 모두 가져갈 것이라는 예상을 하기도 했습니다. 따라서 학계의 이런 믿음 자체에 의문을 가지고 도전하는게 쉬운일이 아니었을 것이라고 생각합니다. 이런 연구를 내놓은 연구자들의 실력과 자신감에서 또 한번 겸손해야함을 느낍니다.


---  
## **Author Information**  

* 홍성래 SungRae Hong
    * Master's Course, KAIST Knowledge Service Engineering 
    * Interested In : SSL, Vision DL, Audio DL
    * Contact : sun.hong@kaist.ac.kr

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation  
* Reference  

