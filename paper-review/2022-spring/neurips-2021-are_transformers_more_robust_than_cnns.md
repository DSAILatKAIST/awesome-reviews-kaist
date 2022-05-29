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
- 논문과 PaperWithCode에 있는 설명이 조금 다른데 PWC를 기준으로 적었다
    - ImageNet-A : ResNet model이 강한 확신으로 틀린 이미지셋. 기계학습 모델이 어려워하는 즉 학습 분포랑은 좀 다른 이미지들의 모임이다. 실제 이미지를 보면 왜 그런 틀린 답을 냈는지 알 것도 같다
    - ImageNet-A 예시
        
        ![Untitled](%5BPresentation%5DAre%20Transformers%20More%20Robust%20Than%20CN%20c8d0616ad5f1492b8c77a31b94d5b362/Untitled%203.png)
        
    - ImageNet-C : 이미지당 다양한 Augmentation이 적용된 이미지셋
    - ImageNet-C 예시
        
        ![Untitled](%5BPresentation%5DAre%20Transformers%20More%20Robust%20Than%20CN%20c8d0616ad5f1492b8c77a31b94d5b362/Untitled%204.png)
        
    - Stylized ImageNet :  이미지당 다양한 texture를 입한 데이터셋
    - Stylized ImageNet 예시
        
        ![Untitled](%5BPresentation%5DAre%20Transformers%20More%20Robust%20Than%20CN%20c8d0616ad5f1492b8c77a31b94d5b362/Untitled%205.png)
        






## **4. Experiment**  

In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

### **Experiment setup**  
* Dataset  
* baseline  
* Evaluation Metric  

### **Result**  
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

