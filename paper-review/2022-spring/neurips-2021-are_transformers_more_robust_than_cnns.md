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
```
💡  이 연구가 다른 Architecture끼리의 강건성을 비교하는 표준이 되길 바란다고 저자는 밝히고 있습니다
```
</aside>



## **3. Method**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  

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

