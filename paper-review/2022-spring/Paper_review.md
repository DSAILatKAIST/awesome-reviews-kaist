---
description : Jason Wei / Finetuning Language Models are Zero-shot Learners / ICLR2022(oral)  
---

# **Finetuning Language Models are Zero-shot Learners** 

## **1. Problem Definition**  

해당 논문이 풀고자 하는 문제는, Language Model (LM)의 zero-shot inference task 이다. 즉, training 과정에서 보지 못했던 task와 data를 inference 과정에서, 추가적인 학습 없이 모델의 성능을 테스트 한다.

Zero-shot learning 은 본래 인간이 언어적인 사고를 할 때, 추가적인 학습 없이도, 다양한 언어처리를 할 수 있다는 것에 기인하여, LM 또한, 언어에서 인간의 지능을 모방할 수 있는 가를 평가하는 중요한 요소이다. 해당 논문은 새로운 Finetuning 방법론을 제안하여 기존 Large Scale LM 들의 Zero-shot 성능을 크게 개선시키는 데 집중한다. 


## **2. Motivation**  

"Language Models are Few-shot Learners" (이하 GPT-3) 는 그 논문 제목 처럼, LM이 Task에 대한 적은 추가적 학습 (Few shot) 혹은 example 나열 만으로도 좋은 성능이 나올 수 있음을 시사하면서, LM의 일반 인공지능 (Artificial General Integlligent) 로의 방향성을 제시하였다. GPT-3는 기존 BERT 기반 LM 들의 연구 방향이 큰 Dataset 에서 Self-supervised learning을 통해 학습 한 후, 각 task에 decoder를 finetuning 한 흐름을 지적하였다. GPT-3는 BERT 기반 LM 들과는 다르게, Decoder finetuning 없이, large scale pre-train 및 few shot learning 을 통해 좀 더 일반적인 LM을 탄생시켰다. 

하지만, GPT-3 조차 Zero-shot 성능은 좋지 못했고, 이 점이 큰 한계로 지적되었다. GPT-3가 LM을 "Few-shot learner" 로 이끌었다면, 이 논문은 그것을 넘어 "Zero-shot learner"로 이끌 수 있는 Novel 한 방법을 제시한다. 



## **3. Method**  

### Instruction Tuning

해당 논문은 Instruction Tuning 이라는 굉장히 간단한 아이디어로 Large Scale LM의 Zero-shot performance를 크게 증가시킨다. 

<img width="140" src=".gitbook/2022-spring-assets/symbol.png">  


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

Please summarize the paper.  
It is free to write all you want. e.g, your opinion, take home message(오늘의 교훈), key idea, and etc.

---  
## **Author Information**  

* Author name  
    * Affiliation  
    * Research Topic

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation  
* Reference  

