## Description

* Xingning Dong et al. / Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation / CVPR 2022

---



##  1. Task Definition

우선 Scene Graph Generation 이 무엇인지 간략히 소개하겠습니다

Scene Graph Generation (SGG) 는, 이미지를 입력으로 받았을 때 이를 그래프로 바꾸어주는 Task 입니다.

![1](/.gitbook/2022-spring-assets/KanghoonYoon_2/Figure1_task.png)


그림1은 SGG의 모델의 과정을 나타냅니다. 구체적으로 설명하면, 사람과 말이 있는 이미지를 입력으로 받아 모델이 그래프를 생성합니다.

이 때 우리가 생성하고 싶은 그래프 G는 **_V, E, R, O_**   총 4가지 컴포넌트를 가지고 있습니다.  

**_V_** 는 노드, object detector의 proposal 로 구성되며  **_E_** 는 edge로, 연관이 있는 object 끼리 연결이 됩니다. 

또한 SGG 에서는 각 노드와 엣지의 label 의 class 가 무엇인지 구분하는 classification Task도 수행합니다.

**_R_** 은 Edge의 Relation class를 뜻하며, **_O_** 은 Object의 class를 뜻합니다.

따라서 최종 얻은 Graph 는  

<object, predicate, subject> (사람, 먹이주다, 말) 와 같은 triplet 의 조합으로 이루어지게 됩니다.


그러면 위의 식으로 부터    

**P(_V | I_ ) - object detector**

**P(_E | V, I_ ) - relation proposal netowrk**

**P(_R, O | V, E, I_ ) - Classification models for entity and predicate.**  

이 3가지를 모델링 하면 저희는 Scene Graph 를 생성할 수 있는 문제를 정의할 수 있게됩니다.

특히나 이 연구의 중점은, Unbiased SGG 로서, 특정 class 에 biased 되지 않고,

다양한 relation을 맞출 수 있도록 (class imbalanced training 과 유사) 하는 SGG 모델을 학습하는 것을 목표로 합니다.




## 2. Motivation

그렇다면 현재 존재하는 SGG 모델은 어떤 연구들이 있고, 또 그 연구들은 어떤 한계점이 있는지 알아보겠습니다.

### Scene Graph Generation

기존 SGG 방법들은 visual context를 반영한 Scene Graph 를 생성하기 위해 많은 노력을 기울였습니다. Scene 의 Object 간의 관계를 반영한 context를 학습하기 위하여 노력합니다.

1) 초기에는 scene 을 표현하는 feature에 대해 연구하였습니다. 그들은 Faster R-CNN object detector 로 추출한 feature를 어떻게 활용하여, 모델을 학습하는지에 더 나아가 language feature (class label의 word)
등을 이용하여, 보다 나은 scene graph context 를 학습하고자 하였습니다. 

2) 현재에는 모델 수준에서, 어떻게 context를 추출할지에 중점을 둔 연구가 많이 발달하였습니다.
그들은 기초적으로 LSTM 과 같은 sequential 모델, GNN 도메인에서 사용하는 meassage propagation scheme,  또는 self-attention network 등을 사용하여 그러한 context를 모델링하였습니다.
하지만, 이렇게 expressive power를 올려도, Scene Graph 데이터에 존재하는 label class의 bias 문제에는 아주 소소한 향상만을 가져왔습니다. 구체적으로 설명하면, 'on'과 같이 빈번히 등장하는
class 에 대해서는 잘 맞추지만 이는 scene graph 생성 관점에서는 의미가 적고,  'standing on'과 같은 tail class에 존재하는 relation에서는 모델이 잘 학습하지 못하지만, 이는 visual context를 잘 표현하는
 중요한 relation 입니다. 따라서, State-of-art SGG 연구들은 unbiased SGG를 만들고자 합니다. 그들은 대게, 1) data resampling 을 통해 모델의 bias를 줄여주거나, 2) re-weight loss 를 사용하여 모델을 학습하고,
또는 3) transfer learning framework 를 이용하여, 지식을 전달해주는 방식으로 bias 문제를 완화 합니다. 본 논문은 3) 과 연관된 연구라고 할 수 있겠습니다.


### 기존 연구의 LIMITATION

**첫째, language semantic 을 학습할 때 concat 과 같이 단순한 방식을 사용한다**
**둘째, 기존의 Unbiased Training 기법은 Tail에 overfit 되어 Head 퍼포먼스를 너무 희생한다**

### 본 연구의 IDEA

**첫째, Multi-Modal Learning 에서의 아키텍쳐를 가져와, language semantic을 보다 효율적으로 추출**
**둘째, Class Incremental Learning 에서의 Expert Training 기법을 차용하여, Head Tail 모두에서 우월한 성능을 가지는 SGG 모델 Training 기법 제안**

## 3. Method

아래 그림은, 제안된 모델의 전체적인 아키텍쳐 구조입니다.

![2](/.gitbook/2022-spring-assets/KanghoonYoon_2/Figure3_overall.png)

1) Proposal Network 를 통과하여, 이미지에서 Visual Feature(Bounding Box, convolutional Feature), Language Fature (Class Label word) 등을 추출합니다.

2) Visual Feature 와 Language Feature를 통해 각각 Object와 Relation의 Emedding을 Encoding 합니다. 이 때 Encoding 을 위해 사용되는 구조가 본 논문의 첫번째 contribution인
Stacked hybrid attention 입니다. 더 자세한건 뒤에서 다루도록 하겠습니다.

3) Encoder에서 얻어낸 Embeeding을 통해서, Object와 Relation 의 Decoder를 각각 학습합니다. 여기선 단순히 Classifier 를 학습한다고 이해하면 될 것 같습니다.
다만, 기존의 연구와의 차이점은 Relation decoding part 의 Group Collaborative Learning 입니다. 이 파트는 Relation의 Class Imabalance를 완화하기 위한 모듈로, 본 논문의 두번째
Contribution 입니다. 이 또한 뒤에서 자세히 다루도록 하겠습니다.



### Stacked Hybrid-Attention (SHA)

SHA는 앞서 언급한대로, 기존의 concatenation, summation 하여 visual/language feature를 사용하는 것이, 둘 사이의 inter-modal / intra-modal 관계를 잡아내는데 불충분하다는 데에서 출발합니다.
더 깊게 생각해보면, visual feature들 사이 (사람 이미지 <-> 말 이미지)에서 존재하는 관계가 있고, 단어들 끼리의 관계 ('human' word <-> 'horse' word) 의 관계가 multi-modal 의 형태로 존재하기 때문에,
단순 summation이 좋지 않다는 것 입니다. SHA는 기존의 multimodal learning 에서의 architecture를 사용하기 때문에 아주 쉽게 이해할 수 있습니다. 아래 그림이 SHA의 구조를 나타낸 그림입니다.

![3](/.gitbook/2022-spring-assets/KanghoonYoon_2/Figure4_SHA.png)

SA 모듈과 CA 모듈이 있는데 이 두 모듈 다 Multe-Head Attention 모듈을 사용한 것 입니다. 둘의 차이는 SA 모듈의 경우 intra-modal refinement르 목적으로, 같은 feature (image면 image) 끼리 넣은 모듈이고,
CA 모듈의 경우 둘다 같이 넣어서 semantic 을 추출한 cross attention 모듈 입니다. 이를 통해, 본 논문은 Feature를 더 잘 활용하여 context를 모델링할 수 있다고 이야기 합니다.


### Group Collaborative Learning (GCL)

Group Collaborative Learning 는 기존 relation 의 class imabalance를 해결하기 위해서 class incremental learning의 구조에 착안하여, SGG 연구에 적용한 사례로 이해할 수 있습니다. 어떻게 Bias를 해결 할 수 있는지
자세히 알아보도록 하겠습니다. 우선 아래 Group Collaborative Learning의 그림을 먼저 보겠습니다.

![4](/.gitbook/2022-spring-assets/KanghoonYoon_2/Figure5_GCL.png)

그림을 보면, 크게 Predicate Class Grouping ~ Collaborative Knowledge Distillation 순으로 여러 과정을 거치게 됩니다. 이 Class Incremental Learning의 핵심 아이디어를 요약하면,
''주어진 Data 가 Imbalanced 하니까, balanced 한 상황에서 여러 모델 (여러 Expert) A, B, ..., E 를 각각 나누어 학습하자. 그러면 A, B, C , D, E 각각의 모듈은 각각 전문적으로 잘 예측하는 class 가 생기고, 그 지식을 한 모델에게
공유 (전이, knowledge distillation) 하여, 모든 class 에 대해 잘 맞출 수 있는 하나의 모델을 만들자'' 입니다.

다소 복잡한 말로 들릴 수 있는데, 전문가 여러명을 나누어서 만들고, 전문가의 여러 지식을 한 학생에게 주입해주자는 것 입니다.

Step 1. Predicate Class Grouping. 전문가를 몇명 둘지를 정하는 것입니다. All Classes의 Distribution이 매우 Long-tail 이라 Imabalance 가 심하지만, 이를 sorting 하여 앞에서부터 잘라 Group을 만들면, 상대적으로 Balance하게 됩니다.
즉, 파란 relation 을 Group 1,  파란색 + 초록색 relation을 Group 2, ... 이런식으로 총 K 개의 Group을 만듭니다. 이 각각의 Group 내에서는 상대적으로 Balanced distribution을 갖게 됩니다.


Step 2. Balanced Sample Preparation 에서는, Group 내에서 적게 등장하는 Class를 좀더 볼수 있도록 해주는 것 입니다. 이 때에는 Under Sampling 만 적용하며, 적게 등장하는 Class는 조금만 Drop 하고, 많이 등장하는 Class를 많이 Drop 하여
그룹내에서의 Balance를 적게 등장하는 애들에게 더 초점을 두도록 합니다.

Step 3. Class Probability Prediction/Parallel Classifier Optimization. 기존, Classifier를 학습하는 것과 동일합니다. Cross Entropy를 사용하여, 총 K 개의 그룹에 대하여 각각 Classifier를 평행하게 학습합니다.

Step 4. Collaborative Knowledge Distillation. 이제 각각의 Classifier 는 전문적인 지식을 보유하고 있습니다. Group 1은 Head Class 의 지식을 많이 가지고 있을 것이고, Group K는 Tail class 의 지식을 많이 가지고 있을 것이며,
그 사이의 Classifier 는 Body Class의 지식을 가지고 있을 것입니다. 이를 줄을 세워 놓고, KL-divergence Loss를 학습하여, 지식을 전이해 줍니다.  지식의 전이 순서는 후에 실험 뒤에서 설명하겠습니다.  우선 Adjacency 방식을 설명하자면,
1번 Clasifier 는 2번 Classifier 에게 지식을 주고, 2번은 3번에게.. 체인 형식으로 지식을 전파해줍니다. 이렇게 되면 최종에 있는 K 번째 classifier 는 모든 지식을 순차적으로 전달 받아, Head~ Tail 모두를 잘 맞출 수 있는 Classifier를 얻게 됩니다.




## 4. Experiment & Result

실험에서는 기존의 실험 세팅에서, 제안된 모델이 얼마나 효과적인지를 검증하고, 각각의 모델 Component 가 효력이 있었는지 검증합니다.


### Metric

Unbiased SGG의 경우 평가 메트릭 mR@K 입니다. top-K triplet (<subject, relation, object>) 를 모델이  추정했을 때, 실제 GT triplet 에서 얼마나
맞추었는지를 평가합니다. 전체 개수의 평균을 재면 R@K, class 별 R@K 를 재고 Class로 나누어주면 meanR@K(mR@K) 가 됩니다.

### Task

Task는 다음과 같은 3가지 입니다.

*SGDET* -   Image -> Object detect / object classification / predicate classification  수행.

	전형적으로 이미지가 주어졌을 때, Graph를 생성하는 태스크 입니다. 세가지 중에 가장 어려운 태스크라고 볼 수 있으며,
	말 그대로 이미지가 그래프 자체로 변환하는 맵핑을 배우는 것 입니다. 따라서, Object Detector, Graph Edge Prediction, Object, relation classifier의
	모든 성능을 다 체크하는 것이라고 할 수 있겠습니다.

*SGCLS* - Ground Truth Box -> object classification / Predicate classification 수행

	이미지가 주어지고, 실제 Bounding Box가 주어졌을 때 Scene Graph를 만드는 태스크 입니다. Object Detector에 Dependent하지 않기 때문에
	위의 SGDET Task보다는 살짝 쉬워진 Task 입니다. 오직 Object, Predicate Classifer의 성능을 측정하는 기준 입니다.

	
*PREDCLS* - Ground Truth Box, object category -> Predciate Classification 수행

	마지막으로, 이미지가 주어지고, 실제 Bounding Box와 Object의 Classs까지 무엇인지 주어졌을 때 Scene Graph를 만드는 태스크 입니다. 
	Object Detector에 Dependent하지 않고, Object의 Class도 이미 알기 때문에 가장 쉬운 태스크입니다. 오직, Predicate Classifer의 성능을 측정하는 기준 입니다.



### Result

![5](/.gitbook/2022-spring-assets/KanghoonYoon_2/Figure6_Table1.png) 

위 표는 mR@K 를 K=20, 50, 100 에 따라 각각의 Task에 비교한 것을 볼 수 있습니다. 본 논문은 SHA와 GCL 을 제안하였는데요, SHA는 모델 인코더의 아키텍쳐 제안이라 본 논문에만 해당하지만,
GCL 의 경우 Training scheme 을 제안한 것이기 때문에 Model agnostic (기존의 다른 논문들에 대해서도 적용할 수 있음) 합니다. 본 논문에선 LSTM 기반으로 Context를 추정하여 Relation을 예측하는
Motif 와 TreeLSTM 구조를 통해 예측하는 VCTree 이 2가지에 GCL 을 적용한 것도 같이 실험을 진행한걸 볼 수 있습니다. 결과를 해석해보면, GCL 을 사용하면 기존의 모델의 mR@K 값도 크게 향상 가능하며,
특히나 제안된 Self-Attention 기반 모델에서, SHA 레이어와 GCL를 함께 사용한 것이 가장 우수했음을 확인할 수 있었습니다.


![6](/.gitbook/2022-spring-assets/KanghoonYoon_2/Figure7_Abl.png) 

위 표는 제안 된 논문의 Component을 잘게 잘라 ablation study를 하고, 각 모델의 컴포넌트의 효용성을 입증하는 단계로 볼 수 있겠습니다.  GCL을 빼버리면, 모델이 쉽게 biased 되는것을 확인 할 수 있고,
Knowledge Distillation 을 통해 모델을 하나로 합쳤을 때, 지식이 전이 되면서 성능이 더욱 향상되는 것을 보아, Transfer learning이 효과적이었음을 알 수 있습니다. 이해 비해 성능향상이 적지만 SHA의 SA와
CA 레이어도 각각 효력이 있었음을 보여주고 있습니다.




![7](/.gitbook/2022-spring-assets/KanghoonYoon_2/Figure8_gcl_example.png)
 
위 그림은 GCL 구조를 실제 여러 파라미터에 대해서 진행해보고, 어떻게 진행되는지 좀더 구체화된 예시를 보여주는 것 입니다. 파라미터를 조절하며, 각각의 group의 수를 바꾸어가며 모델을 학습해 볼 수 있습니다.

이에 따른 결과는 다음과 같습니다.

![8](/.gitbook/2022-spring-assets/KanghoonYoon_2/Figure9_param.png) 

Adjacency 방식보다 Top down 방식이 효과적인 것을 알 수 있고, 그룹을 어떻게 나누냐에 따라서도 성능의 차이가 꽤 나는 것을 볼 수 있습니다. 하지만 다른 그룹에서도, 기존 모델들과 비교했을 때에는  여전히 뛰어난 
성능을 보이기는 하네요.





## 5. Conclusion

본 LooC 논문에서는 Contrastive Learning 에서 Augmentation이 수반하는 Inductive Bias 문제를
제시하고 다루었습니다. 모델이 Augmentation을 학습해도, 해당 Property를 잃지 않고 인지할 수 있도록 프레임워크를 제안하였습니다

### Take home message 
이미지 뿐만 아니라 다른 도메인에서도, Augmentation은 특별하게 중요시 여겨지는 경우가 많습니다
그 상황에 맞게 Inductive Bias가 무엇인지 생각해보고, 개선해본다면 충분히 적용가능하거나 확장가능한 논문으로 보입니다


### Author

**윤강훈 \(Kanghoon Yoon\)** 

* Affiliation \(KAIST Industrial Engineering Department\)
* \(optional\) ph.D students in DSAIL


## Reference & Additional materials

1. Visual translation embedding network for visual relation detection
2. Representation learning for scene graph completion via jointly structural and visual embedding
3. Neural Motifs: Scene Graph Parsing with Global Context
4. Graph R-CNN for Scene Graph Generation.
5. GPS-net: Graph property sensing network for scene graph generation