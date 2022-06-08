---
description: >-
  Deli Chen, Yankai Lim, Wei Li, Peng Li, Jie Zhou, Xu Sum / Measuring and
  Relieving the Over-smoothing Problem for Graph Neural Networks from the
  Topological View / AAAI-2020
---

# MAD

## **1. Problem Definition**

> What is Over-smoothing?

최근 GNN의 발전으로 그래프 기반 여러 문제들을 푸는데 많은 발전이 있었다. 하지만, 이러한 성공에도 불구하고 간과되는 점이 있다. 바로 `Over-smoothing` 이슈다. 여기서 Over-smoothing 이라는 것은, 단어에서 우리가 유추할 수 있듯, 과하게 smoothing이 일어난다는 것이다. 먼저 GNN이 현재까지 성공을 거둔 이유는 그래프 자료구조의 특성인 자기와 비슷한 노드는 서로 앳지로 연결되어있는 이러한 상황을 이주 잘 utilize 한데 있었다. 여기서 자기와 비슷한 노드끼리 연결되어 있는 상황을 우리는 homophily 특성에 대응시킬 수 있고 동시에 이를 smoothness 한 상황이라고 볼 수 있다. 즉, 이러한 smoothness가 적절한 수의 layer를 바탕으로 했을 땐 장점으로 작용하지만 되려 layer를 과도하게 쌓게되면 서로서로 representation이 다 비슷해져서 다른 라벨을 가진 노드끼리 식별성이 떨어지는 그런 상황이 바로 **Over-smoothing** 한 상황이라고 해석할 수 있다. 즉, `과유불급`이 정확히 부합하는 상황이다.

먼저, 시각적으로 이 상황을 이해해 볼 필요가 있다. 아래 첫번째 그림은 Input Graph와 node1, node2, node3, node4에 대한 1-hop neighbor 를 표현한 그림이다. 이때는 node1(노란색), node4(보라색)를 center로 한 그래프가 많이 다른 모습을 볼 수 있다. 이에 자기자신과 비슷한 node 에서 더욱 많은 정보들을 받으며 embedding을 업데이트 할 수 있다. 다만,두번째 그림에서는 상황이 달라진다. 첫번째 그림과 달리 node1(노란색), node4(보라색)를 center로 한 그래프가 많이 비슷한, 즉 공유하고 있는 노드들이 많이 겹치는 모습을 볼 수 있다. 이런 식으로, layer 을 더 쌓을수록 더 많은 노드들을 서로 공유하게 되어 최종 업데이트 된 embedding이 서로서로 비슷하지게 된다.

![](../../.gitbook/2022-spring-assets/SukwonYun\_2/oversmoothing1.png)

![](../../.gitbook/2022-spring-assets/SukwonYun\_2/oversmoothing2.png)

정리하면, Over-smoothing 이 발생하는 상황을 우리는 현재 집중하고 있고 node들의 class를 구분하는 _node classification_ 등의 task 에서 이는 심각한 문제가 되고 따라서 이를 개선할 필요성이 대두된다. 하지만, 현재 많은 연구들이 GNN의 모델링, 효과적인 알고리듬 구축에 신경을 쓰고 있을 뿐, Over-smoothing의 문제를 타겟하고 이를 해결하기 위한 연구는 더딘 상태이다. 그래서 오늘의 Paper는 다음 3가지의 문제에 접근하고자 한다.

1. **Systematic and Quantitative study of over-smoothing issue and verification of key factor that cause over-smoothing**
   * 먼저, Over-smoothing이 발생하는 상황에 대한 이해 그리고 무엇보다 이를 **정량화**하기 위해 Smoothness 그리고 Oversmoothness의 `척도(measure)`를 제시하고 동시에 Over-smoothing을 일으키는 핵심원인을 살펴본다.
2. **Two Quantitative metrics: MAD, MADGap**
   * Smoothness 를 정량화하기 위한 척도로 `MAD(Mean Average Distance)` 그리고 Over-smoothness를 정량화하기 위한 척도로 `MADGap(Mean Average Distance Gap)`을 제시한다.
3. **Two methods: MADReg, AdaEdge**
   * 척도와 더불어, 실제로 Oversmoothing을 방지할 수 있는 2가지의 방법론 MADReg(MADGap-based Regularizer) 그리고 AdaEdge(Adaptive Edge Optimization)을 제시한다.

## **2. Motivation**

> How to measure Over-smoothing? MADGap!

앞서 우리는 GNN에서 Over-smoothing이 문제가 되는 상황을 살펴보았다. 궁극적으로, 문제를 발견하고 이를 해결해야할텐데 그렇다면 여기서 자연스럽게 Motivation이 발생하게 된다. 바로 문제가 되는 `Over-smoothing을 정량화할 수 있냐`는 물음에 대한 답이다. 정량화할 수 있게 된다면 이를 수치적으로 바라볼 수 있게 되고 더욱 구체화하면, **언제 어떻게 어느정도로** 심화되는지 한 층더 심화해서 살펴볼 수 있게 된다.

### (1) MAD

먼저 Smoothness, 그리고 Over-smoothness 두 상황을 구분해서 바라봐야할텐데 Smoothness를 먼저 정량화해보자. 이해를 돕기 위해 직접 수식과 함께 Matrix를 그려보면 아래와 같이 나타낼 수 있다.

![](../../.gitbook/2022-spring-assets/SukwonYun\_2/mad.png)

수식 순서대로 설명을 해보면 아래와 같다.

**(1)** 임베딩 matrix를 토대로 모든 노드 페어 간 cosine similaritiy를 구한 뒤 이를 1에서 빼주어 cosine distance matrix, ![](https://latex.codecogs.com/svg.image?D)를 정의해준다. 여기서 1에서 빼주는 이유는, 우리는 similarity가 아닌 distance로 접근하고 있기 때문이다. (즉, 두 노드 간 similairity가 높을수록 거리는 가깝기 때문)

**(2)** 이후, mask matrix, ![](https://latex.codecogs.com/svg.image?M%5E%7Btgt%7D)와 element-wise 하게 곱하여 내가 원하는 노드 페어만을 살펴볼 수 있게 target matrix, ![](https://latex.codecogs.com/svg.image?D%5E%7Btgt%7D)를 구한다. 이때 target matrix는 우리가 원하는 노드 페어 (i,j) 는 1의 값을 갖고 나머지 페어는 0의 값을 가진다.

**(3)** **행 단위**로 normalize 해주어 `행렬을 벡터로` 축소한다. 이 때, 값이 존재하는 페어의 값은 1로 취급해주어 normalize 해준다.

**(4)** **열 단위**로 normalize 해주어 `벡터를 스칼라로` 축소한다. 이 때, 값이 존재하는 페어의 값은 1로 취급해주어 normalize 해준다. 그리고 최종적으로, 사전에 살정한 target 노드 페어를 바탕으로 한 ![](https://latex.codecogs.com/svg.image?MAD%5E%7Btgt%7D) 을 얻는다.

이렇게 정의된 MAD 값이 정말 유의미한지 우리는 verify 해 볼 필요가 있다. 이에 대한 검증으로 paper는 아래 Figure 를 제시한다. 이를 통해, 우리는 여러 GNN 모델에서 Layer를 깊게 쌓을수록 MAD값이 줄어드는 상황, 즉 노드 페어 간 Mean Average Distance, 거리가 감소하는 상황을 살펴볼 수 있다. 여기서 거리가 감소한다는 것은 서로 가까이 위치한, 즉 서로서로 표현이 비슷한 상황이다. 여기서 더 쉬운 이해를 위해 MAD는 `식별성`에 대응시키면 쉽게 이해할 수 있다.

![](../../.gitbook/2022-spring-assets/SukwonYun\_2/mad\_figure.png)

### (2) Information-to-noise Ratio

이로써 우리는 그래프의 임베딩 표현을 바탕으로 smoothness를 정량화할 수 있는 척도를 얻어냈다. 하지만 우리는 본래 목표했던 Over-smoothing 에 대해 더 이해해 볼 필요가 있다. Paper 는 Over-smoothing 의 원인을 `over-mixing of information and noise` 때문이라고 보고 있다. 즉, smoothness 를 토대로 장점을 얻지만 동시에 이 smoothness 때문에 되려 noise 가 발생할 수 있다는 것이고 이 noise 의 영향력이 커져 Over-smoothing이 일어난다는 것이다. 동시에 또, 현재까지 GNN이 성공적이었던 이유는 information의 비율이 noise의 비율보다 컸기 때문이라고 논문은 주장한다. 여기서 구체화해보면, _information_은 **intra-class** (같은 클래스 내) 그리고 _noise_는 **inter-class** (다른 클래스 간)로 이해할 수 있다.

이러한 information 그리고 noise가 공존하는 상황을 역시 비율로써 정량화할 수 있는데 이를 정리해보면 아래와 같다. `Information-to-noise` 는 전체 노드 페어 중에서 intra-class(같은 클래스 간 연결된 페어)의 비율로 나타낼 수 있고, 예를 들어 2-hop 내에서 Information-to-noise 를 노드 관점, 그리고 그래프 관점에서 각각 구할 수 있다. 둘의 차이는 노드 관점에서는 2-hop내 실제 노드 수를 바탕으로 한다는 점과 그래프 관점에서는 2-hop내 실제 노드 페어 수를 바탕으로 한다는 점이다. 역시 정량화한 Information-to-noise가 유의미한지 확인해볼 필요가 있다. 오른쪽 그림을 보면, Order(Hop)가 높아질수록, Information-to-noise 비율이 급격하게 줄어드는 경향성을 확인할 수 있다. 이를 통해, 우리는 Hop 이 커질수록, intra-class 가 줄어들고(i.e., inter-class 는 늘어남) information에 비해 noise가 더욱 커지는 상황을 확인할 수 있다. 종합하면, 우리는 Over-smoothing의 원인을 **over-mixing of information and noise** 바라보았고 이는 information-to-noise 의 비율로서 살펴볼 수 있었고, 결과적으로 Hop이 커질수록 noise가 커지는 상황을 재차 확인하였다.

![](../../.gitbook/2022-spring-assets/SukwonYun\_2/information\_to\_noise.png)

### (3) MADGap

이렇게 Hop(i.e., Order)이 커질수록 noise가 커지는 상황을 우리는 두 가지 케이스로 구분하여 바라보려고 한다. 바로 order가 작은 상황(i.e., neighboring nodes, 논문에서는 3-hop 이내) 그리고 order가 큰 상황(i.e., remote nodes, 논문에서는 8-hop 이상)이다. 아래 정리된 그림을 통해 보면, ![](https://latex.codecogs.com/svg.image?MADGap)은 멀리 떨어졌을 때의 식별성, ![](https://latex.codecogs.com/svg.image?MAD%5E%7Brmt%7D)과 가까이 있을 때의 식별성, ![](https://latex.codecogs.com/svg.image?MAD%5E%7Bneb%7D)의 차를 통해 정의되는 것을 확인할 수 있고 이를 통해 우리는 드디어 `Over-smoothing 이 언제 발생하는지`, 이를 `수치적으로` 이해할 수 있게된다. 즉, ![](https://latex.codecogs.com/svg.image?MADGap)이 크면 그만큼 멀리 떨어진 노드의 식별성이 좋은 상황이므로, 우리에게 좋은 상황이되고 이와 달리 ![](https://latex.codecogs.com/svg.image?MADGap)이 작거나 음수 값을 가지게 되면 멀리 떨어진 노드들의 식별성이 가까운 노드들보다 안좋은, 바로 **Over-smoothing** 이 발생하는 순간임을 알 수 있다.

![](../../.gitbook/2022-spring-assets/SukwonYun\_2/madgap.png)

정의한 MADGap이 실제로 Over-Smoothing을 잘 대변하는 효과적인 수치인지 역시 확인해 볼 필요가 있다. 저자는 아래 실험들을 통해 효과성을 검증한다. 해석해보면, 많은 경우 MADGap과 Accuracy가 같은 경향성을 가지고 있고 실제로 Pearson 계수까지 1에 가까운 수치를 가지고 있음을 확인할 수 있다. 이를 통해, 우리는 Layer가 커질수록 감소하는 Accuracy 와 같은 경향을 지닌 MADGap이 Measure로서 타당함을 확인할 수 있다. 첨언하면, MADGap 이 감소한다는 것은 멀리 떨어진 노드의 식별성이 떨어져서 Over-smoothing 이 발생하고 있다는 것이다. ![](../../.gitbook/2022-spring-assets/SukwonYun\_2/madgap\_verification.png)

### (4) Topology Affects the Information-to-noise Ratio

앞서 우리는 Oversmoothing 의 원인이 되는 Information-to-noise를 살펴봤고 이를 통해 MADGap을 정의할 수 있었다. 그러면 이러한 Information-to-noise에 영향을 주는 요소를 찾아 볼 필요성이 대두된다. 결론부터 말하면, 바로 Edge를 기반으로 Graph가 생성되는 방식, `Graph topology` 때문이라고 paper는 주장한다. 이는 Graph가 생성된 방식과 우리가 풀고자 하는 문제(e.g., node classification)에서 괴리가 있다는 것인데, 더 구체화하면 node classification에서는 서로 다른 클래스의 노드들을 잘 구분하는게 목표임에 비해, 그래프가 애초에 생성될 때 **inter-class**의 Edge가 너무 많다는 것이다. 이러한, 서로 다른 클래스를 연결해주는 inter-class의 엣지는 message-passing 과정에서 자기 자신 뿐만이 아닌 다른 클래스의 정보도 전파해서 악영향을 끼치게 된다. 여기서 우리는 intra-class의 엣지는 늘려주고, inter-class의 엣지는 제거해주면 우리가 풀고자 하는 문제에 더 적합할 것이라는 Motivation을 얻게 된다. 실제로 저자는 간단한 실험을 통해 이를 입증하는데, 아래 그림에서 _inter-class의 엣지를 제거할수록_, 그리고 _intra-class의 엣지를 증가시킬수록_ 더욱 성능 개선이 있음을 Acc, MAGap 측면에서 확인할 수 있다.

![](../../.gitbook/2022-spring-assets/SukwonYun\_2/topology.png)

## **3. Method**

> Let's alleviate Over-smoothing!

앞선 Motivation에서 우리는 3가지의 새로운 척도인 MAD, Information-to-noise, MADGap를 도입하여서 Smoothness 그리고 Over-smoothness를 수치적으로 정량화하였다. 그렇다면 이렇게 정의한 Over-smoothing을 실제로 개선하기 위해서는 어떻게 접근해야할까? Paper는 두 가지 간단한 방안을 제시한다.

### (1) **MADReg: MADGap as Regularizer**

필자는 여기서 앞서 정의한 척도, `MADGap`이 비로소 빛이 난다고 생각한다. 바로, 우리가 minimize하고 싶은 loss에 추가적인 텀으로, regularizer로 붙여주기만 하면 되기 때문이다. 먼저 식을 살펴보자.

$$
\begin{equation} \mathcal{L} = \sum -l\text{log} p(\hat{l}|\mathbf{X},\mathbf{A},\mathbf{\Theta}) - \lambda \text{MADGap} \end{equation}
$$

기본적인 Cross-Entropy loss 에 추가적으로 MADGap의 Regularizer가 추가되었다. 해석해보면, MADGap이 커질수록 remote nodes의 식별성이 커져서 우리에게 좋은 상황이고, MADGap이 작아질수록 remote nodes의 식별성이 작아지는, 우리에게 안좋은 상황이므로 임베딩이 업데이트되는 과정에서 전체 Loss를 줄이기 위해, MADGap이 커지도록 업데이트가 될 것이다. 또 다른 관점에서는 위에서 Accuarcy와 경향성이 일치하는 면모를 보았기에 이 Accuracy가 커지는 방향으로 업데이트 된다고 생각할 수도 있겠다. 추가적으로, ![](https://latex.codecogs.com/svg.image?%5Clambda)는 MADReg을 조절해주는 constant이다. 종합하면 아주 간단하지만, 위에서 정의한 MAD, MADGap 덕분에 이렇게 간다하면서도 효과적인 방안이 디자인될 수 있었다고 필자는 생각한다.

### (2) **AdaEdge: Adaptive Edge Optimization**

또 다른 방안으로는, 우리가 앞서 살펴보았던 Motivation (2)-4 `Topology Affects the Information-to-noise Ratio`에서 비롯된다. 바로, inter-class의 엣지는 제거해주고 intra-class의 엣지는 증가시켜주는 방향으로 그래프의 구조를 다시 바꿔주는 방향으로 Optimization 되는 것이다. Optimization 알고리듬을 살펴보면 아래와 같다.

![](../../.gitbook/2022-spring-assets/SukwonYun\_2/adaedge.png)

간단히 살펴보면, ADDEDGE 함수에서는 실제 연결이 안되어있는 노드 페어를 대상으로 최종 prediction이 같고, softmax를 태운 결과값이 사전에 정의한 conf+ 의 값보다 크면 해당 페어를 기존 엣지에 추가적으로 연결해주는 역할을 하고 비슷하게 REMOVEDGE 함수는 실제 연결이 되어있는 노드 페어를 대상으로 서로 prediction이 다르고 softmax를 태운 결과값이 사전에 정의한 conf- 의 값보다 크면 해당 노드 페어의 엣지를 제거해주는 역할을 한다. 이 두함수를 바탕으로 매 iteration에서 Graph의 구조는 바뀌어가게된다.

## **4. Experiment**

다음은 위의 2가지 방안, MADReg 그리고 AdaEdge의 효과성을 입증하기 위한 실험이다. 실험은 node-classification을 대상으로 진행하였고 특정 방식의 GNN 모델링 혹은 알고리듬을 제안한게 아닌 각 GNN 모델에서 적용가능한, _Oversmoothing을 개선하는 하나의 Framework_를 제안하였기에 각 모델에 이를 적용해서 효과성을 입증하는 방식으로 실험이 진행되었다.

### **Experiment setup**

* Dataset
  * 총 7가지의 dataset, _Cora, CiteSeer, PubMed, Amazon Photo, Amazon Computers, Coauthor CS, Coauthor Physics_를 사용하였다. Dataset의 Statistics는 아래와 같이 요약된다. ![](../../.gitbook/2022-spring-assets/SukwonYun\_2/data.png)
* baseline
  * 총 10가지의 GNN 베이스라인을 사용하였다. 가장 유명한 GCN부터 출발해서 ChebGCN, HyperGraph, FeatSt, GraphSAGE, GAT, ARMA, HighOrder, DNA, GGNN의 여러 GNN Variant를 다루었다.
* Evaluation Metric
  * Accuracy와 해당 paper에서 제시한 MADGap 을 사용하였다.

### **Result**

**(1) MADReg and AdaEdge Results on **_**CORA/CiteSeer/PubMed**_

![](../../.gitbook/2022-spring-assets/SukwonYun\_2/experiment1.png)

일반적으로 GNN은 2-3 layer 내에서 Best Performace가 나오지만 그 이후 layer 수에서는 performance가 급격하게 감소하게 된다. 이에 저자는 4 layer를 기준으로 삼고 기존 baseline과 baseline에 MADReg 그리고 AdaEdge를 적용해 본 버전을 실험군으로 두었다. 결과를 확인해보면 Acc, MADGap 모두에서 그리고 Cora, CiteSeer, PubMed 데이터에서 Oversmoothing을 개선한 방안인 두 방법론이 효과적임을 확인할 수 있다. 실험은 5개의 split 방법에서 각각 10개의 random seed를 두어 진행하였기에 그려진 box plot이 유의미하다고 볼 수 있다.

**(2) MADReg with different layers and AdaEdge with GNN varaints** ![](../../.gitbook/2022-spring-assets/SukwonYun\_2/experiment2.png)

위에는 MADReg을 GCN에 적용하여서 Acc, MADGap 측면에서 기존 GCN과 대비하여 성능향상이 얼마나 있었는지 살펴본 결과이다. 확실하게 Layer를 많이 쌓을수록 기존 GCN의 성능은 급격하게 떨어지는 면모를 살펴볼 수 있는데 비해 MADReg을 적용한 버전은 상대적으로 그 성능이 떨어지는 기울기가 덜 급격한 것을 확인할 수 있다. 이는 Oversmoothing이 상대적으로 덜 일어나고 있음을 확인할 수 있는 대목이다.

밑에는 여러 GNN varaints에 AdaEdge 방법론을 적용했을 때 성능 증가폭을 살펴 본 결과이다. 확실하게 특정 GNN이 아닌 모든 GNN에서 효과적으로 적용가능한 framework임을 확인할 수 있는 실험으로 해석된다.

## **5. Conclusion**

이로써 우리는 Oversmoothing이 일어나는 원인을 밝히고 이를 수치적으로 정량화할 수 있는 Measure들인 MAD, Information-to-noise Ratio, MADGap를 처음으로 살펴보았다. 더 나아가, Oversmoothing을 해결하기 위한 방법론으로 MADReg 그리고 AdaEDGE까지 살펴보고 그 방법론의 효과성까지 검증할 수 있다. 오늘의 논문을 세 줄 요약하면 다음과 같다.

* (1) GNN에서 Oversmoothing을 파헤치고 왜 일어나는지 분석한 초창기 연구
* (2) MAD(for smoothness), Information-to-noise(key factor of oversmoothing), MADGap(for oversmoothness)
* (3) Oversmoothing을 해결하고자 한 MADReg, AdaEdge

그리고, 필자는 이렇게 글을 마무리 짓고 싶다.

**`Alleviating Oversmooting? Okay, but still it ain't guarantees best performance.`**

## **Author Information**

* Sukwon Yun (윤석원)
  * Master Student in ISySE, KAIST ([DSAIL](http://dsail.kaist.ac.kr))
  * Interested in **Weakness of GNN such as Long-Tail Problem, Over-smoothing Problem and Differential Equations on general NN**
  * Contact: swyun@kaist.ac.kr

## **6. Reference & Additional materials**

* Paper: [https://arxiv.org/abs/1909.03211](https://arxiv.org/abs/1909.03211)
* GCN: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
  * PDF 자료: [https://github.com/SukwonYun/GNN-Papers](https://github.com/SukwonYun/GNN-Papers)
  * 윤훈상 연구원님 자료: [https://www.youtube.com/watch?v=F-JPKccMP7k\&t=635s](https://www.youtube.com/watch?v=F-JPKccMP7k\&t=635s)
* Github Review (본문 사진이 잘 안보일 경우): [https://github.com/SukwonYun/awesome-reviews-kaist/blob/2022-Spring/paper-review/2022-spring/AAAI-2020-MAD.md](https://github.com/SukwonYun/awesome-reviews-kaist/blob/2022-Spring/paper-review/2022-spring/AAAI-2020-MAD.md)
