description

---

Kamilė Stankevičiūtė, Ahmed Alaa, Mihaela van der Schaar / Conformal Time-Series Forecasting / 2021(NeurIPS)

---

# Conformal Time-Series Forecasting

## **1. Problem Definition**

point가 아닌 interval 예측을 하고, 불확실성 간격(uncertainty interval)에 대해 frequentist 범위에 대한 이론적 보장을 제공하는 RNN 기반 시계열 예측 모델인 CF-RNN(Conformal Forecasting RNN)을 제안한다.





## **2. Motivation**

시계열 예측은 주가, 서비스 수요, 의학적 예후 등 응용 영역의 핵심이며 RNN이 이 작업을 수행하는 일반적인 모델이다. RNN을 이용하는 multi-horizon 시계열 예측[1]에 대한 현재 접근 방식은  *point 예측*(시계열 미래 값에 대한 단일 추정치)에 집중하고 있다. 금융 및 의료와 같은 고부담 응용 영역의 경우에는 정확한 위험 평가와 의사결정을 위해서는 *불확실성 추정이 요구*되기 때문에 **point 예측은 충분하지 않다.**

불확실성 추정을 하기 위해 표준 feed-forward 신경망[2]을 이용한 다양한 방법들은 지금까지 제안되어 왔지만 **RNN기반 시계열 모델을 이용한 방법들은 아직 탐구 되지 않았다.** 

RNN에 대한 불확실성 추정을 위해 개발된 이전 방법들에는 베이지안 RNNs, 분위수 RNNs(Quantile RNN), 앙상블, 등각예측(Conformal prediction, CP)가 있다. 그리고 이 방법들은 각각 유발되는 한계점들이 존재한다. 베이지안 RNNs은 기본 아키텍쳐에 대한 많은 변화가 요구되고 분위수 RNNs은 샘플 복잡도가 좋지 않아 분위수 과적합의 위험이 있다. 심층 신경망 앙상블은 일반적으로 불확실성 정량화에 대해 수학적으로 원리가 적용되지 않는다. 또한 계산 복잡성이 높은 경우도 존재한다. 뿐만아니라, 불확실성 간격(uncertainty interval)에 대한 이론적 타당성에 대한 보장을 제공하지 않을 수 있다는 제한점이 존재 한다.


저자가 본격적으로 참고한 이전 연구는 등각예측 방법(CP)이기 때문에 지금부터 이에 대해 집중적으로 알아보겠다. 이는 회귀문제(예시: 시계열 예측) 추가 보정 세트와 기본 모델을 사용하여 귀납적으로 작동하도록 수정되고 이를 귀납적 등각 예측(ICP)라고 한다. 


![그림1](/.gitbook/2022-spring-assets/paperreview_result/figure1.PNG)


(시계열 관찰 패러다임; 왼쪽 부분의 데이터 세트는 단일 시계열로 구성되어 있다고 가정하고 관측값은 시계열 내의 개별 시간 단계이며 이러한 관찰은 시간적으로 종속적이다. 반면에 오른쪽 데이터셋은 독립적인 시계열 세트로 구성되며 전체 시리즈가 관측값으로 처리되고 시계열의 독립성은 교환 가능성을 의미한다.)


시계열 예측을 위해 (I)CP 방법을 적용하는 작업은 다음과 같은 문제점 때문에 거의 수행되지 않았다. 주요 문제는 CP가 데이터 세트 관찰의 모든 순열이 등가 가능성이 있는 교환 가능성을 가정한다는 것이다. 그러나 시계열 내의 시간 단계는 시간 종속성으로 인해 본질적으로 교환할 수 없다(위 그림의 왼쪽). 따라서 주어진 시계열에서 예측 간격을 도출하기 위해 순진하게 CP를 적용하는 것은 방법론적으로 타당하지 않으며 타당성 보장이 부족하다. 한가지 주목할만한 예외는 EnbPI 모델로, 교환 가능성 가정(일부 다른 도입)을 우회하고 부트스트랩추정기의 앙상블을 사용하여 대략적으로 유효한 간격을 제공한다. 그러나 단일 시계열에서 학습하는 것은 실제로 하나의 시계열만 사용할 수 있는 경우에 유용하지만 위 그림의 오른쪽 그림과 같이 데이터 세트에 여러 시계열이 포함되어 있고 공유 패턴이 잠재적으로 악용될 수 있는 설정에서는 최적이 아닐 수 있다고 주장한다. 방법론적으로 더 근거가 있음에도 불구하고 어떤 기존 방법도 후자의 예측 설정에 CP를 적용하지 않았다. 그러나 여러 시계열의 데이터 세트는 점점 더 일반적이고 유용하다.



저자는 이러한 ICP의 특성을 이용하여 본 연구에서는 inductive conformal prediction[3] framework를 시계열 예측으로 확장시켜 기존 방법들의 문제점들(앞 문단에서 언급된 제한점들)을 해결하는 lightweight uncertainty estimation 과정을 제안한다. 이 접근 방식은 모든 multi-horizon forecast predictor와 모든 dataset에 대해 minimal exchangeability 가정을 하면서 불확실성 간격에 대해 frequentist 범위에 대한 이론적 보장을 제공한다. 본 연구를 진행한 이론적 동기에 대해서 더 구체적으로는 3. Method의 CF-RNN: ICP for multi-horizon RNNs에서 보일 것이다.


[1] multi-horizon 시계열 예측: 3. Method 영역의 다중 수평 시계열 예측 부분을 참고

[2] 표준 feed-forward 신경망: 노드의 연결이 루프를 형성하지 않는 인공 신경망 유형

[3] inductive conformal prediction(ICP): 3. Method 영역의 귀납적 등각 예측 부분을 참고






## **3. Method**

저자는 motivation영역에서 언급한 기존 방법들에 대한 제한점을 타파하고자 conformal forecasting RNNs(CF-RNNs) 모델을 제안했다.




**[1. 다중 수평 시계열 예측(Multi-horizon time-series forecasting)]**

![](https://latex.codecogs.com/svg.image?y__{t:t'}) = ![](https://latex.codecogs.com/svg.image?(y__{t},&space;y__{t&plus;1},&space;...&space;,&space;y__{t'}))가 d차원 시계열 관측값이고 ![](https://latex.codecogs.com/svg.image?y__{t:t'}) = ![](https://latex.codecogs.com/svg.image?)가 주어지면, multi-horizon 시계열 예측은 ![](https://latex.codecogs.com/svg.image?\\hat{y}__{t'&plus;1:t'&plus;H})인 미래 값을 예측한다. 이는 H x d 차원이고 H는 예측할 steps의 수(예측 horizon)이다.



중요한 응용 프로그램의 경우 예측과 관련된 불확실성에 관심이 있다. 예측 범위의 각 시간 단계(h)에 대해
ground truth 값 yt+h가 충분히 높은 확률로 ![](https://latex.codecogs.com/svg.image?[{\\hat{y}}{t+h}^{L},&space;{\hat{y}}{t+h}^{U}]), h ∈ {1, . . . , H}) 구간에 포함되도록 한다. 전체 시계열 궤적의 실제 값이 간격 내에 포함되도록 원하는 유의 수준(또는 오류율) α를 수정한다. 

즉, 아래의 식을 만족하게 한다.

![그림1](/.gitbook/2022-spring-assets/paperreview_result/3.1f.png)





**[2. 귀납적 등각 예측(Inductive conformal prediction, ICP)]**

회귀 작업에 대한 귀납적 등각 예측(ICP)에 대한 필요한 배경을 설명하겠다.
관측값 세트 및 새로운 데이터가 주어지면 ICP 절차는 다음과 같은 예측 간격을 반환한다. 그렇기 때문에 **타당성 속성이 충족**된다.
등각 예측 프레임워크는 분포가 없으며(즉, 기본 데이터 D의 분포에 대한 가정이 없음) **교환 가능성 가정이 충족**된다.




**[3. CF-RNN: ICP for multi-horizon RNNs]**

이제 등각 예측 절차의 세부 사항에 대해 설명하겠다.
지금까지 레이블 y ∈ R이 스칼라이지만 다중 수평 시계열 예측이 H(d-차원) 값을 반환하는 경우를 고려했다. 저자는 다중 수평선 예측 간격을 내보내는 것의 타당성을 유지하면서 다중 예측 환경에서도 다룰 수 있도록 ICP 프레임워크를 확장했다. 이를 등각 예측 프레임워크(Conformal prediction framework)라고 한다.
H 조건부 독립 예측은 동일한 임베딩에서 얻어지기 때문에 원하는 오류율 α를 유지하기 위해 임계 보정 점수에 Bonferroni 보정을 적용한다. 특히, 원래의 α를 H로 나누므로 임계 부적합 점수 εˆ1, . . . , εˆH는 대응하는 부적합 점수 분포에서 [(m + 1)(1 − α/H)]-번째 가장 작은 잔차가 된다. 따라서 예측 구간의 결과 집합은 다음과 같다.


![그림1](/.gitbook/2022-spring-assets/paperreview_result/f7.png)


요약하면, CF-RNN(Conformal Forecasting RNN) 모델은 RNN 발행 지점 예측과 불확실성을 도출하기 위한 등각 예측 절차로 구성된다. CF-RNN에서 예측 구간을 구성하는 전체 절차는 아래의 그림에 설명되어 있고 아래의 알고리즘에 요약되어 있다.


![그림1](/.gitbook/2022-spring-assets/paperreview_result/figure2.PNG)



또한, 저자는 등각 예측 절차로 얻은 구간에 대한 타당성을 제공하는 다음 정리를 통해 제안된 접근 방식의 이론적 동기(2. Motivation의 구체적 부분)를 보여주었다.
이는 등각 예측 타당성(Conformal forecasting validity)때문이다.
다음 D를 교환 가능한 시계열 관측치 데이터 세트이고 H-단계 예측값을 동일한 기본 확률 분포에서 얻는다고 가정해보자. 


![그림1](/.gitbook/2022-spring-assets/paperreview_result/f8.png)


그리고 M을 직접 전략을 사용하는 H-단계 예측을 예측하는 순환 신경망이라고 하자. 모든 유의 수준 α ∈ [0, 1]에 대해 ICP 기반 등각 예측 알고리즘으로 얻은 간격(interval)은 최대 α의 오류율을 갖는다. 
즉, 아래 식을 만족하게 된다.


![그림1](/.gitbook/2022-spring-assets/paperreview_result/f9.png)


(증명은 Vovk[51]에서 ICP의 조건부 타당성과 Boole’s inequality에서 따라나온다.)


알고리즘은 다음과 같다:


![그림1](/.gitbook/2022-spring-assets/paperreview_result/algorithm.PNG)








## **4. Experiment**

다양한 합성 데이터와 실제 의료 데이터에 대해 기존에 존재하는 baseline과 본 연구에서 제안한 방법(CF-RNN)과의 3가지 기준으로의 성능 비교를 통해 conformal forecasting framework의 효율성을 본다. 합성데이터의 경우에는 BJ-RNN을 기준으로, 3개의 실제 의료 데이터의 경우에는 MQ-RNN, DP-RNN을 기준으로 비교한다.



### **Experiment setup**

- Dataset
    - 속성이 제어된 합성 데이터
        
        (BJ-RNN은 더 큰 실제 데이터 세트로 확장되지 않기 때문에 BJ-RNN을 다른 방법과 비교하기 위해 더 작은 합성 데이터 세트를 사용)
        
        
        >합성 데이터 만드는 방법
        >
        >: Autoregressive process와 noise process가 포함된 합성 시계열데이터 생성
        >
        >: (수학적으로 표현)
        >
        >먼저 시계열의 추세를 결정하는 자기회귀 프로세스와 데이터 세트의 고유한 불확실성을 나타내는 노이즈 프로세스의 두 가지 구성요소로 구성된 합성 시계열을 생성한다.
        >길이 T의 시계열에 대해 다음과 같이 표현된다.
        >
        >![그림1](/.gitbook/2022-spring-assets/paperreview_result/f10.png)
        >
        >(xt ~ N(μx, σ2x), a = 0.9는 메모리 매개변수, t ~ N(0, σ2t)은 노이즈 프로세스 5개의 정적 노이즈 분산 프로파일 σ2t = 0.1n, n = {1, . . . , 5})
        
  
    - 3개의 현실 의료 데이터들(MIMIC-3, EEG, COVID-19)
        
        > 실시간 시계열의 다양한 시나리오를 나타내기 위해 이러한 데이터세트를 선택했다.


- baseline
    
    모든 아키텍처는 기본 순환 신경망으로 LSTM을 사용하며 직접 다중 수평선 예측을 생성하도록 조정된다.
    
    - The Frequentist blockwise jackknife RNN (BJ-RNN)
    - The multi-quantile RNN (MQ-RNN)
    - The Monte Carlo dropout-based RNN (DP-RNN)
    
    >선정 이유: uncertainty estimation 틀에서 가장 유명하고 대표적인 예시
    
    
- Evaluation Metric
    - Empirical joint coverage(높을수록 좋음)
    - Prediction interval widths(낮을수록 좋음)
    - Bonferroni-corrected and uncorrected empirical coverages
    






### **Result**


#### **[합성 데이터]**

2개의 노이즈 분산 프로필에 대해 2000개의 훈련 시퀀스(CF-RNN이 이 데이터 세트를 1000개의 실제 훈련 및 1000개의 보정 시퀀스로 분할)에서 모델을 훈련합니다. 저자는 예측 간격(H)에 대해 미래 값들(![](https://latex.codecogs.com/svg.image?y__{t'&plus;1:t'&plus;H})을 기본 적용률(coverage rate)인 90%(α =0.1)로 예측하는 것을 목표로 한다(여기서 T = 15, H = 5). 불확실성 추정 모델을 위한 RNN 하이퍼파라미터는 공정한 비교를 위해 이전 연구에서 정해진 대로 정했다.

무작위로 생성된 새로운 데이터 세트로 실험을 5번 반복하여 다양한 realizations에 대한 경험적 joint 범위(coverage)의 변화를 관찰했다.

아래의 표 2와 3은 모델의 joint coverage 불확실성 간격을 비교한다. CF-RNN과 BJ-RNN은 모두 정적(static) 및 시간 의존적인 노이즈 환경 둘 다에서 요구된대로 유한 표본 frequentist 범위 보장을 만족시키면서 90%(α = 0.1)인 목표 공동 커버리지를 경험적으로 능가한다.


![그림1](/.gitbook/2022-spring-assets/paperreview_result/table2.PNG)

![그림1](/.gitbook/2022-spring-assets/paperreview_result/table3.PNG)


표 3은 추가로 CF-RNN 간격이 데이터 세트의 시간적 역학 특성에 적응함을 보여준다. 노이즈가 static일 때, CF-RNN 예측 간격 폭은 증가하는 기본에 따라 크게 변하지 않는다. 반면에 노이즈 프로파일이 시간 종속적일 때, 평균 간격은 데이터 세트 불확실성이 증가함에 따라 넓어진다. 

다른 frequentist 기준선인 BJ-RNN은 CF-RNN보다 간격이 훨씬 더 넓다. 이것은 완벽한 적용 범위를 유지하는 데 중요할 수 있다. 그러나 저자는 적용 범위가 목표 적용 범위를 초과하는 한, 의사 결정에 가장 유익한 정보를 제공하기 위해 간격이 최대한 효율적(좁음)이어야 한다고 한다(무한한 간격은 완벽한 적용 범위를 갖지만 정보를 제공하지 않을 것이라고 간주하면서). 또한 BJ-RNN은 계산하는 데 굉장히 오랜 시간이 걸린다(단 하나의 시드만 포함하고 실제 데이터에 대한 비교에서도 제외되는 이유). 반대로 ICP 절차는 보정 세트(추가 계산 비용 없이 모델을 원하는 범위에 대해 동시에 보정할 수 있음)에서 훈련된 RNN 모델을 실행하기만 하면 되며, 이 시점에서 예측에 불확실성 간격을 추가하는 데 일정한 시간이 걸린다. 

반면에 MQ-RNN 및 DP-RNN과 같은 대안적(비빈도적) 패러다임을 따르는 기준선은 모두 목표 적용 범위를 달성하지 못하며 때때로 적용 범위 비율이 0으로 나온다. 이러한 이유로 두 모델은 더 좁은(더 효율적인) 간격으로 제공되지만 적용 범위 보장이 부족하여 위험이 큰 실제 응용 프로그램에서 덜 유용하다.


![그림1](/.gitbook/2022-spring-assets/paperreview_result/figure3.PNG)


제어된 속성이 있는 데이터에 대한 실험을 통해 원하는 적용 범위와 미래에 얼마나 멀리 예측을 안정적으로 수행할 수 있는지 간의 균형에 대한 추가 통찰력을 얻을 수 있었다. 

위의 그래프는 왼쪽 및 중간 패널은 훈련 데이터 세트 크기에 따른 CF-RNN, MQ-RNN 및 DPRNN 기준선의 평균 성능을 보여 준다. CF-RNN은 제한된 수의 예제로 필요한 joint 범위 비율을 달성하고 유지하는 유일한 모델이다. 또한 더 많은 데이터(더 큰 교정 데이터세트)를 사용하면 불일치 점수의 분포를 더 정확하게 지정할 수 있으므로 간격의 너비가 감소한다. 

마지막으로 오른쪽 패널에서 예측 간격 너비를 고정하고 각 수평선에 대해 H는 CF-RNN에 의해 유지되는 가장 큰 커버리지 수준 1-α를 계산한다. 위의 세번째 그림에서 볼 수 있듯이 목표 범위 수준이 낮으면 먼 미래까지 유효한 예측을 할 수 있으며 이상적인 범위 수준은 예측 지점 근처의 수평선에서만 달성할 수 있습니다. 모든 순환 신경망 모델 M에 대해 전체 추세가 유지된다.





#### **[현실 시계열 데이터]**

CF-RNN, MQ-RNN, DP-RNN 모델들을 아래의 표4에서 처럼 3개의 데이터셋에 대해 훈련시켰다. 


![그림1](/.gitbook/2022-spring-assets/paperreview_result/table4.PNG)



- MIMIC-3
    
    다양한 길이로 백혈구 수에 대한 일일 예측을 한다.
    
- EEG(electroencephal graphy)
    
    세 가지 유형의 시각적 자극에 노출된 건강한 피험자로부터 얻은 다운샘플링된 EEG 신호의 궤적을 예측한다.
    
- COVID-19
    
    영국 지방 당국의 일일 COVID-19 사례를 예측
    


아래의 표5에서 모델의 성능을 확인할 수 있다. 

![그림1](/.gitbook/2022-spring-assets/paperreview_result/table5.PNG)



제안된 CF-RNN 아키텍처의 기본 LSTM 모델은 일부 예제가 보정 절차에 사용되었기 때문에 경쟁 기준과 동일한 하이퍼파라미터를 갖지만 훈련 인스턴스는 더 적었다. 그럼에도 불구하고 CF-RNN은 모든 데이터 세트에 대해 가장 높은 Coverage를 얻었고 목표 Joint Coverage 비율을 경험적으로 달성한 유일한 모델이다. 따라서 제안된 모델의 예측은 데이터 세트 및 시나리오 범위에서 실제로 신뢰할 수 있다. 

반면에 기준 모델은 일부 설정(예: MIMIC-III)에서 더 나은 효율성으로 경쟁력 있는 적용 범위를 갖는 것으로 보이지만 덜 특정 시나리오(예: COVID-19)에서는 신뢰할 수 없는 예측으로 되돌간다. 즉, CF-RNN은 예측 간격 너비를 필요한 대상 범위와 안정적으로 일치하도록 조정하여 예측할 수 없는 데이터 세트의 너비를 늘린다.

아래의 표6에서 모델의 joint coverage, independent coverage에 대한 실험 결과를 통해 CF-RNN 교정(calibration) 절차에서 오류율의 Bonferroni correction 수정의 중요성과 동기(motivation)를 실험적으로 살펴볼 수 있다.


![그림1](/.gitbook/2022-spring-assets/paperreview_result/table6.PNG)


독립적인 범위(independent coverage) 비율이 일반적으로 목표 범위를 달성함에도 불구하고, Bonferroni correction가 없는 경우에는 보정 점수(calibration scores)는 낮은 joint coverage를 나오게 한다. 특히 예측의 어려움으로 선택되었던 COVID-19 데이터 세트가 가장 눈에 띄는 예외이다.







## **5. Conclusion**

본 논문에서는 ICP 프레임워크를 multi-horizon 시계열 예측 문제로 확장하여 frequentist coverage를 위한 이론적 보장하는 lightweight 알고리즘을 제공한다. 또한 이는 실험을 통해서 기존의 기준선보다 좋은 성능을 가지며, 목표 간격 범위를 충족시킨다.

future work로는 예측 간격의 너비를 줄여 보면서 전반적으로 효율성을 높이는 연구를 진행해 볼 수 있다. 또한, 차원을 d = 1일때를 집중적으로 보았기 때문에 다음 연구로 다변수 시계열 의 결과에 대해 연구를 진행 할 수 있을 것이다.

이 논문은 저자가 서술한 대로 글을 읽으면 내용을 이해하기 쉽게 서술해 주었기 때문에 나중에 내가 논문을 쓸 때에도 다시 참고할 만한 논문이라서 내용 뿐 아니라 논문 자체를 읽으면서 흐름을 공부하는 것도 도움이 될 것이다. 이해하기 쉬운 이유는 내가 생각해 보았을 때, 연구의 구체적 설명이 들어가기 이전에 한번씩은 큰 흐름을 먼저 파악할 수 있도록 간단하게 설명을 하여 과정에 대한 뼈대를 생각하게 한다. 그 뼈대에 구체적인 살을 붙여가며 다시 설명해 주는 과정을 다량의 정보 이전에 계속 제공해주어 이 연구를 더 파악하기 편했다고 느꼈다. 또한, 이 연구를 따라가기에 꼭 필요한 굵직한 키워드들의 정보(ex. ICP)들을 자세히 설명해주어서 어떤 연구를 하고 결과가 나왔는지 파악하기 쉬웠다. 

---



## **Author Information**

- Author name: 여정
    - Affiliation: CSD Lab
    - Research Topic: Healthcare Service


## **6. Reference & Additional materials**

Please write the reference. If paper provides the public code or other materials, refer them.

- Github Implementation: [https://github.com/kamilest/conformal-rnn](https://github.com/kamilest/conformal-rnn)
- Reference:
