---

description : Yinda Zhang, Tomas Funkhouser / Deep Depth Completion of a Single RGB-D Image / 2018(CVPR)

---

# **Deep Depth Completion of a Single RGB-D Image**

## **1. Problem Definition**

본  연구에서는  RGB-D image에서  발생하는  깊이(depth) 정보  누락문제를  해결하기  위해  정렬된  색상(color) 이미지를  사용하여  single RGB-D 이미지의  깊이  채널(depth channel)을  완성하기  위한  딥러닝  접근법을  제안한다.

## **2. Motivation**

RGB-Depth 카메라란  RGB-D 이미지(RGB 색영상+깊이(depth)  정보)를  제공하는  카메라를  뜻하며, RGB-D sensor는  생체  인증, 자율주행, 실내로봇, 증강현실  등에서  사용되며, 많은 future application을  가능하게  한다. 즉, RGB-D 이미지는  다양한  곳에  사용되는  주요한  정보를  포함하고  있다.

![motivation](https://github.com/jeong-jjang/awesome-reviews-kaist/tree/2022-Spring/.gitbook/2022-spring-assets/YeoJeong_1/motivation.png)


 위의  그림에서  보이는  것처럼, 상용  심도  카메라(Commodity-grade depth cameras)로  찍힌  RGB-D 이미지를 보면 컬러  이미지(RGB 부분)는 실제 그대로 보여지는 반면에 깊이  이미지는 종종  아래 예시(Depth 부분)의 흰색  영역처럼  깊이  채널(depth channel)의  거대한  누락이  존재한다. 이는 밝은 빛, 멀거나 빛나는 표면, 얇은 물체 그리고 검은 표면에서 주로 발생한다.
 
이와  관련된  이전  연구들을  확인  할  수  있다. 첫번째로  깊이  인페인팅(Depth Inpainting) 방법은  ~하는 . 두번째로  깊이  추정(Depth Estimation) 방법은  .  이 방법들은 작은 공간에 대해서만 작동하거나 컬러이미지만으로 깊이 추정을 위해 설계되었고 이 이전 연구들은  depth completion에는  사용되지  않았다. $$$$

저자는  넓은  공간에대해서도  작동하고  컬러  이미지의  정보  뿐만  아니라  깊이  정보를  모두  이용한  깊이  완성(depth completion)을  하는  딥러닝  접근법을  제시한다.


## **3. Method**

저자가  제안하는  RGB-D 깊이  완성(depth completion) 문제는  2단계를  통해  해결된다.

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/method.png">


**1 단계) 
색상에서만  표면  법선(surface normals) 및  폐색  경계(occlusion boundaries)를  예측**

쉽게  떠올릴  수  있는  접근  방식은  색상  및  관측된  깊이  모두를  신경망으로  직접  훈련하여  완전한  깊이를  요청하는  것이다. 하지만  저자는  표면  법선  및  폐색  경계를  이용하여  문제를  해결한다. 이  이유는  국부적  미분  속성을  이용하는  것으로  단안  컬러  이미지에서  절대  깊이를  추정하는  것보다  훨씬  쉬운  방법이기  때문이다. 절대  깊이를  측정하는  것은  local feature과 global context의  결합을  필요로  하기때문에  이러한  방법은  잘  작동하지  않는다.  이후  2단계에서  진행되는  global 최적화를  통해  절대  깊이를  구할  수  있다.

>표면  법선을  이용하는  이유는  크게  두가지가  있다. 첫번째로, 표면  법선을  추정하는  것이  깊이를  직접  추정하는  것보다  쉽다.  표면  법선은  평면  영역  내의  픽셀에  대해  일정하며  대부분  로컬  셰이딩  또는  텍스처에  의해  결정되고  절대값과  같은  임의의  값보다  수치적으로  회귀하기  쉬운 -1에서 1 사이의  단위  벡터이다.  두번째로는 깊이는  법선에서  강력하게  추정할  수  있다. 법선에서  깊이를  해결하면  전체  장면에  대한  표면  제약  조건을  전체적으로  통합할  수  있다. 법선을  통합하는  방정식은  비선형이고  제약  조건이  있는데, 저자는  그것들을  선형화하고  감각  깊이에서  정규화를  제공하여  효율적이고  강력하고  글로벌한  솔루션을  달성한다.


표면  법선  및  폐색  경계를  예측하기  위해  감독하에  완전  컨볼루션  신경망(fully convolutional neural network)을  훈련한다. 컬러  이미지만  입력으로  이용하는  이유는  깊이  입력을  하게  되면  깊이  채널의  큰  누락의  경우  심층  네트워크를  훈련시키는  것이  어렵기  때문이다. 일반적으로  입력  깊이를  복사하고  보간하는  방법만  학습하고, 네트워크가  색상과  깊이의  오정렬(misalignments)에  적응하는  방법을  배우는  것도  어렵다. 즉, 네트워크는  색상에서  local 기능만  예측한다.

본  연구를  위해  Zhang et.al이  제안한  심층  네트워크  구조를  선택했는데  그  이유는  normal estimation과  boundary detection에서  우수한  성능을  보이기  때문이다. 이  모델은  대칭  인코더  및  디코더가  있는  VGG-16의  back-bone에  구축된  완전  컨볼루션  신경망이다. 또한  로컬  이미지  기능을  학습하는데  중요한  해당  최대  풀링  및  풀링  해제  레이어에  대한  바로  가기  연결  및  공유  풀링  마스크가  장착되어  있다.

저자는  네트워크  훈련의  손실(loss) 결정을  위해  실험을  진행했다. 관측된  픽셀만  혹은  구멍(누락된  픽셀)만을  이용했을  때보다  모든  픽셀을  이용했을  때  그리고  원시  normal보다  랜더링된  normal로  훈련했을  때  성능이  좋다는  결과를  얻어서  랜더링된  모든  픽셀에  대해  계산된  손실을  이용하였다.

**2 단계) 
관측된  깊이의  제공과  1단계의  예측  정보로부터  global 표면  구조의  최적화.**

1단계에서의  얻은  정제되지  않은  구조는  입력  깊이로부터  정규화와  함께  global 최적화를  통해  재구성된다. 표면  법선과  폐색  경계만으로  깊이를  푸는  것은  이론적으로  불가능하기  때문에  raw 깊이  관측이  포함되어  있어야  한다.

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/optimization.png">

목적  함수(_E_)는  4개의  항으로  되어  있고  제곱  오차의  가중치  합으로  정의된다.

(_N_: Surface normal image, 
_B_: Occlusion boundary image , 
_D_: Depth image, 
_D($p$)_: Observed depth at pixel _p_, 
_D$_0(p)$_: Observed raw depth at pixel _p_,
 _N($p$)_: Predicted surface normal , 
 _E$_D$_: Distance between _D$(p)$_ and _D$_0(p)$_, 
 _E$_N$_: Consistency between the estimated depth and _N($p$)_, _E$_S$_: Adjacent pixels to have the same depth)

*본  연구의  최적화는 $λ_D = 10^3,  λ_N = 1 및 λ_S = 10^(−3)$으로 수행되었다.

- 제안된  접근  방식이  다른  접근방식보다  훨씬  더  작은  상대  오차를  가진다.

- 훈련된  네트워크가  관찰된  깊이와  무관하기  때문에  새로운  깊이  센서에  대해  다시  훈련할  필요가  없다는  추가  이점이  있다.



## **4. Experiment**

본 연구에서는 두가지  실험이  진행된다.

첫  번째  실험에서는  서로  다른  테스트  입력, 훈련  데이터, 손실  함수(loss functions), depth 표현(depth representations) 및  최적화(optimization) 방법이  depth 예측  결과에  어떤  영향을  미치는지  조사한다.

두  번째  실험  세트에서는  제안된  접근  방식이  baseline depth inpainting 및  depth estimation 방법과  어떻게  비교되는지  조사한다.

### **Experiment setup**

* Dataset

	SUNCG 데이터  세트에  대해  사전  훈련을  진행했다. 제안된  프레임워크의  성능을  실험하기  위해서는  랜더링된[1] completions(D*)가  포함된  총  117,516개의 RGB-D 이미지가  사용되었으며, 105,432개의 훈련 세트와 평가를  위한  12,084개의  테스트 세트로  분할되었다.

	저자는  이미  존재하는  표면  매쉬(surface meshes)를  이용하는데, 이는  데이터  셋을  생성하기  위해  큰  공간의  다중  뷰  RGB-D 스캔에서  재구성되었다. 기존의  여러  datasets 중  ‘Matterport3D’ dataset을  이용하여  재구성된  매쉬를  랜더링했으며  이것은  여러  카메라의  깊이를  통합하는  것과  같은  결과를  낼  수  있다.
( [1] 랜더링: 컬러  이미지(color image)와  센서  깊이(sensor depth)를  이용해서  ground truth를  만드는  과정)

	>이렇게  생성된  데이터  셋은  해당  연구의  심층 네트워크를 훈련에  있어  몇 가지 유리한 속성이 있고  이는  다음과  같다: 완성된  깊이  이미지(D*)는  일반적으로  누락된  영역이  적고, D*에서  멀리  있는  표면을  원본보다  더  나은  해상도로  제공하며, 원본보다  노이즈가  훨씬  적다.

* baseline

	1. 검증을  통해  최종적인  딥러닝  방법론  결정

	2. 제안된  딥러닝  방법론의  성능  평가

	- Depth Inpainting Methods
		- Smooth
		- joint bilinear filtering (Bilateral)
		-	fast bilateral solver (Fast)
		- global edge-aware energy optimization (TGV)

	- Depth Estimation Methods
		접근 방식이 우리와 가장 유사(예측 도함수 사용)
		- Laina et al. 
		- Chakrabarti et al. 

* Evaluation Metric
	-1. 검증을  통해  최종적인  딥러닝  방법론  결정
		
		1) 네트워크의  Input data 결정
		- 깊이  예측  평가:
			- median error relative to the rendered depth(Rel) 
			- root mean squared error in meters(RMSE)
			- percentages of pixels with predicted depths falling within an interval ([δ = |predicted − true|/true]), where δ is 1.05, 1.10, 1.25, 1.252 , or 1.253 . 

		- 표면  법선  예측  평가:
			- mean
			- median errors (in degrees)
			- the percent- ages of pixels with predicted normals less than thresholds of 11.25, 22.5, and 30 degrees.

	-2. 제안된  딥러닝  방법론의  성능  평가
		
		1)  & 2) 인페인팅(Inpainting) 및 깊이 추정의 성능비교
			- Qualitative evaluation
			- median error relative to the rendered depth(Rel)
			- root mean squared error in meters(RMSE)
			- percentages of pixels with predicted depths falling within an interval ([δ = |predicted − true|/true]), where δ is 1.05, 1.10, 1.25, 1.252 , or 1.253 . 

	
### **Result**



#### 1. 검증을  통해  최종적인  딥러닝  방법론  결정

1. 네트워크의  Input data 결정

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/table1.png">

일반적인 예측 네트워크에 가장 적합한 입력 유형(색상만, 원시 깊이만 또는 둘 다)을 테스트한  결과, 네트워크에  색상(color input)만 제공될 때,  깊이 추정값이  약간 더 우수하다(Rel = 0.089(색상만) vs. 0.090(둘다))는  것과  표면 법선도  더 잘 예측하는 방법을 학습한다(median error = 17.28º(색상만) vs. 23.07º(둘  다))는  것을  발견할  수  있다.

>이  결과는  훈련  시  모든 픽셀 관측된  픽셀만 또는 관측되지 않은 픽셀을  이용하는지에  관계없이 지속된다. 그 이유는 네트워크가 가능한 경우 관찰된 깊이에서 보간하는 방법을 빠르게 학습하여  큰 구멍에서 새로운 깊이를 합성하는 학습을 방해하기 때문이다.


2. 네트워크가 예측하기에 가장 적합한 깊이 표현(depth representation) 결정

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/table2.png">

절대 깊이(D), 표면 법선(N) 및 8개 방향(DD)의 깊이 도함수를 예측하기 위해 네트워크를 개별적으로 훈련시킨 다음, 방정식 1을 최적화하여 깊이를 완성하기 위해 다양한 조합을 사용한다. 결과에서  예측된 법선(N)은 최상의 결과를 제공한다는  결과를  볼  수  있다.(0.167(D), 0.100(미분; DD), 0.092(법선 및 미분; N+DD))에 비해 N의 경우 Rel = 0.089이다.

>법선은 상대적으로 예측하기 쉬운 표면의 방향만을 나타내기 때문이다. 게다가 법선은 깊이 또는 깊이 도함수와 달리 깊이에 따라 크기가 조정되지 않으므로 뷰 범위에서 더 일관된다는  장점이  있다.

3. 폐색  경계의  예측이  도움이  되는지  확인

표 2의 행 2-4의  값을  통해  경계 예측이 없는  경우(No)이고  행 5-7에는 경계 예측이 있는  경우(Yes)의  최적화  평가를  알  수  있다. 결과는 ‘Yes’의  경우가  개선함을 나타낸다(Rel = 0.089(Y) vs. 0.110(N)).

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/figure6.png">

Figure 6에 정성적으로 표시된 것처럼  네트워크가  표면 법선에 잡음이 있거나 부정확한 픽셀인  경우에도  평균적으로 정확하게 예측한다는 것을 의미한다.


4. 제안된  깊이  완성  방식이  입력  깊이의  양에  얼마나  의존하는지  평가

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/figure7.png">

2단계인  최적화를  하기  전에  입력  깊이  이미지의  픽셀  수를  무작위로  마스킹하여  입력  깊이  이미지의  화질을  저하시켰다. Figure 7의 두 개의 그래프 중 왼쪽은 원래 원시 깊이 이미지에서 관측된 픽셀에 대한 깊이 정확도를 나타내는 그래프이고 오른쪽 그래프는 관측되지 않은 픽셀에 대해 깊이 정확도를 나타내는 그래프이다.
더 많은 깊이 샘플을 사용하면 결과가 계속 향상되지만 100픽셀 이후에는 별 차이가 없으며 평균 100개 깊이 샘플을 사용하여 우리 방법은 깊이의 5% 내에서 제어되는 픽셀의 70% 화살표(오차)로 10개의 깊이를 생성합니다.

#### 2. 제안된  딥러닝  방법론의  성능  평가

1. 인페인팅(Inpainting) 방법과의 성능 비교

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/table3.png">

표 3의 결과는 우리의 방법이 inpainting의 baseline을 훨씬 능가함을 보여준다(Rel=0.089(본  연구) vs. 0.103-0.151(baseline)). 심층 네트워크로 표면 법선을 예측하도록 훈련함으로써 제안된 방법은 단순한 기하학적 휴리스틱보다 강력한 데이터 기반 사전으로 깊이를 완성하는 방법을 학습한다. 

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/figure8.png">

테스트를 거친 hand-tuned approaches (Bilateral)과의 차이점은 Figure 8에서 확인할 수 있다.

3. 깊이 추정(Depth Estimation) 방법과의 성능 비교

본 논문에서 제안한 ‘색상과 깊이 정보를 이용하여 깊이를 추정하는 방법’과 ‘색상만으로 깊이를 추정하는 기존 방법(baseline)’을 비교한다. 

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/figure9.png">

정성적  비교는  Figure 9에서의  비교를  통해  확인되는데, 저자의  방법(Ours)이  이미지의  구조와 미세한 디테일을 가장 잘 재현한다는 것을 알 수 있다. 

<img width="140"  src=".gitbook/2022-spring-assets/YeoJeong_1/table4.png">

Table 4의 정량적 결과에 따르면 평가 픽셀이 깊이를 관측했는지 여부(Y/N)에 관계없이 저자의 방법이 다른 방법보다 23-40% 더 우수하다는 것을 알 수 있다. 이러한 결과는 표면 법선을 예측하는 것이 깊이 추정에 대한 유망한 접근 방식임을 시사한다.

## **5. Conclusion**

본  연구는  RGB-D 이미지의 빈  depth channel을 완성하는  딥러닝  프레임워크를  제안하였다. Input RGB-D 이미지가 주어지면 색상으로부터  완전 컨볼루션 신경망을  통해  표면 법선(surface normals)과 폐색 경계(occlusion boundaries)를  예측한 다음, input 깊이에  의해  정규화된 global  선형 최적화로 output 깊이를 도출한다.

연구는  두 가지 주요 연구 기여를 제공한다. 첫째, 표면 법선과 교합 경계가 색상에서 예측된 다음 예측결과로부터  깊이  완성(depth completion)이  해결되는  기존에  없었던  새로운  2단계 프로세스를  제안한다. 둘째, 대규모 표면 재구성에서 렌더링된 데이터에 대해  지도 교육을 통해 깊이 이미지를 완성하는 방법을 배울  수  있다.
또한, 해당  논문에서  제안된  방식이  Depth Inpainting 및 Deep Estimation 방법인  이전 baseline  방식보다 성능이 우수함을 보였다.

한개의  논문이  나오기까지  수많은  생각과  논리적  근거  그리고  실험이  필요하다는  것을  다시  한  번  깨닫게  되었다. 저자는  수학적  근거  혹은  실험으로  증명된  결과를  토대로  연구의  모든  선택을  진행했고  허투루  기준을  잡거나  특정  방식을  채택하지  않았다. 말로  설명하면  당연한  일이지만  연구를  진행하는  과정에  있는  사람에게는  쉽게  간과할  수  있는  부분이기  때문에  내가  연구를  진행할  때, 저자의  연구과정을  기억하며  논리적으로  불완전한  부분은  없는지, 부족한  실험은  없는지  되새겨보며  연구를  진행할  것임을  다짐했다.

---

## **Author Information**

* Author name: Jeong Yeo

* Affiliation: Knowledge Service Engineering

* Research Topic: To be determined

## **6. Reference & Additional materials**

* Github Implementation

* Reference
	* [공식 추가 자료](https://deepcompletion.cs.princeton.edu/)
	* [학회 발표 영상](https://www.youtube.com/watch?v=WrEKJeK-Wow)
![image](https://user-images.githubusercontent.com/100551559/164981125-5e4acba8-d9ff-422b-ba68-2cde3bfed3fa.png)


