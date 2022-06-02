---
description: >-
  Weng et al./ FaceSight; Enabling Hand-to-Face Gesture Interaction on AR
  Glasses with a Downward-Facing Camera Vision / Proceedings of the 2021 CHI
  Conference on Human Factors in Computing Systems
---

# FaceSight

[소개할 연구](https://doi.org/10.1145/3411764.3445484)는 2021년에 CHI (Conference on Human Factors in Computing Systems)에서 발표되었으며,\
AR 환경에서 사람과 컴퓨터 간 상호작용을 돕는 새로운 제스처 인식 기술을 소개하고자 합니다.\
_keyword: Hand-to-Face Gestures; AR Glasses; Computer Vision_

\


## **1. Problem Definition**

> AR안경 기반의 제스처 감지기술

AR(Augmented Reality) 기술이 점점 발전함에 따라, 사용자가 AR 환경에서 컴퓨터와 상호작용하는 기술에 대한 중요도가 증가하고 있습니다.\
마우스나 키보드와 같은 입력장치를 통해 컴퓨터와 상호작용하던 기존의 고정된 환경과 달리,\
AR환경에서 손, 얼굴 등 신체를 활용하여 움직이면서 컴퓨터와 상호작용하는 기술이 점점 발전하고 있습니다.

본 리뷰에서는 AR 안경을 활용하여 **hand-to-face** 제스처를 감지하는, 컴퓨터 비전 기반의 기술인 **`FaceSight`** 를 소개하려 합니다.\
**FaceSight**는 적외선 카메라를 AR 안경 다리에 고정하여 얼굴 아랫부분(뺨, 코, 입, 턱)에서의 손 제스처를 감지하는 기술입니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/fig1\_facesight.PNG?raw=true)\


손으로 뺨을 두드리는 것과 같은 **hand-to-face** 제스처 상호작용은 아래의 장점을 갖고 있습니다.

* _자신의 신체를 활용한다는 점에서_ 언제든 사용 가능하며, 촉각적이며, 거부감이 덜합니다
* 얼굴을 상호작용의 매개로 사용한다는 점에서 직관적이고, 배우기 쉬우며, 넓은 공간 활용이 가능합니다

\
\
\


## **2. Motivation**

### 2.1 Hand-to-Face Interaction을 활용한 기존 연구들 및 한계점

\


\[1] 촉각정보 기반 상호작용

* [Serano et al.](https://doi.org/10.1145/2556288.2556984)는 일상적인 작업을 할 때 뺨과 이마를 활용해서 hand-to-face 상호작용을 할 수 있음을 보였습니다.
* [Lee et al.](https://doi.org/10.1145/3242587.3242642)은 얼굴에서 멀리 떨어진 귀, 목을 활용하면 hand-to-face 상호작용 설계에 적합할 수 있으며, 5개의 손을 모두 사용하는 큰 움직임보다 소수의 손가락을 사용하는 작은 움직임이 입력에 적합할 수 있음을 제안했습니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/serano,lee.png?raw=true)\


* [Itchy Nose](https://doi.org/10.1145/3123021.3123060)는 안경에 전극센서(electrooculography)를 부착해서 코를 밀거나 문지르는 제스처를 인식하도록 개발된 hand-to-nose 상호작용 기술입니다.
* 비슷하게 [CheekInput](https://doi.org/10.1145/3139131.3139146), [FaceRubbing](https://doi.org/10.1145/3174910.3174924) 등 모자나 안경에 센서를 부착해서 얼굴을 당기고, 문지르는 제스처를 인식하는 기술도 개발되었습니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/checkinput,facerubbing.png?raw=true)\


\[2] 청각정보 기반 상호작용

* [PrivateTalk](https://doi.org/10.1145/3332165.3347950)은 음성 데이터를 비교해서 사람이 손으로 입을 가리고 속삭이는 소리를 감지했고, [EarBuddy](https://doi.org/10.1145/3313831.3376836)는 뺨에 손을 두드리는 소리를 감지했습니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/privatetalk,earbuddy.png?raw=true)\


AR 안경을 활용한 이전 연구들은 촉각적이거나 청각적인 정보를 기반으로 제스처를 감지하는 연구들을 대부분 수행했습니다.

* 그러나 기존 촉각 기반의 hand-to-face 상호작용 기술들은 부자연스러울 정도로 과한 제스처만을 인식했으며, 얼굴을 가볍게 쓸어내리는 등의 가벼운 제스처는 인식할 수 없었습니다.
* 청각 기반의 hand-to-face 상호작용 기술 역시 감지 가능한 제스처의 종류, 개수가 제한적이었으며, 뺨을 두드리는 횟수 등의 간단한 제스처만 인식할 수 있었습니다.

기존 연구에서는 제스처가 단순하거나 제한되므로, AR 안경 상호작용을 풍부하게 하기 위해서는 다양한 제스처 인식 기술이 필요합니다.

\


### 2.2 기존 한계점을 보완한 시각정보 기반의 FaceSight 개발

이러한 Research Gap을 줄이기 위해 본 연구에서는 AR 안경 다리에 적외선카메라를 접목해서 착용자의 얼굴을 캡처하는 **FaceSight** 를 개발하였습니다.\
대부분의 기존 감지기술과 달리, **카메라를 활용한 시각정보 기반**의 상호작용 기술은 좀 더 다양하고 복잡한 제스처를 인식할 수 있게 해줍니다.\
본 연구에서는 AR 안경을 착용하여 시각정보를 기반으로 여러 종류의 제스처를 인식할 수 있는 기술을 개발하였습니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/fig1\_facesight.PNG?raw=true)\


> 개발된 FaceSight는 아래의 3가지 이점을 갖고 있습니다:

* 사용자의 얼굴, 손을 고해상도 이미지로 캡쳐 → 풍부하고 섬세한 hand-to-face gesture 감지 가능
* 적외선 광원의 광도를 조절해서 어두운 배경의 전경(얼굴 아랫부분, 손)만 조명 가능 → 컴퓨터 비전 프로세스 단순화, 프라이버시 문제 완화
* AR 안경 다리에 카메라를 부착하면 소형 폼팩터 역할을 함 → 웨어러블 기기의 실용성 증대

> 본 연구는 아래의 3가지 contribution 요소를 갖고 있습니다:

* 시각정보 기반의 제스처 감지 기술인 **FaceSight**개발
* 구분 가능한 21개의 제스처 종류 개발
* hand-to-face gesture를 감지하기 위한 알고리즘 파이프라인 설계 및 구현

\
\
\


## **3. Method**

\


### 3.1 FaceSight 구성요소

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/fig2\_ARglasses.PNG?raw=true)\


FaceSight는 위 그림과 같이 AR안경의 안경코 부근에 비디오 카메라를 장착하는 것입니다. 본 연구에서는 [Nreal Light](https://www.nreal.ai) AR 안경을 사용했습니다.\
그리고 안경에 광각 카메라를 장착하여 사용자의 얼굴 아랫부분(뺨, 코, 입, 턱)을 인식하며, 인식 범위는 아래와 같습니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/fig3\_interactionspace.PNG?raw=true)\


카메라 렌즈 주변에는 적외선 전구 6개가 조명원을 제공하여 완전히 어두운 환경에서도 제스처를 인식할 수 있도록 구성했습니다.\
또한, 적외선 조명값과 카메라의 노출값을 조정함으로써 얼굴 아랫부분만 안정적으로 이미지를 수집할 수 있습니다. 아래 그림은 조명값 및 노출값 설정에 따른 이미지 효과를 보여줍니다.\
이러한 조정은 감지 알고리즘을 강건하고 효율적으로 만들뿐 아니라 발생 가능한 개인 정보 보호 문제를 완화하게 됩니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/fig4\_highcontrast.PNG?raw=true)\


### 3.2 hand-to-face 제스처 상호작용

FaceSight에서 지원할 수 있는 hand-to-face 제스처 상호작용에 대하여 살펴보겠습니다. 본 연구에서는 총 21개의 구분 가능한 제스처를 제시하고 있습니다.

* hand-to-cheek 제스처 : 7개
* hand-to-nose 제스처 : 6개
* hand-to-mouth 제스처 : 4개
* hand-to-chin 제스처 : 4개

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/fig5\_gestures.PNG?raw=true)\


이런 hand-to-face 제스처 종류는 AR 안경에 대한 입력 방법을 풍부하게 할뿐 아니라 상호작용에서의 효율성을 증진하여 사용자의 경험을 향상시킬 수 있는 장점을 갖고 있습니다.

_\[1] 터치 위치_\
카메라 배치를 통해 FaceSight는 뺨, 코, 입, 턱을 포함한 얼굴 아랫부분의 대부분을 구분할 수 있습니다. 1) 뺨은 왼쪽과 오른쪽으로 구분되며, 2) 코는 코끝, 왼쪽과 오른쪽 코볼로구분되며, 3) 입은 왼쪽, 가운데, 오른쪽으로 구분됩니다. 이러한 걸굴 부위는 사람들이 자주 만지고 가장 자연스럽게 사용하는 부위입니다. 다만, 카메라의 위치(AR 안경코)의 제약 때문에 귀 윗부분은 상호작용에 사용할 수없다는 제한사항을 갖고 있습니다.

_\[2] 제스처_\
tapping, swiping은 현대 터치스크린에서 가장 일반적이며 사용자들에게 굉장히 친숙한 입력 방법입니다. FaceSight에서는 한 번의 클릭, 두 번의 클릭, 긴 클릭(몇 초씩) 제스처를 사용합니다.\
또한, 볼과 턱의 매끄러운 표면은 swiping 작업을 하기에 적합합니다. FaceSight에서는 뺨 한쪽을 수직으로 쓸어넘기기, 뺨 양쪽을 수직으로 쓸어넘기기, 턱을 수평으로 쓸어넘기는 제스처를 사용합니다.\
더불어, 카메라를 사용하면 상징적인 제스처를 인식할 수 있다는 장점이 있습니다. 예를 들어, 검지와 새끼손가락을 귀에 붙이는 제스처로 전화를 걸거나, 검지를 입술에 붙여서 장치를 음소거하는 신호로 사용할 수 있습니다.

_\[3] 코 변형 및 손가락 접촉 횟수_\
카메라가 코 바로 위에 있어서 손가락에 의해 코가 밀리거나 움켜쥐었을 때 코의 미세한 변형을 감지할 수 있습니다. 예를 들어, FaceSight에서는 코를 부드럽게 누르는 동작과 코를 강하게 눌러서 일그러지는 동작을 구분할 수 있습니다.\
손가락이 얼굴에 접촉하는 개수를 각각 다른 제스처로 인식하는 것도 현대 터치스크린에서 널리 사용되는 상호작용 기법입니다. 예를 들어, FaceSight에서는 턱에 1개의 손가락이 접촉하는 것과 2개의 손가락이 동시에 접촉하는 제스처를 구분합니다.

\
\
\


## **4. Experiment : 동작 감지 알고리즘**

FaceSight에서 상술한 제스처 종류들을 인식하고 구분하기 위한 알고리즘 파이프라인에 대하여 설명하고, 정확도와 계산 효율성을 평가해보겠습니다.

\


### 4.1 알고리즘 파이프라인

아래 그림은 FaceSight의 인식 파이프라인을 단계별로 보여줍니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/fig6\_recognitionstep.PNG?raw=true)\


적외선 카메라에서 캡처한 gray-scale 이미지가 수집되면, 먼저 여러 밝기 feature를 적용하여 손, 코, 입, 뺨을 구분(segmentation)합니다. 그 다음 hand-to-face 제스처를 감지하기 위한 4단계 알고리즘이 수행됩니다: 2) Detection of touch contact, 3) recognizing touch location (1에서 촉각을 감지한 경우), 4) gesture classification with CNN, 5) determine the required interaction parameter (제스처가 nose pushing이거나 cheek/chin tapping인 경우). 단계별 자세한 과정은 아래에서 설명드리겠습니다.

\


_\[stage0] 손, 코, 입, 뺨 구분(Segmentation)_\
FaceSight는 아랫 그림과 같이 카메라와 조명 설정을 통해 배경(가슴 또는 다른 물체)과 전경(얼굴 아랫부분, 손)을 구분할 수 있습니다. 밝기 임계값, 움직임 정보, 픽셀 강도 등의 정보를 활용하여 손, 코, 입, 뺨을 구분합니다. 먼저밝기 임계값을 적용하여 배경을 제거합니다(a). 이어서, 얼굴 영역이 정적인 반면 손 영역은 움직이며 밝기 변화를 일으킨다는 사실을 이용하여 얼굴과 손을 구분합니다 (b). 코는 항상 영상의 중앙에 위치하며 하단에 연결되어 있고 전구와 가까운 위치에 있어서 밝기가 높고(c), 입은 코 위, 얼굴 하단에 위치하고 있으며(d), 나머지 픽셀(배경, 손, 코, 입이 없는 입력 이미지)을 계산하면 왼쪽과 오른쪽 뺨(e, f)을 구분할 수 있습니다.\
아래 그림은 오른쪽 뺨을 한 번 터치했을 때의 구분(segmentation) 예시입니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/fig7\_segmentationapproach.PNG?raw=true)\


_\[stage1] Detection of touch contact_\
hand-to-face 제스처 상호작용을 위해서는 손이 얼굴에 닿는 시점을 결정하는 것이 중요하며, 이는 손과 얼굴이 겹치는지 여부를 확인함으로써 결정될 수 있습니다: 손 끝이 얼굴 영역 윗부분에 있거나 얼굴 영역 안쪽에 있을 때 접촉이 감지되며, 손끝(fingertip) 움직임이 두개의 연속 프레임 상에서 갑자기 변경될 때에도 접촉이 감지됩니다. FaceSight에서는 코의 중심과 손끝 사이의 거리를 계산하고, 거리가 연속된 두 개의 프레임에서 더 커지는 경우 접촉으로 감지하며, 이를 통해 접촉 감지의 정확도를 높입니다.

_\[stage2] recognizing touch location_\
이미지에서 손과 얼굴의 접촉이 감지외면 접촉이 발생하는 위치를 5가지 범주(코, 입, 턱, 왼쪽뺨, 오른쪽뺨)로 파악합니다. FaceSight에서는 손가락 끝이 닿는 가장 가까운 위치를 접촉 위치로 인식합니다.

_\[stage3] gesture classification with CNN_\
터치 위치가 주어지면 해당 위치에서 수행된 손 동작이 무엇인지를 convolutional neural network (CNN)을 사용하여 구분합니다. 본 연구에서는 코, 입, 턱, 왼쪽뺨, 오른쪽뺨에 대한 이미지에 대하여 CNN 모델을 별도로 훈련시켰습니다. 본 CNN 모델은 2개의 convolutional layer, 2x2 maximum pooling layer, fully connected layer를 포함합니다.\
본 연구에서는 정확도 향상을 위하여 convolutional layer의 파라미터를 조정했습니다: 첫번째 layer는 11x11 kernal size, stride step값은 5, padding값은 3; 두번째 layer는 5x5 kernal size, stride step은 1, padding값은 2. Loss function으로 softmax와 cross-entropy를 사용했으며, 정확도에 대한 지표로 accuracy rate와 false recognition rate을 활용했습니다.\
optimizer는 Adam을 사용했는데, KSE527의 Lecture 6에서 optimization 알고리즘 중에서 뭘 쓸지 모르겠을 때 Adam을 먼저 써보라는 말이 있었던 만큼 광범위하게 쓰이는 알고리즘을 본 연구에서도 사용했습니다. learning rate coefficient은 0.003이었으며, 모델 입력은 200x200의 다운샘플링된 손 영역이며, 모델 출력은 특정 제스처에 해당하는 레이블입니다.

_\[stage4.1] Locating the touching fingertip for continuous input_\
분할된 손 영역을 기반으로 얼굴 영역까지의 거리에 따라 국소 최소값(local minima)을 달성한 윤곽선 상의 점들을 손끝의 후보들로 인식하였으며, 점들 중 가장 낮은 위치에 있는 점을 손끝의 위치로 정의하였습니다. 손끝 위치를 자연스럽게 하기 위하여 instant 프레임과 이전의 두 프레임에서 얻어진 값의 평균을 최종 계산하였습니다. 터치 접촉 감지 단계(stage1)를 실행하기 전에 손끝 위치를 지정합니다.

_\[stage4.2] Estimating the Degree of Nose Deformation_\
코를 미는 동작은 코 부위의 변형이나 움직임을 유발할 수 있으므로, 코 영역의 면적(손끝으로 코볼이 눌려서 감소된 영역)과 코 중심, 코볼 키포인트의 offset을 계산합니다. 예를 들어, 면적 변화에 대해 0.00005의 가중치를, 코 중심에서의 offset은 0.02를, 코볼 키포인트에 대한 offset은 0.04를 실증적으로(empirically) 결정하였습니다. offeset thresholod 단위는 픽셀이었으며, 단위들을 더해서 합이 1.0보다 크면 코를 강하게 누르는 동작으로 인식하였습니다. 참고로, stage4.2는 stage3에서의 인식이 코를 누르는 동작으로 분류된 경우에만 활성화됩니다.

\
\


### 4.2 **Experiment setup**

#### 4.2.1 **Dataset**

> 사람을 대상으로 한 실험에서 수집된 이미지 데이터 (supervised learning)

FaceSight의 접촉 감지와 분류 정확도를 평가하기 위한 실증실험을 수행하였습니다. 본 연구에서는 10명의 참가자(여성 2명, 남성 8명)을 대상으로 하였으며, 그들의 나이는 18세부터 55세까지였습니다 (평균=27.8세). 다양한 얼굴 형태를 가진 사용자로부터 데이터를 수집하는 것을 목표로 하였으며, 안경에 익숙하도록 일상 생활에서 주로 안경을 쓰는 사람들을 대상으로 하였습니다.\
실험 참가자들은 무작위로 60회동안 24개의 제스처를 각각 수행하도록 요구되었으며, 매 번 1분씩 휴식을 보냈습니다. 모든 데이터는 비디오로 녹화되었습니다.

데이터 수집 후 총 14,440개의 hand-to-face 제스처 데이터 샘플이 수집되었습니다(10명x24개 제스처x60회=14,440). CNN 모델을 훈련하기 위한 데이터를 얻기 위해 분할 접근법을 사용하여 비디오로 녹화된 각 프레임으로부터 손 영역을 분리하였으며, 최종적으로 198572개의 이미지를 추출하였습니다. 본 연구에서는 이미지를 수동으로 검사하여 잘못된 제스처를 수행하거나 손을 비디오에서 떼는 등의 부적합한 이미지 4370개(2.2%)는 필터링되었습니다.

수집된 이미지로부터 얼굴 아랫부분(코, 입, 턱, 왼쪽뺨, 오른쪽뺨) 각각에 대한 5개의 데이터세트를 만들었으며, CNN 모델은 아래 표와 같이 각 5개의 데이터 세트에 대해 훈련되었습니다. 전체적으로 코 데이터세트에는 67553개의 이미지, 입 데이터세트에 30368개의 이미지, 턱 데이터세트에 33582개의 이미지, 왼쪽뺨 데이터세트에 51869개의 이미지, 오른쪽뺨 데이터세트에 57747개의 이미지가 있었습니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/table3\_stage3.PNG?raw=true)\


#### 4.2.2 **Evaluation Metric**

본 연구에서는 알고리즘 파이프라인에서 각 단계별 인식 정확도(recognition accuracy)와 계산 효율성(computing efficiency)을 독립적으로 평가했습니다.

인식 정확도는 accuracy, recall, precision, F1-score로 평가되었습니다.

\


* accuracy = ![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/accuracy.PNG?raw=true) \* 100
* recall = ![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/recall.PNG?raw=true) \* 100
* precision = ![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/precision.PNG?raw=true) \* 100
* F1-score= ![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/f1score.PNG?raw=true) \* 100

\


계산 효율성은 각 구성요소가 단일 프레임을 처리하는데 걸리는 평균 시간으로 평가되었습니다. 조건은 1 GTX 1080 Ti NVIDIA GPU, 12GB 메모리, Intel(R) Xeon(R) CPU, 해상도 960x540으로 동일했습니다.

참고로, 본 연구에서는 FaceSight의 성능 평가 시 다른 baseline과의 비교를 수행하지 않았습니다.

\


### **4.3 Result : Algorithm Evaluation**

#### 4.3.1 **touch contact, location 인식 정확도**

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/table1\_stage1.PNG?raw=true)\


touch contact 감지 알고리즘의 인식 정확도는 recall, precesion, F1-score으로 평가되었으며 각각 97.90%, 98.82%, 98.36%였습니다. 대부분의 false negative는 터치가 카메라 뷰의 테두리에 너무 가까워서 손가락 끝이 보이지 않거나 너무 어두워서 인식하지 못한 경우였습니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/table2\_stage2.PNG?raw=true)\


touch location 감지 알고리즘의 인식 정확도는 accuracy으로 평가되었습니다. 총 5개의 위치(코, 입, 턱, 왼쪽뺨, 오른쪽뺨)에 대한 accuracy가 평가되었으며 각각 92.67%, 94.43%, 95.12%, 95.06%, 94.22%였습니다. 전체 평균 정확도는 94.69%였습니다. 대부분의 잘못된 분류는 코-뺨, 입-턱(특히 쉿 하는 제스처나 턱을 잡는 제스처일때), 입-뺨 사이에서 일어났습니다.

#### 4.3.2 **hand-to-gesture, nose pushing/swiping 인식 정확도**

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/table3\_stage3.PNG?raw=true)\


실험 참가자들이 자기만의 독특한 방식으로 제스처를 수행할 수 있고 카메라 뷰에 약간 차이가 있을 수 있음을 고려하여, 본 연구에서는 Leave-One-Out 교차 검증법을 사용했습니다. 모든 모델은 10개의 epoch로 훈련되었으며, 코 96.18% (5 classes), 입 99.53% (3 classes), 턱 94.00% (3 classes), 왼쪽뺨 94.65% (4 classes), 오른쪽뺨 97.73% (5 classes)의 정확도를 보였습니다. 5개 모델의 전체 정확도는 96.42%였습니다.

\
![](https://github.com/bananaorangel/awesome-reviews-kaist/blob/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee\_1/table4\_stage4.PNG?raw=true)\


양쪽 코볼에 있는 서로 다른 nose-pushing 제스처를 인식하는데 94.12% (4 classes)의 정확도를 보였습니다. swiping 제스처에 대한 인식 정확도는 94.67% (6 classes)였습니다.

#### **4.3.3 계산 효율성**

segmentation 알고리즘의 경우 35ms, CNN 분류의 경우 13ms의 계산효율성을 보였습니다. 손끝 위치 확인(fingertip locating) 및 접촉 감지(contact detection)은 모두 1ms의 계산효율성을 보였습니다.

\
\


## **5. Conclusion**

* 본 연구에서는 AR환경에서 hand-to-face 상호작용을 가능하게 하는 새로운 카메라 기반 감지기술인 **FaceSight**를 개발하였습니다. **FaceSight**는 사용자의 얼굴 아랫부분을 고해상도 이미지로 캡처하여 다양하고 섬세한 제스처들을 감지할 수 있습니다. 또한, **FaceSight**의 상호 작용 가능성을 탐색하기 위해 코, 입, 턱, 뺨과 관련된 21개의 hand-to-face 제스처 종류를 제시했습니다. (10개는 본 연구에서 새로 제시된 특별한 제스처입니다)
* 맞춤형 AR 애플리케이션을 구현하기 위해서는 다양한 종류의 제스처가 입력될 수 있어야 한다는 점에서 본 연구에서 제시한 21가지 제스처들은 활용성이 더욱 높아보입니다. **FaceSight**를 통해 AR환경에서 hand-to-face 제스처 상호작용 기술이 한발짝 더 나아갈 것으로 보입니다.
* 더불어, 원자력발전소 환경에서의 인간-컴퓨터 상호작용에 대해 연구하는 입장에서, 본 연구의 **FaceSight** 기술이 원자력발전소 종사자들의 원격 작업에도 활용될 수 있을 것으로 보입니다. 일례로, 원자력발전소 내에서 컴퓨터 하드웨어(본체, 모니터, 키보드 등)를 사용할 수 없는 공정 수리 작업 중일 때 **FaceSight** 기술을 활용하여 작업자가 AR 환경에서 원자력발전소 내 컴퓨터에게 작업 명령을 실행할 수 있을 것입니다. 특히 **FaceSight**는 얼굴 아랫부분과 손에 대한 데이터만 조명하여 컴퓨터 비전 프로세스를 단순화한다는 점에서 원자력발전소에서 중요시하는 보안 문제를 완화할 수 있다는 점에서 큰 이점을 갖고 있습니다.

\


***

## **Author Information**

* Haehyun Lee
  * Affiliation : PhD course in KAIST KSE program
  * Research Topic : Human-Computer Interaction, Human Factors in Nuclear Power Plant
  * ontact email : haehyun\_lee@naver.com

\


## **6. Reference & Additional materials**

* Reference
  * 논문 원문 : [Continuous Alertness Assessments](https://doi.org/10.1145/3411764.3445484)
  * 
