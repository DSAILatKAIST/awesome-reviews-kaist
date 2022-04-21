---
description : Weng et al./ FaceSight; Enabling Hand-to-Face Gesture Interaction on AR Glasses with a Downward-Facing Camera Vision / Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems
---

# **FaceSight: Enabling Hand-to-Face Gesture Interaction on AR Glasses with a Downward-Facing Camera Vision** 

[소개할 연구](https://doi.org/10.1145/3411764.3445484)는 2021년에 CHI (Conference on Human Factors in Computing Systems)에서 발표되었으며,  
AR 환경에서 사람과 컴퓨터 간 상호작용을 돕는 새로운 제스처 인식 기술을 소개하고자 합니다.  
_keyword: Hand-to-Face Gestures; AR Glasses; Computer Vision_

<br>

## **1. Problem Definition**  

본 연구에서는 AR 안경을 활용하여 **hand-to-face** 제스처를 감지하는, 컴퓨터 비전 기반의 기술인 **FaceSight** 를 소개합니다.
**FaceSight**는 적외선 카메라를 AR 안경 다리에 고정하여 얼굴 아랫부분(뺨, 코, 입, 턱)에서의 손 제스처를 감지하는 기술입니다.

★★★★★Figure1 여기엥...사진넣는법...배워야함....

손으로 뺨을 두드리는 것과 같은 **hand-to-face** 제스처 상호작용은 아래의 장점을 갖고 있습니다
- _자신의 신체를 활용한다는 점에서_ 언제든 사용 가능하며, 촉각적이며, 거부감이 덜합니다
- 얼굴을 상호작용의 매개로 사용한다는 점에서 직관적이고, 배우기 쉬우며, 넓은 공간 활용이 가능합니다
- 

<br>

## **2. Motivation**  

Please write the motivation of paper. The paper would tackle the limitations or challenges in each fields.

After writing the motivation, please write the discriminative idea compared to existing works briefly.

#### 2.1 Hand-to-Face Interaction을 활용한 기존 연구들  

[1] 촉각정보 기반 상호작용

 - [Serano et al.](https://doi.org/10.1145/2556288.2556984)는 일상적인 작업을 할 때 뺨과 이마를 활용해서 hand-to-face 상호작용을 할 수 있음을 보였습니다.  
 - [Lee et al.](https://doi.org/10.1145/3242587.3242642)은 얼굴에서 멀리 떨어진 귀, 목을 활용하면 hand-to-face 상호작용 설계에 적합할 수 있으며, 5개의 손을 모두 사용하는 큰 움직임보다 소수의 손가락을 사용하는 작은 움직임이 입력에 적합할 수 있음을 제안했습니다.

★★★★★사진 넣기...어떻게...넣지...?ㅠㅠ

 - [Itchy Nose](https://doi.org/10.1145/3242587.3242642)는 안경에 전극센서(electrooculography)를 부착해서 코를 밀거나 문지르는 제스처를 인식하도록 개발된 hand-to-nose 상호작용 기술입니다. 
 - 비슷하게 [CheekInput](https://doi.org/10.1145/3139131.3139146), [FaceRubbing](https://doi.org/10.1145/3174910.3174924) 등 모자나 안경에 센서를 부착해서 얼굴을 당기고, 문지르는 제스처를 인식하는 기술도 개발되었습니다.

★★★★★사진 넣기

[2] 청각정보 기반 상호작용

 - [PrivateTalk](https://doi.org/10.1145/3332165.3347950)은 음성 데이터를 비교해서 사람이 손으로 입을 가리고 속삭이는 소리를 감지했고, [EarBuddy](https://doi.org/10.1145/3313831.3376836)는 뺨에 손을 두드리는 소리를 감지했습니다.

★★★★★사진 넣기

<br>

#### 2.2 Discriminative idea compared to existing works

AR 안경을 활용한 이전 연구들은 촉각적이거나 청각적인 정보를 기반으로 제스처를 감지하는 연구들을 대부분 수행했습니다.  
 - 그러나 기존 촉각 기반의 hand-to-face 상호작용 기술들은 부자연스러울 정도로 과한 제스처만을 인식했으며, 얼굴을 가볍게 쓸어내리는 등의 가벼운 제스처는 인식할 수 없었습니다.  
 - 소리 기반의 hand-to-face 상호작용 기술 역시 감지 가능한 제스처의 종류, 개수가 제한적이었으며, 뺨을 두드리는 횟수 등의 간단한 제스처만 인식할 수 있었습니다.  

기존 연구에서는 제스처가 단순하거나 제한되므로, AR 안경 상호작용을 풍부하게 하기 위해서는 다양한 제스처 인식 기술이 필요합니다.  

이러한 Research Gap을 줄이기 위해 본 연구에서는 AR 안경 다리에 적외선카메라를 접목해서 착용자의 얼굴을 캡처하는 **FaceSight** 를 개발하였습니다.  
대부분의 기존 감지기술과 달리, **카메라를 활용한 시각정보 기반**의 상호작용 기술은 좀 더 다양하고 복잡한 제스처를 인식할 수 있게 해줍니다.  
본 연구에서는 AR 안경을 착용하여 시각정보를 기반으로 여러 종류의 제스처를 인식할 수 있는 기술을 개발하였습니다.

★★★★★Figure1 여기엥

>개발된 FaceSight는 아래의 3가지 이점을 갖고 있습니다:
 1) 사용자의 얼굴, 손을 고해상도 이미지로 캡쳐 → 풍부하고 섬세한 hand-to-face gesture 감지 가능
 2) 적외선 광원의 광도를 조절해서 어두운 배경의 전경(얼굴 아랫부분, 손)만 조명 가능 → 컴퓨터 비전 프로세스 단순화, 프라이버시 문제 완화
 3) AR 안경 다리에 카메라를 부착하면 소형 폼팩터 역할을 함 → 웨어러블 기기의 실용성 증대

>본 연구는 아래의 3가지 contribution 요소를 갖고 있습니다:
 1) 시각정보 기반의 제스처 감지 기술인 **FaceSight**개발
 2) 구분 가능한 21개의 제스처 종류 개발
 3) hand-to-face gesture를 감지하기 위한 알고리즘 파이프라인 설계 및 구현


<br>

## **3. Method**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  

<br>

#### 3.1 FaceSight 구성요소  

★★★★★Figure2 여기엥

FaceSight는 위 그림과 같이 AR안경의 안경코 부근에 비디오 카메라를 장착하는 것입니다. 본 연구에서는 [Nreal Light](https://www.nreal.ai/) AR 안경을 사용했습니다.  
그리고 안경에 광각 카메라를 장착하여 사용자의 얼굴 아랫부분(뺨, 코, 입, 턱)을 인식하며, 인식 범위는 아래와 같습니다.

★★★★★Figure3 여기엥

카메라 렌즈 주변에는 적외선 전구 6개가 조명원을 제공하여 완전히 어두운 환경에서도 제스처를 인식할 수 있도록 구성했습니다.  
또한, 적외선 조명값과 카메라의 노출값을 조정함으로써 얼굴 아랫부분만 안정적으로 이미지를 수집할 수 있습니다. 아래 그림은 조명값 및 노출값 설정에 따른 이미지 효과를 보여줍니다.  
이러한 조정은 감지 알고리즘을 강건하고 효율적으로 만들뿐 아니라 발생 가능한 개인 정보 보호 문제를 완화하게 됩니다.

★★★★★Figure4 여기엥  


#### 3.2 hand-to-face 제스처 상호작용  

FaceSight에서 지원할 수 있는 hand-to-face 제스처 상호작용에 대하여 살펴보겠습니다. 본 연구에서는 총 21개의 구분 가능한 제스처를 제시하고 있습니다.
 - hand-to-cheek 제스처 : 7개
 - hand-to-nose 제스처 : 6개
 - hand-to-mouth 제스처 : 4개
 - hand-to-chin 제스처 : 4개

★★★★★Figure5 여기엥  

이런 hand-to-face 제스처 종류는 AR 안경에 대한 입력 방법을 풍부하게 할뿐 아니라 상호작용에서의 효율성을 증진하여 사용자의 경험을 향상시킬 수 있는 장점을 갖고 있습니다.

 [1] 터치 위치  
 카메라 배치를 통해 FaceSight는 뺨, 코, 입, 턱을 포함한 얼굴 아랫부분의 대부분을 구분할 수 있습니다. 1) 뺨은 왼쪽과 오른쪽으로 구분되며, 2) 코는 코끝, 왼쪽과 오른쪽 코볼로구분되며, 3) 입은 왼쪽, 가운데, 오른쪽으로 구분됩니다. 이러한 걸굴 부위는 사람들이 자주 만지고 가장 자연스럽게 사용하는 부위입니다. 다만, 카메라의 위치(AR 안경코)의 제약 때문에 귀 윗부분은 상호작용에 사용할 수없다는 제한사항을 갖고 있습니다.
 
 [2] 제스처
 tapping, swiping은 현대 터치스크린에서 가장 일반적이며 사용자들에게 굉장히 친숙한 입력 방법입니다. FaceSight에서는 한 번의 클릭, 두 번의 클릭, 긴 클릭(몇 초씩) 제스처를 사용합니다.  
 또한, 볼과 턱의 매끄러운 표면은 swiping 작업을 하기에 적합합니다. FaceSight에서는 뺨 한쪽을 수직으로 쓸어넘기기, 뺨 양쪽을 수직으로 쓸어넘기기, 턱을 수평으로 쓸어넘기는 제스처를 사용합니다.  
 더불어, 카메라를 사용하면 상징적인 제스처를 인식할 수 있다는 장점이 있습니다. 예를 들어, 검지와 새끼손가락을 귀에 붙이는 제스처로 전화를 걸거나, 검지를 입술에 붙여서 장치를 음소거하는 신호로 사용할 수 있습니다.  
 
 [3] 코 변형 및 손가락 접촉 횟수
 카메라가 코 바로 위에 있어서 손가락에 의해 코가 밀리거나 움켜쥐었을 때 코의 미세한 변형을 감지할 수 있습니다. 예를 들어, FaceSight에서는 코를 부드럽게 누르는 동작과 코를 강하게 눌러서 일그러지는 동작을 구분할 수 있습니다.
 손가락이 얼굴에 접촉하는 개수를 각각 다른 제스처로 인식하는 것도 현대 터치스크린에서 널리 사용되는 상호작용 기법입니다. 예를 들어, FaceSight에서는 턱에 1개의 손가락이 접촉하는 것과 2개의 손가락이 동시에 접촉하는 제스처를 구분합니다.
 

<br>

## **4. Experiment : 동작 감지 알고리즘**  

In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

FaceSight에서 상술한 제스처 종류들을 인식하고 구분하기 위한 알고리즘 파이프라인에 대하여 설명하고, 정확도와 계산 효율성을 평가해보겠습니다.

<br>

#### 4.1 알고리즘 파이프라인  

아래 그림은 FaceSight의 인식 파이프라인을 단계별로 보여줍니다.

★★★★★Figure6 여기엥  

적외선 카메라에서 캡처한 gray-scale 이미지가 수집되면, 먼저 여러 밝기 feature를 적용하여 1) 손, 코, 입, 뺨을 구분(segmentation)합니다. 그 다음 hand-to-face 제스처를 감지하기 위한 4단계 알고리즘이 수행됩니다: 2) Detection of touch contact, 3) recognizing touch location (1에서 촉각을 감지한 경우), 4) gesture classification with CNN, 5) Locating the touching fingertip for continuous input (제스처가 nose pushing이거나 cheek/chin tapping인 경우). 단계별 자세한 과정은 아래에서 설명드리겠습니다.

 [1] 손, 코, 입, 뺨 구분(Segmentation)
 FaceSight는 카메라와 조명 설정을 통해 배경(가슴 또는 다른 물체)과 전경(얼굴 아랫부분, 손)을 구분할 수 있습니다. 먼저밝기 임계값을 적용하여 배경을 제거합니다.


<br>

### **Experiment setup**  
* Dataset  
* baseline  
* Evaluation Metric  

<br>

### **Result**  
Then, show the experiment results which demonstrate the proposed method.  
You can attach the tables or figures, but you don't have to cover all the results.  
  


<br>

## **5. Conclusion**  

Please summarize the paper.  
It is free to write all you want. e.g, your opinion, take home message(오늘의 교훈), key idea, and etc.

<br>

---  
## **Author Information**  

* Author name  
    * Affiliation  
    * Research Topic

<br>

## **6. Reference & Additional materials**  

Please write the reference. If paper provides the public code or other materials, refer them.  

* Github Implementation  
* Reference  

