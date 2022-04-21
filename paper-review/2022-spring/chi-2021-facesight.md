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

(요즘 AR안경이 연구/산업계에 어떻게 활용되는지 추가하면?)

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

개발된 FaceSight는 아래의 3가지 이점을 갖고 있습니다:
 1) 사용자의 얼굴, 손을 고해상도 이미지로 캡쳐함 → 풍부하고 섬세한 hand-to-face gesture 감지 가능
 2) 적외선 광원의 광도를 조절해서 어두운 배경의 전경(코, 볼, 손)만 조명 가능 → 컴퓨터 비전 프로세스 단순화, 프라이버시 문제 완화
 3) AR 안경 다리에 카메라를 부착하면 소형 폼팩터 역할을 함 → 웨어러블 기기의 실용성 증대

본 연구는 아래의 3가지 contribution 요소를 니다:
 1) 시각정보 기반의 제스처 감지 기술인 **FaceSight**개발
 2) 구분 가능한 21개의 제스처 종류 개발
 3) hand-to-face gesture를 감지하기 위한 알고리즘 파이프라인 설계 및 구현


<br>

## **3. Method**  

Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  

<br>

## **4. Experiment**  

In this section, please write the overall experiment results.  
At first, write experiment setup that should be composed of contents.  

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

