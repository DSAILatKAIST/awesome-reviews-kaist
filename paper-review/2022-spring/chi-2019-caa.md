---
description : Tag et al. / Continuous Alertness Assessments; Using EOG Glasses to Unobtrusively Monitor Fatigue Levels In-The-Wild / CHI-2019 conference-year
---

# **Continuous Alertness Assessments** 

[소개할 연구](https://doi.org/10.1145/3290605.3300694)는 2019년에 CHI (Conference on Human Factors in Computing Systems)에서 발표되었으며,  
하루 동안의 눈 깜빡임 빈도를 EOG (Electrooculography)로 측정함으로써 간편하게 대상자의  **`Continuous Alertness Assessments`** 를 모니터링하는 기술을 소개하고자 합니다.  
_keyword: Cognition-Aware Systems, Circadian Computing, Fatigue, Eye Blink, Electrooculography_
  
<br>

## **1. Problem Definition**  
> 피로 수준 모니터링 시스템의 필요성

사람의 집중력과 업무 능력은 24시간 일주기와 관련이 깊기 때문에, 일주기 리듬이 생물학적 사이클에서 벗어날 경우 심각한 건강 문제를 초래할 수 있습니다.  
예를 들어 오랫동안 쉬지 않고 업무를 지속할 경우 실수와 사고를 일으킬 가능성이 증가하며, 이는 교대근무와 장기간 근무가 필요한 조종사, 의료인들에게서 흔히 나타납니다.

수면부족 및 피로감은 업무 효율에 악영향을 미치며, 추론능력, 작업기억 등의 정신기능에도 영향을 줍니다.  
피로감을 스스로 인지하기 어렵기 때문에 **피로 수준을 감지하고 예측할 수 있는 자동화된 시스템** 개발이 필요합니다.  

본 리뷰에서는 상용화된 안경을 활용하여 피로 수준을 모니터링하는, EOG 센서 기반의 **`Continuous Alertness Assessments`** 시스템을 소개하려 합니다.

★![J!NS MEME glasses](https://이미지링크.png)

**`Continuous Alertness Assessments`** 시스템은 아래의 장점을 갖고 있습니다:
 * 피로 수준에 따른 업무 순서 배치로 효율성 증대 가능
 * 사용자에게 휴식이나 수면이 필요한 경우 알림 생성
  
<br>

## **2. Motivation**  
> 불편하지 않고 정확한, EOG 안경 기반의 피로 수준 모니터링 시스템 개발

### **2.1 피로 수준을 측정하는 기존 연구들 및 한계점**  

<br>

  [1] 전통적인 방식의 피로 수준 측정 방법  
  * [Kleitman](https://doi.org/10.1152/ajplegacy.1923.66.1.67)는 항문체온계를 사용하여 신체 내부의 온도를 측정하여 의식의 변화를 측정하였습니다.  
  * [Hofstra and Weerd](https://doi.org/10.1016/j.yebeh.2008.06.002)는 수면 중인 사람을 대상으로 뇌파, 심전도 등을 측정하여 피로 수준을 파악하는 수면다원검사를 소개하였습니다.

★★★항문체온계, 수면다원검사 그림 필요함  

<br>

  [2] 스마트폰을 활용한 피로 수준 측정 방법  
  * [Abdullah et al.](https://doi.org/10.1145/2971648.2971712)은 스마트폰을 사용하여 사용자의 Continuous Alertness를 모니터링하는 시스템을 개발하였습니다.
  * [Dingler et al.](https://doi.org/10.1145/2968219.2968565)은 스마트폰 기반의 집중력 수준 감시 시스템을 개발하여 생산 효율성을 증대하였습니다.
  
★★★스마트폰 그림 필요함  

<br>

피로 수준을 측정한 기존 연구들은 대부분 한계점을 갖고 있었습니다.  
  * 전통적인 방식의 피로 수준 측정 방법은 사용자에게 불쾌감을 주거나 번거로운 장치를 착용해야 했습니다.
  * 스마트폰을 활용한 방법도 스마트폰을 사용할 때만 데이터를 측정할 수 있다는 단점이 있거나, 주의가 스마트폰에 집중되어 실제 하고 있는 업무에 대한 집중도가 떨어지게 됩니다.  

이러한 단점을 보완하고 실제 측정값의 정확도를 높이기 위해, 안구 움직임을 측정 함으로서 불필요하게 관심을 끌지 않으면서 지속적으로 모니터링을 할 수 있는 시스템의 개발이 필요합니다.

<br>

### **2.2 기존 한계점을 보완한 EOG 안경 기반의 피로 수준 측정 시스템**  

이러한 Research Gap을 줄이기 위해 본 연구에서는 EOG 센서가 달린 안경을 활용하여 피로 수준을 측정하는 **`Continuous Alertness Assessments`** 를 개발하였습니다.  

EOG를 통해 얻을 수 있는 안구 운동, 눈깜박임 데이터를 활용하여 피로 수준에 대한 모니터링을 가능하게 해줍니다.  

★★★EOG안경 그림 필요함  

> 개발된 피로 수준 측정 시스템은 아래의 이점을 갖고 있습니다:
  * 센서를 번거로운 방법으로 부착해야 하는 전통적인 방식과 달리, EOG 센서를 안경에 부착함으로써 편안한 방식으로 피로 수준을 측정하는 것이 가능합니다.  
  * 주간/야간, 실내/실외 등 빛의 변화에 관계 없이 피로 수준을 측정할 수 있습니다

> 본 연구는 아래의 3가지 contribution 요소를 갖고 있습니다:
  * 2주 간의 실증 실험을 통해 피로 수준과 눈깜박임 횟수 사이에 관련성이 있음을 제시함
  * EOG 센서 데이터와 그에 따른 눈깜박임 빈도를 활용하여 피로도 수준 변화를 예측하는 모델을 제시함
  * 16명의 피실험자가 시행한 EOG 데이터의 dataset을 제시함

<br>

## **3. Method**  

### **3.1 EOG 안경 기반의 피로 수준 모니터링 시스템 **  



### **3.2 피로 수준 측정 Toolkit**  

★★★toolkit 그림 필요함  


Please write the methodology author have proposed.  
We recommend you to provide example for understanding it more easily.  
  
<br>

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
  
  
<br>


## **5. Conclusion**  
> Summarize the Papaer  
  * 본 연구에서는 상용화된 안경인 `J!NS MEME glasses`에 EOG센서를 부착하여 피로 수준을 모니터링하는, 착용하기 쉬운 시스템을 개발하였습니다.
  * 개발된 시스템은 일상 생활 도중의 눈 깜박임 빈도를 측정함으로써 하루 동안의 피로 수준을 확인할 수 있습니다.
  * 본 연구의 의의는, 더 이상 복잡한 장비를 착용할 필요 없이 상용화된 안경을 착용하는 것만으로 일상생활에서의 피로 수준을 측정하는 시스템을 개발하였다는 것에 있습니다.

> My opinion, Take home message  
  * Human-Computer Interaction (HCI) 분야에서 **안경을 사용하여 신체 및 인지 데이터를 수집하는 연구**들이 활발하게 이뤄지고 있습니다. 기존에 리뷰했던 [FaceSight](https://dsail.gitbook.io/isyse-review/paper-review/2022-spring-paper-review/chi-2021-facesight) 연구도 안경을 활용하여 사용자들의 제스처 데이터를 수집한 바 있습니다.
  ![FaceSight](https://github.com/bananaorangel/awesome-reviews-kaist/raw/2022-Spring/.gitbook/2022-spring-assets/Haehyunlee_1/fig1_facesight.PNG?raw=true)
  * 안경을 활용하여 데이터를 수집하는 경우, 센서들을 신체에 직접 부착해야 하는 불편함이나 번거로움이 적으며 **일상생활에서 사용하기 편리**합니다.  
  이러한 장점들을 기반으로 앞으로도 HCI 분야에서 안경을 사용한 연구들이 활발히 이뤄질 것으로 보입니다.
  * 다만 안경 기반의 연구들의 공통적인 문제점인, **착용 위치에 따른 데이터 노이즈 문제**가 고려되어야 할 것입니다. 동일한 사람이 안경을 다른 위치의 코와 귀에 착용할 경우 안경에서 수집된 데이터에 노이즈가 발생하여 잘못된 알람을 피드백하게될 수 있습니다.  
  따라서 안경 착용 위치가 바뀌더라도 일관적인 결과를 도출할 수 있도록 하는 normalize방법이 중요할 것입니다.
  * 여러 기술들이 개발되고 있지만, **신기술에 취약한 계층들(예, 노인, 장애인 등)** 도 쉽고 편리하게 사용할 수 있도록 이들을 대상으로 한 실험도 활발하게 이뤄져야 할 것으로 보입니다. 대부분의 실험들이 젊은 사람을 대상으로 이뤄지지만, 추후에는 취약계층을 대상으로 한 실험도 증가되어야 할 것으로 보입니다.

  
<br>

---  
## **Author Information**

* Haehyun Lee
  * Affiliation : PhD course in KAIST KSE program
  * Research Topic : Human-Computer Interaction, Human Factors in Nuclear Power Plant
  * Contact email : haehyun_lee@naver.com
  
<br>

## **6. Reference & Additional materials**  
* 논문 원문 : [Continuous Alertness Assessments](https://doi.org/10.1145/3290605.3300694)
* Github Implementation  
  * [Fatigue_EOG_Raw](https://github.com/tagbenja/Fatigue_EOG_Raw)
* Reference  
  * 안경 : [J!NS MEME glasses](https://jinsmeme.com/en/)
  * 모바일 toolkit : [Dingler](https://doi.org/10.1145)
