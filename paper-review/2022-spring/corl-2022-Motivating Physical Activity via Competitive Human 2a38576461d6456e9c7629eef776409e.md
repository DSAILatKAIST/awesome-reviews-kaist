---
description : Boling Yang / Motivating Physical Activity via Competitive Human-Robot Interaction / CORL-2021
---

# Motivating Physical Activity via Competitive Human-Robot Interaction

## 1. Problem Definition

이 논문에서는 사람-로봇 간의 경쟁 환경에서 사람과 비슷한 수준의 로봇을 만드는 것을 목표로 한다. multi-agent RL을 통해서 로봇의 policy를 도출하려고 하였는데, 그러기 위해서는 학습 단계에서 사람과 비슷한 수준의 수준의, 다양한 전략을 가진 로봇이 필요하다.

이 논문에서는 다양한 policy를 가진 로봇을 도출하기 위한 방법을 제안하였고, 다양하게 policy가 도출 되었는지를 분석하였다.

## 2. Motivation

사람은 몇몇 예시만 보고도 빠르게 패턴을 인식하고, 기술을 습득한다. 기존 몇몇 연구에서는 사람들이 Human-Robot Interaction 문제에서 빠르게 로봇에 적응해서 좋은 성능을 보이는 것을 보여주었다. 그러나, 사람이 로봇의 행동을 쉽게 예측하고, 빠르게 optimal counter-strategy를 찾아낸다면, 로봇을 상대로 하는 게임은 별로 어렵지 않다. 그래서 이 논문에서는 게임을 어렵게 하기 위해, 로봇의 게임 스타일을 다양하게 만드려고 시도했다.

## 3. Method

- Learning to Compete

이들은 agent의 학습을 2개의 phase로 나누어서 진행하였다. 전체적인 알고리즘은 아래 그림과 같다.

![Untitled](Motivating%20Physical%20Activity%20via%20Competitive%20Human%202a38576461d6456e9c7629eef776409e/Untitled.png)

phase1 - Learning to Move and Play: 첫 번째 phase는 pre-training process와 같다. motor skill을 이용해서 움직이는 것과 게임의 규칙을 학습하는 과정이라고 생각하면 된다. 여기서는 양 쪽의 에이전트중 한쪽의 에이전트만 수렴할 때까지 학습이 된다. 이후 반대의 에이전트가 수렴할 때까지 학습이 된다. 이들은 이 과정을 2회 진행하였다. phase1이 끝날때의 policy를 warm-start policy라고 부른다.

phase2 - Creating Characterized Policies: multi-agent system에서 각 agent는 서로 경쟁하면서 학습될 때, 매우 달라지는 전략과 행동이 학습된다. 두 번째 phase에서는 이 사실을 이용해서 랜덤한 특성을 가진 전략을 만들어내는 것이 목적이다. 각 agent는 warm-start policy에서 시작되어 서로 번갈아가면서 학습이 된다. 하지만, 학습할 때, 상대방은 phase 2 안에서 생성된 policy중 랜덤하게 선택이 된다. 

- Selecting Agents With Distinct Gameplay Styles.

학습이 종료된 뒤, 사람과 플레이할때 사용할 policy들을 정해야한다. 이들은 phase2에서 저장된 parameter들중 다른 스타일을 가진 policy를 선택하려고 policy를 분석하는 방법을 다음과 같이 적용하였다: 우선, 랜덤으로 6개의 기준이 되는 policy를 선정하였다. 그리고, 모든 policy를 이 6개의 policy와 100회씩 겨룬 600개의 게임 데이터를 평균낸 값을 각 poliy의 특성이라고 생각하였다. 이때 게임 데이터는 게임 시작과 끝의 x, y, z 차이, 평균 속도, 평균 가속도 등등이 포함되어 있다. 이렇게 얻어낸 policy 특성을 PCA를 통해서 가장 separable한 policy 3개를 선정하였다.

## 4. Experiments

- Dataset

Mujoco 환경을 이용한 펜싱 게임

- Baseline

Warm-start policy, selected policy 1, 2, 3.

- Evaluation Metric

game score

저자들은 선택한 3개의 policy와 warm-start policy를 이용해서 사람과 테스트를 해보았는데, 예상과 다르게, 로봇의 전략을 바꾸는 것이 사람이 학습하는걸 어렵게 하지 않았다고 한다. 

대신 이들은 앞서 3개의 전략을 얻어낼때 사용한 방식으로 사람들의 정책을 분석해 보았는데, 다양한 시도를 한 사람일수록 더 높은 점수를 얻는 현상을 관측했다고 이야기한다. (점수 편차가 큰 사람들은 위험을 감수하며 다양한 전략을 썼고, 장기적으로 더 좋은 보상을 얻었다고 한다.)

![Untitled](Motivating%20Physical%20Activity%20via%20Competitive%20Human%202a38576461d6456e9c7629eef776409e/Untitled%201.png)

## 5. Conclusions

이 논문에서는 경쟁적인 HRI에서 사람과 경쟁할만한 로봇을 만들기 위한 방법을 제안하였다. 제안된 방법은 다양한 경쟁 시나리오에서 적용될 수 있을 것 같다. 처음 생각한것과는 다르게, learning 외적인 실험, 분석이 많아서 조금 아쉬운 논문 같다.

## Author Information

- Kanghoon Lee

## 6. Reference

- YANG, Boling, et al. Motivating Physical Activity via Competitive Human-Robot Interaction. In: *Conference on Robot Learning*. PMLR, 2022. p. 839-849.
