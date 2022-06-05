**[다중 수평 시계열 예측(Multi-horizon time-series forecasting)]**
![]([https://latex.codecogs.com/svg.image?y__{t:t'}](https://latex.codecogs.com/svg.image?y__%7Bt:t%27%7D)) = ![]([https://latex.codecogs.com/svg.image?(y__{t},&space;y__{t&plus;1},&space;...&space;,&space;y__{t'}](https://latex.codecogs.com/svg.image?(y__%7Bt%7D,&space;y__%7Bt&plus;1%7D,&space;...&space;,&space;y__%7Bt%27%7D)))가 d차원 시계열 관측값이고 ![]([https://latex.codecogs.com/svg.image?y__{t:t'}](https://latex.codecogs.com/svg.image?y__%7Bt:t%27%7D)) = ![]([https://latex.codecogs.com/svg.image?](https://latex.codecogs.com/svg.image?(y__%7Bt%7D,&space;y__%7Bt&plus;1%7D,&space;...&space;,&space;y__%7Bt%27%7D))가 주어지면, multi-horizon 시계열 예측은 ![]([https://latex.codecogs.com/svg.image?\\hat{y}__{t'&plus;1:t'&plus;H}](https://latex.codecogs.com/svg.image?%5C%5Chat%7By%7D__%7Bt%27&plus;1:t%27&plus;H%7D))인 미래 값을 예측한다. 이는 H x d 차원이고 H는 예측할 steps의 수(예측 horizon)이다.

중요한 응용 프로그램의 경우 예측과 관련된 불확실성에 관심이 있다. 예측 범위의 각 시간 단계(h)에 대해
ground truth 값 yt+h가 충분히 높은 확률로 ![]([https://latex.codecogs.com/svg.image?[{\\hat{y}](https://latex.codecogs.com/svg.image?%5B%7B%5C%5Chat%7By%7D)}*{t+h}^{L},&space;{\hat{y}}*{t+h}^{U}]), h ∈ {1, . . . , H} 구간에 포함되도록 한다. 전체 시계열 궤적의 실제 값이 간격 내에 포함되도록 원하는 유의 수준(또는 오류율) α를 수정한다. 

즉, 아래의 식을 만족하게 한다.
