**[다중 수평 시계열 예측(Multi-horizon time-series forecasting)]**


![](https://latex.codecogs.com/svg.image?y__{t:t'}) = ![](https://latex.codecogs.com/svg.image?(y__%7Bt%7D,&space;y__%7Bt&plus;1%7D,&space;...&space;,&space;y__%7Bt%27%7D))가 d차원 시계열 관측값이고 ![](https://latex.codecogs.com/svg.image?y__{t:t'}) = ![]([https://latex.codecogs.com/svg.image?](https://latex.codecogs.com/svg.image?(y__%7Bt%7D,&space;y__%7Bt&plus;1%7D,&space;...&space;,&space;y__%7Bt%27%7D))가 주어지면, multi-horizon 시계열 예측은 ![]([https://latex.codecogs.com/svg.image?\\hat{y}__{t'&plus;1:t'&plus;H}](https://latex.codecogs.com/svg.image?%5C%5Chat%7By%7D__%7Bt%27&plus;1:t%27&plus;H%7D))인 미래 값을 예측한다. 이는 H x d 차원이고 H는 예측할 steps의 수(예측 horizon)이다.
