# 시험 결과


## DQN 방식 시험
`삼천당제약` 2024년 분봉 OHLCV 데이터로 7, 8월분 학습용, 9, 10월분 검증 데이터로 해서 원래 소스로 학습함

### 학습 및 테스트 절차
* 학습 명령
```
python train_model.py --cuda\
  --data data/ohlcv_min_000250_07-08.csv\
  --val data/ohlcv_min_000250_09-10.csv\
  --batch-size 1024 -r OHLCV
```
* 학습 종료
```
Episode 259600: reward=9, steps=117, speed=0.0 f/s, elapsed=0:00:00
4980000: tst: {'episode_reward': 1.4479348522043591, 'episode_steps': 50.59, 'order_profits': 1.468489826933821, 'order_steps': 28.15}
4980000: val: {'episode_reward': -0.14473396025820845, 'episode_steps': 20.8, 'order_profits': -0.14498113666997411, 'order_steps': 9.11}
Episode 259700: reward=-1, steps=16, speed=0.0 f/s, elapsed=0:00:00
Episode 259800: reward=-1, steps=3, speed=0.0 f/s, elapsed=0:00:00
Episode 259900: reward=0, steps=12, speed=0.0 f/s, elapsed=0:00:00
Episode 260000: reward=0, steps=17, speed=0.0 f/s, elapsed=0:00:00
4990000: tst: {'episode_reward': 1.4853472236520715, 'episode_steps': 45.06, 'order_profits': 1.5019101096595904, 'order_steps': 23.89}
4990000: val: {'episode_reward': -0.2338862795764962, 'episode_steps': 23.21, 'order_profits': -0.23430106809191747, 'order_steps': 10.14}
Episode 260100: reward=-0, steps=20, speed=0.0 f/s, elapsed=0:00:00
Episode 260200: reward=1, steps=24, speed=0.0 f/s, elapsed=0:00:00
Episode 260300: reward=-0, steps=19, speed=0.0 f/s, elapsed=0:00:00
Reached maximum iterations: 5000000. Stopping training.
5000000: tst: {'episode_reward': 1.4891418213834122, 'episode_steps': 43.21, 'order_profits': 1.503887214891026, 'order_steps': 26.78}
5000000: val: {'episode_reward': -0.2005800509043571, 'episode_steps': 25.05, 'order_profits': -0.19988482720452477, 'order_steps': 13.58}
```

* 학습했던 데이터와 동일한 데이터로 모델 성능 시험  
학습에 사용했던 데이터로 시험함으로 성능이 잘 나옴
```
$ python run_model.py -d data/ohlcv_min_000250_07-08.csv -m saves/simple-OHLCV/mean_value-0.992.data -n OHLCV -b 10

100: reward=5.520
200: reward=7.195
300: reward=8.154
400: reward=11.120
500: reward=11.795
600: reward=13.508
700: reward=16.636
800: reward=24.308
900: reward=28.888
```

* 학습에 사용하지 않은 validation 데이터로 모델 성능 시험  
성능이 잘 나오지 않음
```
$ python run_model.py -d data/ohlcv_min_000250_09-10.csv -m saves/simple-OHLCV/mean_value-0.992.data -n OHLCV -b 10
100: reward=-0.945
200: reward=-1.312
300: reward=-0.832
400: reward=-1.976
500: reward=-3.246
600: reward=-4.649
700: reward=-5.442
800: reward=-10.017
900: reward=-11.654
1000: reward=-12.654
1100: reward=-12.221
1200: reward=-8.473
1300: reward=-10.973
1400: reward=-11.936
1500: reward=-13.662
1600: reward=-20.152
1700: reward=-18.361
1800: reward=-19.104
1900: reward=-20.659
2000: reward=-21.677
2100: reward=-21.321
```
### 시험 결과
학습에 사용되었던 데이터로 했을때는 성능이 향상되는 것으로 보아서 학습의 유효성은 있으나
학습에 사용되지 않은 데이터로는 성능이 아주 좋지 않음으로 현재의 모델 아키텍처 또는 학습 방법으로는 많은 부족한 점이 있어 보임
