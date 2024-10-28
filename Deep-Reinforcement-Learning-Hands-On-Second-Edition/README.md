# `Dueling DQN`


## 설치

### 필요한 패키지
```
pytorch-ignite           0.5.1
torch                    2.5.0

```

* 패키지 설치  
```
pip install pandas numpy matplotlib gym torch tensorboardX ptan pytorch-ignite opencv-python
pip install opencv-python
```

* gym 버전에 따른 오류 수정
```
  File "~/venv/lib/python3.10/site-packages/ptan/experience.py", line 176, in __iter__
    for exp in super(ExperienceSourceFirstLast, self).__iter__():
  File "~/venv/lib/python3.10/site-packages/ptan/experience.py", line 94, in __iter__
    next_state, r, is_done, _ = env.step(action_n[0])
ValueError: too many values to unpack (expected 4)
```
  * experience.py 소스 코드를 직접 수정  
```
next_state, r, is_done, _, _ = env.step(action_n[0])
```
