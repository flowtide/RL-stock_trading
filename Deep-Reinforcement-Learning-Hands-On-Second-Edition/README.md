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

* 실행 로그 예  
```
703000: Best mean value updated 0.565 -> 0.575
704000: Best mean value updated 0.575 -> 0.603
710000: tst: {'episode_reward': 1.9369473235378316, 'episode_steps': 60.11, 'order_profits': 1.9621413079790497, 'order_steps': 18.27}
710000: val: {'episode_reward': -0.5275394627695476, 'episode_steps': 134.41, 'order_profits': -0.5173107509361136, 'order_steps': 22.04040404040404}
720000: tst: {'episode_reward': 1.8840912843513657, 'episode_steps': 41.39, 'order_profits': 1.8990451628222726, 'order_steps': 13.07}
720000: val: {'episode_reward': -0.6230097443726101, 'episode_steps': 105.24, 'order_profits': -0.6474673000412609, 'order_steps': 12.927835051546392}
728000: Best mean value updated 0.603 -> 0.606
730000: Best mean value updated 0.606 -> 0.627
730000: tst: {'episode_reward': 1.5247612091601654, 'episode_steps': 56.38, 'order_profits': 1.5385894033246326, 'order_steps': 22.58}
730000: val: {'episode_reward': -0.08247111678089096, 'episode_steps': 87.99, 'order_profits': -0.07266313432956673, 'order_steps': 19.282828282828284}
737000: Best mean value updated 0.627 -> 0.637
740000: tst: {'episode_reward': 1.558767480865891, 'episode_steps': 45.79, 'order_profits': 1.56748012481607, 'order_steps': 17.07}
740000: val: {'episode_reward': -0.4258539956971336, 'episode_steps': 115.31, 'order_profits': -0.4204420484278508, 'order_steps': 9.404040404040405}
750000: tst: {'episode_reward': 1.6260890687164709, 'episode_steps': 45.63, 'order_profits': 1.637533701091149, 'order_steps': 23.3}
750000: val: {'episode_reward': 0.37982448498590266, 'episode_steps': 119.36, 'order_profits': 0.42384544168667704, 'order_steps': 26.171717171717173}
760000: tst: {'episode_reward': 1.2089370968439022, 'episode_steps': 35.03, 'order_profits': 1.1883550124527833, 'order_steps': 12.65}
760000: val: {'episode_reward': -0.288170662150584, 'episode_steps': 112.16, 'order_profits': -0.2961268638623724, 'order_steps': 5.714285714285714}
770000: tst: {'episode_reward': 1.4685933573016627, 'episode_steps': 44.03, 'order_profits': 1.4836248005864645, 'order_steps': 13.59}
770000: val: {'episode_reward': -0.157410110824707, 'episode_steps': 92.98, 'order_profits': -0.1608321959432969, 'order_steps': 9.646464646464647}
778000: Best mean value updated 0.637 -> 0.679
780000: tst: {'episode_reward': 1.1169638412550018, 'episode_steps': 46.05, 'order_profits': 1.1262456595353234, 'order_steps': 24.35}
780000: val: {'episode_reward': 0.21005969162780289, 'episode_steps': 110.85, 'order_profits': 0.20598168796100325, 'order_steps': 20.448979591836736}
781000: Best mean value updated 0.679 -> 0.708
783000: Best mean value updated 0.708 -> 0.731
```

* GPT로 실행 로그 분석  
```
740000: val: {'episode_reward': -0.4258539956971336, 'episode_steps': 115.31, 'order_profits': -0.4204420484278508, 'order_steps': 9.404040404040405}
```
Understanding Each Component:

Iteration Number:

740000:
This indicates that the output corresponds to iteration 740,000 in the training process.
At this point, the agent has completed 740,000 training iterations (updates to the neural network).

Dataset:

val:
The results are from the validation dataset.
The validation set is used to evaluate the agent's performance on unseen data to assess generalization and prevent overfitting.
Metrics Reported:

'episode_reward': -0.4258539956971336:
This is the average total reward accumulated over episodes in the validation set.
A negative value indicates that, on average, the agent is losing money during these episodes.

'episode_steps': 115.31:
This is the average number of steps per episode.
The agent's episodes in the validation set last about 115.31 steps on average.

'order_profits': -0.4204420484278508:
This represents the average cumulative profit from executed orders during the episodes.
A negative value indicates that the agent's trades are, on average, unprofitable.

'order_steps': 9.404040404040405:
This is the average number of steps per order.
On average, the agent holds a position for about 9.4 steps before closing it.
