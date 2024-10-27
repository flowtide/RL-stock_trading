import numpy as np
import torch
import torch.nn as nn
import warnings
from typing import Iterable
from datetime import datetime, timedelta

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, global_step_from_engine


@torch.no_grad()
def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.asarray(exp.state)  # Removed copy=False
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)  # The result will be masked anyway
        else:
            last_states.append(np.asarray(exp.last_state))  # Removed copy=False

    return np.asarray(states), np.asarray(actions), np.asarray(rewards, dtype=np.float32), \
           np.asarray(dones, dtype=np.uint8), np.asarray(last_states)


def calc_loss(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_actions = net(next_states_v).max(1)[1]
    next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def batch_generator(buffer, initial, batch_size):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)

def setup_ignite(engine: Engine, exp_source, run_name: str, extra_metrics: Iterable[str] = ()):
    warnings.simplefilter("ignore", category=UserWarning)

    # Initialize Tensorboard logging
    now = datetime.now().isoformat(timespec='minutes')
    logdir = f"runs/{now}-{run_name}"
    tb_logger = TensorboardLogger(log_dir=logdir)

    # Attach RunningAverage of loss to engine
    RunningAverage(output_transform=lambda v: v['loss']).attach(engine, "avg_loss")

    # Log every iteration to TensorBoard
    tb_logger.attach_output_handler(
        engine,
        event_name=Events.ITERATION_COMPLETED(every=1000),
        tag="train",
        metric_names=["avg_loss", "avg_fps"] + list(extra_metrics),
        global_step_transform=global_step_from_engine(engine)
    )

    # Attach custom handler for end of episodes
    @engine.on(Events.ITERATION_COMPLETED)
    def log_episode_metrics(trainer: Engine):
        if getattr(trainer.state, "episode_done", False):
            passed = trainer.state.metrics.get("time_passed", 0)
            print(f"Episode {trainer.state.episode}: reward={trainer.state.episode_reward:.0f}, "
                  f"steps={trainer.state.episode_steps}, "
                  f"speed={trainer.state.metrics.get('avg_fps', 0):.1f} f/s, "
                  f"elapsed={timedelta(seconds=int(passed))}")

    return tb_logger