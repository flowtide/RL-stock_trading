#!/usr/bin/env python3
import ptan
import pathlib
import argparse
import gym.wrappers
import numpy as np

import torch
import torch.optim as optim

from ignite.engine import Engine, Events
from ignite.contrib.handlers import TensorboardLogger

from lib import environ, data, models, common, validation

from gym.envs.registration import register

register(
    id="StocksEnv-v0",
    entry_point="lib.environ:StocksEnv",
)

SAVES_DIR = pathlib.Path("saves")

EPS_START = 1.0
EPS_FINAL = 0.1
EPS_STEPS = 1_000_000

GAMMA = 0.99

REPLAY_SIZE = 100_000
REPLAY_INITIAL = 10_000
REWARD_STEPS = 2
LEARNING_RATE = 0.0001
STATES_TO_EVALUATE = 1_000

class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, epsilon_start, epsilon_final, epsilon_frames):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_frames = epsilon_frames
        self.frame(0)

    def frame(self, frame):
        epsilon = max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)
        self.epsilon_greedy_selector.epsilon = epsilon

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nocuda", help="Disable cuda", default=True, action="store_false")
    parser.add_argument("-d", "--data", required=True, help="Stocks file or directory for training data")
    parser.add_argument("-v", "--val", required=True, help="Validation data file or directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--max-iterations", type=int, default=5_000_000, help="Maximum training iterations")
    parser.add_argument("--bars-count", type=int, default=50, help="Number of bars to include in the state")
    parser.add_argument("-r", "--run", required=True, help="Run name")
    parser.add_argument("-m", "--model", choices=["DQNConv1D", "SimpleFFDQN"], required=True,
                        help="Choose model architecture: DQNConv1D or SimpleFFDQN")
    args = parser.parse_args()
    device = torch.device("cpu" if args.nocuda else "gpu")

    BATCH_SIZE = args.batch_size  # Use the batch size from command-line arguments
    MAX_ITERATIONS = args.max_iterations  # Use the max iterations from command-line arguments
    BARS_COUNT = args.bars_count  # Use the bars count from command-line arguments
    state_1d = (args.model == "DQNConv1D")  # Determine the state_1d mode based on model choice

    saves_path = SAVES_DIR / f"{args.model}-{args.run}"
    saves_path.mkdir(parents=True, exist_ok=True)

    data_path = pathlib.Path(args.data)
    val_path = pathlib.Path(args.val)

    # Load data directly from file or directory specified in `--data`
    if data_path.is_file():
        stock_data = {"YNDX": data.load_relative(data_path)}
        env = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, state_1d=state_1d)
        env_tst = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, state_1d=state_1d)
    elif data_path.is_dir():
        env = environ.StocksEnv.from_dir(data_path, bars_count=BARS_COUNT, state_1d=state_1d)
        env_tst = environ.StocksEnv.from_dir(data_path, bars_count=BARS_COUNT, state_1d=state_1d)
    else:
        raise RuntimeError("No data to train on")

    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    val_data = {"YNDX": data.load_relative(val_path)}
    env_val = environ.StocksEnv(val_data, bars_count=BARS_COUNT, state_1d=state_1d)

    # Choose model architecture based on the --model argument
    if args.model == "DQNConv1D":
        net = models.DQNConv1D(env.observation_space.shape, env.action_space.n).to(device)
    else:
        net = models.SimpleFFDQN(env.observation_space.shape[0], env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(EPS_START)
    eps_tracker = EpsilonTracker(selector, EPS_START, EPS_FINAL, EPS_STEPS)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, GAMMA, steps_count=REWARD_STEPS)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, REPLAY_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    def process_batch(engine, batch):
        optimizer.zero_grad()
        loss_v = common.calc_loss(batch, net, tgt_net.target_model, gamma=GAMMA ** REWARD_STEPS, device=device)
        loss_v.backward()
        optimizer.step()
        eps_tracker.frame(engine.state.iteration)

        if getattr(engine.state, "eval_states", None) is None:
            eval_states = buffer.sample(STATES_TO_EVALUATE)
            eval_states = [np.asarray(transition.state) for transition in eval_states]
            engine.state.eval_states = np.asarray(eval_states)

        return {
            "loss": loss_v.item(),
            "epsilon": selector.epsilon,
        }

    engine = Engine(process_batch)
    tb = common.setup_ignite(
        engine, exp_source, f"{args.model}-{args.run}", extra_metrics=('values_mean',))

    # Sync target network and log evaluation metrics every 1000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=1000))
    def sync_eval(engine: Engine):
        tgt_net.sync()
        mean_val = common.calc_values_of_states(
            engine.state.eval_states, net, device=device)
        engine.state.metrics["values_mean"] = mean_val
        if getattr(engine.state, "best_mean_val", None) is None:
            engine.state.best_mean_val = mean_val
        if engine.state.best_mean_val < mean_val:
            print(f"{engine.state.iteration}: Best mean value updated "
                  f"{engine.state.best_mean_val:.3f} -> {mean_val:.3f}")
            path = saves_path / f"mean_value-{mean_val:.3f}.data"
            torch.save(net.state_dict(), path)
            engine.state.best_mean_val = mean_val

    # Add a termination condition
    @engine.on(Events.ITERATION_COMPLETED)
    def check_termination_condition(engine: Engine):
        if engine.state.iteration >= MAX_ITERATIONS:
            print(f"Reached maximum iterations: {MAX_ITERATIONS}. Stopping training.")
            engine.terminate()

    # Event handler to track episodes
    @engine.on(Events.ITERATION_COMPLETED)
    def track_episodes(engine: Engine):
        rewards_steps = exp_source.pop_rewards_steps()
        if rewards_steps:
            for reward, steps in rewards_steps:
                engine.state.episode_done = True
                engine.state.episode_reward = reward
                engine.state.episode_steps = steps
                if not hasattr(engine.state, 'episode'):
                    engine.state.episode = 0
                engine.state.episode += 1
        else:
            engine.state.episode_done = False
        
    # Validation handler to evaluate performance every 10,000 iterations
    @engine.on(Events.ITERATION_COMPLETED(every=10000))
    def validate(engine: Engine):
        res = validation.validation_run(env_tst, net, device=device)
        print(f"{engine.state.iteration}: tst: {res}")
        for key, val in res.items():
            engine.state.metrics[key + "_tst"] = val
        res = validation.validation_run(env_val, net, device=device)
        print(f"{engine.state.iteration}: val: {res}")
        for key, val in res.items():
            engine.state.metrics[key + "_val"] = val
        val_reward = res['episode_reward']
        if getattr(engine.state, "best_val_reward", None) is None:
            engine.state.best_val_reward = val_reward
        if engine.state.best_val_reward < val_reward:
            print(f"Best validation reward updated: {engine.state.best_val_reward:.3f} "
                  f"-> {val_reward:.3f}, model saved")
            engine.state.best_val_reward = val_reward
            path = saves_path / f"val_reward-{val_reward:.3f}.data"
            torch.save(net.state_dict(), path)

    # Run the engine
    engine.run(common.batch_generator(buffer, REPLAY_INITIAL, BATCH_SIZE))