"""main executable file for xawr"""
import os
import logging
from itertools import repeat
import gymnasium as gym
import torch
import numpy as np
from util import generate_gif
from util.wrappers import TrainMonitor
from util.buffer import Experience, Trajectory
from collections import deque
# pylint: disable=invalid-name
from XAWR.xawr import XAWRAgent as XAWR_torch

Agent = XAWR_torch
logging.basicConfig(level=logging.INFO)

torch.manual_seed(0)
np.random.seed(0)

EPSILON_DECAY_STEPS = 100


def main(
    n_episodes=20000,
    max_t=500,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.996,
    score_term_rules=lambda s: False,
    time_interval="25ms"
):
  # pylint: disable=line-too-long
  """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon

    """
  scores = []  # list containing score from each episode
  scores_window = deque(maxlen=100)  # last 100 scores
  eps = eps_start
  env = gym.make("LunarLander-v3", render_mode="rgb_array")
  # env = gym.make("CartPole-v1", render_mode="rgb_array")
  # max_t = 200

  env = TrainMonitor(env, tensorboard_dir="./logs", tensorboard_write_all=True)

  gamma = 0.995
  lr_actor = 0.001
  lr_critic = 0.001
  batch_size = 64
  beta = 0.01

  td_lambda = 0.95
  awr_beta = 0.05
  awr_min_weight = None
  awr_max_weight = 20
  learn_iteration = 10
  num_workers = 32

  grad_clip = 0.5
  norm_factor = 10
  value_network_scale = True
  l2_loss_weight = 0.01

  gumbel_loss_beta = 2
  gumbel_loss_clip = None

  agent = Agent(
      state_dims=env.observation_space.shape[0],
      action_space=env.action_space.n,
      lr_actor=lr_actor,
      lr_critic=lr_critic,
      gamma=gamma,
      batch_size=batch_size,
      forget_experience=False,
      mem_size=100000,
      beta=beta,
      td_lambda=td_lambda,
      awr_beta=awr_beta,
      awr_min_weight=awr_min_weight,
      awr_max_weight=awr_max_weight,
      grad_clip=grad_clip,
      norm_factor=norm_factor,
      value_network_scale=value_network_scale,
      l2_loss_weight=l2_loss_weight,
      gumbel_loss_beta=gumbel_loss_beta,
      gumbel_loss_clip=gumbel_loss_clip,
  )
  dump_gif_dir = f"images/{agent.__class__.__name__}/{agent.__class__.__name__}_{{}}.gif"

  policy_loss, val_loss = np.nan, np.nan

  for i_episode in range(1, n_episodes + 1):
    for _ in range(num_workers):
      state, _ = env.reset()
      score = 0
      traj = Trajectory()
      for _, _ in enumerate(repeat(0, max_t)):
        action = agent.take_action(state=state)
        next_state, reward, done, _, _ = env.step(action)
        traj.enqueue(Experience(state, action, reward, next_state, done))

        state = next_state
        score += reward

        if done or score_term_rules(score):
          break

        scores_window.append(score)  ## save the most recent score
        scores.append(score)  ## sae the most recent score
        eps = max(eps * eps_decay, eps_end)  ## decrease the epsilon
        print(" " * os.get_terminal_size().columns, end="\r")
        print(
            f"\rEpisode {i_episode}\t"
            f"Average Score {np.mean(scores_window):.2f}\t"
            f"policy loss {policy_loss:.9f}\t"
            f"value loss {val_loss:.9f}",
            end="\r"
        )

      agent.remember(traj)

    policy_loss, val_loss = agent.learn(learn_iteration)

    if i_episode and i_episode % 100 == 0:
      print(" " * os.get_terminal_size().columns, end="\r")
      print(
          f"\rEpisode {i_episode}\tAverage Score {np.mean(scores_window):.2f}"
      )
      generate_gif(
          env,
          filepath=dump_gif_dir.format(i_episode),
          policy=lambda s: agent.take_action(s, 0),
          duration=float(time_interval.split("ms")[0]),
          max_episode_steps=max_t
      )

  return scores


if __name__ == "__main__":
  main()
