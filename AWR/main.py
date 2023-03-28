"""main executable file for A2C"""
import os
import logging
import gym
import torch
import numpy as np
from util import generate_gif
from util.wrappers import TrainMonitor
from util.buffer import Experience, Trajectory
from collections import deque
# pylint: disable=invalid-name
from AWR.awr import AWRAgent as AWR_torch

Agent = AWR_torch
logging.basicConfig(level=logging.INFO)

torch.manual_seed(0)
np.random.seed(0)

EPSILON_DECAY_STEPS = 100


def main(
    n_episodes=2000, max_t=2000, eps_start=1.0, eps_end=0.01, eps_decay=0.996
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
  env = gym.make("LunarLander-v2", render_mode="rgb_array")
  # env = gym.make("CartPole-v1", render_mode="rgb_array")

  env = TrainMonitor(env, tensorboard_dir="./logs", tensorboard_write_all=True)

  gamma = 0.995
  lr_actor = 0.002
  lr_critic = 0.002
  batch_size = 64
  beta = 0

  gamma = 0.99
  lr_actor = 0.001
  lr_critic = 0.001
  batch_size = 64

  td_lambda = 1.0
  awr_beta = 1.0
  awr_min_weight = 0
  awr_max_weight = 20
  learn_iteration = 100

  agent = Agent(
      state_dims=env.observation_space.shape[0],
      action_space=env.action_space.n,
      lr_actor=lr_actor,
      lr_critic=lr_critic,
      gamma=gamma,
      batch_size=batch_size,
      forget_experience=False,
      beta=beta,
      td_lambda=td_lambda,
      awr_beta=awr_beta,
      awr_min_weight=awr_min_weight,
      awr_max_weight=awr_max_weight,
  )
  dump_gif_dir = f"images/{agent.__class__.__name__}/{agent.__class__.__name__}_{{}}.gif"

  for i_episode in range(1, n_episodes + 1):
    state, _ = env.reset()
    score = 0
    traj = Trajectory()
    for _ in range(max_t):
      action = agent.take_action(state=state)
      next_state, reward, done, _, _ = env.step(action)
      traj.enqueue(Experience(state, action, reward, next_state, done))

      state = next_state
      score += reward

      if done:
        break

      scores_window.append(score)  ## save the most recent score
      scores.append(score)  ## sae the most recent score
      eps = max(eps * eps_decay, eps_end)  ## decrease the epsilon
      print(" " * os.get_terminal_size().columns, end="\r")
      print(
          f"\rEpisode {i_episode}\tAverage Score {np.mean(scores_window):.2f}",
          end="\r"
      )

    agent.remember(traj)
    agent.learn(learn_iteration)

    if i_episode and i_episode % 100 == 0:
      print(" " * os.get_terminal_size().columns, end="\r")
      print(
          f"\rEpisode {i_episode}\tAverage Score {np.mean(scores_window):.2f}"
      )
      generate_gif(
          env,
          filepath=dump_gif_dir.format(i_episode),
          policy=lambda s: agent.take_action(s, 0),
          max_episode_steps=max_t
      )

  return scores


if __name__ == "__main__":
  main()
