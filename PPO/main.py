"""main executable file for ppo"""
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
from PPO.ppo import PPOAgent as PPO_torch


# pylint: disable=unused-argument
def main(
    game="LunarLander-v2",
    backend="torch",  # "torch", "tf", "jax
    gamma=0.995,
    lr_actor=0.001,
    lr_critic=0.01,
    batch_size=32,
    n_steps=0,
    gae_lambda=0.95,
    clip_eps=0.2,
    num_workers=32,
    beta=0.01,
    iteration=1000,
    n_episodes=20000,
    mem_size=100000,
    forget_experience=False,
    max_t=500,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.996,
    time_interval="25ms",
    dump_gif_dir=None,
    log_level="INFO",
    score_term_rules=None,
    tensorboard_dir="logs",
    tensorboard_write_all=True,
    **kwargs
):
  backends = {"torch": PPO_torch, "jax": None, "tf2": None}
  torch.manual_seed(0)
  np.random.seed(0)
  logging.basicConfig(level=log_level.upper())

  # pylint: disable=line-too-long
  scores = []  # list containing score from each episode
  scores_window = deque(maxlen=100)  # last 100 scores
  eps = eps_start

  env = gym.make(game, render_mode="rgb_array")
  env = TrainMonitor(
      env,
      tensorboard_dir=tensorboard_dir,
      tensorboard_write_all=tensorboard_write_all
  )

  agent_class = backends.get(backend, None)
  if agent_class is None:
    raise NotImplementedError(
        f"backend {backend} is not implemented yet, "
        f"please choose from {list(backends.keys())}"
    )

  agent = agent_class(
      state_dims=env.observation_space.shape[0],
      action_space=env.action_space.n,
      lr_actor=lr_actor,
      lr_critic=lr_critic,
      gamma=gamma,
      batch_size=batch_size,
      mem_size=mem_size,
      forget_experience=forget_experience,
      n_steps=n_steps,
      gae_lambda=gae_lambda,
      beta=beta,
      clip_eps=clip_eps,
  )
  dump_gif_dir = (
      f"images/{agent.__class__.__name__}/{agent.__class__.__name__}_{{}}.gif"
      if not dump_gif_dir else dump_gif_dir
  )
  score_term_rules = score_term_rules if callable(
      score_term_rules
  ) else lambda s: False
  policy_loss, val_loss = np.nan, np.nan

  for i_episode in range(1, n_episodes + 1):
    agent.update_actor_old()

    for _ in range(num_workers):
      state, _ = env.reset()
      score = 0
      traj = Trajectory()
      for _, _ in enumerate(repeat(0, max_t)):
        action = agent.take_action(state=state)
        next_state, reward, done, _, _ = env.step(action)
        log_prob = agent.log_prob(action)
        traj.enqueue(
            Experience(state, action, reward, next_state, done, log_prob)
        )

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
            f"value loss {val_loss:.2f}",
            end="\r"
        )

      agent.remember(traj)
    policy_loss, val_loss = agent.learn(iteration=iteration)

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
