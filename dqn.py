import warnings
import numpy as np
from collections import deque
from copy import deepcopy
import gym
from util import generate_gif
from util.wrappers import TrainMonitor

def main():
  # env = gym.make("CartPole-v1", render_mode='human')
  # env = gym.make("CartPole-v1")
  # env = TrainMonitor(env, tensorboard_dir="./logs", tensorboard_write_all=True)

  # observation = env.reset()

  # for t in range(10):
  #   observation = env.reset()
  #   for _ in range(200):
  #     action = env.action_space.sample()
  #     observation, reward, done, info = env.step(action)
  #     # observation, reward, done, info, _ = env.step(action)
  #     env.render()

  #     if done:
  #       observation = env.reset()
  #   generate_gif(env, filepath=f"random{t}.gif")

  # env.close()


if __name__ == '__main__':
  main()
