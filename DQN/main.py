import logging
import gym
import numpy as np
from util import generate_gif
from util.wrappers import TrainMonitor
from util.buffer import Experience
from DQN.dqn_torch import DQNAgent as DQN_torch

DQNAgent = DQN_torch
logging.basicConfig(level=logging.INFO)

def main():
  env = gym.make("CartPole-v1", render_mode='rgb_array')
  env = TrainMonitor(env, tensorboard_dir="./logs", tensorboard_write_all=True)

  observation, info = env.reset()
  agent = DQNAgent(
      state_dims=observation.shape[-1], action_space=env.action_space.n,
      lr=0.001,
      forget_experience=False,
      batch_size=4,
      gamma=0.99,
      # sample_ratio = 0.6
  )

  for t in range(1000):
    s, info = env.reset()
    for _ in range(200):
      action = agent.take_action(state=observation)
      s_nxt, reward, done, truncated, info = env.step(action)
      env.render()
      agent.remember(Experience(s, action, reward, s_nxt, done))
      loss = agent.learn()

      if done:
        break
      s = s_nxt

    if t % 5 == 0:
      agent.update_targe_q()

    print(f"loss: {loss}")

    if t and t % 100 == 0:
      generate_gif(
          env,
          filepath=f"images/DQN/dqn_{t}.gif",
          policy=agent.call,
          # policy=lambda s: agent.take_action(s, 0),
          max_episode_steps=200
      )

  env.close()


if __name__ == "__main__":
  main()
