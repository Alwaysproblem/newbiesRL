"""main executable file for DQN"""
import logging
import gym
from util import generate_gif
from util.wrappers import TrainMonitor
from util.buffer import Experience
from DQN.dqn_torch import DQNAgent as DQN_torch

DQNAgent = DQN_torch
logging.basicConfig(level=logging.INFO)

EPSILON_DECAY_STEPS = 100


def main():
  env = gym.make("CartPole-v1", render_mode="rgb_array")
  env = TrainMonitor(env, tensorboard_dir="./logs", tensorboard_write_all=True)

  agent = DQNAgent(
      state_dims=env.observation_space.shape[0],
      action_space=env.action_space.n,
      lr=0.001,
      forget_experience=False,
      batch_size=128,
      gamma=0.9,
      # sample_ratio = 0.7
  )

  epsilon = 0.9

  for t in range(10000):
    s, info = env.reset()  # pylint: disable=unused-variable
    epsilon = max(epsilon - epsilon / EPSILON_DECAY_STEPS, 0.1)
    for _ in range(200):
      action = agent.take_action(state=s, epsilon=epsilon)
      s_nxt, reward, done, truncated, info = env.step(action)  # pylint: disable=unused-variable
      env.render()
      agent.remember(Experience(s, action, reward, s_nxt, done))
      loss = agent.learn()

      if done:
        break

      s = s_nxt

      if t % 50 == 0:
        agent.update_targe_q()

    print(f"loss: {loss}")

    if t and t % 100 == 0:
      generate_gif(
          env,
          filepath=f"images/DQN/dqn_{t}.gif",
          policy=lambda s: agent.take_action(s, 0),
          max_episode_steps=200
      )

  env.close()


if __name__ == "__main__":
  main()
