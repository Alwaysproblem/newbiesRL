import gymnasium as gym
from util import generate_gif
from util.wrappers import TrainMonitor

# env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = TrainMonitor(env, tensorboard_dir="./logs", tensorboard_write_all=True)

observation, _ = env.reset()

for t in range(10):
  observation, _ = env.reset()
  for _ in range(200):
    action = env.action_space.sample()
    observation, reward, done, truncated, info = env.step(action)
    env.render()

    if done:
      break
  generate_gif(env, filepath=f"random{t}.gif")

env.close()
