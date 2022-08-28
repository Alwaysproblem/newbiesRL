import gym

env = gym.make("CartPole-v1")
observation = env.reset()

for _ in range(1000):
  action = env.action_space.sample()
  observation, reward, done, info = env.step(action)
  env.render()
  if done:
    observation = env.reset()
env.close()