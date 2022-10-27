from collections import namedtuple
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util import generate_gif
from util.wrappers import TrainMonitor
from util.buffer import ReplayBuffer
from util.agent import Agent

Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class Q(nn.Module):

  def __init__(self, state_dim, action_space, hidden_size=None):
    super().__init__()
    self.action_space = action_space
    self.hidden_size = ((5, 5, 5) if not hidden_size else hidden_size)

    self.hidden_layers = [
        (nn.Linear(in_size, out_size), nn.GELU())
        for in_size, out_size in zip((state_dim, ) +
                                     self.hidden_size, self.hidden_size)
    ]
    self.output_layer = nn.Linear(self.hidden_layers[-1], action_space)

  def forward(self, state):
    x = state
    for l, a in self.hidden_layers:
      x = a(l(x))
    o = self.output_layer(x)
    return o

  def Q_function(self, state, action):
    return action * self.forward(state)


class DQNAgent(Agent):

  def __init__(
      self, state_dims, action_space, gamma=0.99, lr=0.001, batch_size=10
  ):
    self.state_dims = state_dims
    self.action_space = action_space
    self.gamma = gamma
    self.lr = lr
    self.batch_size = batch_size
    self.Q = Q(state_dim=state_dims, action_space=action_space)
    self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)
    self.Q_target = Q(state_dim=state_dims, action_space=action_space)
    self.replay_buffer = ReplayBuffer()

  def call(self, state):
    with torch.no_grad():
      table = self.Q(state)
      action = table.argmax()
    return action.item()

  def explore(self, state):
    return np.random.choice(range(self.action_space))

  def _cal_q_loss(self, experiences):
    states = torch.Tensor([e.state for e in experiences])
    actions = torch.Tensor([e.action for e in experiences])
    rewards = torch.Tensor([e.reward for e in experiences])
    next_states = torch.Tensor([e.next_state for e in experiences])
    dones = torch.Tensor([e.done for e in experiences])
    with torch.no_grad():
      next_actions = F.one_hot(self.Q(next_states).argmax())
      target_qvalue = rewards + (
          1 - dones
      ) * self.gamma * next_actions * self.Q_target(next_states)

    return F.mse_loss(self.Q.Q_function(states, actions), target_qvalue)

  def learn_from(self):
    experiences = self.replay_buffer.sample_from(
        self.batch_size, drop_samples=True
    )
    loss = self._cal_q_loss(experiences=experiences)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()

  def remember(self, scenario: Experience):
    self.replay_buffer.enqueue(scenario)

  def update_targe_Q(self):
    self.Q_target.load_state_dict(self.Q.state_dict())


def main():
  # env = gym.make("CartPole-v1", render_mode='human')
  env = gym.make("CartPole-v1")
  env = TrainMonitor(env, tensorboard_dir="./logs", tensorboard_write_all=True)

  observation = env.reset()

  for t in range(10):
    observation = env.reset()
    for _ in range(200):
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      env.render()

      if done:
        observation = env.reset()  # noqa: F841
    generate_gif(env, filepath=f"random{t}.gif")

  env.close()


if __name__ == '__main__':
  main()
