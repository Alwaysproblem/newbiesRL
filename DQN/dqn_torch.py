"""DQN implementation with pytorch."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from util.buffer import ReplayBuffer
from util.agent import Agent
from util.buffer import Experience


class Q(nn.Module):
  """Added the Q-value network for function approximation."""

  def __init__(self, state_dim, action_space, hidden_size=None):
    super().__init__()
    self.action_space = action_space
    self.hidden_size = ((5, 5, 5) if not hidden_size else hidden_size)

    self.hidden_layers = [
        (nn.Linear(in_size, out_size), nn.ReLU())
        for in_size, out_size in zip((state_dim, ) +
                                     self.hidden_size, self.hidden_size)
    ]
    self.output_layer = nn.Linear(self.hidden_size[-1], action_space)

  def forward(self, state):
    x = state
    for linear_layer, activation_layer in self.hidden_layers:
      x = activation_layer(linear_layer(x))
    o = self.output_layer(x)
    return o

  def q_value(self, state, action):
    return action * self.forward(state)


class DQNAgent(Agent):

  def __init__(
      self,
      state_dims,
      action_space,
      gamma=0.99,
      lr=0.001,
      batch_size=10,
      epsilon=0.01,
      mem_size=None,
      forget_experience=True,
      sample_ratio=None
  ):
    self.state_dims = state_dims
    self.action_space = action_space
    self.gamma = gamma
    self.lr = lr
    self.batch_size = batch_size
    self.epsilon = epsilon
    self.q = Q(state_dim=state_dims, action_space=action_space)
    self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)
    self.q_target = Q(state_dim=state_dims, action_space=action_space)
    self.replay_buffer = ReplayBuffer(max_size=mem_size)
    self.forget_experience = forget_experience
    self.q_target.load_state_dict(self.q.state_dict())
    self.loss = nn.SmoothL1Loss()
    self.sample_ratio = sample_ratio

  def call(self, state):
    with torch.no_grad():
      table = self.q(torch.Tensor(state))
      action = table.argmax()
    return action.item()

  def explore(self, state):
    return np.random.choice(range(self.action_space))

  def _cal_q_loss(self, experiences):
    states = torch.Tensor(np.vstack([e.state for e in experiences]))
    actions = torch.Tensor(np.vstack([e.action for e in experiences]))
    rewards = torch.Tensor(np.vstack([e.reward for e in experiences]))
    next_states = torch.Tensor(np.vstack([e.next_state for e in experiences]))
    dones = torch.Tensor(np.vstack([e.done for e in experiences]))

    with torch.no_grad():
      target_qvalue = rewards + (1 - dones) * self.gamma * torch.max(
          self.q_target(next_states), dim=1, keepdim=True
      )[0]

    actions_onehot = F.one_hot(
        actions.to(torch.int64).squeeze(), self.action_space
    )
    q_values = torch.sum(
        self.q.q_value(states, actions_onehot), axis=1, keepdim=True
    )
    loss = self.loss(q_values, target_qvalue)
    return loss

  def _learn(self):
    experiences = self.replay_buffer.sample_from(
        sample_ratio=self.sample_ratio, drop_samples=self.forget_experience
    )
    loss = self._cal_q_loss(experiences=experiences)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss.item()

  def learn(self):
    return self._learn()

  def remember(self, scenario: Experience):
    self.replay_buffer.enqueue(scenario)

  def update_targe_q(self):
    self.q_target.load_state_dict(self.q.state_dict())

  def take_action(self, state, epsilon=None):
    if not epsilon:
      epsilon = self.epsilon
    if np.random.random(size=()) < self.epsilon:
      action = self.explore(state)
    else:
      with torch.no_grad():
        action = torch.argmax(self.q(torch.Tensor(state))).numpy()
    return action

  def save(self, path):
    torch.save(self.q.state_dict(), path)

  def load(self, path):
    self.q.load_state_dict(torch.load(path))
