"""DQN implementation with pytorch."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from util.agent import Agent
from util.buffer import ReplayBuffer, Trajectory


def standardize(v):
  """Method to standardize a rank-1 np array."""
  assert len(v) > 1, "Cannot standardize vector of size 1"
  v_std = (v - v.mean()) / (v.std() + 1e-08)
  return v_std


class Actor(nn.Module):
  """ Actor (Policy) Model."""

  def __init__(self, state_dim, action_space, seed=0, fc1_unit=64, fc2_unit=64):
    """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
    super().__init__()  ## calls __init__ method of nn.Module class
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_dim, fc1_unit)
    self.fc2 = nn.Linear(fc1_unit, fc2_unit)
    self.fc_policy = nn.Linear(fc2_unit, action_space)

  def forward(self, x):
    """
        Build a network that maps state -> action values.
        """
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    pi = F.softmax(self.fc_policy(x), dim=1)
    return pi


class Critic(nn.Module):
  """ Critic (Policy) Model."""

  def __init__(
      self, state_dim, action_space=1, seed=0, fc1_unit=64, fc2_unit=64
  ):
    """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
    super().__init__()  ## calls __init__ method of nn.Module class
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_dim, fc1_unit)
    self.fc2 = nn.Linear(fc1_unit, fc2_unit)
    self.fc3 = nn.Linear(fc2_unit, action_space)

  def forward(self, x):
    """
        Build a network that maps state -> action values.
        """
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CAgent(Agent):
  """Interacts with and learns form environment."""

  def __init__(
      self,
      state_dims,
      action_space,
      gamma=0.99,
      lr_actor=0.001,
      lr_critic=0.001,
      batch_size=64,
      epsilon=0.01,
      mem_size=None,
      forget_experience=True,
      n_steps=0,
      gae_lambda=None,
      beta=0,
      seed=0
  ):

    self.state_dims = state_dims
    self.action_space = action_space
    self.gamma = gamma
    self.batch_size = batch_size
    self.epsilon = epsilon
    self.seed = np.random.seed(seed)
    self.n_steps = n_steps
    self.gae_lambda = gae_lambda
    self.lr_actor = lr_actor
    self.lr_critic = lr_critic
    self.beta = beta

    #Q- Network
    self.actor = Actor(state_dims, action_space).to(device)
    self.critic = Critic(state_dims).to(device)

    self.actor_optimizer = torch.optim.Adam(
        self.actor.parameters(), lr=self.lr_actor
    )
    self.critic_optimizer = torch.optim.Adam(
        self.critic.parameters(), lr=self.lr_critic
    )

    # Replay memory
    self.memory = ReplayBuffer(max_size=mem_size)

    self.forget_experience = forget_experience

    self.val_loss = nn.MSELoss()
    self.policy_loss = nn.MSELoss()

  def learn(self, trajectory: Trajectory):
    states = torch.from_numpy(np.vstack([e.state for e in trajectory])
                              ).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in trajectory])
                               ).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in trajectory])
                               ).float().to(device)
    next_states = torch.from_numpy(
        np.vstack([e.next_state for e in trajectory])
    ).float().to(device)
    terminates = torch.from_numpy(np.vstack([e.done for e in trajectory])
                                  ).float().to(device)
    if self.n_steps > 0:
      advs, v_targets = self.calc_nstep_advs_v_target(
          states, rewards, next_states, terminates
      )
    else:
      advs, v_targets = self.calc_gae_advs_v_target(
          states, rewards, next_states, terminates
      )

    val_loss = self.val_loss(self.critic.forward(states), v_targets)
    _, action_dist = self.action(state=states)

    policy_loss = torch.mean(
        -advs * action_dist.log_prob(actions.T).T -
        self.beta * action_dist.entropy()
    )

    self.actor_optimizer.zero_grad()
    self.critic_optimizer.zero_grad()
    policy_loss.backward()
    val_loss.backward()
    self.actor_optimizer.step()
    self.critic_optimizer.step()
    return policy_loss, val_loss

  def calc_nstep_advs_v_target(self, states, rewards, next_states, terminates):
    with torch.no_grad():
      next_v_pred = self.critic.forward(next_states[-1])
    v_preds = self.critic.forward(states).detach()
    n_steps_rets = self.calc_nstep_return(
        rewards=rewards, dones=terminates, next_v_pred=next_v_pred
    )
    advs = n_steps_rets - v_preds
    v_targets = n_steps_rets
    return advs, v_targets

  def calc_nstep_return(self, rewards, dones, next_v_pred):
    rets = torch.zeros_like(rewards).to(device)
    if self.n_steps > 0:
      future_ret = next_v_pred
      not_dones = 1 - dones
      for i in reversed(range(self.n_steps)):
        rets[i] = future_ret = rewards[
            i] + self.gamma * future_ret * not_dones[i]

    return rets

  def calc_gae_advs_v_target(self, states, rewards, next_states, terminates):
    if self.gae_lambda is None:
      return np.nan
    with torch.no_grad():
      next_v_pred = self.critic.forward(next_states[-1])
    v_preds = self.critic.forward(states).detach()
    v_preds_all = torch.concat((v_preds, next_v_pred.unsqueeze(0)), dim=0)
    advs = self.calc_gaes(rewards, terminates, v_preds_all)
    v_target = advs + v_preds
    return standardize(advs), v_target

  def calc_gaes(self, rewards, dones, v_preds):
    T = len(rewards)  # pylint: disable=invalid-name
    gaes = torch.zeros_like(rewards, device=device)
    future_gae = torch.tensor(0.0, dtype=rewards.dtype, device=device)
    not_dones = 1 - dones  # to reset at episode boundary by multiplying 0
    deltas = rewards + self.gamma * v_preds[1:] * not_dones - v_preds[:-1]
    coef = self.gamma * self.gae_lambda
    for t in reversed(range(T)):
      gaes[t] = future_gae = deltas[t] + coef * not_dones[t] * future_gae
    return gaes

  def action(self, state, mode="eval"):
    if mode == "train":
      self.actor.train()
    else:
      self.actor.eval()

    pi = self.actor.forward(state)
    dist = Categorical(pi)
    action = dist.sample()
    return action.cpu().data.numpy(), dist

  def take_action(self, state, _=0):
    """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)

    with torch.no_grad():
      action_values, *_ = self.action(state)

    return action_values.item()

  def remember(self, scenario: Trajectory):
    self.memory.enqueue(scenario)