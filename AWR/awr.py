"""AWR implementation with pytorch."""
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


class AWRAgent(Agent):
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
      td_lambda=1.0,
      beta=0,
      awr_beta=0.5,
      awr_min_weight=0.1,
      awr_max_weight=0.1,
      seed=0
  ):

    self.state_dims = state_dims
    self.action_space = action_space
    self.gamma = gamma
    self.batch_size = batch_size
    self.epsilon = epsilon
    self.seed = np.random.seed(seed)
    self.td_lambda = td_lambda
    self.lr_actor = lr_actor
    self.lr_critic = lr_critic
    self.beta = beta
    self.awr_beta = awr_beta
    self.awr_min_weight = awr_min_weight
    self.awr_max_weight = awr_max_weight

    #Q- Network
    self.actor = Actor(state_dims, action_space).to(device)
    self.critic = Critic(state_dims).to(device)

    self.actor_optimizer = torch.optim.AdamW(
        self.actor.parameters(), lr=self.lr_actor
    )
    self.critic_optimizer = torch.optim.AdamW(
        self.critic.parameters(), lr=self.lr_critic
    )

    # Replay memory
    self.memory = ReplayBuffer(max_size=mem_size)

    self.forget_experience = forget_experience

    self.val_loss = nn.MSELoss()
    self.policy_loss = nn.MSELoss()

  def learn(self, iteration: int = 10):
    """Update value parameters using given batch of experience tuples."""
    polcy_loss, val_loss = np.nan, np.nan
    if len(self.memory) < self.batch_size:
      return polcy_loss, val_loss

    polcy_loss = []
    val_loss = []
    for _ in range(iteration):
      trajectory = self.memory.sample_from()[0]
      polcy_loss_, val_loss_ = self._learn(trajectory)
      polcy_loss.append(polcy_loss_.cpu().data.numpy())
      val_loss.append(val_loss_.cpu().data.numpy())

    return np.array(polcy_loss).mean(), np.array(val_loss).mean()

  def _learn(self, trajectory: Trajectory):
    states = torch.from_numpy(np.vstack([e.state for e in trajectory])
                              ).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in trajectory])
                               ).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in trajectory])
                               ).float().to(device)
    next_states = torch.from_numpy(
        np.vstack([e.next_state for e in trajectory])
    ).float().to(device)
    _ = torch.from_numpy(np.vstack([e.done
                                    for e in trajectory])).float().to(device)

    mcre = self.monte_carlo_rewards_estimates(rewards, states, next_states)
    mcre = standardize(mcre)
    val_loss = self.val_loss(mcre.detach(), self.critic.forward(states))
    _, action_dist = self.action(state=states, mode="train")

    advs = mcre - self.critic.forward(states).detach()
    advs = standardize(advs)
    adv_weight = torch.clip(
        torch.exp(1 / self.awr_beta * advs), self.awr_min_weight,
        self.awr_max_weight
    )

    policy_loss = -torch.mean(
        adv_weight.detach() * action_dist.log_prob(actions.T).T -
        self.beta * action_dist.entropy()
    )

    self.actor_optimizer.zero_grad()
    self.critic_optimizer.zero_grad()
    policy_loss.backward()
    val_loss.backward()
    self.actor_optimizer.step()
    self.critic_optimizer.step()
    return policy_loss, val_loss

  def monte_carlo_rewards_estimates(self, rewards, states, next_states):
    # computes td-lambda return of path
    T = len(rewards)  # pylint: disable=invalid-name

    v_preds = self.critic.forward(states).detach()
    next_v_pred = self.critic.forward(next_states[-1]).detach()
    val_t = torch.concat((v_preds, next_v_pred.unsqueeze(0)), dim=0)

    assert len(val_t) == T + 1

    return_t = torch.zeros(T)
    last_val = rewards[-1] + self.gamma * val_t[-1]
    return_t[-1] = last_val

    for i in reversed(range(0, T - 1)):
      curr_r = rewards[i]
      next_ret = return_t[i + 1]
      curr_val = curr_r + self.gamma * ((1.0 - self.td_lambda) * val_t[i + 1] +
                                        self.td_lambda * next_ret)
      return_t[i] = curr_val

    return return_t.unsqueeze(1).to(device)

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
