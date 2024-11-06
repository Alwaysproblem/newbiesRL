"""AWR implementation with pytorch."""
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from util.agent import Agent
from util.buffer import ReplayBuffer, Trajectory
from util.algo import standardize, scale_down_values, scale_up_values


class Actor(nn.Module):
  """ Actor (Policy) Model."""

  def __init__(
      self,
      state_dim,
      action_space,
      seed=0,
      fc1_unit=256,
      fc2_unit=256,
      init_weight_gain=np.sqrt(2),
      init_policy_weight_gain=0.01,
      init_bias=0
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
    self.fc_policy = nn.Linear(fc2_unit, action_space)

    nn.init.orthogonal_(self.fc1.weight, gain=init_weight_gain)
    nn.init.orthogonal_(self.fc2.weight, gain=init_weight_gain)
    nn.init.uniform_(
        self.fc_policy.weight, -init_policy_weight_gain, init_policy_weight_gain
    )

    nn.init.constant_(self.fc1.bias, init_bias)
    nn.init.constant_(self.fc2.bias, init_bias)
    nn.init.constant_(self.fc_policy.bias, init_bias)

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
      self,
      state_dim,
      action_space=1,
      seed=0,
      fc1_unit=256,
      fc2_unit=256,
      init_weight_gain=np.sqrt(2),
      init_value_weight_gain=1,
      init_bias=0
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

    nn.init.xavier_uniform_(self.fc1.weight, gain=init_weight_gain)
    nn.init.xavier_uniform_(self.fc2.weight, gain=init_weight_gain)
    nn.init.xavier_uniform_(self.fc3.weight, gain=init_value_weight_gain)

    nn.init.constant_(self.fc1.bias, init_bias)
    nn.init.constant_(self.fc2.bias, init_bias)
    nn.init.constant_(self.fc3.bias, init_bias)

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
      awr_min_weight=None,
      awr_max_weight=None,
      grad_clip=0.5,
      norm_factor=10,
      value_network_scale=True,
      critic_train_step=10,
      actor_train_step=100,
      l2_loss_weight=0.01,
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

    self.grad_clip = grad_clip
    self.norm_factor = norm_factor
    self.value_network_scale = 1 / (
        1 - self.gamma
    ) if value_network_scale else 1.0

    self.global_cnt = 0
    self.global_mean = 0
    self.global_std = 1

    self.l2_loss_weight = l2_loss_weight

    self.critic_train_step = critic_train_step
    self.actor_train_step = actor_train_step

    #Q- Network
    self.actor = Actor(state_dims, action_space).to(device)
    self.critic = Critic(state_dims).to(device)

    self.actor_optimizer = torch.optim.Adam(
        self.actor.parameters(), lr=self.lr_actor, eps=1e-5
    )
    self.critic_optimizer = torch.optim.Adam(
        self.critic.parameters(), lr=self.lr_critic, eps=1e-5
    )

    # Replay memory
    self.memory = ReplayBuffer(max_size=mem_size)

    self.forget_experience = forget_experience

    self.val_loss = nn.MSELoss()
    self.policy_loss = nn.MSELoss()

    self.scale_up_values = partial(
        scale_up_values,
        mean=0,
        std=self.value_network_scale,
        norm_factor=self.norm_factor
    )
    self.scale_down_values = partial(
        scale_down_values,
        mean=0,
        std=self.value_network_scale,
        norm_factor=self.norm_factor
    )

  def learn(self, iteration: int = 10, replace=True):
    """Update value parameters using given batch of experience tuples."""
    polcy_loss, val_loss = np.nan, np.nan
    if len(self.memory) < self.batch_size:
      return polcy_loss, val_loss

    polcy_loss = []
    val_loss = []
    trajectories = self.memory.sample_from(
        num_samples=self.batch_size, replace=replace
    )
    if not trajectories:
      return np.nan, np.nan

    for _ in range(iteration):
      for trajectory in trajectories:
        val_loss_ = self._update_critic(trajectory)
        val_loss.append(val_loss_.cpu().data.numpy())

      for trajectory in trajectories:
        polcy_loss_ = self._update_actor(trajectory)
        polcy_loss.append(polcy_loss_.cpu().data.numpy())

    return np.array(polcy_loss).mean(), np.array(val_loss).mean()

  def _update_critic(self, trajectory: Trajectory):
    states = torch.from_numpy(np.vstack([e.state for e in trajectory])
                              ).float().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in trajectory])
                               ).float().to(device)

    mcre = self.td_lambda_estimate(rewards, states)
    # mcre = standardize(mcre)
    n_mcre = self.scale_down_values(mcre)
    val_loss = 0.5 * self.val_loss(n_mcre.detach(), self.critic.forward(states))
    self.critic_optimizer.zero_grad()
    val_loss.backward()
    self.critic_optimizer.step()
    return val_loss

  def _update_actor(self, trajectory: Trajectory):
    states = torch.from_numpy(np.vstack([e.state for e in trajectory])
                              ).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in trajectory])
                               ).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in trajectory])
                               ).float().to(device)
    mcre = self.td_lambda_estimate(rewards, states)
    advs = mcre - self.scale_up_values(self.critic.forward(states)).detach()
    advs = standardize(advs)
    adv_weight = torch.clip(
        torch.exp(1 / self.awr_beta * advs), self.awr_min_weight,
        self.awr_max_weight
    )
    _, action_dist = self.action(state=states, mode="train")
    l2_loss = torch.sum(torch.square(action_dist.logits), dim=-1)

    policy_loss = -torch.mean(
        adv_weight.detach() * action_dist.log_prob(actions.T).T
    ) + self.beta * action_dist.entropy().mean(
    ) + self.l2_loss_weight * 0.5 * l2_loss.mean()

    self.actor_optimizer.zero_grad()
    policy_loss.backward()
    # nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
    self.actor_optimizer.step()

    return policy_loss

  def td_lambda_estimate(self, rewards, states):
    # computes td-lambda return of path
    T = len(rewards)  # pylint: disable=invalid-name

    v_preds = self.scale_up_values(self.critic.forward(states)).detach()
    val_t = torch.concat((v_preds, torch.Tensor([[0]]).to(device)), dim=0)

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
