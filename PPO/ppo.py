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

  def __init__(
      self, state_dim, action_space, seed=0, fc1_unit=256, fc2_unit=256
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
      self, state_dim, action_space=1, seed=0, fc1_unit=256, fc2_unit=256
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


class PPOAgent(Agent):
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
      clip_eps=0.2,
      beta=0,
      seed=0,
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
    self.clip_eps = clip_eps

    #Q- Network
    self.actor = Actor(state_dims, action_space).to(device)
    self.actor_old = Actor(state_dims, action_space).to(device)
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

    advs, v_targets = self.calc_adv_and_v_target(
        states, rewards, next_states, terminates
    )

    val_loss = self.val_loss(self.critic.forward(states), v_targets.detach())
    # compute the old policy distribution
    with torch.no_grad():
      _, action_dist_old = self.action(
          state=states, actor_old=True, mode="eval"
      )
      log_prob_old = action_dist_old.log_prob(actions.T).detach()

    # compute the policy distribution
    _, action_dist = self.action(state=states, actor_old=False, mode="train")

    # For implementation of the π(aₜ|sₜ) / π(aₜ|sₜ)[old]
    # Here, we use the exp(log(π(aₜ|sₜ)) - log(π(aₜ|sₜ)[old]))
    importance_ratio = torch.exp(
        action_dist.log_prob(actions.T) - log_prob_old
    ).T
    # importance_ratio = torch.exp(
    #     action_dist.log_prob(actions.T) - log_prob_old
    # ).T

    # clip it into (1 - ϵ) (1 + ϵ)
    clip_ratio = torch.clamp(
        importance_ratio, min=1 - self.clip_eps, max=1 + self.clip_eps
    )

    j_clip = -torch.min(importance_ratio, clip_ratio) * advs.detach()

    policy_loss = torch.mean(j_clip - self.beta * action_dist.entropy())

    self.actor_optimizer.zero_grad()
    self.critic_optimizer.zero_grad()
    policy_loss.backward()
    val_loss.backward()
    self.actor_optimizer.step()
    self.critic_optimizer.step()
    return policy_loss, val_loss

  def calc_adv_and_v_target(self, states, rewards, next_states, terminates):
    if self.n_steps > 0:
      return self.calc_nstep_advs_v_target(
          states, rewards, next_states, terminates
      )
    else:
      return self.calc_gae_advs_v_target(
          states, rewards, next_states, terminates
      )

  def calc_nstep_advs_v_target(self, states, rewards, next_states, terminates):
    """calculate the n-stpes advantage and V_target.

    Args:
        states: the current states, shape [batch size, states shape]
        rewards: the rewards shape: [batch size, 1]
        next_states: the next_states after states, shape
        terminates: this will specify the game status for each states


    Returns:
        advantage : the advantage for action, shape: [batch size, 1]
        v_targets : the target of value function , shape: [batch size, 1]
    """
    with torch.no_grad():
      next_v_pred = self.critic.forward(next_states)
    v_preds = self.critic.forward(states).detach()
    n_steps_rets = self.calc_nstep_return(
        rewards=rewards, dones=terminates, next_v_pred=next_v_pred
    )
    advs = n_steps_rets - v_preds
    v_targets = n_steps_rets
    return standardize(advs), v_targets

  def calc_nstep_return(self, rewards, dones, next_v_pred):
    T = len(rewards)  #pylint: disable=invalid-name
    rets = torch.zeros_like(rewards).to(device)
    _ = 1 - dones

    for i in range(T):
      rets[i] = torch.unsqueeze(
          self.gamma ** torch.arange(len(rewards[i:min(self.n_steps + i, T)])
                                     ).to(device),
          dim=0
      ) @ rewards[i:min(self.n_steps + i, T)]

    if T > self.n_steps:
      value_n_steps = self.gamma ** self.n_steps * next_v_pred[self.n_steps:]
      rets = torch.cat([
          value_n_steps,
          torch.zeros(size=(self.n_steps, 1)).to(device)
      ]) + rets

    return rets

  def calc_gae_advs_v_target(self, states, rewards, next_states, terminates):
    """calculate the GAE (Generalized Advantage Estimation) and V_target.

    Args:
        states: the current states, shape [batch size, states shape]
        rewards: the rewards shape: [batch size, 1]
        next_states: the next_states after states, shape
        terminates: this will specify the game status for each states


    Returns:
        advantage : the advantage for action, shape: [batch size, 1]
        v_targets : the target of value function , shape: [batch size, 1]
    """
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

  def action(self, state, mode="eval", actor_old=True):
    actor = self.actor_old if actor_old else self.actor
    if mode == "train":
      actor.train()
    else:
      actor.eval()

    pi = actor.forward(state)
    dist = Categorical(pi)
    action = dist.sample()
    return action.cpu().data.numpy(), dist

  def take_action(self, state, actor_old=True, _=0):
    """Returns action for given state as per current policy
        Params
        =======
            state (array_like): current state
            epsilon (float): epsilon, for epsilon-greedy action selection
        """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)

    with torch.no_grad():
      action_values, *_ = self.action(state, actor_old=actor_old)

    return action_values.item()

  def remember(self, scenario: Trajectory):
    self.memory.enqueue(scenario)

  def update_actor_old(self):
    return self.actor_old.load_state_dict(self.actor.state_dict())