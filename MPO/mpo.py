"""MPO implementation with pytorch."""
import numpy as np
import torch
from torch import nn
from torch.distributions import kl_divergence
from torch.distributions import Categorical
from torch.nn import functional as F

from util.agent import Agent
from util.buffer import ReplayBuffer, Trajectory


def standardize(v):
  """Method to standardize a rank-1 np array."""
  assert len(v) > 1, "Cannot standardize vector of size 1"
  v_std = (v - v.mean()) / (v.std() + 1e-08)
  return v_std


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    nn.init.orthogonal_(self.fc_policy.weight, gain=init_policy_weight_gain)

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
    self.action_space = action_space
    self.fc1 = nn.Linear(state_dim + action_space, fc1_unit)
    self.fc1_ln = nn.LayerNorm(fc1_unit)
    self.fc2 = nn.Linear(fc1_unit, fc2_unit)
    self.fc2_ln = nn.LayerNorm(fc2_unit)
    self.fc3 = nn.Linear(fc2_unit, 1)

    nn.init.orthogonal_(self.fc1.weight, gain=init_weight_gain)
    nn.init.orthogonal_(self.fc2.weight, gain=init_weight_gain)
    nn.init.orthogonal_(self.fc3.weight, gain=init_value_weight_gain)

    nn.init.constant_(self.fc1.bias, init_bias)
    nn.init.constant_(self.fc2.bias, init_bias)
    nn.init.constant_(self.fc3.bias, init_bias)

  def forward(self, x, y):
    """
    Build a network that maps state -> action values.
    """
    y = F.one_hot(y, self.action_space).squeeze().float().to(device)
    x = torch.concat([x, y], dim=1)
    x = F.relu(self.fc1_ln(self.fc1(x)))
    x = F.relu(self.fc2_ln(self.fc2(x)))
    return self.fc3(x)


class MPOAgent(Agent):
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
      grad_clip=0.5,
      init_eta=1.0,
      lr_eta=0.001,
      eta_epsilon=0.1,
      action_sample_round=10,
      kl_epsilon=0.01,
      kl_alpha=1.,
      kl_alpha_max=1.0,
      kl_clip_min=0.0,
      kl_clip_max=1.0,
      improved_policy_iteration=5,
      update_tau=0.005,
      seed=0,
  ):

    self.state_dims = state_dims
    self.action_space = action_space
    self.gamma = gamma
    self.batch_size = batch_size
    self.epsilon = epsilon
    self.seed = np.random.seed(seed)
    self.lr_actor = lr_actor
    self.lr_critic = lr_critic
    self.lr_eta = lr_eta
    self.grad_clip = grad_clip
    self.dist_class = Categorical
    self.eta_epsilon = eta_epsilon
    self.action_sample_round = action_sample_round
    self.kl_epsilon = kl_epsilon
    self.kl_alpha_scaler = kl_alpha
    self.kl_alpha = torch.tensor(0., requires_grad=False).to(device)
    self.kl_clip_min = kl_clip_min
    self.kl_clip_max = kl_clip_max
    self.kl_alpha_max = kl_alpha_max
    self.update_tau = update_tau
    self.improved_policy_iteration = improved_policy_iteration

    # Actor Network
    self.actor = Actor(state_dims, action_space).to(device)
    self.actor_target = Actor(state_dims, action_space).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())

    #Q Network
    self.critic = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target = Critic(self.state_dims, self.action_space).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())

    self.actor_optimizer = torch.optim.Adam(
        self.actor.parameters(), lr=self.lr_actor
    )
    self.critic_optimizer = torch.optim.Adam(
        self.critic.parameters(), lr=self.lr_critic
    )

    self.eta = torch.tensor(init_eta).to(device)
    self.eta.requires_grad = True
    self.eta_optimizer = torch.optim.Adam([self.eta], lr=self.lr_eta)

    # Replay memory
    self.memory = ReplayBuffer(max_size=mem_size)
    self.forget_experience = forget_experience
    self.val_loss = nn.MSELoss()

  def learn(self, iteration: int = 10, replace=True):
    """Update value parameters using given batch of experience tuples."""
    polcy_loss, val_loss, eta_loss = np.nan, np.nan, np.nan
    if len(self.memory) < iteration:
      return polcy_loss, val_loss, eta_loss

    polcy_loss = []
    val_loss = []
    eta_loss = []
    trajectories = self.memory.sample_from(
        num_samples=iteration, replace=replace
    )
    if not trajectories:
      return polcy_loss, val_loss, eta_loss
    for trajectory in trajectories:
      polcy_loss_, val_loss_, eta_loss_ = self._learn(trajectory)
      polcy_loss.append(polcy_loss_.cpu().data.numpy())
      val_loss.append(val_loss_.cpu().data.numpy())
      eta_loss.append(eta_loss_.cpu().data.numpy())

    return (
        np.array(polcy_loss).mean(),
        np.array(val_loss).mean(),
        np.array(eta_loss).mean(),
    )

  def policy_evaluation(
      self, states, actions, rewards, next_states, terminates
  ):
    self.critic.train()
    self.critic_target.eval()

    # sampled actions from π(aₜ₊₁|sₜ₊₁)
    _, dist = self.action(next_states, target_policy=True)
    sampled_actions = dist.sample().reshape(actions.shape)

    # Compute the target Q value with
    # rewards + gamma * Q(sₜ₊₁, {sampled actions})
    target_q = self.critic_target.forward(next_states, sampled_actions)
    target_q = rewards + ((1 - terminates) * self.gamma * target_q).detach()

    # Get current Q estimate
    current_q = self.critic.forward(states, actions)

    val_loss = self.val_loss(current_q, target_q)
    self.critic_optimizer.zero_grad()
    val_loss.backward()
    nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
    self.critic_optimizer.step()

    return val_loss

  def find_action_weights(self, states, actions):
    _, action_dist = self.action(states, target_policy=True)
    sample_actions = []
    for _ in range(self.action_sample_round):
      sample_action = action_dist.sample().reshape(actions.shape)
      sample_actions.append(sample_action)
    sample_actions = torch.cat(sample_actions, dim=0)  # shape [BxN, action_dim]
    tiled_states = torch.tile(
        states, (self.action_sample_round, 1)
    )  # shape [BxN, state_dim]
    target_q = self.critic_target.forward(
        tiled_states, sample_actions
    )  # shape [BxN, 1]
    target_q = target_q.reshape(-1, self.action_sample_round).detach(
    )  # shape [B, N]

    # This is for numberic stability.
    # The original code should be like:
    # eta_loss = self.eta * self.eta_epsilon + self.eta * torch.log(
    #     torch.exp(target_q / self.eta).mean(dim=-1)
    # ).mean()

    max_q = target_q.max(dim=-1, keepdim=True).values
    eta_loss = self.eta * self.eta_epsilon + self.eta * torch.log(
        torch.exp((target_q - max_q) / self.eta).mean(dim=-1)
    ).mean() + torch.mean(max_q)

    self.eta_optimizer.zero_grad()
    eta_loss.backward()
    nn.utils.clip_grad_norm_(self.eta, self.grad_clip)
    self.eta_optimizer.step()

    action_weights = torch.softmax(target_q / self.eta, dim=-1)  # shape [B, N]

    return action_weights, sample_actions, tiled_states, eta_loss

  def fit_an_improved_policy(
      self, action_weights, sample_actions, tiled_states
  ):
    _, action_dist = self.action(tiled_states, mode="train")
    log_prob = action_dist.log_prob(sample_actions.detach().T
                                    ).T.reshape(-1, self.action_sample_round)
    policy_loss = torch.mean(log_prob * action_weights.detach())

    with torch.no_grad():
      _, action_dist_old = self.action(tiled_states, target_policy=True)

    kl = kl_divergence(action_dist_old, action_dist).mean()
    kl = torch.clamp(kl, min=self.kl_clip_min, max=self.kl_clip_max)

    if self.kl_alpha_scaler > 0:
      self.kl_alpha -= self.kl_alpha_scaler * (self.kl_epsilon - kl).detach()
      self.kl_alpha = torch.clamp(
          self.kl_alpha, min=1e-8, max=self.kl_alpha_max
      )

    policy_loss = -(policy_loss + self.kl_alpha * (self.kl_epsilon - kl))

    self.actor_optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
    self.actor_optimizer.step()

    return policy_loss

  def policy_improvement(self, states, actions):
    # step 2
    (action_weights, sample_actions, tiled_states,
     eta_loss) = self.find_action_weights(states, actions)
    # step 3
    for _ in range(self.improved_policy_iteration):
      policy_loss = self.fit_an_improved_policy(
          action_weights, sample_actions, tiled_states
      )
    return policy_loss, eta_loss

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
    terminates = torch.from_numpy(np.vstack([e.done for e in trajectory])
                                  ).float().to(device)

    val_loss = self.policy_evaluation(
        states, actions, rewards, next_states, terminates
    )

    policy_loss, eta_loss = self.policy_improvement(states, actions)

    # update target networks
    self.update_critic_target_network()
    self.update_actor_target_network()

    return policy_loss, val_loss, eta_loss

  def action(self, state, mode="eval", target_policy=False):
    actor = self.actor_target if target_policy else self.actor
    if not target_policy:
      if mode == "train":
        actor.train()
      else:
        actor.eval()
    else:
      actor.eval()

    pi = actor.forward(state)
    dist = self.dist_class(pi)
    action = dist.sample()
    self.dist = dist
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

  def log_prob(self, action):
    return self.dist.log_prob(torch.Tensor([action]).to(device)
                              ).data.cpu().item()

  def remember(self, scenario: Trajectory):
    self.memory.enqueue(scenario)

  def soft_update(self, local_model, target_model):
    """
      Soft update model parameters.
      θ_target = τ * θ_local + (1 - τ) * θ_target
      Token from
      https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/exercise/dqn_agent.py
    """
    for target_param, local_param in zip(
        target_model.parameters(), local_model.parameters()
    ):
      target_param.data.copy_(
          self.update_tau * local_param.data +
          (1.0 - self.update_tau) * target_param.data
      )

  def update_critic_target_network(self):
    self.soft_update(self.critic, self.critic_target)

  def update_actor_target_network(self):
    self.soft_update(self.actor, self.actor_target)
